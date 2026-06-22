#!/usr/bin/env python3
"""Analyze a trained UncExpertFusionHead the way we dissected DINOv2-vs-DPT.

Given a fusion config + trained head + test images (and the per-mode registered_poses.txt produced
by register_eval_expert_modes.py), this reproduces, in one command, the full diagnosis:

  1. THE BAR        - per-mode pose (forced expert i vs fused); does fused beat the best single expert?
  2. REPROJECTION   - per-expert per-patch reprojection error (median + sub-2px tail) and gate weights.
  3. DEPTH          - per-expert camera-frame depth spread + cross-expert depth disagreement.
  4. CORRELATION    - within-expert Spearman(reproj, pose-error) and the between-expert inversion
                      (% images expert has better reproj vs % better pose).
  5. VERDICT        - is the gate aligned or inverted w.r.t. pose? is reprojection a valid cross-expert
                      signal here, or does it disagree with pose?

Example:
  ~/dace_env310/bin/python scripts/analyze_fusion.py \
    --config-path configs_custom/MoGU_N2_frozen_total_tau0_scalar_unc.yaml \
    --head-path  ~/dace/outputs/<session>_head.pt \
    --rgb-glob   "~/dace/datasets/cambridge/shopfacade/test/rgb/*.png" \
    --reg-dir    ~/dace/outputs --session-id <session> \
    --device cuda --use-half --out-json report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Reuse the proven helpers from the diagnostic module (same scripts/ dir).
from diagnose_heads_unc import (
    build_encoders_from_cfg,
    ensure_ace_g_importable,
    expand,
    load_head,
    load_img_tensor,
    print_stage_header,
    run_concat_features,
)

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML required") from exc


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))


def pose_from_registered_line(line: str):
    """Parse 'path qw qx qy qz tx ty tz focal conf' (w2c) -> (name, C_c2w[3], R_c2w[3,3])."""
    from scipy.spatial.transform import Rotation

    p = line.split()
    name = os.path.basename(p[0])
    qw, qx, qy, qz = map(float, p[1:5])
    tx, ty, tz = map(float, p[5:8])
    r_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t_w2c = np.array([tx, ty, tz])
    return name, (-r_w2c.T @ t_w2c), r_w2c.T


def load_registered(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            name, c, r = pose_from_registered_line(line)
            out[name] = (c, r)
    return out


def pose_error(c_est, r_est, gt_c2w) -> tuple[float, float]:
    c_gt, r_gt = gt_c2w[:3, 3], gt_c2w[:3, :3]
    terr = float(np.linalg.norm(c_est - c_gt))
    cosv = np.clip((np.trace(r_est.T @ r_gt) - 1) / 2, -1, 1)
    return terr, float(np.degrees(np.arccos(cosv)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default="~/dace/ace-g")
    p.add_argument("--config-path", required=True)
    p.add_argument("--head-path", required=True)
    p.add_argument("--rgb-glob", required=True, help="Test RGB glob (poses/ and calibration/ siblings expected)")
    p.add_argument("--reg-dir", default=None, help="Dir with {session}__expert{i}/fused_registered_poses.txt")
    p.add_argument("--session-id", default=None, help="Base session id (without the __mode suffix)")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--use-half", action="store_true")
    p.add_argument("--valid-cap-px", type=float, default=100.0)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_stage_header("ANALYZE FUSION")
    ensure_ace_g_importable(args.repo_root)
    from ace_g import data_io, losses, utils

    cfg = yaml.safe_load(expand(args.config_path).read_text(encoding="utf-8"))
    encs = build_encoders_from_cfg(cfg)
    head = load_head(args.repo_root, args.head_path)
    if not hasattr(getattr(head, "config", None), "sanity_check_force_expert"):
        raise SystemExit("Expected an UncExpertFusionHead (config.sanity_check_force_expert).")
    k = int(head.config.num_experts)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.use_half and device == "cuda"
    for i in range(len(encs)):
        encs[i] = encs[i].to(device).eval()
    head = head.to(device).eval()

    paths = sorted(glob.glob(os.path.expanduser(args.rgb_glob)))[: args.num_images]
    if not paths:
        raise SystemExit(f"No images matched: {args.rgb_glob}")
    modes = [f"expert{i}" for i in range(k)] + ["fused"]
    forces = {f"expert{i}": i for i in range(k)}
    forces["fused"] = None

    # ---- optional pose files ----
    reg: dict[str, dict] = {}
    if args.reg_dir and args.session_id:
        for m in modes:
            f = Path(os.path.expanduser(args.reg_dir)) / f"{args.session_id}__{m}_registered_poses.txt"
            if f.exists():
                reg[m] = load_registered(f)
        if reg:
            print(f"Loaded registered poses for modes: {list(reg)}")
        else:
            print("WARNING: no registered_poses files found; pose/correlation sections skipped.")

    cap = float(args.valid_cap_px)
    dists = {m: [] for m in modes}
    depth = {m: [] for m in modes}
    per_img_reproj = {m: {} for m in modes}
    weight_means = np.zeros(k)
    cross_dz = []
    n = 0
    orig = head.config.sanity_check_force_expert

    with torch.no_grad():
        for pth in paths:
            pp = Path(pth)
            x = load_img_tensor(pth, args.max_side).to(device)
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
            pose_c2w = data_io.load_pose(pp.parent.parent / "poses" / f"{pp.stem}.txt")
            w2c = pose_c2w.inverse()[:3, :4].unsqueeze(0).to(device)
            calib = data_io.load_calibration(pp.parent.parent / "calibration" / f"{pp.stem}.txt")
            im = Image.open(pth)
            wo, ho = im.size
            scale = args.max_side / float(max(wo, ho)) if max(wo, ho) > args.max_side else 1.0
            if isinstance(calib, float):
                fx = fy = calib
                cx, cy = wo / 2, ho / 2
            else:
                fx, fy, cx, cy = calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]
            K = torch.eye(3)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx * scale, fy * scale, cx * scale, cy * scale

            sub = int(round(x.shape[-2] / feats.shape[-2]))
            grid = utils.get_pixel_grid(sub).to(device)[:, : feats.shape[-2], : feats.shape[-1]]
            _, hf, wf = grid.shape
            tgt_px = grid.reshape(2, -1).permute(1, 0)
            npat = hf * wf
            tgt_c = torch.zeros((npat, 3), device=device)
            w2c_e = w2c.expand(npat, 3, 4)
            K_e = K.unsqueeze(0).to(device).expand(npat, 3, 3)
            r2, t2 = w2c[0, 2, :3], w2c[0, 2, 3]

            img_d = {}
            for m in modes:
                head.config.sanity_check_force_expert = forces[m]
                with torch.autocast("cuda", enabled=use_half):
                    y, _ = head(feats)
                pred = y.permute(0, 2, 3, 1).reshape(-1, 3).float()
                _, _, _, _, d2d = losses.compute_loss(
                    pred_coords=pred, pred_uncertainties=None, w2c_b34=w2c_e,
                    image_from_camera_b33=K_e, target_pixels=tgt_px, target_coords=tgt_c,
                    supervision_type="2d", use_depth_as_prior=False,
                )
                dn = d2d.detach().float().cpu().numpy().reshape(-1)
                zn = ((pred * r2).sum(-1) + t2).detach().float().cpu().numpy().reshape(-1)
                dists[m].append(dn)
                depth[m].append(zn)
                img_d[m] = dn
                fin = dn[np.isfinite(dn)]
                per_img_reproj[m][pp.name] = float(np.median(fin)) if fin.size else float("nan")
                if forces[m] is None:
                    weight_means += head.last_moe_weights[0].mean(dim=(1, 2, 3)).detach().float().cpu().numpy()
            if k >= 2:
                v = np.isfinite(img_d["expert0"]) & np.isfinite(img_d["expert1"]) & (img_d["expert0"] < cap) & (img_d["expert1"] < cap)
                if v.any():
                    cross_dz.append(np.abs(depth["expert0"][-1][v] - depth["expert1"][-1][v]))
            n += 1
    head.config.sanity_check_force_expert = orig
    weight_means /= max(n, 1)

    # ---- pose per mode ----
    pose_err = {}
    if reg:
        gtdir = Path(paths[0]).parent.parent / "poses"
        for m in modes:
            if m not in reg:
                continue
            errs = {}
            for name, (c, r) in reg[m].items():
                gtf = gtdir / f"{os.path.splitext(name)[0]}.txt"
                if gtf.exists():
                    errs[name] = pose_error(c, r, data_io.load_pose(gtf).numpy())
            pose_err[m] = errs

    report = {"images": n, "k": k, "gate_weight_mean": weight_means.tolist(), "modes": {}}

    print("\n================== 1. THE BAR (pose) ==================")
    best_single = None
    for m in modes:
        d = np.concatenate(dists[m])
        fin = d[np.isfinite(d)]
        med_reproj = float(np.median(fin))
        row = {"reproj_median_px": med_reproj, "frac_lt2px": float((fin < 2).mean())}
        if m in pose_err and pose_err[m]:
            tr = np.array([v[0] for v in pose_err[m].values()])
            ro = np.array([v[1] for v in pose_err[m].values()])
            row["pose_median_cm"] = float(np.median(tr) * 100)
            row["pose_median_deg"] = float(np.median(ro))
            wt = f"gate_w={weight_means[forces[m]]:.3f}" if forces[m] is not None else ""
            print(f"  {m:8s}: pose={row['pose_median_cm']:6.2f}cm/{row['pose_median_deg']:.2f}deg   reproj_med={med_reproj:6.2f}px  %<2px={100*row['frac_lt2px']:4.1f}  {wt}")
            if forces[m] is not None and (best_single is None or row["pose_median_cm"] < best_single[1]):
                best_single = (m, row["pose_median_cm"])
        report["modes"][m] = row

    if best_single and "fused" in pose_err:
        fused_cm = report["modes"]["fused"]["pose_median_cm"]
        verdict = "BEATS" if fused_cm < best_single[1] else "LOSES TO"
        print(f"\n  VERDICT: fused ({fused_cm:.2f}cm) {verdict} best single {best_single[0]} ({best_single[1]:.2f}cm).")

    print("\n========= 2. DEPTH STRUCTURE (z, m) =========")
    for m in modes:
        zv = np.concatenate(depth[m])
        zv = zv[np.isfinite(zv) & (zv > 0.1) & (zv < 1000)]
        report["modes"][m]["depth_std"] = float(np.std(zv)) if zv.size else float("nan")
        print(f"  {m:8s}: std_z={np.std(zv):6.2f}  spread(p10-90)={np.percentile(zv,90)-np.percentile(zv,10):6.2f}")
    if cross_dz:
        dz = np.concatenate(cross_dz)
        report["expert0_vs_1_depth_median_abs_dz_m"] = float(np.median(dz))
        print(f"  expert0 vs expert1 depth disagreement on commonly-good patches: median|dz|={np.median(dz):.3f}m  p90={np.percentile(dz,90):.3f}m")

    if reg and k >= 2 and "expert0" in pose_err and "expert1" in pose_err:
        print("\n===== 3. CORRELATION (objective vs metric) =====")
        for m in modes:
            if m not in pose_err:
                continue
            names = [nm for nm in per_img_reproj[m] if nm in pose_err[m]]
            rp = [per_img_reproj[m][nm] for nm in names]
            tr = [pose_err[m][nm][0] for nm in names]
            s = spearman(np.array(rp), np.array(tr))
            report["modes"][m]["spearman_reproj_pose"] = s
            print(f"  within {m:8s}: Spearman(reproj, pose-error) = {s:.3f}")
        common = [nm for nm in pose_err["expert0"] if nm in pose_err["expert1"] and nm in per_img_reproj["expert0"]]
        r0 = np.array([per_img_reproj["expert0"][nm] for nm in common])
        r1 = np.array([per_img_reproj["expert1"][nm] for nm in common])
        p0 = np.array([pose_err["expert0"][nm][0] for nm in common])
        p1 = np.array([pose_err["expert1"][nm][0] for nm in common])
        reproj_better1 = float(np.mean(r1 < r0))
        pose_better1 = float(np.mean(p1 < p0))
        report["between_expert"] = {"expert1_better_reproj_frac": reproj_better1, "expert1_better_pose_frac": pose_better1}
        print(f"\n  between experts (N={len(common)}): expert1 better REPROJ on {100*reproj_better1:.0f}% of images,")
        print(f"                                   but better POSE on only {100*pose_better1:.0f}%.")
        inverted = (reproj_better1 > 0.5) != (pose_better1 > 0.5)
        print("\n  VERDICT: reprojection ordering is " + ("INVERTED vs pose between experts -> the gate's" if inverted else "ALIGNED with pose between experts -> the gate's"))
        print("           reprojection-based signal " + ("CANNOT" if inverted else "can") + " pick the better expert. " +
              ("A 3D/pose-aware gate target is required." if inverted else "A reprojection/uncertainty gate may work."))

    if args.out_json:
        Path(os.path.expanduser(args.out_json)).write_text(json.dumps(report, indent=2))
        print(f"\nwrote_json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
