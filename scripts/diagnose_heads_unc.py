#!/usr/bin/env python3
"""Diagnostics migrated from notebook cells 31-35.

Subcommands:
- compare-means: compare learned head means across two checkpoints
- compare-logs: compare register/eval logs across session prefixes
- gate-weights: inspect per-expert softmax weights for late-fusion heads
- uncertainty-decisions: inspect MoGU/UncExpertFusionHead uncertainty and patch decisions
- gate-last-layer: inspect gating last-layer params, and optionally probe one image
- print-config: print selected head config fields
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

from _common import (
    ScriptError,
    expand,
    find_first_glob,
    parse_text_metrics,
    print_stage_header,
    read_text_if_exists,
    repo_src_dir,
    tail_text,
    write_csv,
)


def ensure_ace_g_importable(repo_root: str | Path) -> None:
    src = repo_src_dir(repo_root)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def setup_wandb_logging(args: argparse.Namespace, job_type: str):
    if not getattr(args, "wandb_log", False):
        return None
    import wandb

    project = getattr(args, "wandb_project", None)
    entity = getattr(args, "wandb_entity", None)

    # Try to infer entity/project from head_path if not provided
    head_path = getattr(args, "head_path", "")
    if str(head_path).startswith("wandb://") and not (project and entity):
        art_path = str(head_path)[len("wandb://"):]
        parts = art_path.split("/")
        if len(parts) >= 3:
            entity = entity or parts[0]
            project = project or parts[1]

    run = wandb.init(project=project, entity=entity, job_type=job_type)
    
    if str(head_path).startswith("wandb://"):
        art_path = str(head_path)[len("wandb://"):]
        run.use_artifact(art_path)
    return run


def load_head(repo_root: str | Path, head_path: str | Path):
    ensure_ace_g_importable(repo_root)
    from ace_g import scr_heads  # type: ignore

    head_path_str = str(head_path)
    if head_path_str.startswith("wandb://"):
        import wandb
        artifact_path = head_path_str[len("wandb://"):]
        print(f"Downloading wandb artifact: {artifact_path} ...")
        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_path)
            download_dir = artifact.download()
            pt_files = list(Path(download_dir).glob("*.pt"))
            if not pt_files:
                raise ScriptError(f"No .pt file found in artifact {artifact_path}")
            p = pt_files[0]
            print(f"Using downloaded artifact head: {p}")
        except Exception as e:
            raise ScriptError(f"Error fetching wandb artifact: {e}")
    else:
        p = expand(head_path)

    if not p.is_file():
        raise ScriptError(f"Missing head checkpoint: {p}")
    return scr_heads.create_head(p)


def cmd_compare_means(args: argparse.Namespace) -> int:
    print_stage_header("COMPARE HEAD MEANS")
    h0 = load_head(args.repo_root, args.head_a).cpu().eval()
    h1 = load_head(args.repo_root, args.head_b).cpu().eval()

    mean0 = h0.mean.view(-1).tolist()
    mean1 = h1.mean.view(-1).tolist()
    max_abs_diff = float((h0.mean - h1.mean).abs().max().item())
    out = {
        "head_a": str(expand(args.head_a)),
        "head_b": str(expand(args.head_b)),
        "mean_a": mean0,
        "mean_b": mean1,
        "max_abs_diff": max_abs_diff,
    }
    print(json.dumps(out, indent=2))
    return 0


def find_log(log_dir: Path, session_prefix: str, kind: str) -> Path | None:
    return find_first_glob(
        [
            log_dir / f"{session_prefix}__{kind}.log",
            log_dir / f"{session_prefix}*__{kind}.log",
            log_dir / f"{session_prefix}*{kind}*.log",
        ]
    )


def cmd_compare_logs(args: argparse.Namespace) -> int:
    print_stage_header("COMPARE REGISTER/EVAL LOGS")
    log_dir = expand(args.log_dir)
    session_pairs: list[tuple[str, str]] = []
    for item in args.session:
        if "=" not in item:
            raise ScriptError(f"Expected --session LABEL=PREFIX, got: {item}")
        label, prefix = item.split("=", 1)
        session_pairs.append((label, prefix))

    rows: list[dict[str, Any]] = []
    tail_bundle: dict[str, str] = {}
    for label, prefix in session_pairs:
        reg_p = find_log(log_dir, prefix, "register")
        eval_p = find_log(log_dir, prefix, "eval")
        reg_txt = read_text_if_exists(reg_p)
        eval_txt = read_text_if_exists(eval_p)
        reg_m = parse_text_metrics(reg_txt)
        eval_m = parse_text_metrics(eval_txt)
        row = {
            "label": label,
            "session_prefix": prefix,
            "register_log": str(reg_p) if reg_p else None,
            "eval_log": str(eval_p) if eval_p else None,
            "reg_present": reg_m.get("present", False),
            "reg_crashed": reg_m.get("crashed", False),
            "reg_registered": reg_m.get("registered_xy", None),
            "eval_present": eval_m.get("present", False),
            "eval_crashed": eval_m.get("crashed", False),
            "median_m": eval_m.get("median_m", None),
            "median_deg": eval_m.get("median_deg", None),
            "mean_m": eval_m.get("mean_m", None),
            "mean_deg": eval_m.get("mean_deg", None),
            "success_pct": eval_m.get("success_pct", None),
        }
        rows.append(row)
        tail_bundle[f"{label}/register"] = tail_text(reg_txt, args.tail_lines)
        tail_bundle[f"{label}/eval"] = tail_text(eval_txt, args.tail_lines)

    print(json.dumps(rows, indent=2))
    print("\n=== LOG TAILS ===")
    for key, tail in tail_bundle.items():
        print(f"\n--- {key} ---")
        print(tail or "(no log / empty)")

    if args.out_csv:
        out_csv = write_csv(args.out_csv, rows)
        print(f"wrote_csv: {out_csv}")
    return 0


def _import_obj(obj_type: str):
    mod, name = obj_type.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def load_img_tensor(path: str | Path, max_side: int | None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / float(s)
            im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def build_encoders_from_cfg(cfg: dict) -> list[torch.nn.Module]:
    enc_cfg = cfg["sst"]["encoder"]
    obj_type = enc_cfg["obj_type"]

    if obj_type.endswith("ListMultiEncoder"):
        encs = []
        for e in enc_cfg["kwargs"]["encoders"]:
            cls = _import_obj(e["obj_type"])
            encs.append(cls(**(e.get("kwargs", {}) or {})))
        return encs

    if obj_type.endswith("MultiDinoEncoder"):
        encs = []
        main = enc_cfg["kwargs"]["main"]
        fusion = enc_cfg["kwargs"].get("fusion", []) or []
        cls = _import_obj(main["obj_type"])
        encs.append(cls(**(main.get("kwargs", {}) or {})))
        for e in fusion:
            cls = _import_obj(e["obj_type"])
            encs.append(cls(**(e.get("kwargs", {}) or {})))
        return encs

    raise ScriptError(f"Unsupported encoder obj_type for diagnostics: {obj_type}")


def run_concat_features(encs: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    feats = []
    h0 = w0 = None
    for enc in encs:
        f = enc(x)
        if h0 is None:
            h0, w0 = f.shape[-2], f.shape[-1]
        elif f.shape[-2] != h0 or f.shape[-1] != w0:
            raise ScriptError(f"Spatial mismatch across encoders: {(h0, w0)} vs {(f.shape[-2], f.shape[-1])}")
        feats.append(f)
    return torch.cat(feats, dim=1)


def extract_softmax_weights(head, patch_embeddings: torch.Tensor) -> torch.Tensor:
    cfg = head.config
    k = getattr(cfg, "num_experts", getattr(cfg, "num_encoders", None))
    if k is None:
        raise ScriptError("Head config has neither num_experts nor num_encoders")
    main_index = getattr(cfg, "main_index", 0)
    gate_input = getattr(cfg, "gate_input", "main")
    weights_per_patch = getattr(cfg, "weights_per_patch", False)
    temperature = float(getattr(cfg, "temperature", 1.0))

    b, kc, h, w = patch_embeddings.shape
    c = kc // k
    feats = patch_embeddings.view(b, k, c, h, w)
    if gate_input == "main":
        g_in = feats[:, main_index]
    elif gate_input == "concat":
        g_in = patch_embeddings
    else:
        raise ScriptError(f"Unsupported gate_input: {gate_input}")

    logits = head.gate(g_in)
    if weights_per_patch:
        wts = torch.softmax((logits / temperature).float(), dim=1).to(logits.dtype)
    else:
        lg = logits.mean(dim=(2, 3))
        wg = torch.softmax((lg / temperature).float(), dim=1).to(logits.dtype)
        wts = wg[:, :, None, None].expand(-1, -1, h, w)
    return wts



def extract_uncertainty_decisions(head, patch_embeddings: torch.Tensor):
    """Run an UncExpertFusionHead and return y_total, preds, sigmas, weights."""
    with torch.no_grad():
        y_total, _ = head(patch_embeddings)

    preds = getattr(head, "last_expert_preds", None)
    sigmas = getattr(head, "last_sq_sigmas", None)
    weights = getattr(head, "last_moe_weights", None)

    if preds is None or sigmas is None or weights is None:
        raise ScriptError(
            "Head did not expose last_expert_preds/last_sq_sigmas/last_moe_weights. "
            "This diagnostic expects UncExpertFusionHead.forward() to store these tensors."
        )
    if preds.shape != sigmas.shape:
        raise ScriptError(f"preds/sigmas shape mismatch: {tuple(preds.shape)} vs {tuple(sigmas.shape)}")
    if weights.ndim != sigmas.ndim:
        raise ScriptError(f"weights/sigmas ndim mismatch: {tuple(weights.shape)} vs {tuple(sigmas.shape)}")
    if weights.shape[0] != sigmas.shape[0] or weights.shape[1] != sigmas.shape[1]:
        raise ScriptError(f"weights/sigmas B,K mismatch: {tuple(weights.shape)} vs {tuple(sigmas.shape)}")
    if weights.shape[2] not in (1, sigmas.shape[2]):
        raise ScriptError(
            "Expected weights coord dim to be 1 or D. "
            f"got weights={tuple(weights.shape)}, sigmas={tuple(sigmas.shape)}"
        )
    return y_total, preds, sigmas, weights


def _bincount_frac(idx: np.ndarray, k: int) -> np.ndarray:
    idx = np.asarray(idx).reshape(-1)
    if idx.size == 0:
        return np.zeros(k, dtype=np.float64)
    return np.bincount(idx, minlength=k).astype(np.float64) / float(idx.size)


def summarize_uncertainty_decisions(
    sigmas_bkd: torch.Tensor,
    weights_bkd: torch.Tensor,
    *,
    image_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize one image worth of MoGU uncertainty/routing."""
    if sigmas_bkd.ndim != 4 or weights_bkd.ndim != 4:
        raise ScriptError(
            f"Expected per-image tensors [K,D,h,w], got sigmas={tuple(sigmas_bkd.shape)}, "
            f"weights={tuple(weights_bkd.shape)}"
        )

    sigmas = sigmas_bkd.detach().float().cpu()
    weights = weights_bkd.detach().float().cpu()

    k, d, h, w = sigmas.shape
    wd = weights.shape[1]
    coord_names = ["x", "y", "z"][:d]

    decision_coord = weights.argmax(dim=0).numpy()  # [wd,h,w]
    decision_patch = weights.mean(dim=1).argmax(dim=0).numpy()  # [h,w]
    patch_win_frac = _bincount_frac(decision_patch, k)

    coord_win_fracs: dict[str, np.ndarray] = {}
    if wd == d:
        for ci, cname in enumerate(coord_names):
            coord_win_fracs[cname] = _bincount_frac(decision_coord[ci], k)
    else:
        coord_win_fracs["scalar"] = _bincount_frac(decision_coord[0], k)

    rows: list[dict[str, Any]] = []
    for ei in range(k):
        var_e = sigmas[ei]
        std_e = torch.sqrt(var_e.clamp_min(0.0))
        w_e = weights[ei] if wd == d else weights[ei].expand(d, h, w)

        row: dict[str, Any] = {
            "image": image_name,
            "expert": ei,
            "var_mean": float(var_e.mean().item()),
            "var_min": float(var_e.amin().item()),
            "var_max": float(var_e.amax().item()),
            "std_mean": float(std_e.mean().item()),
            "weight_mean": float(w_e.mean().item()),
            "weight_min": float(w_e.amin().item()),
            "weight_max": float(w_e.amax().item()),
            "patch_win_frac": float(patch_win_frac[ei]),
        }
        for ci, cname in enumerate(coord_names):
            row[f"var_{cname}_mean"] = float(var_e[ci].mean().item())
            row[f"std_{cname}_mean"] = float(std_e[ci].mean().item())
            row[f"weight_{cname}_mean"] = float(w_e[ci].mean().item())
            if cname in coord_win_fracs:
                row[f"{cname}_win_frac"] = float(coord_win_fracs[cname][ei])
        if "scalar" in coord_win_fracs:
            row["scalar_win_frac"] = float(coord_win_fracs["scalar"][ei])
        rows.append(row)

    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0).mean()
    image_summary = {
        "image": image_name,
        "num_experts": k,
        "height_patches": h,
        "width_patches": w,
        "weight_coord_dim": wd,
        "weight_entropy": float(entropy.item()),
        "patch_decision_hist": patch_win_frac.tolist(),
    }
    return rows, image_summary


def save_uncertainty_maps(
    out_dir: Path,
    image_name: str,
    sigmas_kdhw: torch.Tensor,
    weights_kdhw: torch.Tensor,
    preds_kdhw: torch.Tensor | None = None,
) -> Path:
    """Save raw per-patch maps for offline plotting/debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    sigmas = sigmas_kdhw.detach().float().cpu().numpy()
    weights = weights_kdhw.detach().float().cpu().numpy()
    decision_patch = weights.mean(axis=1).argmax(axis=0).astype(np.int64)
    decision_coord = weights.argmax(axis=0).astype(np.int64)

    payload: dict[str, Any] = {
        "sq_sigma_k3hw": sigmas,
        "weights_kdhw": weights,
        "decision_patch_hw": decision_patch,
        "decision_coord_dhw": decision_coord,
    }
    if preds_kdhw is not None:
        payload["preds_k3hw"] = preds_kdhw.detach().float().cpu().numpy()

    out_path = out_dir / f"{stem}_mogu_uncertainty_decisions.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def cmd_uncertainty_decisions(args: argparse.Namespace) -> int:
    print_stage_header("UNCERTAINTY DECISIONS")
    ensure_ace_g_importable(args.repo_root)
    run = setup_wandb_logging(args, "uncertainty-decisions")

    cfg_path = expand(args.config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    encs = build_encoders_from_cfg(cfg)
    head = load_head(args.repo_root, args.head_path)

    cfg_obj_type = getattr(getattr(head, "config", None), "obj_type", "")
    if "UncExpertFusionHead" not in str(cfg_obj_type):
        print(
            "WARNING: head config does not look like UncExpertFusionHead. "
            f"obj_type={cfg_obj_type!r}. Continuing, but forward() must expose last_sq_sigmas/last_moe_weights."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.use_half and device == "cuda"

    for i in range(len(encs)):
        encs[i] = encs[i].to(device).eval()
    head = head.to(device).eval()

    paths = sorted(glob.glob(args.rgb_glob))
    if getattr(args, "random_sample", False):
        import random
        random.seed(getattr(args, "seed", 42))
        random.shuffle(paths)
    paths = paths[: args.num_images]

    if not paths:
        if run:
            run.finish(exit_code=1)
        raise ScriptError(f"No images matched: {args.rgb_glob}")

    print(f"config_path: {cfg_path}")
    print(f"head_path  : {expand(args.head_path)}")
    print(f"device     : {device}")
    print(f"use_half   : {use_half}")
    print(f"images     : {len(paths)}")
    print(f"save_maps  : {args.save_maps}")

    all_rows: list[dict[str, Any]] = []
    all_weight_hists: list[list[np.ndarray]] = []
    all_var_hists: list[list[np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    out_dir = expand(args.out_dir) if args.out_dir else None

    with torch.no_grad():
        for p in paths:
            img_name = Path(p).name
            x = load_img_tensor(p, args.max_side).to(device)
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
                _, preds, sigmas, weights = extract_uncertainty_decisions(head, feats)

            sig_i = sigmas[0]
            w_i = weights[0]
            pred_i = preds[0]

            rows, summary = summarize_uncertainty_decisions(sig_i, w_i, image_name=img_name)
            all_rows.extend(rows)
            summaries.append(summary)

            k = sig_i.shape[0]
            if not all_weight_hists:
                all_weight_hists = [[] for _ in range(k)]
                all_var_hists = [[] for _ in range(k)]
            for ei in range(k):
                all_weight_hists[ei].append(w_i[ei].detach().float().cpu().numpy().reshape(-1))
                all_var_hists[ei].append(sig_i[ei].detach().float().cpu().numpy().reshape(-1))

            print(f"\n--- {img_name} ---")
            print(
                f"patch_grid={summary['height_patches']}x{summary['width_patches']} "
                f"weight_coord_dim={summary['weight_coord_dim']} "
                f"entropy={summary['weight_entropy']:.4f}"
            )
            for row in rows:
                msg = (
                    f"expert[{row['expert']}] "
                    f"var_mean={row['var_mean']:.6g} std_mean={row['std_mean']:.6g} "
                    f"w_mean={row['weight_mean']:.4f} w_min={row['weight_min']:.4f} w_max={row['weight_max']:.4f} "
                    f"patch_win%={100.0 * row['patch_win_frac']:.1f}"
                )
                if "x_win_frac" in row:
                    msg += (
                        f" xyz_win%="
                        f"{100.0 * row['x_win_frac']:.1f}/"
                        f"{100.0 * row['y_win_frac']:.1f}/"
                        f"{100.0 * row['z_win_frac']:.1f}"
                    )
                print(msg)

            if args.save_maps:
                if out_dir is None:
                    raise ScriptError("--save-maps requires --out-dir")
                saved = save_uncertainty_maps(out_dir, img_name, sig_i, w_i, pred_i)
                print(f"saved_maps: {saved}")

    if args.out_csv:
        out_csv = write_csv(args.out_csv, all_rows)
        print(f"wrote_csv: {out_csv}")

    if args.out_json:
        out_json = expand(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"images": summaries, "rows": all_rows}, indent=2), encoding="utf-8")
        print(f"wrote_json: {out_json}")

    if run:
        import wandb
        log_dict: dict[str, Any] = {"uncertainty_decision_summary": summaries}
        for ei in range(len(all_weight_hists)):
            wdata = np.concatenate(all_weight_hists[ei])
            vdata = np.concatenate(all_var_hists[ei])
            try:
                log_dict[f"expert_{ei}_mogu_weights"] = wandb.Histogram(wdata)
                log_dict[f"expert_{ei}_sq_sigma"] = wandb.Histogram(vdata)
            except ValueError:
                whist, wedges = np.histogram(wdata, bins=max(1, min(64, len(np.unique(wdata)))))
                vhist, vedges = np.histogram(vdata, bins=max(1, min(64, len(np.unique(vdata)))))
                log_dict[f"expert_{ei}_mogu_weights"] = wandb.Histogram(np_histogram=(whist.tolist(), wedges.tolist()))
                log_dict[f"expert_{ei}_sq_sigma"] = wandb.Histogram(np_histogram=(vhist.tolist(), vedges.tolist()))
        run.log(log_dict)
        run.finish()

    print("\nDone.")
    return 0

def cmd_gate_weights(args: argparse.Namespace) -> int:
    print_stage_header("GATE WEIGHTS")
    ensure_ace_g_importable(args.repo_root)
    run = setup_wandb_logging(args, "gate-weights")
    eval_reg = getattr(args, "eval_registration", False)
    if eval_reg:
        from ace_g import data_io, losses, utils



    cfg_path = expand(args.config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    encs = build_encoders_from_cfg(cfg)
    head = load_head(args.repo_root, args.head_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.use_half and device == "cuda"

    for i in range(len(encs)):
        encs[i] = encs[i].to(device).eval()
    head = head.to(device).eval()

    paths = sorted(glob.glob(args.rgb_glob))
    if getattr(args, "random_sample", False):
        import random
        random.seed(getattr(args, "seed", 42))
        random.shuffle(paths)
    paths = paths[: args.num_images]
    
    if not paths:
        if run: run.finish(exit_code=1)
        raise ScriptError(f"No images matched: {args.rgb_glob}")

    print(f"config_path: {cfg_path}")
    print(f"head_path  : {expand(args.head_path)}")
    print(f"device     : {device}")
    print(f"use_half   : {use_half}")
    print(f"eval_reg   : {eval_reg}")
    print(f"images     : {len(paths)}")
    all_wts = []

    with torch.no_grad():
        for p in paths:
            img_name = Path(p).name
            x = load_img_tensor(p, args.max_side).to(device)
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
                wts = extract_softmax_weights(head, feats)
            w = wts[0]
            k = w.shape[0]
            if not all_wts:
                all_wts = [[] for _ in range(k)]
            for idx in range(k):
                all_wts[idx].append(w[idx].detach().cpu().numpy().flatten())
            mean = w.mean(dim=(1, 2)).detach().float().cpu().numpy()
            mn = w.amin(dim=(1, 2)).detach().float().cpu().numpy()
            mx = w.amax(dim=(1, 2)).detach().float().cpu().numpy()
            winner = w.argmax(dim=0).detach().cpu().numpy()
            win_frac = np.bincount(winner.reshape(-1), minlength=k) / winner.size

            if eval_reg:
                p_path = Path(p)
                pose_path = p_path.parent.parent / "poses" / f"{p_path.stem}.txt"
                calib_path = p_path.parent.parent / "calibration" / f"{p_path.stem}.txt"
                if not pose_path.exists() or not calib_path.exists():
                    raise ScriptError(f"Missing GT pose or calib for {p}")

                pose_c2w = data_io.load_pose(pose_path)
                w2c_44 = pose_c2w.inverse()
                w2c_b34 = w2c_44[:3, :4].unsqueeze(0).to(device)

                calib = data_io.load_calibration(calib_path)
                im = Image.open(p)
                w_orig, h_orig = im.size
                s = max(w_orig, h_orig)
                scale = args.max_side / float(s) if s > args.max_side else 1.0

                if isinstance(calib, float):
                    fx = fy = calib
                    cx = w_orig / 2
                    cy = h_orig / 2
                else:
                    fx = calib[0, 0]
                    fy = calib[1, 1]
                    cx = calib[0, 2]
                    cy = calib[1, 2]

                intrinsics = torch.eye(3)
                intrinsics[0, 0] = fx * scale
                intrinsics[1, 1] = fy * scale
                intrinsics[0, 2] = cx * scale
                intrinsics[1, 2] = cy * scale
                image_from_camera_b33 = intrinsics.unsqueeze(0).to(device)

                sub_h = int(round(x.shape[-2] / feats.shape[-2]))
                pixel_grid_2hw = utils.get_pixel_grid(sub_h).to(device)
                pixel_grid_2hw = pixel_grid_2hw[:, :feats.shape[-2], :feats.shape[-1]]
                _, h_f, w_f = pixel_grid_2hw.shape
                target_px_b2 = pixel_grid_2hw.reshape(2, -1).permute(1, 0)
                target_coords_b3 = torch.zeros((h_f * w_f, 3), device=device)

                experts_coords = []
                b_f, kc_f, h_f_feat, w_f_feat = feats.shape
                c_f = kc_f // k
                feats_reshaped = feats.view(b_f, k, c_f, h_f_feat, w_f_feat)
                with torch.autocast("cuda", enabled=use_half):
                    for idx in range(k):
                        yi, _ = head.expert_heads[idx](feats_reshaped[:, idx])
                        pred_coords_b3 = yi.permute(0, 2, 3, 1).reshape(-1, 3).float()
                        experts_coords.append(pred_coords_b3)

                    y_total, _ = head(feats)
                    pred_coords_total_b3 = y_total.permute(0, 2, 3, 1).reshape(-1, 3).float()

                n_patches = h_f * w_f
                w2c_b34_exp = w2c_b34.expand(n_patches, 3, 4)
                image_from_camera_b33_exp = image_from_camera_b33.expand(n_patches, 3, 3)

                reg_losses = []
                for idx in range(k):
                    _, _, _, _, dists_2d = losses.compute_loss(
                        pred_coords=experts_coords[idx],
                        pred_uncertainties=None,
                        w2c_b34=w2c_b34_exp,
                        image_from_camera_b33=image_from_camera_b33_exp,
                        target_pixels=target_px_b2,
                        target_coords=target_coords_b3,
                        supervision_type="2d",
                        use_depth_as_prior=False,
                    )
                    reg_losses.append(dists_2d.mean().item())

                _, _, _, _, dists_2d_tot = losses.compute_loss(
                    pred_coords=pred_coords_total_b3,
                    pred_uncertainties=None,
                    w2c_b34=w2c_b34_exp,
                    image_from_camera_b33=image_from_camera_b33_exp,
                    target_pixels=target_px_b2,
                    target_coords=target_coords_b3,
                    supervision_type="2d",
                    use_depth_as_prior=False,
                )
                total_reg_loss = dists_2d_tot.mean().item()

            print(f"\n--- {img_name} ---")
            for idx in range(k):
                msg = f"expert[{idx}] mean={mean[idx]:.4f} min={mn[idx]:.4f} max={mx[idx]:.4f} win%={100 * win_frac[idx]:.1f}"
                if eval_reg:
                    msg += f" reg_loss_px={reg_losses[idx]:.2f}"
                print(msg)
            
            if eval_reg:
                print(f"Total combined reg_loss_px={total_reg_loss:.2f}")

    if run:
        import wandb
        log_dict = {}
        for idx in range(len(all_wts)):
            data = np.concatenate(all_wts[idx])
            try:
                log_dict[f"expert_{idx}_weights"] = wandb.Histogram(data)
            except ValueError:
                unique_vals = np.unique(data)
                num_bins = max(1, min(64, len(unique_vals)))
                hist, edges = np.histogram(data, bins=num_bins)
                log_dict[f"expert_{idx}_weights"] = wandb.Histogram(np_histogram=(hist.tolist(), edges.tolist()))
        run.log(log_dict)
        run.finish()
        
    print("\nDone.")
    return 0


def cmd_gate_last_layer(args: argparse.Namespace) -> int:
    print_stage_header("GATE LAST LAYER")
    run = setup_wandb_logging(args, "gate-last-layer")

    head = load_head(args.repo_root, args.head_path).cpu().eval()
    if not hasattr(head, "gate"):
        if run: run.finish(exit_code=1)
        raise ScriptError("Loaded head has no .gate attribute")
    last = head.gate[-1]
    if not isinstance(last, torch.nn.Conv2d):
        if run: run.finish(exit_code=1)
        raise ScriptError(f"Expected head.gate[-1] to be Conv2d, got {type(last)}")

    bias = last.bias.detach().cpu().numpy() if last.bias is not None else None
    weight_sum = last.weight.detach().cpu().sum(dim=(1, 2, 3)).numpy()
    weight_l2 = torch.linalg.vector_norm(last.weight.detach().cpu().reshape(last.weight.shape[0], -1), dim=1).numpy()

    out = {
        "last_conv": str(last),
        "bias": bias.tolist() if bias is not None else None,
        "weight_sum_per_expert": weight_sum.tolist(),
        "weight_l2_per_expert": weight_l2.tolist(),
        "bias_only_softmax": torch.softmax(last.bias.detach().cpu(), dim=0).numpy().tolist() if last.bias is not None else None,
    }
    print(json.dumps(out, indent=2))
    
    if run:
        run.config.update({"gate_last_layer_stats": out})

    if args.probe_config_path and args.probe_rgb_glob:
        ensure_ace_g_importable(args.repo_root)
        cfg = yaml.safe_load(expand(args.probe_config_path).read_text(encoding="utf-8"))
        encs = build_encoders_from_cfg(cfg)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_half = args.use_half and device == "cuda"
        for i in range(len(encs)):
            encs[i] = encs[i].to(device).eval()
        head = head.to(device).eval()

        images = sorted(glob.glob(args.probe_rgb_glob))[:1]
        if not images:
            if run: run.finish(exit_code=1)
            raise ScriptError(f"No images matched probe glob: {args.probe_rgb_glob}")
        x = load_img_tensor(images[0], args.max_side).to(device)

        captured: dict[str, torch.Tensor] = {}

        def hook_fn(module, inputs, output):
            captured["pre_last"] = inputs[0].detach().cpu()
            captured["logits"] = output.detach().cpu()

        hook = last.register_forward_hook(hook_fn)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
                _ = extract_softmax_weights(head, feats)
        hook.remove()

        h = captured.get("pre_last")
        logits = captured.get("logits")
        if h is None or logits is None:
            if run: run.finish(exit_code=1)
            raise ScriptError("Failed to capture last-layer input/logits during probe")

        w = last.weight.detach().cpu()
        b = last.bias.detach().cpu() if last.bias is not None else torch.zeros(w.shape[0])
        logits_from_w = torch.nn.functional.conv2d(h, w, bias=None)
        logits_from_b = b.view(1, -1, 1, 1).expand_as(logits_from_w)
        total = logits_from_w + logits_from_b
        probs = torch.softmax(logits, dim=1)
        probe_stats = {
            "probe_image": images[0],
            "mean_abs_w_dot_h_per_expert": logits_from_w.abs().mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_abs_bias_per_expert": logits_from_b.abs().mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_total_logits_per_expert": total.mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_logits_over_batch": logits.mean(dim=(2, 3)).mean(dim=0).numpy().tolist(),
            "std_logits_over_images": logits.mean(dim=(2, 3)).std(dim=0).numpy().tolist(),
            "mean_probs_over_batch": probs.mean(dim=(2, 3)).mean(dim=0).numpy().tolist(),
            "std_probs_over_images": probs.mean(dim=(2, 3)).std(dim=0).numpy().tolist(),
        }
        print("\nPROBE_STATS")
        print(json.dumps(probe_stats, indent=2))
        
        if run:
            run.log({"probe_stats": probe_stats})
            
    if run:
        run.finish()
        
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    print_stage_header("PRINT HEAD CONFIG")
    head = load_head(args.repo_root, args.head_path).cpu().eval()
    cfg = head.config
    out = {
        "weights_per_patch": getattr(cfg, "weights_per_patch", None),
        "gate_input": getattr(cfg, "gate_input", None),
        "temperature": getattr(cfg, "temperature", None),
        "num_experts": getattr(cfg, "num_experts", getattr(cfg, "num_encoders", None)),
        "main_index": getattr(cfg, "main_index", None),
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="~/dace/ace-g", help="ACE-G repo root")
    p.add_argument("--wandb-log", action="store_true", help="Log diagnostic results to W&B")
    p.add_argument("--wandb-project", default=None, help="W&B project")
    p.add_argument("--wandb-entity", default=None, help="W&B entity")

    sp = p.add_subparsers(dest="cmd", required=True)

    p_means = sp.add_parser("compare-means", help="Compare two head mean tensors")
    p_means.add_argument("--head-a", required=True)
    p_means.add_argument("--head-b", required=True)
    p_means.set_defaults(func=cmd_compare_means)

    p_logs = sp.add_parser("compare-logs", help="Compare register/eval logs for session prefixes")
    p_logs.add_argument("--log-dir", default="~/dace/outputs/logs")
    p_logs.add_argument("--session", action="append", required=True, help="LABEL=SESSION_PREFIX")
    p_logs.add_argument("--tail-lines", type=int, default=40)
    p_logs.add_argument("--out-csv", default=None)
    p_logs.set_defaults(func=cmd_compare_logs)

    p_gw = sp.add_parser("gate-weights", help="Inspect late-fusion softmax weights over sample images")
    p_gw.add_argument("--config-path", required=True, help="Path to the late-fusion config YAML")
    p_gw.add_argument("--head-path", required=True, help="Path to the trained head checkpoint")
    p_gw.add_argument("--rgb-glob", required=True, help="Glob for sample RGB test images")
    p_gw.add_argument("--num-images", type=int, default=25)
    p_gw.add_argument("--random-sample", action="store_true", help="Randomly sample images instead of picking the first N")
    p_gw.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p_gw.add_argument("--max-side", type=int, default=640)
    p_gw.add_argument("--device", default=None)
    p_gw.add_argument("--use-half", action="store_true")
    p_gw.add_argument("--eval-registration", action="store_true", help="Evaluate patch-level 2D registration loss against ground truth")
    p_gw.set_defaults(func=cmd_gate_weights)

    p_ud = sp.add_parser(
        "uncertainty-decisions",
        help="Inspect UncExpertFusionHead per-expert uncertainty and patch decisions",
    )
    p_ud.add_argument("--config-path", required=True, help="Path to the MoGU/fusion config YAML")
    p_ud.add_argument("--head-path", required=True, help="Path to the trained UncExpertFusionHead checkpoint")
    p_ud.add_argument("--rgb-glob", required=True, help="Glob for sample RGB test images")
    p_ud.add_argument("--num-images", type=int, default=25)
    p_ud.add_argument("--random-sample", action="store_true", help="Randomly sample images instead of picking the first N")
    p_ud.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p_ud.add_argument("--max-side", type=int, default=640)
    p_ud.add_argument("--device", default=None)
    p_ud.add_argument("--use-half", action="store_true")
    p_ud.add_argument("--out-csv", default=None, help="Optional CSV path for per-image/per-expert summary rows")
    p_ud.add_argument("--out-json", default=None, help="Optional JSON path for summary rows and image summaries")
    p_ud.add_argument("--out-dir", default=None, help="Output directory used with --save-maps")
    p_ud.add_argument("--save-maps", action="store_true", help="Save .npz maps: sq_sigma, weights, and patch decisions")
    p_ud.set_defaults(func=cmd_uncertainty_decisions)

    p_gl = sp.add_parser("gate-last-layer", help="Inspect gating last-layer params, optionally with a probe image")
    p_gl.add_argument("--head-path", required=True)
    p_gl.add_argument("--probe-config-path", default=None, help="Optional config path to probe a single image through the gate")
    p_gl.add_argument("--probe-rgb-glob", default=None, help="Optional image glob for gate probing")
    p_gl.add_argument("--max-side", type=int, default=640)
    p_gl.add_argument("--device", default=None)
    p_gl.add_argument("--use-half", action="store_true")
    p_gl.set_defaults(func=cmd_gate_last_layer)

    p_cfg = sp.add_parser("print-config", help="Print selected head config fields")
    p_cfg.add_argument("--head-path", required=True)
    p_cfg.set_defaults(func=cmd_print_config)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
