#!/usr/bin/env python3
"""End-to-end smoking gun for UncExpertFusionHead.

The question this answers, with an answer known BY CONSTRUCTION (not derived from any
diagnostic script):

    If the gate is fed a *correct* per-patch signal, does the real, unmodified fusion
    stack turn that into "fused beats the best single expert" on the real metric?

How it is unfakeable:
  * Known answer by construction. We plant complementary experts: a deterministic
    checkerboard mask decides which expert is the *good* one on each patch. On the
    good patches an expert returns a genuinely good prediction (expert-0's real
    scene-coords on the real features); on its bad patches it returns that prediction
    plus a large known world-space offset, so its reprojection error explodes there.
    Each single expert is therefore corrupted on half its patches; a working gate that
    follows the planted (oracle) uncertainty must select the good expert everywhere, so
    fused error must be strictly LOWER than either single expert.
  * Real code path. We replace ONLY the leaf expert modules. Everything downstream is
    the real, unmodified `UncExpertFusionHead.forward`: the var clamp, the
    log-inverse-variance gate, `gate_temperature`, the softmax, the weighted sum, all
    reshaping. The metric is the real `losses.compute_loss` reprojection error against
    GT pose + calibration.
  * Predicted direction. We assert a RELATIONSHIP (fused < best single by a margin), so a
    constant bias in the metric cannot make a broken system pass.

Interpretation:
  PASS -> the fusion + gate + metric machinery is sound. A correct per-patch gate signal
          DOES produce "fused < best single". Your real-data shortfall is therefore NOT
          in this stack -- it is upstream: the UncHeads do not *emit* a discriminative
          sigma, or the experts have no complementary headroom. (Confirm with the
          oracle-gate / overfit tests.)
  FAIL -> the bug is INSIDE this stack (fusion arithmetic, gate sign, clamp, temperature,
          or the metric wiring), independent of whether your other diagnostics are
          reliable. Fix here first.

Scope: this is an *inference* smoking gun -- it proves the machinery converts a correct
gate signal into a win. Whether the UncHeads can *learn* to emit such a signal is the
separate trainability question (overfit-a-handful test).

Run:
    ~/dace_env310/bin/python scripts/smoking_gun_fusion.py \
        --config-path configs_custom/MoGU_N2_frozen_total_tau0_scalar_unc.yaml \
        --head-path ~/dace/outputs/heads/<fused-head>.pt \
        --rgb-glob "~/dace/datasets/cambridge/shopfacade/test/rgb/*.jpg"
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import yaml

# Reuse the *same* loading / feature path the diagnostics use, so this test exercises the
# real encoders -> features pipeline rather than a parallel reimplementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ScriptError, expand, print_stage_header  # noqa: E402
from diagnose_heads_unc import (  # noqa: E402
    build_encoders_from_cfg,
    ensure_ace_g_importable,
    load_head,
    load_img_tensor,
    run_concat_features,
)


class _PlantedExpert(nn.Module):
    """Leaf-only stand-in for a fusion expert.

    Mirrors the real expert contract `forward(feats) -> (pred[B,3,h,w], sq_sigma[B,1,h,w])`,
    but ignores `feats` and returns planted values: a good prediction where this expert
    "owns" the patch, a corrupted one elsewhere, plus an oracle low/high variance that
    tells the gate exactly which is which.
    """

    def __init__(
        self,
        p_good: torch.Tensor,      # [1, 3, h, w] shared good prediction
        mask_good: torch.Tensor,   # [1, 1, h, w] bool: True where THIS expert is the good one
        corrupt_offset_m: float,
        sigma_lo: float,
        sigma_hi: float,
    ) -> None:
        super().__init__()
        self.register_buffer("p_good", p_good)
        self.register_buffer("mask_good", mask_good)
        pred_bad = p_good + float(corrupt_offset_m)
        self.register_buffer("p_bad", pred_bad)
        sigma = torch.where(
            mask_good,
            torch.full_like(mask_good, float(sigma_lo), dtype=p_good.dtype),
            torch.full_like(mask_good, float(sigma_hi), dtype=p_good.dtype),
        )
        self.register_buffer("sigma", sigma)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        pred = torch.where(self.mask_good, self.p_good, self.p_bad)
        return pred.expand(b, -1, -1, -1).clone(), self.sigma.expand(b, -1, -1, -1).clone()


def _checkerboard(h: int, w: int, device) -> torch.Tensor:
    """[1,1,h,w] bool checkerboard: True where (i+j) is even."""
    ii = torch.arange(h, device=device).view(h, 1)
    jj = torch.arange(w, device=device).view(1, w)
    return (((ii + jj) % 2) == 0).view(1, 1, h, w)


def _patch_reproj_px(
    losses_mod,
    pred_b3: torch.Tensor,       # [N, 3] world scene-coords (row-major over h,w)
    w2c_exp: torch.Tensor,       # [N, 3, 4]
    k_exp: torch.Tensor,         # [N, 3, 3]
    target_px_b2: torch.Tensor,  # [N, 2]
) -> np.ndarray:
    target_coords_b3 = torch.zeros((pred_b3.shape[0], 3), device=pred_b3.device)
    _, _, _, _, d2d = losses_mod.compute_loss(
        pred_coords=pred_b3,
        pred_uncertainties=None,
        w2c_b34=w2c_exp,
        image_from_camera_b33=k_exp,
        target_pixels=target_px_b2,
        target_coords=target_coords_b3,
        supervision_type="2d",
        use_depth_as_prior=False,
    )
    return d2d.detach().float().cpu().numpy().reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default="~/dace/ace-g")
    ap.add_argument("--config-path", required=True, help="MoGU/fusion config YAML (for encoders)")
    ap.add_argument("--head-path", required=True, help="Trained UncExpertFusionHead checkpoint")
    ap.add_argument("--rgb-glob", required=True, help="Glob for sample RGB images (with sibling poses/ and calibration/)")
    ap.add_argument("--num-images", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-half", action="store_true")
    ap.add_argument("--corrupt-offset-m", type=float, default=5.0, help="World-space offset added on a patch's bad expert")
    ap.add_argument("--margin", type=float, default=2.0, help="Required factor: best_single_median >= margin * fused_median to PASS")
    args = ap.parse_args()

    print_stage_header("SMOKING GUN: PLANTED COMPLEMENTARITY (end-to-end)")
    ensure_ace_g_importable(args.repo_root)
    from ace_g import data_io, losses, utils

    cfg_path = expand(args.config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    encs = build_encoders_from_cfg(cfg)
    head = load_head(args.repo_root, args.head_path)

    if not hasattr(head, "experts") or not hasattr(head.config, "num_experts"):
        raise ScriptError("smoking-gun expects an UncExpertFusionHead (head.experts / config.num_experts).")
    k = int(head.config.num_experts)
    if k != 2:
        raise ScriptError(f"This planted test assumes exactly 2 experts; head has {k}.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.use_half and device == "cuda"
    for i in range(len(encs)):
        encs[i] = encs[i].to(device).eval()
    head = head.to(device).eval()

    # Oracle variances live inside the real [var_min, var_max] clamp so the gate sees a
    # clean, in-range low/high signal (no reliance on the clamp to manufacture contrast).
    var_min = float(getattr(head.config, "var_min", 0.1))
    var_max = float(getattr(head.config, "var_max", 100.0))
    sigma_lo, sigma_hi = var_min, var_max
    gate_t = float(getattr(head.config, "gate_temperature", 1.0))

    paths = sorted(glob.glob(expand(args.rgb_glob).as_posix()))[: args.num_images]
    if not paths:
        raise ScriptError(f"No images matched: {args.rgb_glob}")

    print(f"config_path: {cfg_path}")
    print(f"head_path  : {expand(args.head_path)}")
    print(f"device={device} use_half={use_half} images={len(paths)} k={k}")
    print(f"gate_temperature={gate_t}  oracle sigma_lo={sigma_lo} sigma_hi={sigma_hi}  corrupt_offset={args.corrupt_offset_m} m")

    orig_experts = head.experts
    orig_force = head.config.sanity_check_force_expert
    fused_d, e0_d, e1_d = [], [], []
    good_weight_means = []

    try:
        with torch.no_grad():
            for p in paths:
                p_path = Path(p)
                pose_path = p_path.parent.parent / "poses" / f"{p_path.stem}.txt"
                calib_path = p_path.parent.parent / "calibration" / f"{p_path.stem}.txt"
                if not pose_path.exists() or not calib_path.exists():
                    raise ScriptError(f"Missing GT pose or calib for {p}")

                x = load_img_tensor(p, args.max_side).to(device)
                with torch.autocast("cuda", enabled=use_half):
                    feats = run_concat_features(encs, x)

                # --- Geometry for the real reprojection metric (mirror repro-vs-pose) ---
                pose_c2w = data_io.load_pose(pose_path)
                w2c_b34 = pose_c2w.inverse()[:3, :4].unsqueeze(0).to(device)
                calib = data_io.load_calibration(calib_path)
                im = Image.open(p)
                w_orig, h_orig = im.size
                s = max(w_orig, h_orig)
                scale = args.max_side / float(s) if s > args.max_side else 1.0
                if isinstance(calib, float):
                    fx = fy = calib
                    cx, cy = w_orig / 2, h_orig / 2
                else:
                    fx, fy, cx, cy = calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]
                intrinsics = torch.eye(3)
                intrinsics[0, 0] = fx * scale
                intrinsics[1, 1] = fy * scale
                intrinsics[0, 2] = cx * scale
                intrinsics[1, 2] = cy * scale

                sub_h = int(round(x.shape[-2] / feats.shape[-2]))
                pixel_grid_2hw = utils.get_pixel_grid(sub_h).to(device)
                pixel_grid_2hw = pixel_grid_2hw[:, : feats.shape[-2], : feats.shape[-1]]
                _, h_f, w_f = pixel_grid_2hw.shape
                target_px_b2 = pixel_grid_2hw.reshape(2, -1).permute(1, 0)
                n_patches = h_f * w_f
                w2c_exp = w2c_b34.expand(n_patches, 3, 4)
                k_exp = intrinsics.unsqueeze(0).to(device).expand(n_patches, 3, 3)

                # --- Shared good prediction = expert-0's real scene-coords on real feats ---
                head.experts = orig_experts
                head.config.sanity_check_force_expert = None
                with torch.autocast("cuda", enabled=use_half):
                    head(feats)
                p_good = head.last_expert_preds[:, 0].float().detach()  # [1, 3, h_f, w_f]

                # --- Plant complementary experts (leaf-only swap) ---
                mask0 = _checkerboard(h_f, w_f, device)   # expert 0 good where True
                mask1 = ~mask0                            # expert 1 good elsewhere
                head.experts = nn.ModuleList([
                    _PlantedExpert(p_good, mask0, args.corrupt_offset_m, sigma_lo, sigma_hi).to(device),
                    _PlantedExpert(p_good, mask1, args.corrupt_offset_m, sigma_lo, sigma_hi).to(device),
                ])

                # Fused (real gate decides per patch) and each single expert baseline.
                head.config.sanity_check_force_expert = None
                with torch.autocast("cuda", enabled=use_half):
                    y_fused, _ = head(feats)
                w = head.last_moe_weights  # [1, K, 1, h, w]
                good_w = torch.where(mask0, w[:, 0], w[:, 1])  # weight given to whichever expert is good
                good_weight_means.append(float(good_w.mean().item()))

                head.config.sanity_check_force_expert = 0
                with torch.autocast("cuda", enabled=use_half):
                    y_e0, _ = head(feats)
                head.config.sanity_check_force_expert = 1
                with torch.autocast("cuda", enabled=use_half):
                    y_e1, _ = head(feats)

                def _flat(y):
                    return y.permute(0, 2, 3, 1).reshape(-1, 3).float()

                fused_d.append(_patch_reproj_px(losses, _flat(y_fused), w2c_exp, k_exp, target_px_b2))
                e0_d.append(_patch_reproj_px(losses, _flat(y_e0), w2c_exp, k_exp, target_px_b2))
                e1_d.append(_patch_reproj_px(losses, _flat(y_e1), w2c_exp, k_exp, target_px_b2))
    finally:
        head.experts = orig_experts
        head.config.sanity_check_force_expert = orig_force

    def _med(parts):
        a = np.concatenate(parts)
        a = a[np.isfinite(a)]
        return float(np.median(a))

    fused_m, e0_m, e1_m = _med(fused_d), _med(e0_d), _med(e1_d)
    best_single = min(e0_m, e1_m)
    good_w_mean = float(np.mean(good_weight_means))

    print("\n================ PER-PATCH REPROJECTION ERROR (median px, L1) ================")
    print(f"  expert0 (corrupted on half): {e0_m:8.3f}")
    print(f"  expert1 (corrupted on half): {e1_m:8.3f}")
    print(f"  best single                : {best_single:8.3f}")
    print(f"  FUSED                      : {fused_m:8.3f}")
    print(f"  gate weight on the good expert (mean): {good_w_mean:.4f}  (oracle target ~1.0)")

    # Smoking gun: a working stack must drive fused well below the best single expert,
    # because the planted oracle uncertainty makes the right choice obvious on every patch.
    passed = fused_m * args.margin <= best_single and good_w_mean > 0.9
    print("\n----------------------------------------------------------------")
    if passed:
        print(f"PASS: fused ({fused_m:.3f}) beats best single ({best_single:.3f}) by >= {args.margin:g}x, "
              f"and gate concentrated on the good expert ({good_w_mean:.3f}).")
        print("=> The fusion + gate + metric machinery is sound. A correct per-patch signal")
        print("   produces 'fused < best single'. Your real-data shortfall is UPSTREAM:")
        print("   the UncHeads do not emit a discriminative sigma, or there is no headroom.")
    else:
        print(f"FAIL: fused={fused_m:.3f} best_single={best_single:.3f} (need fused*{args.margin:g} <= best_single), "
              f"gate_on_good={good_w_mean:.3f} (need >0.9).")
        print("=> The bug is INSIDE the fusion stack (gate sign/clamp/temperature, weighted")
        print("   sum, or metric wiring) -- not in your diagnostics. Fix here first.")
    print("----------------------------------------------------------------")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
