#!/usr/bin/env python3
"""
Evaluate a K=2 MoE gating classifier for ACE-G-style camera localization.

Updates vs. the first eval script:
- Robustly loads either a raw state_dict or the checkpoint dict saved by the updated trainer.
- Infers d_model from the checkpoint when possible.
- Supports hard and soft alpha outputs; soft alpha is treated as first-class.
- Logs oracle, hard, and soft-alpha distributions.
- Logs alpha agreement metrics and patch-level reprojection regret vs. oracle.
- Uses torch.no_grad() during evaluation to avoid accidental autograd tensors/memory growth.
- Handles the final eval chunk safely and uses the actual chunk size, not a fixed batch size.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Add repo root to python path to import ace_g and _common.
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "scripts"))

import _common  # noqa: F401
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yoco
import torch
import torch.nn as nn
import tqdm

from ace_g import encoders, scr_heads, regressors, datasets, buffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)
SCRIPT_VERSION = "eval_gating_classifier_no_alpha_bins_v3_cfg_none_fix_2026_05_24"


def cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Return cfg[key], but treat missing/None as default."""
    value = cfg.get(key, default)
    return default if value is None else value


def cfg_get_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    value = cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class GatingClassifier(nn.Module):
    """K=2 gating classifier mapping two 1024-d embeddings to 11 alpha-bin logits."""

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_ln = nn.LayerNorm(1024)
        self.shared_proj = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder_id_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.encoder_id_embed, std=0.02)

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 11),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 3 or z.shape[1] != 2 or z.shape[2] != 1024:
            raise ValueError(f"Expected z shape [B, 2, 1024], got {tuple(z.shape)}")

        batch_size = z.shape[0]
        x = self.input_ln(z.float())
        x = self.shared_proj(x)
        x = x + self.encoder_id_embed[None, :, :]
        x = x.reshape(batch_size, -1)
        logits = self.classifier(x)

        probs = logits.softmax(dim=-1)
        pred_class = logits.argmax(dim=-1)
        bins = torch.linspace(0.0, 1.0, 11, device=logits.device, dtype=logits.dtype)
        alpha_hard = bins[pred_class]
        alpha_soft = probs @ bins

        weights_hard = torch.stack([alpha_hard, 1.0 - alpha_hard], dim=-1)
        weights_soft = torch.stack([alpha_soft, 1.0 - alpha_soft], dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "pred_class": pred_class,
            "alpha": alpha_hard,          # backward-compatible alias
            "alpha_hard": alpha_hard,
            "alpha_soft": alpha_soft,
            "weights": weights_hard,      # backward-compatible alias
            "weights_hard": weights_hard,
            "weights_soft": weights_soft,
        }


def _torch_load_any(path: Path, map_location: torch.device) -> Any:
    """Compatibility wrapper for torch.load across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _extract_state_dict_and_meta(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Support raw state_dict and common checkpoint dict formats."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Raw state_dict: keys are module parameter names.
    if any(k in ckpt for k in ["input_ln.weight", "encoder_id_embed", "shared_proj.0.weight"]):
        return _strip_module_prefix(ckpt), {}

    for key in ["model_state_dict", "state_dict", "model"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            meta = {k: v for k, v in ckpt.items() if k != key}
            return _strip_module_prefix(ckpt[key]), meta

    raise KeyError(
        "Could not find model weights in checkpoint. Expected raw state_dict or one of: "
        "model_state_dict/state_dict/model."
    )


def _infer_d_model_from_state_dict(state_dict: Dict[str, torch.Tensor], default: int = 256) -> int:
    if "encoder_id_embed" in state_dict:
        return int(state_dict["encoder_id_embed"].shape[1])
    if "shared_proj.0.weight" in state_dict:
        return int(state_dict["shared_proj.0.weight"].shape[0])
    return default


def load_gating_classifier(gating_head_path: str, device: torch.device, d_model_arg: int | None, dropout: float) -> GatingClassifier:
    """Load the K=2 gating classifier.

    `alpha_bins` is not learned, so this eval model creates it dynamically in forward().
    If an older checkpoint contains alpha_bins, we drop it before strict loading.
    If a newer checkpoint does not contain alpha_bins, nothing special is needed.
    """
    path = Path(gating_head_path)
    _logger.info(f"Loading trained gating classifier from {path}...")
    ckpt = _torch_load_any(path, map_location=device)
    state_dict, meta = _extract_state_dict_and_meta(ckpt)

    # Remove deterministic non-learned compatibility buffer if present.
    state_dict = dict(state_dict)
    state_dict.pop("alpha_bins", None)

    d_model = int(d_model_arg) if d_model_arg is not None else _infer_d_model_from_state_dict(state_dict)
    _logger.info(f"Using gating d_model={d_model}, dropout={dropout}")
    if meta:
        _logger.info(f"Checkpoint metadata keys: {sorted(meta.keys())}")

    model = GatingClassifier(d_model=d_model, dropout=dropout).to(device)

    state_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in state_dict.items()
    }

    incompatible = model.load_state_dict(state_dict, strict=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if missing or unexpected:
        raise RuntimeError(
            "Gating checkpoint/state_dict mismatch in learned params. "
            f"Missing={missing}, Unexpected={unexpected}"
        )

    model.eval()
    _logger.info("Loaded gating checkpoint successfully. alpha_bins are initialized dynamically, not loaded.")
    return model

def load_model(config_path: str, device: torch.device):
    """Load expert coordinate-head regressor model following repo structure."""
    _logger.info(f"Loading configuration from {config_path}...")
    cfg = yoco.load_config_from_file(
        str(config_path),
        search_paths=[".", "./configs", "./configs_custom", "./configs_auto"],
    )

    reg_cfg = cfg.get("reg", {})
    encoder_cfg = reg_cfg.get("encoder")
    if encoder_cfg is None:
        raise ValueError(f"Could not find encoder config in {config_path}")

    _logger.info("Initializing encoder...")
    encoder = encoders.create_encoder(encoder_cfg)

    head_path = reg_cfg.get("head_path")
    if head_path is None:
        raise ValueError(f"Could not find head_path in {config_path}")
    head_path = Path(head_path)
    if not head_path.is_absolute():
        head_path = repo_root / head_path

    _logger.info(f"Loading expert head checkpoint from {head_path}...")
    head = scr_heads.create_head(head_path)

    regressor = regressors.Regressor(encoder, head).to(device)
    regressor.eval()

    map_embeddings = None
    mean = None
    map_path = reg_cfg.get("map_path")
    if map_path is not None:
        map_path = Path(map_path)
        if not map_path.is_absolute():
            map_path = repo_root / map_path
        if map_path.is_file():
            _logger.info(f"Loading map file from {map_path}...")
            map_dict = _torch_load_any(map_path, map_location=device)
            map_embeddings = map_dict.get("map_embeddings")
            mean = map_dict.get("mean")
        else:
            _logger.warning(f"map_path {map_path} is specified but file does not exist.")

    sst_cfg = cfg.get("sst", {})
    return regressor, map_embeddings, mean, sst_cfg


def build_dataset(dataset_root: str, scene: str, split: str, subsample_factor: int, use_half: bool = True):
    """Load CamLocDataset split matching _common paths."""
    from _common import scene_split_paths

    paths = scene_split_paths(dataset_root, scene, split)
    ds_config = datasets.CamLocDataset.Config(
        rgb_files=paths["rgb"],
        pose_files=paths["poses"],
        calibration_files=paths["calibration"],
        use_color=True,
        subsample_factor=subsample_factor,
        use_half=use_half,
        calibration_source="dataset",
        force_depth=False,
    )
    return datasets.CamLocDataset(ds_config)


def predict_coordinates(regressor, features_bc, map_embeddings, means, use_half: bool = True) -> torch.Tensor:
    """Forward pass to extract 3D coordinate predictions from an expert regressor."""
    b_eval, dim_features = features_bc.shape
    if b_eval % 512 != 0:
        raise ValueError(f"Batch size {b_eval} must be a multiple of 512.")

    features_nchw = features_bc.view(-1, 16, 32, dim_features).permute(0, 3, 1, 2).contiguous()

    with torch.no_grad(), torch.autocast("cuda", enabled=use_half):
        pred_coords_w_n3hw, _ = regressor.get_scene_coordinates(
            features_nchw,
            map_embeddings=map_embeddings,
            means=means,
        )

    pred_coords_w_b3 = pred_coords_w_n3hw.permute(0, 2, 3, 1).flatten(0, 2).float()
    return pred_coords_w_b3


def compute_reprojection_errors(
    predicted_coords: torch.Tensor,
    target_px: torch.Tensor,
    poses_inv: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Per-patch reprojection error in pixels; invalid behind-camera points get 10000."""
    n = predicted_coords.shape[0]
    ones = torch.ones(n, 1, device=predicted_coords.device, dtype=predicted_coords.dtype)
    p_w_hom = torch.cat([predicted_coords, ones], dim=1)
    p_c = torch.bmm(poses_inv[:, :3, :].float(), p_w_hom.unsqueeze(-1)).squeeze(-1)
    p_pixels_hom = torch.bmm(intrinsics.float(), p_c.unsqueeze(-1)).squeeze(-1)
    pred_px = p_pixels_hom[:, :2] / p_pixels_hom[:, 2:3].clamp_min(1e-8)
    repro_err = torch.linalg.norm(pred_px - target_px.float(), dim=1)
    repro_err[p_c[:, 2] < 0.1] = 10000.0
    return repro_err


def quantize_alpha_to_class(alpha: torch.Tensor) -> torch.Tensor:
    return torch.round(alpha.clamp(0.0, 1.0) * 10.0).long().clamp(0, 10)


def alpha_histogram(classes: torch.Tensor) -> np.ndarray:
    return torch.bincount(classes.detach().cpu(), minlength=11).cpu().numpy()


def plot_histogram(path: Path, counts: np.ndarray, title: str):
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 1)
    plt.figure(figsize=(10, 5))
    plt.bar(alphas, counts, width=0.07, edgecolor="black", alpha=0.85)
    plt.xlabel(r"Alpha bin $\alpha$", fontsize=12)
    plt.ylabel("Number of patches", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(alphas)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def summarize_alpha_metrics(
    oracle_class: torch.Tensor,
    hard_class: torch.Tensor,
    soft_alpha: torch.Tensor,
    oracle_alpha: torch.Tensor,
) -> Dict[str, float]:
    soft_class = quantize_alpha_to_class(soft_alpha)
    hard_alpha = hard_class.float() / 10.0

    return {
        "hard_exact_acc": float((hard_class == oracle_class).float().mean().item()),
        "hard_off_by_1_acc": float(((hard_class - oracle_class).abs() <= 1).float().mean().item()),
        "hard_alpha_mae": float((hard_alpha - oracle_alpha).abs().mean().item()),
        "soft_quantized_exact_acc": float((soft_class == oracle_class).float().mean().item()),
        "soft_quantized_off_by_1_acc": float(((soft_class - oracle_class).abs() <= 1).float().mean().item()),
        "soft_alpha_mae": float((soft_alpha - oracle_alpha).abs().mean().item()),
        "mean_oracle_alpha": float(oracle_alpha.mean().item()),
        "mean_hard_alpha": float(hard_alpha.mean().item()),
        "mean_soft_alpha": float(soft_alpha.mean().item()),
        "std_soft_alpha": float(soft_alpha.std(unbiased=False).item()),
    }


def median_pair(values_t: Tuple[float, float]) -> str:
    return f"{values_t[0]:6.2f} cm,  {values_t[1]:4.2f} deg"


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE K=2 gating classifier for camera localization.")
    parser.add_argument("--model0", required=True, type=str, help="Path to first model YAML map config.")
    parser.add_argument("--model1", required=True, type=str, help="Path to second model YAML map config.")
    parser.add_argument("--gating_head", required=True, type=str, help="Path to trained gating head checkpoint.")
    parser.add_argument("--dataset_root", required=True, type=str, help="Path to datasets root folder.")
    parser.add_argument("--scene", required=True, type=str, help="Scene name.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save logs and plots.")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use.")
    parser.add_argument("--max_buffer_size", type=int, default=None, help="Override maximum buffer size.")
    parser.add_argument("--samples_per_image", type=int, default=None, help="Override patches sampled per image.")
    parser.add_argument("--eval_batch_size", type=int, default=51200, help="Patch eval batch size; rounded down to multiple of 512.")
    parser.add_argument("--split", default="test", choices=["train", "test", "both"], help="Split to evaluate.")
    parser.add_argument("--gating_d_model", type=int, default=None, help="Override gate d_model; otherwise infer from checkpoint.")
    parser.add_argument("--gating_dropout", type=float, default=0.0, help="Dropout used to instantiate gate at eval; inactive in eval mode.")
    parser.add_argument("--save_npz", action="store_true", help="Save per-patch alphas/errors for later analysis.")
    
    parser.add_argument("--wandb-entity", type=str, default="yahav6893")
    parser.add_argument("--wandb-project", type=str, default="DACE_gating")
    parser.add_argument("--disable-wandb", action="store_true")
    
    args = parser.parse_args()

    if not args.disable_wandb:
        try:
            import wandb
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"gating_eval_{args.scene}_{timestamp}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=run_name,
                tags=[args.scene, "gating_eval"],
            )
        except Exception as e:
            _logger.warning(f"Failed to initialize W&B: {e}. Proceeding without W&B.")
            args.disable_wandb = True

    _logger.info(f"Eval script version: {SCRIPT_VERSION}")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regressor0, map_embeddings0, mean0, sst_cfg0 = load_model(args.model0, device)
    regressor1, map_embeddings1, mean1, sst_cfg1 = load_model(args.model1, device)
    if regressor0.subsample_factor != regressor1.subsample_factor:
        raise ValueError(
            f"Expert subsample mismatch: {regressor0.subsample_factor} vs {regressor1.subsample_factor}"
        )
    subsample_factor = regressor0.subsample_factor

    gating_model = load_gating_classifier(args.gating_head, device, args.gating_d_model, args.gating_dropout)

    max_buffer_size = args.max_buffer_size if args.max_buffer_size is not None else int(cfg_get(sst_cfg0, "max_buffer_size", 4_000_000))
    samples_per_image = args.samples_per_image if args.samples_per_image is not None else int(cfg_get(sst_cfg0, "samples_per_image", 1024))
    use_half = cfg_get_bool(sst_cfg0, "use_half", True)
    seed = int(cfg_get(sst_cfg0, "base_seed", 2089))
    buffer_creation_workers = int(cfg_get(sst_cfg0, "buffer_creation_workers", 0))
    buffer_creation_batch_size = int(cfg_get(sst_cfg0, "buffer_creation_batch_size", 4))

    _logger.info(
        "Eval buffer config: max_buffer_size=%s, samples_per_image=%s, use_half=%s, "
        "seed=%s, buffer_creation_workers=%s, buffer_creation_batch_size=%s",
        max_buffer_size, samples_per_image, use_half, seed, buffer_creation_workers, buffer_creation_batch_size,
    )

    alpha_bins_np = np.round(np.arange(0.0, 1.01, 0.1), 1)
    alpha_bins = torch.linspace(0.0, 1.0, 11, device=device)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        _logger.info(f"\n================== Evaluating split: {split} ==================")

        dataset = build_dataset(args.dataset_root, args.scene, split, subsample_factor, use_half)

        _logger.info("Creating patch buffer for Expert 0...")
        buf0 = buffer.create_buffer(
            dataset=dataset,
            encoder=regressor0.encoder,
            max_buffer_size=max_buffer_size,
            buffer_device=device.type,
            max_dataset_passes=1,
            num_data_loader_workers=buffer_creation_workers,
            num_samples_per_image=samples_per_image,
            use_half=use_half,
            seed=seed,
            batch_size=buffer_creation_batch_size,
        )

        _logger.info("Creating patch buffer for Expert 1...")
        buf1 = buffer.create_buffer(
            dataset=dataset,
            encoder=regressor1.encoder,
            max_buffer_size=max_buffer_size,
            buffer_device=device.type,
            max_dataset_passes=1,
            num_data_loader_workers=buffer_creation_workers,
            num_samples_per_image=samples_per_image,
            use_half=use_half,
            seed=seed,
            batch_size=buffer_creation_batch_size,
        )

        min_len = min(len(buf0.features), len(buf1.features))
        b_total = (min_len // 512) * 512
        if b_total == 0:
            raise ValueError(f"Not enough samples in buffer ({min_len}) to form a batch of 512.")
        _logger.info(f"Clipping evaluation buffer to {b_total} patches (nearest multiple of 512).")

        features0 = buf0.features[:b_total].to(device)
        features1 = buf1.features[:b_total].to(device)
        target_px = buf0.target_px[:b_total].to(device)
        poses_inv = buf0.poses_inv[:b_total].to(device)
        intrinsics = buf0.intrinsics[:b_total].to(device)
        pose_idx = buf0.pose_idx[:b_total].to(device)

        pred0_all = torch.zeros((b_total, 3), dtype=torch.float32, device=device)
        pred1_all = torch.zeros((b_total, 3), dtype=torch.float32, device=device)
        gating_alphas_hard = torch.zeros(b_total, dtype=torch.float32, device=device)
        gating_alphas_soft = torch.zeros(b_total, dtype=torch.float32, device=device)
        repro_errors_all = torch.zeros((11, b_total), dtype=torch.float32, device=device)

        eval_batch_size = min(args.eval_batch_size, b_total)
        eval_batch_size = max(512, (eval_batch_size // 512) * 512)
        _logger.info(f"Using eval_batch_size={eval_batch_size}")

        for start in tqdm.trange(0, b_total, eval_batch_size, desc="Running expert + gating evaluation"):
            end = min(start + eval_batch_size, b_total)
            # Because b_total and eval_batch_size are multiples of 512, all chunks should be multiples too.
            if (end - start) % 512 != 0:
                end = start + ((end - start) // 512) * 512
            if end <= start:
                continue

            with torch.no_grad():
                pred0_batch = predict_coordinates(regressor0, features0[start:end], map_embeddings0, mean0, use_half)
                pred1_batch = predict_coordinates(regressor1, features1[start:end], map_embeddings1, mean1, use_half)

                pred0_all[start:end] = pred0_batch
                pred1_all[start:end] = pred1_batch

                z_batch = torch.stack([features0[start:end], features1[start:end]], dim=1).float()
                gating_out = gating_model(z_batch)
                gating_alphas_hard[start:end] = gating_out["alpha_hard"]
                gating_alphas_soft[start:end] = gating_out["alpha_soft"]

                for a_idx, alpha in enumerate(alpha_bins):
                    final_pred_batch = alpha * pred0_batch + (1.0 - alpha) * pred1_batch
                    repro_errors_all[a_idx, start:end] = compute_reprojection_errors(
                        final_pred_batch,
                        target_px[start:end],
                        poses_inv[start:end],
                        intrinsics[start:end],
                    )

        # Oracle best alpha with deterministic tie-break: lowest error -> closest to 0.5 -> smaller class index.
        min_errors = repro_errors_all.min(dim=0, keepdim=True).values
        is_min = repro_errors_all <= (min_errors + 1e-6)
        large_penalty = torch.full_like(repro_errors_all, 1e9)
        penalty = torch.where(is_min, torch.zeros_like(repro_errors_all), large_penalty)
        dist_to_05 = torch.abs(alpha_bins[:, None] - 0.5)
        class_indices = torch.arange(11, dtype=torch.float32, device=device)[:, None]
        priority_scores = penalty + dist_to_05 * 1000.0 + class_indices
        oracle_class = torch.argmin(priority_scores, dim=0)
        oracle_alphas = alpha_bins[oracle_class]

        hard_class = quantize_alpha_to_class(gating_alphas_hard)
        soft_class = quantize_alpha_to_class(gating_alphas_soft)

        patch_ids = torch.arange(b_total, device=device)
        oracle_reproj = repro_errors_all[oracle_class, patch_ids]
        hard_reproj = repro_errors_all[hard_class, patch_ids]
        expert0_reproj = repro_errors_all[10, :]
        expert1_reproj = repro_errors_all[0, :]
        soft_pred_all = gating_alphas_soft[:, None] * pred0_all + (1.0 - gating_alphas_soft[:, None]) * pred1_all
        soft_reproj = compute_reprojection_errors(soft_pred_all, target_px, poses_inv, intrinsics)

        import dsacstar

        def evaluate_pose_errors(predicted_coords: torch.Tensor) -> Tuple[float, float]:
            unique_pose_indices = torch.unique(pose_idx).detach().cpu().numpy()
            translation_errors = []
            rotation_errors = []

            for p_idx in unique_pose_indices:
                mask = (pose_idx == p_idx).squeeze()
                px_img = target_px[mask].detach()
                crd_img = predicted_coords[mask].detach().float()

                non_zero_mask = (crd_img != 0).any(dim=1)
                px_img = px_img[non_zero_mask]
                crd_img = crd_img[non_zero_mask]
                if len(crd_img) < 10:
                    continue

                intr_img = intrinsics[mask][0].detach().cpu().numpy()
                w2c_gt = poses_inv[mask][0].detach().cpu().numpy()
                c2w_gt = np.linalg.inv(w2c_gt)

                x_grid = torch.round(px_img[:, 0] / subsample_factor - 0.5).long().clamp(min=0)
                y_grid = torch.round(px_img[:, 1] / subsample_factor - 0.5).long().clamp(min=0)
                h_grid = int(y_grid.max().item()) + 1
                w_grid = int(x_grid.max().item()) + 1

                scene_coordinates_b3hw = torch.zeros((1, 3, h_grid, w_grid), dtype=torch.float32, device="cpu")
                scene_coordinates_b3hw[0, :, y_grid.cpu(), x_grid.cpu()] = crd_img.cpu().T

                focal_length = float(intr_img[0, 0])
                pp_x = float(intr_img[0, 2])
                pp_y = float(intr_img[1, 2])
                out_pose_c2w = torch.zeros((4, 4), dtype=torch.float32, device="cpu")

                dsacstar.forward_rgb(
                    scene_coordinates_b3hw,
                    out_pose_c2w,
                    64,
                    10.0,
                    focal_length,
                    pp_x,
                    pp_y,
                    100.0,
                    100.0,
                    subsample_factor,
                    2089,
                )

                c2w_est = out_pose_c2w.cpu().numpy()
                if np.allclose(c2w_est, 0):
                    continue

                t_err = np.linalg.norm(c2w_gt[0:3, 3] - c2w_est[0:3, 3]) * 100.0
                gt_r = c2w_gt[0:3, 0:3]
                out_r = c2w_est[0:3, 0:3]
                r_err = np.matmul(out_r, np.transpose(gt_r))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = float(np.linalg.norm(r_err) * 180.0 / np.pi)

                translation_errors.append(t_err)
                rotation_errors.append(r_err)

            if len(translation_errors) == 0:
                return 999.0, 999.0
            return float(np.median(translation_errors)), float(np.median(rotation_errors))

        _logger.info("Evaluating Expert 0 (alpha = 1.0)...")
        med_expert0 = evaluate_pose_errors(pred0_all)

        _logger.info("Evaluating Expert 1 (alpha = 0.0)...")
        med_expert1 = evaluate_pose_errors(pred1_all)

        _logger.info("Evaluating Oracle Best alpha blend...")
        oracle_pred = oracle_alphas[:, None] * pred0_all + (1.0 - oracle_alphas[:, None]) * pred1_all
        med_oracle = evaluate_pose_errors(oracle_pred)

        _logger.info("Evaluating Learned Gating Head (Hard argmax blend)...")
        hard_pred = gating_alphas_hard[:, None] * pred0_all + (1.0 - gating_alphas_hard[:, None]) * pred1_all
        med_hard = evaluate_pose_errors(hard_pred)

        _logger.info("Evaluating Learned Gating Head (Soft probability blend)...")
        med_soft = evaluate_pose_errors(soft_pred_all)

        hist_oracle = alpha_histogram(oracle_class)
        hist_hard = alpha_histogram(hard_class)
        hist_soft_quant = alpha_histogram(soft_class)
        alpha_metrics = summarize_alpha_metrics(oracle_class, hard_class, gating_alphas_soft, oracle_alphas)

        repro_metrics = {
            "mean_reproj_expert0": float(expert0_reproj.mean().item()),
            "mean_reproj_expert1": float(expert1_reproj.mean().item()),
            "mean_reproj_oracle": float(oracle_reproj.mean().item()),
            "mean_reproj_hard": float(hard_reproj.mean().item()),
            "mean_reproj_soft": float(soft_reproj.mean().item()),
            "mean_reproj_regret_hard_minus_oracle": float((hard_reproj - oracle_reproj).mean().item()),
            "mean_reproj_regret_soft_minus_oracle": float((soft_reproj - oracle_reproj).mean().item()),
            "median_reproj_oracle": float(oracle_reproj.median().item()),
            "median_reproj_hard": float(hard_reproj.median().item()),
            "median_reproj_soft": float(soft_reproj.median().item()),
        }

        log_path = output_dir / f"gating_eval_{split}.log"
        json_path = output_dir / f"gating_eval_{split}.json"
        with open(log_path, "w", encoding="utf-8") as log_file:
            def log_and_print(msg: str):
                _logger.info(msg)
                log_file.write(msg + "\n")

            log_and_print("-" * 65)
            log_and_print(f" MoE Gating Evaluation Results - Split: {split}")
            log_and_print(f" Scene: {args.scene}")
            log_and_print(f" Total Patches Evaluated: {b_total}")
            log_and_print("-" * 65)
            log_and_print(" Camera Localization Median Pose Errors:")
            log_and_print(f"   Expert 0 (alpha = 1.0):      {median_pair(med_expert0)}")
            log_and_print(f"   Expert 1 (alpha = 0.0):      {median_pair(med_expert1)}")
            log_and_print(f"   Oracle Best Alpha Blend:     {median_pair(med_oracle)}")
            log_and_print(f"   Gating Head (Hard Blend):    {median_pair(med_hard)}")
            log_and_print(f"   Gating Head (Soft Blend):    {median_pair(med_soft)}")
            log_and_print("-" * 65)
            log_and_print(" Alpha Agreement Metrics vs Oracle:")
            log_and_print(f"   Hard exact acc:              {100.0 * alpha_metrics['hard_exact_acc']:6.2f}%")
            log_and_print(f"   Hard off-by-1 acc:           {100.0 * alpha_metrics['hard_off_by_1_acc']:6.2f}%")
            log_and_print(f"   Hard alpha MAE:              {alpha_metrics['hard_alpha_mae']:.4f}")
            log_and_print(f"   Soft quantized exact acc:    {100.0 * alpha_metrics['soft_quantized_exact_acc']:6.2f}%")
            log_and_print(f"   Soft quantized off-by-1 acc: {100.0 * alpha_metrics['soft_quantized_off_by_1_acc']:6.2f}%")
            log_and_print(f"   Soft alpha MAE:              {alpha_metrics['soft_alpha_mae']:.4f}")
            log_and_print(f"   Mean oracle alpha:           {alpha_metrics['mean_oracle_alpha']:.4f}")
            log_and_print(f"   Mean hard alpha:             {alpha_metrics['mean_hard_alpha']:.4f}")
            log_and_print(f"   Mean soft alpha:             {alpha_metrics['mean_soft_alpha']:.4f}")
            log_and_print("-" * 65)
            log_and_print(" Patch Reprojection Metrics:")
            for key, value in repro_metrics.items():
                log_and_print(f"   {key}: {value:.4f}")
            log_and_print("-" * 65)

            def print_dist(title: str, counts: np.ndarray):
                log_and_print(f" {title}:")
                for idx in range(11):
                    pct = (counts[idx] / b_total) * 100.0
                    log_and_print(f"   alpha = {idx / 10.0:.1f}:   {counts[idx]:8d} patches ({pct:6.2f}%)")
                log_and_print("-" * 65)

            print_dist("Oracle Alpha Distribution", hist_oracle)
            print_dist("Gating Hard Alpha Distribution", hist_hard)
            print_dist("Gating Soft Alpha Quantized Distribution", hist_soft_quant)

        summary = {
            "split": split,
            "scene": args.scene,
            "total_patches": int(b_total),
            "pose_median_errors": {
                "expert0_alpha_1": {"cm": med_expert0[0], "deg": med_expert0[1]},
                "expert1_alpha_0": {"cm": med_expert1[0], "deg": med_expert1[1]},
                "oracle": {"cm": med_oracle[0], "deg": med_oracle[1]},
                "gating_hard": {"cm": med_hard[0], "deg": med_hard[1]},
                "gating_soft": {"cm": med_soft[0], "deg": med_soft[1]},
            },
            "alpha_metrics": alpha_metrics,
            "reprojection_metrics": repro_metrics,
            "hist_oracle": hist_oracle.tolist(),
            "hist_hard": hist_hard.tolist(),
            "hist_soft_quantized": hist_soft_quant.tolist(),
            "args": vars(args),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _logger.info(f"Saved eval log to {log_path}")
        _logger.info(f"Saved eval JSON to {json_path}")

        plot_histogram(output_dir / f"gating_eval_{split}_oracle_histogram.png", hist_oracle, f"Oracle Alpha Distribution ({split})")
        plot_histogram(output_dir / f"gating_eval_{split}_hard_histogram.png", hist_hard, f"Gating Hard Alpha Distribution ({split})")
        plot_histogram(output_dir / f"gating_eval_{split}_soft_quant_histogram.png", hist_soft_quant, f"Gating Soft Alpha Quantized Distribution ({split})")

        if args.save_npz:
            npz_path = output_dir / f"gating_eval_{split}_per_patch.npz"
            np.savez_compressed(
                npz_path,
                oracle_class=oracle_class.detach().cpu().numpy(),
                oracle_alpha=oracle_alphas.detach().cpu().numpy(),
                hard_class=hard_class.detach().cpu().numpy(),
                hard_alpha=gating_alphas_hard.detach().cpu().numpy(),
                soft_alpha=gating_alphas_soft.detach().cpu().numpy(),
                oracle_reproj=oracle_reproj.detach().cpu().numpy(),
                hard_reproj=hard_reproj.detach().cpu().numpy(),
                soft_reproj=soft_reproj.detach().cpu().numpy(),
                expert0_reproj=expert0_reproj.detach().cpu().numpy(),
                expert1_reproj=expert1_reproj.detach().cpu().numpy(),
            )
            _logger.info(f"Saved per-patch NPZ to {npz_path}")

        if not args.disable_wandb:
            try:
                import wandb
                wandb_log_dict = {
                    f"gating_eval/{split}/total_patches": int(b_total),
                    f"gating_eval/{split}/pose_err/expert0_alpha_1_cm": med_expert0[0],
                    f"gating_eval/{split}/pose_err/expert0_alpha_1_deg": med_expert0[1],
                    f"gating_eval/{split}/pose_err/expert1_alpha_0_cm": med_expert1[0],
                    f"gating_eval/{split}/pose_err/expert1_alpha_0_deg": med_expert1[1],
                    f"gating_eval/{split}/pose_err/oracle_cm": med_oracle[0],
                    f"gating_eval/{split}/pose_err/oracle_deg": med_oracle[1],
                    f"gating_eval/{split}/pose_err/gating_hard_cm": med_hard[0],
                    f"gating_eval/{split}/pose_err/gating_hard_deg": med_hard[1],
                    f"gating_eval/{split}/pose_err/gating_soft_cm": med_soft[0],
                    f"gating_eval/{split}/pose_err/gating_soft_deg": med_soft[1],
                }
                for k, v in alpha_metrics.items():
                    wandb_log_dict[f"gating_eval/{split}/alpha/{k}"] = v
                for k, v in repro_metrics.items():
                    wandb_log_dict[f"gating_eval/{split}/reproj/{k}"] = v

                oracle_hist_path = output_dir / f"gating_eval_{split}_oracle_histogram.png"
                hard_hist_path = output_dir / f"gating_eval_{split}_hard_histogram.png"
                soft_hist_path = output_dir / f"gating_eval_{split}_soft_quant_histogram.png"

                if oracle_hist_path.is_file():
                    wandb_log_dict[f"gating_eval/{split}/plots/oracle_hist"] = wandb.Image(str(oracle_hist_path))
                if hard_hist_path.is_file():
                    wandb_log_dict[f"gating_eval/{split}/plots/hard_hist"] = wandb.Image(str(hard_hist_path))
                if soft_hist_path.is_file():
                    wandb_log_dict[f"gating_eval/{split}/plots/soft_hist"] = wandb.Image(str(soft_hist_path))

                wandb.log(wandb_log_dict)

                art = wandb.Artifact(name=f"gating_eval_{args.scene}_{split}", type="eval-results")
                art.add_file(str(json_path))
                art.add_file(str(log_path))
                wandb.log_artifact(art)
            except Exception as e:
                _logger.warning(f"Failed to log to W&B: {e}")

    if not args.disable_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
