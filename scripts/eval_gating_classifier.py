#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Add repo root to python path to import ace_g and _common
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "scripts"))

import _common
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yoco
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from ace_g import encoders, scr_heads, regressors, datasets, buffer, utils

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)


class GatingClassifier(nn.Module):
    """Gating network classifier mapping two head embeddings to an 11-class alpha distribution."""
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_ln = nn.LayerNorm(1024)
        self.shared_proj = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.encoder_id_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.encoder_id_embed, std=0.02)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 11)
        )

    def forward(self, z: torch.Tensor) -> dict:
        B = z.shape[0]
        z = self.input_ln(z)
        z = self.shared_proj(z)
        z = z + self.encoder_id_embed
        z_flat = z.view(B, -1)
        logits = self.classifier(z_flat)
        
        pred_class = logits.argmax(dim=-1)
        alpha = pred_class.float() / 10.0
        weights = torch.stack([alpha, 1.0 - alpha], dim=-1)
        
        probs = logits.softmax(dim=-1)
        alpha_soft = probs @ torch.linspace(0, 1, 11, device=z.device)
        
        return {
            "logits": logits,
            "pred_class": pred_class,
            "alpha": alpha,
            "weights": weights,
            "alpha_soft": alpha_soft
        }


def load_model(config_path, device="cuda"):
    """Load expert coordinate head regressor model following repo structure."""
    _logger.info(f"Loading configuration from {config_path}...")
    cfg = yoco.load_config_from_file(str(config_path), search_paths=[".", "./configs", "./configs_custom", "./configs_auto"])
    
    # Resolving reg config
    reg_cfg = cfg.get("reg", {})
    
    # Extract encoder config
    encoder_cfg = reg_cfg.get("encoder")
    if encoder_cfg is None:
        raise ValueError(f"Could not find encoder config in {config_path}")
        
    _logger.info("Initializing encoder...")
    encoder = encoders.create_encoder(encoder_cfg)
    
    # Load head
    head_path = reg_cfg.get("head_path")
    if head_path is None:
        raise ValueError(f"Could not find head_path in {config_path}")
    head_path = Path(head_path)
    if not head_path.is_absolute():
        head_path = repo_root / head_path
    
    _logger.info(f"Loading expert head checkpoints from {head_path}...")
    head = scr_heads.create_head(head_path)
    
    # Create regressor
    regressor = regressors.Regressor(encoder, head)
    regressor = regressor.to(device)
    regressor.eval()
    
    # Load map path or embeddings if present
    map_embeddings = None
    mean = None
    
    map_path = reg_cfg.get("map_path")
    if map_path is not None:
        map_path = Path(map_path)
        if not map_path.is_absolute():
            map_path = repo_root / map_path
        if map_path.is_file():
            _logger.info(f"Loading map file from {map_path}...")
            map_dict = torch.load(map_path, map_location=device, weights_only=False)
            map_embeddings = map_dict.get("map_embeddings")
            mean = map_dict.get("mean")
        else:
            _logger.warning(f"map_path {map_path} is specified but file does not exist.")
            
    # Also extract the buffer parameters from the "sst" section if present
    sst_cfg = cfg.get("sst", {})
    
    return regressor, map_embeddings, mean, sst_cfg


def build_dataset(dataset_root, scene, split, subsample_factor, use_half=True):
    """Load CamLocDataset split matching the _common paths."""
    from _common import scene_split_paths
    paths = scene_split_paths(dataset_root, scene, split)
    
    # Create dataset config
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
    
    dataset = datasets.CamLocDataset(ds_config)
    return dataset


def predict_coordinates(regressor, features_bc, map_embeddings, means, use_half=True):
    """Forward pass to extract 3D coordinate predictions from expert regressor."""
    B_eval, dim_features = features_bc.shape
    assert B_eval % 512 == 0, f"Batch size {B_eval} must be a multiple of 512."
    
    features_nchw = features_bc.view(-1, 16, 32, dim_features).permute(0, 3, 1, 2)
    
    with torch.autocast("cuda", enabled=use_half):
        pred_coords_w_n3hw, _ = regressor.get_scene_coordinates(
            features_nchw,
            map_embeddings=map_embeddings,
            means=means,
        )
        
    pred_coords_w_b3 = pred_coords_w_n3hw.permute(0, 2, 3, 1).flatten(0, 2).float()
    return pred_coords_w_b3


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE Gating network for Camera Localization.")
    parser.add_argument("--model0", required=True, type=str, help="Path to first model YAML map config.")
    parser.add_argument("--model1", required=True, type=str, help="Path to second model YAML map config.")
    parser.add_argument("--gating_head", required=True, type=str, help="Path to trained gating head weights.")
    parser.add_argument("--dataset_root", required=True, type=str, help="Path to datasets root folder.")
    parser.add_argument("--scene", required=True, type=str, help="Cambridge scene name.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save logs and plots.")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use.")
    parser.add_argument("--max_buffer_size", type=int, default=None, help="Override maximum buffer size.")
    parser.add_argument("--samples_per_image", type=int, default=None, help="Override patches sampled per image.")
    parser.add_argument("--split", default="test", choices=["train", "test", "both"], help="Which split to evaluate: train, test, or both.")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load expert models
    regressor0, map_embeddings0, mean0, sst_cfg0 = load_model(args.model0, device)
    regressor1, map_embeddings1, mean1, sst_cfg1 = load_model(args.model1, device)
    assert regressor0.subsample_factor == regressor1.subsample_factor
    subsample_factor = regressor0.subsample_factor

    # 2. Load and initialize gating head
    _logger.info(f"Loading trained gating classifier from {args.gating_head}...")
    gating_model = GatingClassifier().to(device)
    gating_model.load_state_dict(torch.load(args.gating_head, map_location=device))
    gating_model.eval()

    max_buffer_size = args.max_buffer_size or sst_cfg0.get("max_buffer_size", 4_000_000)
    samples_per_image = args.samples_per_image or sst_cfg0.get("samples_per_image", 1024)
    use_half = sst_cfg0.get("use_half", True)
    seed = sst_cfg0.get("base_seed", 2089)
    buffer_creation_workers = sst_cfg0.get("buffer_creation_workers", 0)
    buffer_creation_batch_size = sst_cfg0.get("buffer_creation_batch_size", 4)

    alphas = np.round(np.arange(0.0, 1.01, 0.1), 1)
    
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    for split in splits:
        _logger.info(f"\n================== Evaluating split: {split} ==================")
        
        # Build dataset & buffers
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
        B_total = (min_len // 512) * 512
        if B_total == 0:
            raise ValueError(f"Not enough samples in buffer ({min_len}) to form a batch of size 512.")
            
        _logger.info(f"Clipping evaluation buffer to {B_total} patches (nearest multiple of 512).")
        
        features0 = buf0.features[:B_total].to(device)
        features1 = buf1.features[:B_total].to(device)
        target_px = buf0.target_px[:B_total].to(device)
        poses_inv = buf0.poses_inv[:B_total].to(device)
        intrinsics = buf0.intrinsics[:B_total].to(device)
        pose_idx = buf0.pose_idx[:B_total].to(device)

        # Storage for all predicted coordinates
        pred0_all = torch.zeros((B_total, 3), dtype=torch.float32, device=device)
        pred1_all = torch.zeros((B_total, 3), dtype=torch.float32, device=device)
        
        # Storage for predicted alphas from Gating Network
        gating_alphas_hard = torch.zeros(B_total, dtype=torch.float32, device=device)
        gating_alphas_soft = torch.zeros(B_total, dtype=torch.float32, device=device)
        
        # Storage for reprojection errors to compute the oracle best alpha
        repro_errors_all = torch.zeros((len(alphas), B_total), device=device)

        eval_batch_size = 51200
        if B_total < eval_batch_size:
            eval_batch_size = B_total
            
        num_batches = B_total // eval_batch_size
        
        for b_idx in tqdm.trange(num_batches, desc="Running gating-informed expert evaluation"):
            start = b_idx * eval_batch_size
            end = start + eval_batch_size
            
            # Predict coordinates for both models
            pred0_batch = predict_coordinates(regressor0, features0[start:end], map_embeddings0, mean0, use_half)
            pred1_batch = predict_coordinates(regressor1, features1[start:end], map_embeddings1, mean1, use_half)
            
            pred0_all[start:end] = pred0_batch
            pred1_all[start:end] = pred1_batch
            
            target_px_batch = target_px[start:end]
            poses_inv_batch = poses_inv[start:end]
            intrinsics_batch = intrinsics[start:end]
            
            # Predict blending alphas via gating network
            z_batch = torch.stack([features0[start:end], features1[start:end]], dim=1).float()
            with torch.no_grad():
                gating_out = gating_model(z_batch)
                gating_alphas_hard[start:end] = gating_out["alpha"]
                gating_alphas_soft[start:end] = gating_out["alpha_soft"]

            # Compute reprojection errors to determine oracle
            for a_idx, alpha in enumerate(alphas):
                final_pred_batch = alpha * pred0_batch + (1.0 - alpha) * pred1_batch
                p_w_hom = torch.cat([final_pred_batch, torch.ones(eval_batch_size, 1, device=device)], dim=1)
                p_c = torch.bmm(poses_inv_batch[:, :3, :], p_w_hom.unsqueeze(-1)).squeeze(-1)
                p_pixels_hom = torch.bmm(intrinsics_batch, p_c.unsqueeze(-1)).squeeze(-1)
                pred_px = p_pixels_hom[:, :2] / p_pixels_hom[:, 2:3]
                
                repro_err = torch.linalg.norm(pred_px - target_px_batch, dim=1)
                behind_camera_mask = p_c[:, 2] < 0.1
                repro_err[behind_camera_mask] = 10000.0
                
                repro_errors_all[a_idx, start:end] = repro_err

        # Compute Oracle Best Alpha via tie-breaking formula
        min_errors_global = torch.min(repro_errors_all, dim=0, keepdim=True).values
        is_min_global = (repro_errors_all <= min_errors_global + 1e-6)
        large_penalty = 1e9
        penalty_global = torch.where(is_min_global, 0.0, large_penalty)
        dist_to_05_global = torch.tensor([abs(a / 10.0 - 0.5) for a in range(11)], dtype=torch.float32, device=device).unsqueeze(-1)
        class_indices_global = torch.arange(11, dtype=torch.float32, device=device).unsqueeze(-1)
        priority_scores_global = penalty_global + dist_to_05_global * 1000.0 + class_indices_global
        best_alpha_indices = torch.argmin(priority_scores_global, dim=0)
        oracle_alphas = torch.from_numpy(alphas[best_alpha_indices.cpu().numpy()]).to(device)

        import dsacstar

        # Camera Localization RANSAC Helper
        def evaluate_pose_errors(predicted_coords):
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
                
                x_grid = torch.round(px_img[:, 0] / subsample_factor - 0.5).long()
                y_grid = torch.round(px_img[:, 1] / subsample_factor - 0.5).long()
                x_grid = torch.clamp(x_grid, min=0)
                y_grid = torch.clamp(y_grid, min=0)
                
                H_grid = int(y_grid.max().item()) + 1
                W_grid = int(x_grid.max().item()) + 1
                
                scene_coordinates_B3HW = torch.zeros((1, 3, H_grid, W_grid), dtype=torch.float32, device="cpu")
                scene_coordinates_B3HW[0, :, y_grid.cpu(), x_grid.cpu()] = crd_img.cpu().T
                
                focal_length = float(intr_img[0, 0])
                ppX = float(intr_img[0, 2])
                ppY = float(intr_img[1, 2])
                
                out_pose_c2w = torch.zeros((4, 4), dtype=torch.float32, device="cpu")
                
                dsacstar.forward_rgb(
                    scene_coordinates_B3HW,
                    out_pose_c2w,
                    64,
                    10.0,
                    focal_length,
                    ppX,
                    ppY,
                    100.0,
                    100.0,
                    subsample_factor,
                    2089,
                )
                
                c2w_est = out_pose_c2w.cpu().numpy()
                if np.allclose(c2w_est, 0):
                    continue
                    
                t_err = np.linalg.norm(c2w_gt[0:3, 3] - c2w_est[0:3, 3]) * 100.0
                gt_R = c2w_gt[0:3, 0:3]
                out_R = c2w_est[0:3, 0:3]
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = float(np.linalg.norm(r_err) * 180 / np.pi)
                
                translation_errors.append(t_err)
                rotation_errors.append(r_err)
                
            if len(translation_errors) == 0:
                return 999.0, 999.0
            return np.median(translation_errors), np.median(rotation_errors)

        # 1. Expert 0 alone
        _logger.info("Evaluating Expert 0 (alpha = 1.0)...")
        med_t0, med_r0 = evaluate_pose_errors(pred0_all)

        # 2. Expert 1 alone
        _logger.info("Evaluating Expert 1 (alpha = 0.0)...")
        med_t1, med_r1 = evaluate_pose_errors(pred1_all)

        # 3. Oracle Best alpha blend
        _logger.info("Evaluating Oracle Best alpha blend...")
        oracle_pred = oracle_alphas.unsqueeze(-1) * pred0_all + (1.0 - oracle_alphas.unsqueeze(-1)) * pred1_all
        med_t_oracle, med_r_oracle = evaluate_pose_errors(oracle_pred)

        # 4. Gating Head Hard Blend
        _logger.info("Evaluating Learned Gating Head (Hard argmax blend)...")
        hard_pred = gating_alphas_hard.unsqueeze(-1) * pred0_all + (1.0 - gating_alphas_hard.unsqueeze(-1)) * pred1_all
        med_t_hard, med_r_hard = evaluate_pose_errors(hard_pred)

        # 5. Gating Head Soft Blend
        _logger.info("Evaluating Learned Gating Head (Soft probability blend)...")
        soft_pred = gating_alphas_soft.unsqueeze(-1) * pred0_all + (1.0 - gating_alphas_soft.unsqueeze(-1)) * pred1_all
        med_t_soft, med_r_soft = evaluate_pose_errors(soft_pred)

        # Frequency distribution metrics for predicted alphas
        hist_hard = torch.bincount((gating_alphas_hard * 10).long(), minlength=11).cpu().numpy()

        log_path = output_dir / f"gating_eval_{split}.log"
        with open(log_path, "w") as log_file:
            def log_and_print(msg):
                _logger.info(msg)
                log_file.write(msg + "\n")

            log_and_print("-" * 65)
            log_and_print(f" MoE Gating Evaluation Results - Split: {split}")
            log_and_print(f" Scene: {args.scene}")
            log_and_print(f" Total Patches Evaluated: {B_total}")
            log_and_print("-" * 65)
            log_and_print(" Camera Localization Median Pose Errors:")
            log_and_print(f"   Expert 0 (alpha = 1.0):      {med_t0:6.2f} cm,  {med_r0:4.2f} deg")
            log_and_print(f"   Expert 1 (alpha = 0.0):      {med_t1:6.2f} cm,  {med_r1:4.2f} deg")
            log_and_print(f"   Oracle Best Alpha Blend:     {med_t_oracle:6.2f} cm,  {med_r_oracle:4.2f} deg")
            log_and_print(f"   Gating Head (Hard Blend):    {med_t_hard:6.2f} cm,  {med_r_hard:4.2f} deg")
            log_and_print(f"   Gating Head (Soft Blend):    {med_t_soft:6.2f} cm,  {med_r_soft:4.2f} deg")
            log_and_print("-" * 65)
            log_and_print(" Gating Head Predicted Hard Alpha Distribution:")
            for idx in range(11):
                pct = (hist_hard[idx] / B_total) * 100.0
                log_and_print(f"   alpha = {idx/10.0:.1f}:   {hist_hard[idx]:8d} patches ({pct:6.2f}%)")
            log_and_print("-" * 65)

        # Plot predicted hard alpha distribution
        plot_path = output_dir / f"gating_eval_{split}_histogram.png"
        plt.figure(figsize=(10, 5))
        plt.bar(alphas, hist_hard, width=0.07, color='#2c3e50', edgecolor='black', alpha=0.85)
        plt.xlabel(r"Predicted expert weight $\alpha$", fontsize=12)
        plt.ylabel("Number of Patches", fontsize=12)
        plt.title(f"Gating Network Predicted Alpha Frequency ({split.capitalize()} Split)", fontsize=14, fontweight='bold')
        plt.xticks(alphas)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        _logger.info(f"Saved predicted alpha histogram plot to {plot_path}")


if __name__ == "__main__":
    main()
