#!/usr/bin/env python3
"""Offline blend/fusion test script to find optimal blending parameter alpha.

Evaluates:
  final_pred = alpha * pred0 + (1 - alpha) * pred1
across alpha in [0.0, 0.1, ..., 1.0] using a patch buffer.
"""

import argparse
import logging
import pathlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yoco
import cv2

# Add repo root to python path to import ace_g and _common
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "scripts"))

import _common
from ace_g import encoders, scr_heads, regressors, datasets, buffer, utils

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)


def load_model(config_path, device="cuda"):
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
    B_eval, dim_features = features_bc.shape
    assert B_eval % 512 == 0, f"Batch size {B_eval} must be a multiple of 512."
    
    # Reshape to fake NCHW shape
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
    parser = argparse.ArgumentParser(description="Offline Fusion grid search on blending parameter alpha.")
    parser.add_argument("--model0", required=True, type=str, help="Path to first model YAML map config.")
    parser.add_argument("--model1", required=True, type=str, help="Path to second model YAML map config.")
    parser.add_argument("--dataset_root", required=True, type=str, help="Path to datasets root folder.")
    parser.add_argument("--scene", required=True, type=str, help="Cambridge scene name (e.g. shopfacade).")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save logs and plots.")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use (e.g. cuda, cpu).")
    parser.add_argument("--max_buffer_size", type=int, default=None, help="Override maximum buffer size.")
    parser.add_argument("--samples_per_image", type=int, default=None, help="Override patches sampled per image.")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    regressor0, map_embeddings0, mean0, sst_cfg0 = load_model(args.model0, device)
    regressor1, map_embeddings1, mean1, sst_cfg1 = load_model(args.model1, device)

    # Assert that they use same subsample factor
    assert regressor0.subsample_factor == regressor1.subsample_factor, \
        f"Subsample factors must match! Model 0: {regressor0.subsample_factor}, Model 1: {regressor1.subsample_factor}"

    subsample_factor = regressor0.subsample_factor

    # 2. Extract buffer params from config
    max_buffer_size = args.max_buffer_size or sst_cfg0.get("max_buffer_size", 4_000_000)
    samples_per_image = args.samples_per_image or sst_cfg0.get("samples_per_image", 1024)
    use_half = sst_cfg0.get("use_half", True)
    seed = sst_cfg0.get("base_seed", 2089)
    buffer_creation_workers = sst_cfg0.get("buffer_creation_workers", 0)
    buffer_creation_batch_size = sst_cfg0.get("buffer_creation_batch_size", 4)

    _logger.info("Buffer configuration:")
    _logger.info(f"  max_buffer_size: {max_buffer_size}")
    _logger.info(f"  samples_per_image: {samples_per_image}")
    _logger.info(f"  use_half: {use_half}")
    _logger.info(f"  seed: {seed}")
    _logger.info(f"  buffer_creation_workers: {buffer_creation_workers}")
    _logger.info(f"  buffer_creation_batch_size: {buffer_creation_batch_size}")

    # Evaluate on both splits
    splits = ["train", "test"]
    for split in splits:
        _logger.info(f"================== Processing split: {split} ==================")
        
        # Build dataset
        dataset = build_dataset(args.dataset_root, args.scene, split, subsample_factor, use_half)
        
        # Build buffers for both models (identical coordinate samples, poses, and indices)
        _logger.info("Creating patch buffer for Model 0...")
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
        
        _logger.info("Creating patch buffer for Model 1...")
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

        # Clip both buffers to the same size (nearest multiple of 512)
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

        # Run grid search
        alphas = np.round(np.arange(0.0, 1.01, 0.1), 1)
        repro_errors_all = torch.zeros((len(alphas), B_total), device=device)
        pred0_all = torch.zeros((B_total, 3), dtype=torch.float32, device=device)
        pred1_all = torch.zeros((B_total, 3), dtype=torch.float32, device=device)
        
        eval_batch_size = 51200
        if B_total < eval_batch_size:
            eval_batch_size = B_total
            
        num_batches = B_total // eval_batch_size
        
        for b_idx in tqdm.trange(num_batches, desc="Evaluating blended predictions"):
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
            
            for a_idx, alpha in enumerate(alphas):
                final_pred_batch = alpha * pred0_batch + (1.0 - alpha) * pred1_batch
                
                # Project to homogeneous 2D
                p_w_hom = torch.cat([final_pred_batch, torch.ones(eval_batch_size, 1, device=device)], dim=1)
                
                # Transform to camera space
                p_c = torch.bmm(poses_inv_batch[:, :3, :], p_w_hom.unsqueeze(-1)).squeeze(-1)
                
                # Project to pixel space
                p_pixels_hom = torch.bmm(intrinsics_batch, p_c.unsqueeze(-1)).squeeze(-1)
                
                # Dehomogenize
                z_clamp = p_pixels_hom[:, 2:3].clamp(min=0.1)
                pred_px = p_pixels_hom[:, :2] / z_clamp
                
                # Reprojection error is the L2 pixel distance
                repro_err = torch.linalg.norm(pred_px - target_px_batch, dim=1)
                
                # Assign a large penalty to behind-the-camera predictions
                behind_camera_mask = p_c[:, 2] < 0.1
                repro_err[behind_camera_mask] = 10000.0
                
                repro_errors_all[a_idx, start:end] = repro_err

        # Find the best alpha for each patch using tie-break rule:
        # lowest error -> closest to 0.5 -> smaller class index
        min_errors_global = torch.min(repro_errors_all, dim=0, keepdim=True).values  # (1, B_total)
        is_min_global = (repro_errors_all <= min_errors_global + 1e-6)                # (11, B_total)
        
        large_penalty = 1e9
        penalty_global = torch.where(is_min_global, 0.0, large_penalty)               # (11, B_total)
        
        dist_to_05_global = torch.tensor([abs(a / 10.0 - 0.5) for a in range(11)], dtype=torch.float32, device=device).unsqueeze(-1)  # (11, 1)
        class_indices_global = torch.arange(11, dtype=torch.float32, device=device).unsqueeze(-1)                                    # (11, 1)
        
        priority_scores_global = penalty_global + dist_to_05_global * 1000.0 + class_indices_global  # (11, B_total)
        best_alpha_indices = torch.argmin(priority_scores_global, dim=0)                             # (B_total,)
        best_alphas = alphas[best_alpha_indices.cpu().numpy()]

        import dsacstar

        # Helper to compute camera pose errors via dsacstar RANSAC
        def evaluate_pose_errors(predicted_coords):
            unique_pose_indices = torch.unique(pose_idx).detach().cpu().numpy()
            translation_errors = []
            rotation_errors = []
            
            for p_idx in unique_pose_indices:
                mask = (pose_idx == p_idx).squeeze()
                
                px_img = target_px[mask].detach()
                crd_img = predicted_coords[mask].detach().float()
                
                # Filter out any points where the predicted 3D coordinate is exactly [0, 0, 0]
                non_zero_mask = (crd_img != 0).any(dim=1)
                px_img = px_img[non_zero_mask]
                crd_img = crd_img[non_zero_mask]
                
                # dsacstar requires at least 10 valid pixels to perform PnP RANSAC
                if len(crd_img) < 10:
                    continue
                    
                intr_img = intrinsics[mask][0].detach().cpu().numpy()
                w2c_gt = poses_inv[mask][0].detach().cpu().numpy()
                c2w_gt = np.linalg.inv(w2c_gt)
                
                # Reconstruct coordinates into H, W grid for dsacstar
                x_grid = torch.round(px_img[:, 0] / subsample_factor - 0.5).long()
                y_grid = torch.round(px_img[:, 1] / subsample_factor - 0.5).long()
                
                # Ensure indices are in-bounds
                x_grid = torch.clamp(x_grid, min=0)
                y_grid = torch.clamp(y_grid, min=0)
                
                H_grid = int(y_grid.max().item()) + 1
                W_grid = int(x_grid.max().item()) + 1
                
                # Create grid on CPU
                scene_coordinates_B3HW = torch.zeros((1, 3, H_grid, W_grid), dtype=torch.float32, device="cpu")
                scene_coordinates_B3HW[0, :, y_grid.cpu(), x_grid.cpu()] = crd_img.cpu().T
                
                focal_length = float(intr_img[0, 0])
                ppX = float(intr_img[0, 2])
                ppY = float(intr_img[1, 2])
                
                out_pose_c2w = torch.zeros((4, 4), dtype=torch.float32, device="cpu")
                
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_B3HW,
                    out_pose_c2w,
                    64,          # hypotheses
                    10.0,        # threshold in pixels
                    focal_length,
                    ppX,
                    ppY,
                    100.0,       # inlier_alpha
                    100.0,       # max_pixel_error
                    subsample_factor,
                    2089,        # base_seed
                )
                
                c2w_est = out_pose_c2w.cpu().numpy()
                
                # If pose is all zeros or invalid (dsacstar failure)
                if np.allclose(c2w_est, 0):
                    continue
                    
                t_err = np.linalg.norm(c2w_gt[0:3, 3] - c2w_est[0:3, 3]) * 100.0
                
                gt_R = c2w_gt[0:3, 0:3]
                out_R = c2w_est[0:3, 0:3]
                r_err = np.matmul(out_R, np.transpose(gt_R))
                
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = float(np.linalg.norm(r_err) * 180 / np.pi)
                
                translation_errors.append(t_err)
                rotation_errors.append(r_err)
                
            if len(translation_errors) == 0:
                return float('inf'), float('inf')
                
            return np.median(translation_errors), np.median(rotation_errors)

        _logger.info("Computing camera pose error metrics via PnP RANSAC...")
        # 1. Model 0 alone (alpha = 1.0)
        med_t0, med_r0 = evaluate_pose_errors(pred0_all)
        # 2. Model 1 alone (alpha = 0.0)
        med_t1, med_r1 = evaluate_pose_errors(pred1_all)
        # 3. Best Alpha Blend (dynamic blend per patch)
        best_alphas_tensor = torch.tensor(best_alphas, dtype=torch.float32, device=device).unsqueeze(-1)
        best_pred_coords = best_alphas_tensor * pred0_all + (1.0 - best_alphas_tensor) * pred1_all
        med_t_best, med_r_best = evaluate_pose_errors(best_pred_coords)

        # Log statistics
        log_path = output_dir / f"offline_alpha_test_{split}.log"
        with open(log_path, "w") as log_file:
            def log_and_print(msg):
                _logger.info(msg)
                log_file.write(msg + "\n")
                
            log_and_print(f"Offline Blend Test Results - Split: {split}")
            log_and_print(f"Scene: {args.scene}")
            log_and_print(f"Total Patches Evaluated: {B_total}")
            log_and_print(f"Model 0: {args.model0}")
            log_and_print(f"Model 1: {args.model1}")
            log_and_print("-" * 50)
            
            mean_alpha = np.mean(best_alphas)
            median_alpha = np.median(best_alphas)
            std_alpha = np.std(best_alphas)
            log_and_print(f"Mean best alpha: {mean_alpha:.4f}")
            log_and_print(f"Median best alpha: {median_alpha:.4f}")
            log_and_print(f"Std dev best alpha: {std_alpha:.4f}")
            log_and_print("-" * 50)
            
            log_and_print("Camera Localization Median Pose Errors:")
            log_and_print(f"  Model 0 (alpha = 1.0): {med_t0:6.2f} cm, {med_r0:5.2f} deg")
            log_and_print(f"  Model 1 (alpha = 0.0): {med_t1:6.2f} cm, {med_r1:5.2f} deg")
            log_and_print(f"  Best Alpha Blend:      {med_t_best:6.2f} cm, {med_r_best:5.2f} deg")
            log_and_print("-" * 50)
            
            log_and_print("Alpha Frequency Distribution:")
            unique_alphas, counts = np.unique(best_alphas, return_counts=True)
            alpha_counts = dict(zip(unique_alphas, counts))
            
            for alpha in alphas:
                count = alpha_counts.get(alpha, 0)
                pct = (count / B_total) * 100.0
                log_and_print(f"  alpha = {alpha:.1f}: {count:8d} patches ({pct:6.2f}%)")
                
        _logger.info(f"Saved statistics log to {log_path}")

        # Plot frequency histogram
        plot_path = output_dir / f"offline_alpha_test_{split}_histogram.png"
        plt.figure(figsize=(10, 6))
        
        bin_edges = np.arange(-0.05, 1.15, 0.1)
        plt.hist(best_alphas, bins=bin_edges, rwidth=0.8, color='skyblue', edgecolor='black', alpha=0.8, density=False)
        plt.xticks(alphas)
        plt.xlabel(r"Optimal Blending Parameter $\alpha$")
        plt.ylabel("Number of Patches")
        plt.title(f"Distribution of Optimal Blending Parameter Alpha (Split: {split}, Scene: {args.scene})\n"
                  r"$\mathbf{x}_{blend} = \alpha \cdot \mathbf{x}_{model0} + (1-\alpha) \cdot \mathbf{x}_{model1}$")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        _logger.info(f"Saved histogram plot to {plot_path}")
        
        # =========================================================================
        # PATCH-LEVEL DATASET GENERATION AND SAVING
        # =========================================================================
        _logger.info(f"Generating patch-level alpha fusion dataset for split: {split}...")
        
        # 1. z: FloatTensor[N, 2, 1024]
        z = torch.stack([features0, features1], dim=1).float()
        
        # 2. target_class: LongTensor[N]
        target_classes = best_alpha_indices                 # (B_total,)
        
        # 3. alpha: FloatTensor[N]
        alphas_target = target_classes.float() / 10.0                          # (B_total,)
        
        # 4. error_curve: FloatTensor[N, 11]
        error_curves = repro_errors_all.T.float()                              # (B_total, 11)
        
        # Transfer all tensors to CPU for verification checks and saving
        z_cpu = z.cpu()
        target_classes_cpu = target_classes.cpu()
        alphas_target_cpu = alphas_target.cpu()
        error_curves_cpu = error_curves.cpu()
        
        # Required checks
        assert z_cpu.shape == (B_total, 2, 1024), f"z shape {z_cpu.shape} != [N, 2, 1024]"
        assert target_classes_cpu.min() >= 0, f"target_class.min() {target_classes_cpu.min()} < 0"
        assert target_classes_cpu.max() <= 10, f"target_class.max() {target_classes_cpu.max()} > 10"
        assert torch.allclose(alphas_target_cpu, target_classes_cpu.float() / 10.0), "alpha != target_class / 10"
        assert torch.isfinite(z_cpu).all(), "z contains non-finite values (NaN/Inf)"
        assert torch.isfinite(error_curves_cpu).all(), "error_curve contains non-finite values (NaN/Inf)"
        
        # Debug print per split
        _logger.info(f"============ Dataset Debug Stats (Split: {split}) ============")
        _logger.info(f"  N samples: {B_total}")
        
        # Class histogram 0..10
        hist = torch.histc(target_classes_cpu.float(), bins=11, min=0.0, max=10.0)
        _logger.info("  Class histogram 0..10:")
        for idx in range(11):
            _logger.info(f"    Class {idx:2d} (alpha={idx/10.0:.1f}): {int(hist[idx]):8d} samples ({hist[idx]/B_total*100.0:6.2f}%)")
            
        mean_alpha_val = alphas_target_cpu.mean().item()
        pct_alpha_0 = (target_classes_cpu == 0).sum().item() / B_total * 100.0
        pct_alpha_10 = (target_classes_cpu == 10).sum().item() / B_total * 100.0
        
        mean_best_err = min_errors_global.cpu().mean().item()
        mean_head0_err = repro_errors_all[10].cpu().mean().item()  # alpha=1.0 is index 10 (DINOv2)
        mean_head1_err = repro_errors_all[0].cpu().mean().item()   # alpha=0.0 is index 0  (DPTv2)
        
        _logger.info(f"  Mean alpha: {mean_alpha_val:.4f}")
        _logger.info(f"  % alpha=0 (DPTv2 only): {pct_alpha_0:.2f}%")
        _logger.info(f"  % alpha=1 (DINOv2 only): {pct_alpha_10:.2f}%")
        _logger.info(f"  Mean best error: {mean_best_err:.4f} pixels")
        _logger.info(f"  Mean head0-only error (DINOv2): {mean_head0_err:.4f} pixels")
        _logger.info(f"  Mean head1-only error (DPTv2): {mean_head1_err:.4f} pixels")
        _logger.info("=============================================================")
        
        # Save dataset dictionary
        dataset_save_path = output_dir / f"{args.scene}_dataset_{split}.pt"
        torch.save({
            "z": z_cpu,
            "target_class": target_classes_cpu,
            "alpha": alphas_target_cpu,
            "error_curve": error_curves_cpu
        }, dataset_save_path)
        _logger.info(f"Successfully saved dataset to: {dataset_save_path}")
        _logger.info("-" * 60)


if __name__ == "__main__":
    main()
