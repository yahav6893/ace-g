# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Script to optimize head and / or map embeddings for a single scene.

Produces the following files:
    {output_dir}/{session_id}_head.pt: Trained head network.
    {output_dir}/{session_id}_map.pt: Trained map embeddings (if used).
    {output_dir}/{session_id}_log.txt: Text log of training progress.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import time
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
import torch
import tqdm
import yoco

from ace_g import buffer, configuration, datasets, encoders, losses, regressors, schedule, scr_heads, utils, vis_utils

os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402


_logger = logging.getLogger(__name__)


class SingleSceneTrainer:
    """Trainer for a single scene."""

    @dataclasses.dataclass
    class Config(configuration.GlobalConfig):
        """Configuration for one single-scene training run."""

        encoder: encoders.EncoderConfig | dict
        """See encoders.EncoderConfig."""
        head: scr_heads.HeadConfig
        """See scr_heads.HeadConfig."""
        dataset: datasets.CamLocDataset.Config | None = None
        """See dataset.CamLocDataset.Config.
        The following attributes are overwritten by arguments of this config:
            use_color, use_half, force_depth, use_aug, aug_rotation, aug_scale_max, aug_scale_min"""
        repro_loss_hard_clamp: int = 1000
        """Hard clamping threshold for the reprojection losses."""
        repro_loss_soft_clamp: int = 50
        """Soft clamping threshold for the reprojection losses."""
        repro_loss_soft_clamp_min: int = 1
        """Minimum value of the soft clamping threshold when using a schedule."""
        repro_loss_type: Literal["l1", "l1+sqrt", "l1+logl1", "tanh", "dyntanh"] | None = "dyntanh"
        """Loss function on the reprojection error. Dyn varies the soft clamping threshold."""
        repro_loss_schedule: Literal["circle", "linear"] = "circle"
        """How to decrease the softclamp threshold during training, circle is slower first."""
        loss_fn_2d: str | None = "50*torch.tanh(x/50)"
        """2D loss function; not used if use_uncertainty of head is True."""
        loss_fn_3d: str | None = None
        """3D loss function; not used if use_uncertainty of head is True."""
        depth_min: float = 0.1
        """Enforce minimum depth of network predictions."""
        depth_target: float = 10
        """Default depth to regularize training."""
        depth_max: float = 1_000
        """Enforce maximum depth of network predictions."""
        calibration_source: Literal["pose_file", "external", "dataset", "heuristic"] = "dataset"
        """Source of camera calibration. See CamLocDataset for details."""
        external_focal_length: float | None = None
        """Set external focal length value. Only used if calibration_source is set to 'external'."""
        use_aug: bool = True
        """Whether to use augmentation on image level."""
        aug_rotation: float = 15
        """Max inplane rotation angle."""
        aug_scale: float = 1.5
        """Max scale factor."""
        use_feature_aug: bool = False
        """Whether to apply feature dropout and random noise to the input features."""
        feature_aug_drop: float = 0.1
        """Feature drop out probability if enabled."""
        feature_aug_std: float = 0.0
        """Feature random noise standard deviation if enabled."""
        map_emb_drop: float = 0.0
        """Dropout probability for map embeddings."""
        max_buffer_size: int = 8_000_000
        """Maximum number of patches in the training buffer."""
        max_dataset_passes: int = 10
        """Maximum number of repetitions of mapping images (with different augmentations)."""
        samples_per_image: int = 1024
        """Number of patches drawn from each image when creating the buffer.
        Higher will lead to faster buffer creation since fewer images are processed to reach the target buffer size."""
        use_depth: bool = False
        """Whether to use depth information for supervision if available."""
        use_pose_seed: float = -1
        """Use a single image with identity pose as seed, float value [0-1] represents image ID relative to dataset
        size, -1: do not use seed."""
        supervision_type: Literal["2d", "3d"] = "2d"
        """How to supervise scene coordinates. Note 2D still uses depth as prior if available and use_depth is True."""
        num_iterations: int = 25_000
        """Number of training iterations."""
        use_half: bool = True
        """Whether to use half precision for training."""
        train_head: bool = True
        """Whether to train the head network."""
        train_map_embs: bool = True
        """Whether to treat patches independently during mapping."""
        schedule_config: schedule.ScheduleACE.Config = dataclasses.field(default_factory=schedule.ScheduleACE.Config)
        """See schedule.ScheduleACE.Config"""
        test_run: bool = False
        """Overwrite parameters such that training quickly concludes."""
        base_seed: int = 2089
        """Random seed for reproducibility."""
        device: str = "cuda"
        """Device to run training on."""
        buffer_device: str | None = None
        """Device to store the feature buffer on. Using CPU allows training on smaller GPUs, but is slower. If None,
        same as device."""
        num_map_embs: int = 1024
        """Number of map embeddings to use."""
        output_period: int = 100
        """Print training statistics every n iterations, also render_visualization frame frequency."""
        output_dir: pathlib.Path = pathlib.Path("outputs")
        """Target output dir for the trained network and / or map embeddings."""
        session_id: str | None = None
        """Custom session name used to generate output files; generated from time and options if not provided."""
        use_color: bool | None = None
        """Whether to use color information for training. If None, color is used if the encoder supports it."""
        buffer_creation_workers: int | None = None
        """Number of workers for the data loader, set according to number of CPU cores."""
        buffer_creation_batch_size: int = 4
        """Batch size used in the DataLoader for buffer creation."""
        batch_size: int = 512 * 10
        """Number of patches for each parameter update."""
        use_rerun: bool = False
        """Whether to log to Rerun."""
        rrd_path: pathlib.Path | None = None
        """If set, save Rerun's rrd to this path; otherwise, use connect."""
        rerun_spawn: bool = False
        """If True, spawn a new Rerun viewer, otherwise use connect."""

    def __init__(self, config: Config) -> None:
        """Initialize the ACE trainer.

        Args:
            config: The global configuration. See train_single_scene for details.
        """
        self.config = config

        assert self.config.dataset is not None, "Dataset configuration is required."

        if self.config.test_run:
            self.config.max_buffer_size = 10_000
            self.config.num_iterations = 10
            # TODO check if gpu memory would be sufficient with original parameters

        _logger.info(f"Using device for training: {self.config.device}")
        _logger.info(f"ACE feature buffer device: {self.config.buffer_device}")

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = False

        if "WANDB_RUN_ID" in os.environ and "WANDB_PROJECT" in os.environ:
            import wandb
            # In a subprocess, wandb init with resume="allow" automatically merges perfectly into the active run 
            # or tracks seamlessly alongside it.
            wandb.init(resume="allow")
            self._use_wandb = True
        else:
            self._use_wandb = False

        # Setup randomness for reproducibility.
        _logger.info(f"Setting random seed to {self.config.base_seed}")
        utils.set_seed(self.config.base_seed)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.config.base_seed + 8191)

        # Determine whether we will generate ground truth scene coordinate from depth maps.
        self.require_depth = self.config.use_pose_seed >= 0  # generate depth if not available

        # Where to save all the output files.
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True)

        # Disable multi-threaded data loading in case we have to predict depth maps in the dataset class.
        if self.require_depth and self.dataset.depth_files is None:
            _logger.info(
                "Disabling multi-threaded data loading because we cannot run multiple depth inference passes "
                "simultaneously."
            )
            self.config.buffer_creation_workers = 0

        # Create encoder.
        encoder = encoders.create_encoder(self.config.encoder)  # type: ignore

        # Create dataset (potentially based on encoder).
        self.config.dataset.use_color = (
            self.config.use_color if self.config.use_color is not None else encoder.supports_rgb
        )
        self.config.dataset.use_half = self.config.use_half
        self.config.dataset.force_depth = self.require_depth
        self.config.dataset.use_aug = self.config.use_aug
        self.config.dataset.aug_rotation = self.config.aug_rotation
        self.config.dataset.aug_scale_max = self.config.aug_scale
        self.config.dataset.aug_scale_min = 1 / self.config.aug_scale
        self.config.dataset.calibration_source = self.config.calibration_source
        self.dataset = datasets.CamLocDataset(self.config.dataset)

        # Create head.
        if not isinstance(self.config.head, pathlib.Path):
            self.config.head.dim_in = encoder.dim_out
            self.config.head.mean = self.dataset.mean_cam_center if "MLPHead" in self.config.head.obj_type else None

        print(self.config.head)
        head = scr_heads.create_head(self.config.head)

        # Create regressor.
        self.regressor = regressors.Regressor(encoder, head)
        self.regressor.to(self.config.device)

        # Create map embeddings if needed.
        if isinstance(self.regressor.head, scr_heads.TransformerHead):
            self.map_embeddings = torch.randn(
                self.config.num_map_embs,
                self.regressor.head.dim_map_emb,
                dtype=torch.float32,
                device=self.config.device,
            )
            with torch.no_grad():
                self.map_embeddings *= 0.01
            self.mean = self.dataset.mean_cam_center.to(self.config.device)
        else:
            self.map_embeddings = None
            self.mean = None

        # Set requires_grad for all parameters
        self.regressor.encoder.requires_grad_(False)
        self.regressor.head.requires_grad_(self.config.train_head)
        if self.map_embeddings is not None:
            self.map_embeddings.requires_grad_(self.config.train_map_embs)

        # Set subsample factor for the dataset
        self.dataset.set_subsample_factor(self.regressor.subsample_factor)

        # Create iterable of trained parameters
        trained_parameters = []
        if self.config.train_head:
            if hasattr(self.regressor.head, 'get_param_groups'):
                trained_parameters.extend(
                    self.regressor.head.get_param_groups(
                        min_lr=self.config.schedule_config.learning_rate_min,
                        max_lr=self.config.schedule_config.learning_rate_max
                    )
                )
            else:
                trained_parameters.append({"params": list(self.regressor.head.parameters()), "name": "head"})
        if self.config.train_map_embs and self.map_embeddings is not None:
            trained_parameters.append({"params": [self.map_embeddings], "name": "map_embeddings"})

        self.regressor.train()

        if self.config.external_focal_length is not None:
            self.dataset.set_external_focal_length(self.config.external_focal_length)

        _logger.info(
            "Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
                self.config.dataset.rgb_files,
                len(self.dataset),
                self.dataset.mean_cam_center[0],
                self.dataset.mean_cam_center[1],
                self.dataset.mean_cam_center[2],
            )
        )

        # Set session id
        if self.config.session_id is None:
            self.session_id = utils.generate_session_id(
                self.dataset.scene_name,
                self.regressor.encoder.__class__.__name__,
                self.regressor.head.__class__.__name__,
            )
        else:
            self.session_id = self.config.session_id

        # Setup optimization parameters and learning rate scheduler.
        self.training_scheduler = schedule.ScheduleACE(
            parameters=trained_parameters,
            use_scaler=self.config.use_half,
            num_iterations=self.config.num_iterations,
            config=self.config.schedule_config,
        )

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = utils.get_pixel_grid(self.regressor.subsample_factor)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.config.device)

        if self.config.repro_loss_type is not None:
            self.loss_fn_2d = losses.ReproLoss(
                total_iterations=self.config.num_iterations,
                soft_clamp=self.config.repro_loss_soft_clamp,
                soft_clamp_min=self.config.repro_loss_soft_clamp_min,
                type=self.config.repro_loss_type,
                circle_schedule=(self.config.repro_loss_schedule == "circle"),
            )
        else:
            self.loss_fn_2d = self.config.loss_fn_2d

        _logger.info(f"Using {self.config.supervision_type=}")
        if self.config.supervision_type == "2d":
            _logger.info(f"Using {self.loss_fn_2d} as 2D loss.")
        if self.config.supervision_type == "3d":
            assert self.config.use_depth, "Any type of 3D supervision requires use_depth to be True."
            _logger.info(f"Using {self.config.loss_fn_3d} as 3D loss.")

        # Setup Rerun
        if self.config.use_rerun:
            vis_utils.rr_init("ace_g_mapping", self.config.rrd_path, self.config.rerun_spawn)
            _setup_mapping_blueprint()

    def train(self) -> dict:
        """Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """
        buffer_creation_time = 0.0
        optimization_time = 0.0

        self.iteration = 0
        self.epoch = 0
        self.start_time = time.time()

        buffer_device = self.config.buffer_device if self.config.buffer_device is not None else self.config.device

        # Create training buffer.
        buffer_start_time = time.time()
        self.training_buffer = buffer.create_buffer(
            dataset=self.dataset,
            encoder=self.regressor.encoder,
            max_buffer_size=self.config.max_buffer_size,
            buffer_device=buffer_device,
            max_dataset_passes=self.config.max_dataset_passes,
            num_data_loader_workers=self.config.buffer_creation_workers,
            batch_size=self.config.buffer_creation_batch_size,
            num_samples_per_image=self.config.samples_per_image,
            use_half=self.config.use_half,
            seed=self.config.base_seed,
        )
        buffer_end_time = time.time()
        buffer_creation_time += buffer_end_time - buffer_start_time
        self.training_buffer_size = len(self.training_buffer.features)
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Setup a training log - first derive a log file from the map file
        log_path = self.config.output_dir / (self.session_id + "_log.txt")
        self.log_file = open(log_path, "w")

        # Train the regression head.
        self.training_pbar = tqdm.tqdm(
            total=self.training_scheduler.max_iterations, desc="Training", dynamic_ncols=True
        )
        while True:
            self.epoch += 1

            epoch_start_time = time.time()
            continue_training = self.run_epoch()
            optimization_time += time.time() - epoch_start_time

            if not continue_training:
                break

        # Save trained model.
        out_dict = self.save_model()
        self.log_file.close()

        end_time = time.time()
        _logger.info(
            f"Done without errors. "
            f"Creating buffer time: {buffer_creation_time:.1f} seconds. "
            f"Training time: {optimization_time:.1f} seconds. "
            f"Total time: {end_time - self.start_time:.1f} seconds."
        )

        return out_dict

    def run_epoch(self) -> bool:
        """Run one epoch of training, shuffling the feature buffer and iterating over it.

        Returns:
            bool: False if max iterations have been reached, True otherwise.
        """
        # check whether training has finished, number of total iterations might have been reduced
        if self.iteration >= self.training_scheduler.max_iterations:
            return False

        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.training_buffer_size, self.config.batch_size):
            rr.set_time("iteration", sequence=self.iteration - self.config.num_iterations)

            batch_end = batch_start + self.config.batch_size

            # Drop last batch if not full.
            if batch_end > self.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            # If the buffer lives in main memory we also move it to the GPU, otherwise the
            # .to() is a no-op.
            # .contiguous() is needed to for a faster forward pass through the head later.
            # If the buffer was in main memory, .to() already makes the GPU copy contiguous,
            # so the .contiguous() is actually a no-op.
            self.training_step(
                self.training_buffer.features[random_batch_indices].to(self.config.device).contiguous(),
                self.training_buffer.target_px[random_batch_indices].to(self.config.device).contiguous(),
                self.training_buffer.aug_poses_inv[random_batch_indices].to(self.config.device).contiguous(),
                self.training_buffer.poses_inv[random_batch_indices].to(self.config.device).contiguous(),
                self.training_buffer.intrinsics[random_batch_indices].to(self.config.device).contiguous(),
                self.training_buffer.target_crds[random_batch_indices].to(self.config.device).contiguous(),
            )
            self.iteration += 1

            self.training_pbar.update(1)

        return True

    def training_step(
        self,
        features_bc: torch.Tensor,
        target_px_b2: torch.Tensor,
        c2augc_b34: torch.Tensor,
        w2c_b44: torch.Tensor,
        image_from_camera_b33: torch.Tensor,
        target_coords_b3: torch.Tensor,
    ) -> None:
        """Run one iteration of training, computing the reprojection error and minimizing it.

        Notation for reference frames:
            w = world
            c = original camera (OpenCV / RDF convention)
            augc = augmented camera (OpenCV / RDF convention)
        """
        # check whether to start cooldown
        self.training_scheduler.check_and_set_cooldown(self.iteration)

        if hasattr(self.regressor.head, 'unfreeze_experts_if_needed'):
            self.regressor.head.unfreeze_experts_if_needed(self.iteration)

        # check whether training has finished, number of total iterations might have been reduced
        if self.iteration >= self.training_scheduler.max_iterations:
            return

        dim_features = features_bc.shape[1]

        # Reshape to a "fake" NCHW shape, since it's faster to run through the network compared to the original shape.
        # Note N != B after this reshaping, hence the different letter.
        features_nchw = features_bc[None, None, ...].view(-1, 16, 32, dim_features).permute(0, 3, 1, 2)

        if self.config.use_feature_aug:
            features_nchw = utils.augment_features(
                features_nchw, self.config.feature_aug_drop, self.config.feature_aug_std
            )

        map_embeddings = self.map_embeddings
        if map_embeddings is not None and self.config.map_emb_drop > 0:
            map_embeddings = utils.augment_features(map_embeddings, self.config.map_emb_drop, 0.0)

        with torch.autocast("cuda", enabled=self.config.use_half):
            pred_coords_w_n3hw, pred_uncertainties_n1hw = self.regressor.get_scene_coordinates(
                features_nchw,
                map_embeddings=map_embeddings,
                means=self.mean,
            )

        # Back to the original shape. Convert to float32 as well.
        pred_coords_w_b3 = pred_coords_w_n3hw.permute(0, 2, 3, 1).flatten(0, 2).float()
        pred_uncertainties_b1 = (
            pred_uncertainties_n1hw.permute(0, 2, 3, 1).flatten(0, 2).float()
            if pred_uncertainties_n1hw is not None
            else None
        )

        if self.config.use_rerun:
            pred_coords_w_b3_np = pred_coords_w_b3.numpy(force=True)

            if pred_uncertainties_b1 is not None:
                vis_utils.rr_log_points_with_scalar(
                    "pred_coords",
                    pred_coords_w_b3_np,
                    pred_uncertainties_b1.numpy(force=True),
                    vmin=0.0,
                    vmax=0.5,
                    radii=rr.Radius(0.01),
                )
            else:
                rr.log("pred_coords", rr.Points3D(pred_coords_w_b3_np, radii=rr.Radius(0.01)))

        # combine augmentation poses and original camera poses
        w2augc_b34 = torch.bmm(c2augc_b34, w2c_b44)

        loss, losses_3d, losses_2d, dists_3d, dists_2d = losses.compute_loss(
            pred_coords=pred_coords_w_b3,
            pred_uncertainties=pred_uncertainties_b1,
            w2c_b34=w2augc_b34,
            image_from_camera_b33=image_from_camera_b33,
            target_pixels=target_px_b2,
            target_coords=target_coords_b3,
            supervision_type=self.config.supervision_type,
            # disable depth prior for 3D only to avoid extra masking (we just want to be able to monitor 2D metrics)
            use_depth_as_prior=self.config.use_depth and self.config.supervision_type in {"2d", "2d+3d"},
            const_depth_prior=self.config.depth_target,
            depth_min=self.config.depth_min,
            depth_max=self.config.depth_max,
            max_reprojection_error=self.config.repro_loss_hard_clamp,
            loss_fn_2d=self.loss_fn_2d,
            loss_fn_3d=self.config.loss_fn_3d,
            iteration=self.iteration,
        )

        losses.log_metrics(loss, losses_3d, losses_2d, dists_3d, dists_2d)

        if hasattr(self.regressor.head, 'last_l2_reg_loss') and self.regressor.head.last_l2_reg_loss is not None:
            loss = loss + self.regressor.head.last_l2_reg_loss
            if self.config.use_rerun:
                rr.log("loss_l2_reg", rr.Scalars(self.regressor.head.last_l2_reg_loss.item()))

        head_cfg = getattr(self.regressor.head, "config", None)
        mogu_loss_weight = float(getattr(head_cfg, "mogu_loss_weight", 0.0))

        if mogu_loss_weight > 0.0:
            if not hasattr(self.regressor.head, "compute_mogu_loss"):
                raise RuntimeError(
                    "mogu_loss_weight > 0, but head has no compute_mogu_loss(). "
                    "Use UncExpertFusionHead or disable mogu_loss_weight."
                )

            loss_mogu = self.regressor.head.compute_mogu_loss(target_coords_b3)

            loss = loss + mogu_loss_weight * loss_mogu

            rr.log("loss_mogu", rr.Scalars(loss_mogu.item()))
            rr.log("loss_mogu_weighted", rr.Scalars((mogu_loss_weight * loss_mogu).item()))            

        if getattr(self, '_use_wandb', False):
            import wandb
            wb_log_dict = {}
            if hasattr(self.regressor.head, 'last_l2_reg_loss') and self.regressor.head.last_l2_reg_loss is not None:
                wb_log_dict["train/loss_l2_reg"] = self.regressor.head.last_l2_reg_loss.item()
            
            for i, pg in enumerate(self.training_scheduler.optimizer.param_groups):
                name = pg.get("name", f"group_{i}")
                wb_log_dict[f"train/lr_{name}"] = pg["lr"]
                
            if wb_log_dict:
                wandb.log(wb_log_dict, commit=False)

        if torch.any(torch.isnan(loss)):
            _logger.info("nan loss detected (step skipped)")

        # Set gradient buffers to zero
        self.training_scheduler.zero_grad(set_to_none=True)

        # Calculate gradients of loss
        self.training_scheduler.backward(loss)

        # Update the weights
        self.training_scheduler.step()

        if self.iteration % self.config.output_period == 0:
            # Print status.
            time_since_start = time.time() - self.start_time

            _logger.info(
                f"Iteration: {self.iteration:6d}|{self.training_scheduler.max_iterations:6d} / "
                f"Epoch {self.epoch:03d}, "
                f"Loss: {loss:.1f}, "
                # f"Batch inliers ({self.config.learning_rate_cooldown_trigger_px_threshold}px): "
                # f"{batch_inliers * 100:.1f}%, "
                f"Time: {time_since_start:.0f}s"
            )

            # write the main information to the log file
            log_str = f"{self.iteration} {time_since_start} {loss}"

            self.log_file.write(log_str + "\n")

    def save_model(self) -> dict:
        """Save the trained model to disk."""
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.config.output_path_prefix)

        # Save the head if they can't be reconstructed from the config alone.
        # That is, either the head was trained or no path was given (i.e., randomly initialized head).
        save_head = self.config.train_head or not isinstance(self.config.head, pathlib.Path)
        save_map_embeddings = self.config.train_map_embs and self.regressor.uses_map_embeddings

        if save_head:
            head_path = str(self.config.output_dir / (self.session_id + "_head.pt"))
        elif not save_head and not isinstance(self.config.head, pathlib.Path):
            head_path = None
        else:
            head_path = str(self.config.head)

        if save_head and head_path is not None:
            # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as
            # mentioned in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained
            # encoder file.
            head_state_dict = self.regressor.head.state_dict()
            for k, v in head_state_dict.items():
                # we store the arguments to construct the module as an extra state, this makes it easy to reconstruct
                # the module from just the state_dict
                if "_extra_state" in k:
                    continue
                head_state_dict[k] = v.half()
            torch.save(head_state_dict, head_path)
            _logger.info(f"Saved trained head weights to: {head_path}")

        map_path = str(self.config.output_dir / (self.session_id + "_map.pt")) if save_map_embeddings else None
        if save_map_embeddings and map_path is not None:
            map_dict = {}
            map_dict["map_embeddings"] = self.map_embeddings
            map_dict["mean"] = self.mean
            torch.save(map_dict, map_path)
            _logger.info(f"Saved map embeddings to: {map_path}")

        out_dict = {
            "sst": utils.primitive(configuration.asdict(self.config, remove_global_config=True)),
            "reg": {
                **(self.config.reg if self.config.reg is not None else {}),  # this allows specifying reg options early
                "encoder": self.config.encoder,
                "head_path": head_path,
                "map_path": map_path,
                "session_id": self.session_id,
            },
        }

        utils.save_yaml(utils.primitive(out_dict), self.map_yaml)
        return out_dict

    @property
    def map_yaml(self) -> pathlib.Path:
        """Return the path to the map YAML file."""
        return self.config.output_dir / (self.session_id + "_map.yaml")


def _setup_mapping_blueprint() -> None:
    """Setup the blueprint for the mapping visualization."""
    timeseries = rrb.Vertical(
        contents=[
            rrb.TimeSeriesView(
                name="Loss",
                contents="loss",
            ),
            rrb.Horizontal(
                contents=[
                    rrb.TimeSeriesView(
                        name="Inliers by reprojection error",
                        contents=["inliers_5px", "inliers_10px", "inliers_20px"],
                    ),
                    rrb.TimeSeriesView(
                        name="Reprojection error quantiles (px)",
                        contents=["quantile_2d_0.1", "quantile_2d_0.2", "quantile_2d_0.5"],
                        axis_y=rrb.ScalarAxis(range=(0, 200), zoom_lock=True),
                    ),
                ]
            ),
        ]
    )
    blueprint = rrb.Blueprint(
        rrb.Grid(
            contents=[
                rrb.Spatial3DView(
                    name="3D View",
                ),
                timeseries,
            ],
        )
    )
    rr.send_blueprint(blueprint)


if __name__ == "__main__":
    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train scene coordinate regression on a single scene",
    )
    parser.add_argument("--config", nargs="+", help="Config file(s) to load.")

    config_dict = yoco.load_config_from_args(
        parser, search_paths=[".", "./configs", os.path.normpath(os.path.join(os.path.dirname(__file__), "configs"))]
    )
    config = configuration.fromdict(SingleSceneTrainer.Config, config_dict, defaults_key="sst")

    trainer = SingleSceneTrainer(config)
    trainer.train()
