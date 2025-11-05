# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Script which registers (i.e., localizes) images to a map given by an SCR network.

Produces the following files:
    {output_dir}/{session_id}_registered_poses.txt: Contains the registered poses and errors.
    {output_dir}/{session_id}_mapping.pkl: Contains the state of the visualizer for rendering.
    {output_dir}/{session_id}_register.pkl: Contains the state of the visualizer for rendering.
"""

import argparse
import copy
import dataclasses
import logging
import os
import pathlib
import time
from typing import Literal, Tuple

import numpy as np
import rerun as rr
import torch
import tqdm
import yoco
from torch.utils.data import DataLoader

import dsacstar
from ace_g import configuration, data_io, datasets, encoders, eval_poses_utils, regressors, scr_heads, utils

os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402


_logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class RegistrationConfig(configuration.GlobalConfig):
    """Configuration for the register_images function."""

    # Dataset configuration.
    dataset: datasets.CamLocDataset.Config
    """Dataset configuration."""
    use_color: bool | None = None
    """Whether to use color images. If None, color is used if the encoder supports it."""
    external_focal_length: float | None = None
    """Use this focal length instead of the one from the dataset; only used if dataset.calibration_source is set to
    'external'."""

    # Network + map configuration.
    encoder: encoders.EncoderConfig | dict
    """Encoder configuration."""
    head_path: pathlib.Path
    """Path to the head state dict."""
    map_path: pathlib.Path | None = None
    """Path to the map file containing map embeddings and mean."""

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    hypotheses: int = 64
    """Number of hypotheses, i.e. number of RANSAC iterations."""
    threshold: int | float = 10  # Inlier threshold in pixels (RGB) or centimeters (RGB-D).
    inlier_alpha: int | float = 100
    """Alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means
    softer."""
    max_pixel_error: int | float = 100
    """Maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all
    measurements; error is clamped to this value for stability."""
    uncertainty_filter: Literal["threshold", "quantile"] | None = None
    """Set type of uncertainty filter."""
    uncertainty_filter_threshold: float = 0.02
    """Initially only consider scene coordinates with estimated uncertainty below this threshold. Double threshold until
    enough points are below the threshold."""
    uncertainty_filter_threshold_min_points: int = 200
    """Double the threshold until this many points are below the threshold."""
    uncertainty_filter_quantile: float = 0.1
    """Set the uncertainty threshold to a fixed factor of this quantile."""
    uncertainty_filter_quantile_factor: float = 1.5
    """Factor to multiply the quantile-based threshold with."""

    # Logging and output.
    output_dir: pathlib.Path = pathlib.Path("./outputs")
    """Output dir for the output files."""
    session_id: str | None = None
    """Custom session name used to generate output files; generated from time and options if not provided."""

    # Misc configuration.
    register_gt_coords: bool = False
    """Register to ground truth coordinates instead of predictions; useful for debugging."""
    device: str = "cuda"
    """Device to use for inference."""
    base_seed: int = 1305
    """Seed to control randomness."""
    test_run: bool = False
    """Overwrites parameters such that the script finishes in a few seconds."""
    max_estimates: int = -1
    """Max number of images to consider."""
    num_data_workers: int = 12
    """Number of data loading workers, set according to the number of available CPU cores."""
    rr_prefix: str = ""
    """Added to all rerun entity paths (can be useful for merging rrd files)."""


def register_images(
    config: RegistrationConfig,
    regressor: regressors.Regressor | None = None,
    map_embeddings: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
) -> Tuple[dict[str, tuple[np.ndarray, float]], dict]:
    """Register images to a map given by an SCR network.

    Can be used either with via the command line like this
        register_images(create_register_images_parser().parse_args())
    or from code like this
        register_images({ "rgb_files": "datasets/scene/*.jpg", "test_run": True, ... })

    Args:
        config: Configuration for the registration.
        regressor: A regressor to use for registration. If None, a regressor is created from the config.
        map_embeddings: Map embeddings to use for registration. If None, the embeddings are loaded from the config.
        mean: Mean tensor to use for registration.

    Returns:
        A dictionary mapping image file paths to tuples of estimated cam-to-world matrices with shape (4,4) and their
        confidence values.
    """
    assert config.dataset is not None, "Dataset configuration is required."

    # set random seeds
    utils.set_seed(config.base_seed)

    # Load encoder
    if regressor is None:
        encoder = encoders.create_encoder(config.encoder)  # type: ignore

        # Load head
        head = scr_heads.create_head(config.head_path)

        # Create regressor.
        regressor = regressors.Regressor(encoder, head)
        regressor = regressor.to(config.device)

    # Load map embeddings if specified
    if config.map_path is not None:
        map_dict = torch.load(config.map_path, map_location=config.device, weights_only=False)
        if map_embeddings is None:
            map_embeddings = map_dict.get("map_embeddings")
        if mean is None:
            mean = map_dict.get("mean")
        _logger.info(f"Loaded map from: {config.map_path}")

    # Setup dataset.
    dataset_config = copy.deepcopy(config.dataset)
    dataset_config.use_color = config.use_color if config.use_color is not None else regressor.encoder.supports_rgb
    dataset_config.pose_files = None  # No poses needed for registration; and also don't remove images without poses.

    dataset = datasets.CamLocDataset(dataset_config)
    if config.test_run:
        dataset.valid_file_indices = dataset.valid_file_indices[::10]

    _logger.info(f"Test images found: {len(dataset)}")

    _logger.info(f"Using focal length source: {config.dataset.calibration_source}")

    # Overwrite dataset heuristic focal length with external value if provided.
    if config.external_focal_length is not None:
        dataset.set_external_focal_length(config.external_focal_length)

        if config.dataset.calibration_source == "external":
            _logger.info(f"Using external focal length: {config.external_focal_length}")

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(
        dataset, shuffle=True, num_workers=config.num_data_workers, timeout=60 if config.num_data_workers > 0 else 0
    )

    # Set session id.
    if config.session_id is not None:
        session_id = config.session_id
    else:
        session_id = utils.generate_session_id(
            dataset.scene_name,
            regressor.encoder.__class__.__name__,
            regressor.head.__class__.__name__,
        )

    # Set subsample factor for the dataset
    dataset.set_subsample_factor(regressor.subsample_factor)

    # Setup for evaluation.
    regressor.eval()

    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = config.output_dir / (session_id + "_registered_poses.txt")
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    pose_log = open(pose_log_file, "w", 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    ace_estimates = []

    # Testing loop.
    with torch.no_grad():
        for image_B1HW, _, _, _, intrinsics_B33, _, gt_coords, filenames, indices in tqdm.tqdm(
            testset_loader, desc="Register images", dynamic_ncols=True
        ):
            batch_start_time = time.time()

            if config.register_gt_coords:
                scene_coordinates_B3HW = gt_coords.to(config.device, non_blocking=True)
                scene_coordinates_B3HW = scene_coordinates_B3HW[:, :, :32, :41]
            else:
                image_B1HW = image_B1HW.to(config.device, non_blocking=True)

                # Predict scene coordinates.
                with torch.autocast("cuda", enabled=True):
                    scene_coordinates_B3HW, uncertainty_B1HW = regressor(
                        images=image_B1HW, map_embeddings=map_embeddings, means=mean
                    )

                if uncertainty_B1HW is not None and config.uncertainty_filter is not None:
                    if config.uncertainty_filter == "threshold":
                        uncertainty_threshold = torch.tensor(config.uncertainty_filter_threshold)
                        while True:
                            used_points = (uncertainty_B1HW < uncertainty_threshold).sum()
                            if used_points > config.uncertainty_filter_threshold_min_points:
                                break
                            uncertainty_threshold *= 2
                            _logger.info(f"Increasing uncertainty threshold to {uncertainty_threshold}")
                    elif config.uncertainty_filter == "quantile":
                        uncertainty_threshold = torch.quantile(uncertainty_B1HW, config.uncertainty_filter_quantile)
                        uncertainty_threshold *= config.uncertainty_filter_quantile_factor
                    else:
                        raise ValueError(f"Invalid uncertainty filter: {config.uncertainty_filter}")

                    mask = uncertainty_B1HW < uncertainty_threshold
                    scene_coordinates_B3HW = scene_coordinates_B3HW * mask
                    used_points = mask.sum()

                    print(f"Using {mask.sum()} points with uncertainty below {uncertainty_threshold}")
                    rr.set_time("iteration", sequence=indices[0])
                    rr.set_time("percentage", duration=float(indices[0] / len(dataset)))
                    rr.log(config.rr_prefix + "used_points", rr.Scalars(used_points.numpy(force=True).item()))
                    rr.log(
                        config.rr_prefix + "uncertainty_threshold",
                        rr.Scalars(uncertainty_threshold.numpy(force=True).item()),
                    )

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for scene_coordinates_3HW, intrinsics_33, frame_path, index in zip(
                scene_coordinates_B3HW, intrinsics_B33, filenames, indices
            ):
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])
                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                # Allocate output variable.
                out_pose_c2w = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose_c2w,
                    config.hypotheses,
                    config.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    config.inlier_alpha,
                    config.max_pixel_error,
                    regressor.subsample_factor,
                    config.base_seed,
                )

                # Store estimates.
                ace_estimates.append(
                    eval_poses_utils.TestEstimate(
                        pose_c2w_est=out_pose_c2w.numpy().copy(),
                        pose_c2w_gt=None,
                        calibration=dataset.get_calibration(index),
                        confidence=inlier_count,
                        image_file=frame_path,
                    )
                )

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

            if 0 < config.max_estimates <= len(ace_estimates):
                _logger.info(f"Stopping at {len(ace_estimates)} estimates.")
                break

    # Process estimates and write them to file.
    for estimate in ace_estimates:
        pose_c2w_est = estimate.pose_c2w_est

        _logger.info(f"Frame: {estimate.image_file}, Confidence: {estimate.confidence}")

        # Write estimated pose to pose file (inverse).
        data_io.write_pose_to_ace_pose_file(
            pose_log,
            rgb_file=estimate.image_file,
            pose_w2c=np.linalg.inv(pose_c2w_est),
            confidence=estimate.confidence,
            calibration=estimate.calibration,
        )

    # Compute average time.
    avg_time = avg_batch_time / num_batches
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    pose_log.close()

    out_dict = {
        "sst": config.sst,
        "reg": utils.primitive(configuration.asdict(config, remove_global_config=True)),
        "eva": {
            "gt_pose_files": config.dataset.pose_files,
            "include_intervals": config.dataset.include_intervals,
            "ace_pose_file": pose_log_file,
            "session_id": session_id,
        },
    }

    # Write to yaml file
    output_file = config.output_dir / f"{session_id}_reg.yaml"
    utils.save_yaml(utils.primitive(out_dict), output_file)

    return {est.image_file: (est.pose_c2w_est, est.confidence) for est in ace_estimates}, out_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Register images to a map given by an SCR network.",
    )
    parser.add_argument("--config", nargs="+", help="Config file(s) to load.")

    config_dict = yoco.load_config_from_args(parser, search_paths=[".", "./configs"])
    config = configuration.fromdict(RegistrationConfig, config_dict, defaults_key="reg")

    register_images(config)
