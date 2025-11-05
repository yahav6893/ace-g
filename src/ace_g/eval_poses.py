# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Evaluate estimated poses.

Produces the following files:
    {output_dir}/{session_id}_eval.txt: Contains the evaluation results.
    {output_dir}/{session_id}_eval.yaml: Contains the evaluation results and configuration.
"""

import argparse
import collections
import dataclasses
import logging
import math
import pathlib
from typing import Tuple

import cv2
import numpy as np
import tqdm
import yoco

from ace_g import configuration, data_io, eval_poses_utils, utils

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationConfig(configuration.GlobalConfig):
    """Configuration for the eval_poses function."""

    ace_pose_file: pathlib.Path
    """Path to an ACE pose file with one line per image."""
    gt_pose_files: str
    """Glob pattern for pose files, e.g. 'datasets/scene/*.txt', each file is assumed to contain a 
    4x4) pose matrix, cam2world, correspondence with rgb files in the ACE pose file is assumed by 
    alphabetical order."""
    include_intervals: list[float] | None = None
    """Include only frames within these intervals, specified as an even-length list of alternating start and end ratios.
    If None, all frames are included."""
    estimate_alignment: bool = False
    """Estimate rigid body transformation between estimates and ground truth."""
    estimate_alignment_scale: bool = False
    """Estimate similarity transformation when estimating alignment."""
    estimate_alignment_conf_threshold: float = 500
    """Only consider pose estimates with higher confidence when estimating the alignment."""

    pose_error_thresh_t: list[float] = dataclasses.field(default_factory=lambda: [0.05, 0.1, 0.2])
    """Pose threshold(s) (translation) for evaluation and alignment; only first pair used for alignment."""
    pose_error_thresh_r: list[float] = dataclasses.field(default_factory=lambda: [5, 10, 20])
    """Pose threshold(s) (rotation) for evaluation and alignment; only first pair used for alignment."""

    test_run: bool = False
    """Overwrites parameters such that the script finishes in a few seconds; useful to try experiment scripts."""
    session_id: str | None = None
    """Custom session name used to generate output files; generated from pose file if not set."""
    output_dir: pathlib.Path = pathlib.Path("./outputs")
    """Target output dir for the trained network and / or map embeddings."""


def eval_poses(
    config: EvaluationConfig,
    estimates: dict[str, tuple[np.ndarray, float]] | None = None,
) -> Tuple[dict[str, float], dict]:
    """Evaluate estimated poses.

    Can be used either with via the command line like this
        eval_poses(create_eval_poses_parser().parse_args())
    or from code like this
        eval_poses({"ace_pose_file": "path/to/ace_pose_file.txt", "gt_pose_files": "path/to/pose_files/*.txt", ...})

    Args:
        config: Evaluation configuration.
        estimates:
            Dictionary mapping image names to tuple of cam-to-world pose matrix and confidence. If None the poses are
            read from the ACE pose file.

    Returns:
        Dictionary of metrics. Keys are metric names and values are the corresponding metric values.
    """
    if estimates is not None:
        _logger.info("Using estimates provided as argument.")
        ace_estimates = estimates
    else:
        assert config.ace_pose_file is not None, "Either estimates or ace_pose_file must be provided."
        _logger.info("Reading ACE pose file.")
        with open(config.ace_pose_file, "r") as f:
            ace_pose_data = f.readlines()

        # Dict mapping file name to ACE estimate
        ace_estimates = {}

        # parse pose file data
        for pose_line in ace_pose_data:
            file_name, pose_w2c, _, confidence = data_io.parse_ace_pose_line(pose_line)
            pose_c2w = np.linalg.inv(pose_w2c)
            ace_estimates[file_name] = (pose_c2w, confidence)

        _logger.info(f"Read {len(ace_estimates)} poses from: {config.ace_pose_file}")

    # sort ACE estimates by file names
    sorted_ace_poses = [ace_estimates[key] for key in sorted(ace_estimates.keys())]

    # load ground truth poses, sorted by file name
    sorted_gt_poses = data_io.load_pose_files(config.gt_pose_files)

    num_poses = len(sorted_gt_poses)
    if config.include_intervals is not None:
        # only evaluate poses within specified intervals
        sorted_gt_poses = data_io.apply_include_intervals(config.include_intervals, sorted_gt_poses)
        _logger.info(
            f"Filtered eval poses using intervals {config.include_intervals}. Reduced poses from {num_poses} to "
            f"{len(sorted_gt_poses)}."
        )

    if config.test_run:
        sorted_gt_poses = sorted_gt_poses[::10]

    # Remove invalid ground truth poses.
    sorted_ace_poses = data_io.filter_invalid_poses(poses=sorted_gt_poses, items=sorted_ace_poses)
    sorted_gt_poses = data_io.filter_invalid_poses(poses=sorted_gt_poses, items=sorted_gt_poses)

    # convert torch to numpy
    sorted_gt_poses = [pose.numpy() for pose in sorted_gt_poses]

    _logger.info(f"Evaluating {len(sorted_gt_poses)} ground truth poses.")

    assert len(sorted_ace_poses) == len(sorted_gt_poses), "Number of ground-truth poses and ACE estimates do not match."

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain threshold from their GT pose.
    deg_cm_to_accuracy = collections.defaultdict(float)

    if config.estimate_alignment:
        # alignment needs a list of pose correspondences with confidences
        pose_correspondences = []

        # walk through ACE estimates and GT poses in parallel
        for (ace_pose, ace_confidence), gt_pose in zip(sorted_ace_poses, sorted_gt_poses):
            pose_correspondences.append(
                (
                    eval_poses_utils.TestEstimate(
                        pose_c2w_est=ace_pose,
                        pose_c2w_gt=gt_pose,
                        confidence=ace_confidence,
                        image_file=None,
                        calibration=None,
                    )
                )
            )

        alignment_transformation, alignment_scale = eval_poses_utils.estimate_alignment(
            estimates=pose_correspondences,
            confidence_threshold=config.estimate_alignment_conf_threshold,
            estimate_scale=config.estimate_alignment_scale,
            inlier_threshold_r=config.pose_error_thresh_r[0],
            inlier_threshold_t=config.pose_error_thresh_t[0],
        )

        if alignment_transformation is None:
            _logger.info(f"Alignment requested but failed. Setting all pose errors to {math.inf}.")
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.0

    # Evaluation Loop
    for (ace_pose, ace_confidence), gt_pose in tqdm.tqdm(
        zip(sorted_ace_poses, sorted_gt_poses), dynamic_ncols=True, desc="Evaluating poses"
    ):
        if alignment_transformation is not None:
            # Apply alignment transformation to GT pose
            gt_pose = alignment_transformation @ gt_pose

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - ace_pose[0:3, 3]))

            # Correct translation scale with the inverse alignment scale (since we align GT with estimates)
            t_err = t_err / alignment_scale

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = ace_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            t_err, r_err = math.inf, math.inf

        _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100)  # in cm

        # Check various thresholds.
        for r_thresh, t_thresh in zip(config.pose_error_thresh_r, config.pose_error_thresh_t):
            if r_err < r_thresh and t_err < t_thresh:
                deg_cm_to_accuracy[r_thresh, t_thresh] += 1

    total_frames = len(rErrs)
    assert total_frames == len(sorted_ace_poses)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx].item()
    median_tErr = tErrs[median_idx]

    # Compute final accuracy.
    for r_thresh, t_thresh in zip(config.pose_error_thresh_r, config.pose_error_thresh_t):
        accuracy = deg_cm_to_accuracy[r_thresh, t_thresh] / total_frames * 100
        deg_cm_to_accuracy[r_thresh, t_thresh] = accuracy

    _logger.info("===================================================")
    _logger.info("Test complete.")

    for r_thresh, t_thresh in zip(config.pose_error_thresh_r, config.pose_error_thresh_t):
        accuracy = deg_cm_to_accuracy[r_thresh, t_thresh]
        _logger.info(f"Accuracy ({r_thresh}deg, {t_thresh * 100}cm): {accuracy:.1f}%")
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")

    # Set session id.
    if config.session_id is not None:
        session_id = config.session_id
    elif config.ace_pose_file is not None:
        session_id = config.ace_pose_file.stem.replace("_registered_poses", "")
    else:
        session_id = utils.generate_session_id(pathlib.Path(config.gt_pose_files).parts[-4])

    # Write to file
    output_file = config.output_dir / f"{session_id}_eval.txt"
    with open(output_file, "a") as f:
        f.write("===================================================\n")
        f.write(f"Results for {config.ace_pose_file} and {config.gt_pose_files}\n")
        for r_thresh, t_thresh in zip(config.pose_error_thresh_r, config.pose_error_thresh_t):
            accuracy = deg_cm_to_accuracy[r_thresh, t_thresh]
            f.write(
                f"Accuracy ({config.pose_error_thresh_r}deg, {config.pose_error_thresh_t * 100}cm): {accuracy:.1f}%\n"
            )
        f.write(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm\n")

    metric_dict = {
        "median_error_deg": median_rErr,
        "median_error_cm": median_tErr,
    }
    for r_thresh, t_thresh in zip(config.pose_error_thresh_r, config.pose_error_thresh_t):
        metric_dict[f"acc_{int(r_thresh)}deg_{int(t_thresh * 100)}cm"] = deg_cm_to_accuracy[r_thresh, t_thresh]

    out_dict = {
        "sst": config.sst,
        "reg": config.reg,
        "eva": utils.primitive(configuration.asdict(config, remove_global_config=True)),
        "res": metric_dict,
    }

    # Write to yaml file
    output_file = config.output_dir / f"{session_id}_eval.yaml"
    utils.save_yaml(utils.primitive(out_dict), output_file)

    return metric_dict, out_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compute pose error metrics for a set of estimated poses.",
    )
    parser.add_argument("--config", nargs="+", help="Config file(s) to load.")

    config_dict = yoco.load_config_from_args(parser, search_paths=[".", "./configs"])
    config = configuration.fromdict(EvaluationConfig, config_dict, defaults_key="eva")

    eval_poses(config)
