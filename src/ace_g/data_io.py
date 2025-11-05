# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Utility functions for loading dataset files."""

import glob
import logging
import time
import pathlib
from typing import Sequence, TextIO, cast

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation

_logger = logging.getLogger(__name__)


def load_pose(pose_file: str | pathlib.Path) -> torch.Tensor:
    """Load a pose from a file.

    The pose file should contain a 4x4 matrix.
    The pose is loaded using numpy's loadtxt function, converted to a torch tensor, and returned.

    Args:
        pose_file: The path to the pose file.

    Returns:
        The pose as a 4x4 torch tensor.
    """
    pose = np.loadtxt(pose_file)
    pose = torch.from_numpy(pose).float()
    return pose


def load_calibration(calibration_file: str | pathlib.Path) -> float | np.ndarray:
    """Load the focal length from a calibration file.

    The calibration file can either contain the focal length directly or a calibration matrix.

    If the calibration file contains a single value, it is assumed to be the focal length.
    If the calibration file contains two values, it is assumed to be the x and y focal lengths and the average is
    returned.
    If the calibration file contains a 3x3 matrix, it is assumed to be a calibration matrix and it is returned as is.

    Args:
        calibration_file: The path to the calibration file.

    Returns:
        The focal length or calibration matrix.
    """
    calibration_data = np.loadtxt(calibration_file)

    if calibration_data.shape == (2,):  # fx, fy
        return float(np.loadtxt(calibration_file)[0])
    elif calibration_data.shape == (3, 3):  # calibration matrix
        # code does not support different fx and fy; set them to the average
        f = (calibration_data[0, 0] + calibration_data[1, 1]) / 2
        if calibration_data[0, 0] != calibration_data[1, 1]:
            _logger.warning(
                f"Calibration matrix has different x and y focal lengths: {calibration_data[0, 0]} and "
                f"{calibration_data[1, 1]} and the average is used."
            )
        calibration_data[0, 0] = f
        calibration_data[1, 1] = f
        return calibration_data
    else:
        # assume calibration file contains focal length only
        return float(np.loadtxt(calibration_file))


def get_files_from_glob(glob_pattern: str) -> list[str]:
    """Get a list of files from a glob pattern, sorted alphabetically."""
    start_time = time.time()
    files = glob.glob(glob_pattern)
    files = sorted(files)
    _logger.info(f"Found {len(files)} files from glob {glob_pattern} in {time.time() - start_time:.2f}s.")

    if len(files) == 0:
        raise FileNotFoundError(f"No files found for glob pattern: {glob_pattern}")

    return files


def load_pose_files(glob_pattern: str) -> list[torch.Tensor]:
    """Load pose files by resolving the glob pattern (sorted alphabetically), return as a list of 4x4 torch tensors."""
    start_time = time.time()
    pose_files = sorted(glob.glob(glob_pattern))
    poses = [load_pose(pose_file) for pose_file in pose_files]
    _logger.info(f"Loaded {len(poses)} poses from glob {glob_pattern} in {time.time() - start_time:.2f}s.")
    return poses


def load_calibration_files(glob_pattern: str) -> list[float | np.ndarray]:
    """Load calibration files by resolving the glob pattern (sorted alphabetically), return as a list of floats."""
    start_time = time.time()
    calibration_files = sorted(glob.glob(glob_pattern))
    calibrations = [load_calibration(calibration_file) for calibration_file in calibration_files]
    run_time = time.time() - start_time
    _logger.info(f"Loaded {len(calibrations)} calibration files from glob {glob_pattern} in {run_time:.2f}s.")
    return calibrations


def check_pose(pose: torch.Tensor) -> bool:
    """Check if a pose is valid.

    A pose is considered valid if it does not contain NaN or inf values.

    Args:
        pose: The pose as a 4x4 torch tensor.

    Returns:
        True if the pose is valid, False otherwise.
    """
    return not torch.isnan(pose).any() and not torch.isinf(pose).any()


def filter_invalid_poses(poses: Sequence[torch.Tensor], items: Sequence) -> list:
    """Remove each item with an invalid pose from items.

    An invalid pose is a pose that contains NaN or inf values.

    Args:
        poses: The poses to check. Length must match items.
        items: The items to filter. Length must match poses.
    """
    assert len(poses) == len(items), "Length of poses and items must match."
    filtered_items = []
    for pose, item in zip(poses, items):
        if check_pose(pose):
            filtered_items.append(item)
    return filtered_items


def remove_invalid_poses(
    poses: Sequence[torch.Tensor],
    rgb_files: Sequence[str],
    depth_files: Sequence[str],
    calibrations: Sequence[float | np.ndarray],
) -> tuple[list[torch.Tensor], list[str], list[str], list[float | np.ndarray]]:
    """Remove each invalid pose from poses and the corresponding RGB file from rgb_files.

    An invalid pose is a pose that contains NaN or inf values.
    """
    valid_poses = []
    valid_rgb_files = []
    valid_depth_files = []
    valid_calibrations = []

    for i, pose in enumerate(poses):
        rgb_file = rgb_files[i]
        if not check_pose(pose):
            _logger.info(f"Pose for {rgb_file} contains NaN or inf values, skipping.")
        else:
            valid_poses.append(pose)
            valid_rgb_files.append(rgb_file)
            if len(calibrations) > 0:
                valid_calibrations.append(calibrations[i])
            if len(depth_files) > 0:
                valid_depth_files.append(depth_files[i])

    return valid_poses, valid_rgb_files, valid_depth_files, valid_calibrations


def load_dataset_ace(
    pose_file: str, confidence_threshold: float
) -> tuple[list[str], list[torch.Tensor], list[float | np.ndarray]]:
    """Load a dataset from a pose file. The pose file should contain lines with 10 tokens each.

    Poses are assumed to be world-to-cam.
    The tokens represent the following information:
        - mapping file
        - quaternion rotation (w, x, y, z)
        - translation (x, y, z)
        - focal length
        - confidence value

    Only entries with a confidence value above the provided threshold are included in the output.

    Args:
        pose_file: The path to the pose file.
        confidence_threshold: The minimum confidence value for an entry to be included in the output.

    Returns:
        A tuple containing three lists:
            - rgb_files: The paths to the RGB files.
            - poses: The poses as 4x4 torch tensors, cam-to-world.
            - focal_lengths: The focal lengths.
    """
    with open(pose_file, "r") as f:
        pose_lines = f.readlines()

        rgb_files = []
        poses = []
        calibrations = []

        for pose_line in pose_lines:
            file_name, pose_w2c, calibration, confidence = parse_ace_pose_line(pose_line)

            # read confidence values and compare to threshold
            if confidence < confidence_threshold:
                continue

            # pose files contain world-to-cam but we need cam-to-world
            pose_c2w = np.linalg.inv(pose_w2c)
            pose_c2w = torch.from_numpy(pose_c2w).float()

            rgb_files.append(file_name)
            calibrations.append(calibration)
            poses.append(pose_c2w)

    return rgb_files, poses, calibrations


def write_pose_to_ace_pose_file(
    out_pose_file: TextIO, rgb_file: str, pose_w2c: np.ndarray, confidence: float, calibration: float | np.ndarray
) -> None:
    """Write a pose to a pose file.

    The pose is converted from a numpy matrix to a quaternion and translation and world-to-cam convention.
    The pose file line format is as follows:
        - RGB file path
        - Quaternion rotation (w, x, y, z)
        - Translation (x, y, z)
        - Focal length or flattened intrinsic matrix
        - Confidence value

    Args:
        out_pose_file: The output file to write the pose to. Must be open for writing.
        rgb_file: The path to the RGB file.
        pose_w2c: The pose as a numpy matrix, 4x4 or 3x4, world-to-cam.
        confidence: The confidence value.
        calibration: The focal length or intrinsic matrix.
    """
    # convert Numpy pose matrix to quaternion and translation
    R_33 = pose_w2c[:3, :3]
    q_xyzw = Rotation.from_matrix(R_33).as_quat()
    t_xyz = pose_w2c[:3, 3]

    if isinstance(calibration, np.ndarray):
        calibration_str = " ".join([str(f) for f in calibration.flatten()])
    else:
        calibration_str = f"{calibration}"

    # write to pose file
    pose_str = (
        f"{rgb_file} "
        f"{q_xyzw[3]} {q_xyzw[0]} {q_xyzw[1]} {q_xyzw[2]} "
        f"{t_xyz[0]} {t_xyz[1]} {t_xyz[2]} {calibration_str} {confidence}\n"
    )

    out_pose_file.write(pose_str)


def parse_ace_pose_line(pose_line: str) -> tuple[str, torch.Tensor, float | np.ndarray, float]:
    """Parse a pose line from a pose file.

    Inverse of write_pose_to_pose_file.

    Returns:
        file_name: The file name.
        pose_w2c: The camera pose as a world-to-cam transformation, shape (4,4).
        calibration: The focal length or intrinsic matrix.
        confidence: The confidence value.
    """
    # image info as: file_name, q_w, q_x, q_y, q_z, t_x, t_y, t_z, focal_length or flattened intrinsics, confidence
    LENGTH_W_INTRINSICS = 18
    LENGTH_WO_INTRINSICS = 10
    pose_tokens = pose_line.split()

    assert len(pose_tokens) in [LENGTH_WO_INTRINSICS, LENGTH_W_INTRINSICS], (
        f"Expected 10 or 18 tokens per line in pose file, got {len(pose_tokens)}."
    )

    # read file name and confidence
    file_name = pose_tokens[0]
    confidence = float(pose_tokens[-1])

    # read pose
    q_wxyz = [float(t) for t in pose_tokens[1:5]]
    t_xyz = [float(t) for t in pose_tokens[5:8]]

    # quaternion to rotation matrix
    rot_w2c = Rotation.from_quat(q_wxyz[1:] + [q_wxyz[0]]).as_matrix()

    # construct full pose matrix
    pose_w2c = np.eye(4)
    pose_w2c[:3, :3] = rot_w2c
    pose_w2c[:3, 3] = t_xyz

    if len(pose_tokens) == LENGTH_W_INTRINSICS:
        calibration = np.array([float(t) for t in pose_tokens[8:17]]).reshape(3, 3)
    else:
        calibration = float(pose_tokens[-2])

    return file_name, pose_w2c, calibration, confidence


def get_depth_model(init: bool = False) -> torch.nn.Module:
    """Load the pretrained ZoeDepth model from the isl-org/ZoeDepth repository.

    Use torch.hub.load to load the model directly from GitHub.

    Args:
        init: Force reload the model from the repository.

    Returns:
        The pretrained ZoeDepth model.
    """
    # Warm up dependency in the torch hub cache.
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=init, trust_repo="check")
    repo = "isl-org/ZoeDepth"

    # Zoe_N
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=init, trust_repo="check")

    # Zoe_K
    # model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True, force_reload=init, trust_repo="check")

    # Zoe_NK (best performing model).
    model_zoe_nk = cast(
        torch.nn.Module, torch.hub.load(repo, "ZoeD_NK", pretrained=True, force_reload=init, trust_repo="check")
    )
    model_zoe_nk.eval().cuda()
    _logger.info("Loaded pretrained ZoeDepth model.")

    return model_zoe_nk


def estimate_depth(model: torch.nn.Module, image_pil: Image.Image) -> np.ndarray:
    """Estimate depth from an RGB image using the ZoeDepth model.

    Args:
        model: The ZoeDepth model.
        image_pil: The RGB image as a numpy array (HxWx3).

    Returns:
        The estimated depth as a numpy array (in m, HxW).
    """
    # Convert to tensor.
    image_BCHW = TF.to_tensor(image_pil).unsqueeze(0).cuda()

    # Run forward pass (on CPU)
    with torch.no_grad():
        depth_B1HW = model.infer(image_BCHW)  # type: ignore

    # Convert to numpy.
    depth_HW = depth_B1HW.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)

    return depth_HW


def apply_include_intervals(include_intervals: list[float], sorted_data: list) -> list:
    """Apply include intervals to data. Only keep data points that fall within the specified intervals."""
    return [data for i, data in enumerate(sorted_data) if in_intervals(i / len(sorted_data), include_intervals)]


def in_intervals(x: float, intervals: list[float]) -> bool:
    """Check if a value x is within any of the intervals."""
    for interval in zip(intervals[::2], intervals[1::2]):
        if interval[0] <= x < interval[1]:
            return True
    return False


def invert_intervals(intervals: list[float]) -> list[float]:
    """Invert intervals, i.e., return the complement of the intervals."""
    inverted_intervals = []

    if intervals[0] > 0:
        inverted_intervals.extend((0, intervals[0]))

    for prev_end, next_start in zip(intervals[1::2], intervals[2::2]):
        inverted_intervals.extend((prev_end, next_start))

    if intervals[-1] < 1:
        inverted_intervals.extend((intervals[-1], 1))

    return inverted_intervals


def remove_interval(intervals: list[float], remove_interval: list[float]) -> list[float]:
    """Remove an interval from intervals."""
    new_intervals = []
    for start, end in zip(intervals[::2], intervals[1::2]):
        if start >= remove_interval[1] or end <= remove_interval[0]:
            new_intervals.extend((start, end))
        elif start < remove_interval[0] and end > remove_interval[1]:
            new_intervals.extend((start, remove_interval[0], remove_interval[1], end))
        elif start <= remove_interval[0]:
            new_intervals.extend((start, remove_interval[0]))
        elif end >= remove_interval[1]:
            new_intervals.extend((remove_interval[1], end))
    return new_intervals


def subdivide_intervals(intervals: list[float], max_length: float) -> list[float]:
    """Subdivide intervals into smaller intervals of at most max_length."""
    new_intervals = []
    for start, end in zip(intervals[::2], intervals[1::2]):
        if end - start <= max_length:
            new_intervals.extend((start, end))
        else:
            num_subintervals = int((end - start) / max_length) + 1
            subinterval_length = (end - start) / num_subintervals
            for i in range(num_subintervals):
                new_intervals.extend((start + i * subinterval_length, start + (i + 1) * subinterval_length))
    return new_intervals
