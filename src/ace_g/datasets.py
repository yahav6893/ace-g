# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Generic dataset class for camera localization."""

import dataclasses
import logging
import math
import os
import pathlib
import random
import warnings
from typing import List, Literal, NamedTuple, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from skimage import color, io
from skimage.transform import resize, rotate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as torch_transforms

from ace_g import data_io, utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

_logger = logging.getLogger(__name__)


class CamLocDataset(Dataset):
    """Dataset class for ACE-G.

    Datasets are based on glob patterns that, at the minimum, define the path to the RGB images.

    Additionally, the following files can be provided:
        - Pose files
        - Depth files
        - Calibration files

    The resulting files need to sort consistently to ensure that different files are matched correctly.

    Returns images, calibration and poses. Optionally, ground truth scene coordinates from depth.

    See Config below for details.
    """

    @dataclasses.dataclass
    class Config:
        """Configuration for the dataset."""

        # Glob patterns for dataset files
        rgb_files: str
        """Glob pattern that matches RGB files."""
        pose_files: str | None = None
        """Glob pattern that matches pose files associated with RGB files."""
        depth_files: str | None = None
        """Glob pattern that matches depth files associated with RGB files. Dataset assumes that depth files can be
        resized to match the RGB images, i.e., it is the depth as seen from the RGB camera."""
        calibration_files: str | None = None
        """Glob pattern that matches per-image calibration files. Optional, see calibration_source."""

        # Parameters that should typically match the image encoder architecture
        use_color: bool = False
        """If True, the image is converted to grayscale."""
        subsample_factor: int | None = None
        """Subsample factor of the regressor (i.e., input size / output size). If None, no subsampling is applied."""

        # Other parameters
        use_half: bool = True
        """If True, the image is converted to half-precision floats."""
        calibration_source: Literal["pose_file", "dataset", "heuristic", "external"] | None = "dataset"
        """Source of the calibration. Has to be one of "pose_file", "dataset", "heuristic", "external"."""
        include_intervals: List[float] | None = None
        """Intervals of the dataset to include. Must be a sorted list with an even number of elements, where each pair
        defines the start and end of an interval. E.g., [0.5, 0.75] would only use frames from 50% (incl.) to 75%
        (excl.) of the dataset. The used file types (i.e., rgb, pose, depth, calibration) have to be alphabetically
        sortable for this to be consecutive frames. If None, the whole dataset is used. This can be used to split a
        single sequence into mapping and query parts."""
        mirror: bool = False
        """If True, images and corresponding poses will be flipped horizontally, such that, the up-axis remains up."""
        scale: float | None = None
        """Scale factor for the dataset. If None, no scaling is applied."""
        skip: int | None = None
        """Skip frames in the dataset. If None, no frames are skipped."""
        force_depth: bool = False
        """If True, estimate depth with ZoeDepth when no depth files are provided."""

        # Augmentation parameters
        use_aug: bool = False
        """Use random data augmentation."""
        aug_rotation: float = 15
        """Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees."""
        aug_scale_min: float = 2 / 3
        """Lower limit of image scale factor for uniform sampling."""
        aug_scale_max: float = 3 / 2
        """Upper limit of image scale factor for uniform sampling."""
        aug_black_white: float = 0.1
        """Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]"""
        aug_color: float = 0.3
        """Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]"""
        image_short_size: int = 480
        """RGB images are rescaled such that the short side has this length. If augmentation is disabled, and in the
        range [aug_scale_min * image_short_size, aug_scale_max * image_short_size] otherwise."""

    class Item(NamedTuple):
        """Item returned by the dataset."""

        image: torch.Tensor
        mask: torch.Tensor
        w2c: torch.Tensor
        """World-to-camera transformation matrix."""
        c2augc: torch.Tensor
        """Camera-to-augmented camera transformation matrix."""
        intrinsics: torch.Tensor
        intrinsics_inv: torch.Tensor
        coords: torch.Tensor
        file_names: list[str] | str
        idx: torch.LongTensor | int

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Create the dataset.

        Args:
            config: Configuration for the dataset. See CamLocDataset.Config for details.
        """
        self.config = config
        self.use_depth = self.config.force_depth or self.config.depth_files is not None
        self.subsample_factor = None

        _logger.info(f"Using color: {self.config.use_color}")

        self.root_dir = pathlib.Path(self.config.rgb_files).parent.parent

        # an external focal length can be provided using a setter function to overwrite the focal length
        self.external_focal_length = None

        if self.config.calibration_source == "heuristic":
            _logger.info("Overwriting focal length with heuristic derived from image dimensions.")

        # Loading dataset depending on what arguments are provided.
        _logger.info(f"Loading RGB files from: {self.config.rgb_files}")
        self.rgb_files = data_io.get_files_from_glob(self.config.rgb_files)
        self.depth_files = (
            data_io.get_files_from_glob(self.config.depth_files) if self.config.depth_files is not None else []
        )

        if self.use_depth and len(self.depth_files) == 0:
            _logger.info("No depth files provided. Estimating depth with ZoeDepth.")
            self.depth_model = data_io.get_depth_model()

        poses = data_io.load_pose_files(self.config.pose_files) if self.config.pose_files is not None else []
        calibrations = self.load_calibrations() if self.config.calibration_source == "dataset" else []

        self.poses = poses if self.config.pose_files is not None else []

        # set focal lengths
        if self.config.calibration_source == "dataset":
            self.calibrations = calibrations
        elif self.config.calibration_source in {"external", "heuristic"}:
            self.calibrations = []
        else:
            raise ValueError(
                "Focal length source when not using pose_file must be 'dataset', 'external' or 'heuristic'."
            )

        if len(self.poses) > 0:
            # Remove invalid poses and corresponding RGB files.
            self.poses, self.rgb_files, self.depth_files, self.calibrations = data_io.remove_invalid_poses(
                self.poses,
                self.rgb_files,
                self.depth_files,
                self.calibrations,
            )

        num_images_rgb = len(self.rgb_files)

        # Apply include intervals if set.
        if self.config.include_intervals is not None:
            assert sorted(self.config.include_intervals) == self.config.include_intervals, (
                "Include intervals must be sorted."
            )
            assert len(self.config.include_intervals) % 2 == 0, (
                "Include intervals must have an even number of elements."
            )

            self.rgb_files = data_io.apply_include_intervals(self.config.include_intervals, self.rgb_files)
            self.poses = data_io.apply_include_intervals(self.config.include_intervals, self.poses)
            self.calibrations = data_io.apply_include_intervals(self.config.include_intervals, self.calibrations)
            self.depth_files = data_io.apply_include_intervals(self.config.include_intervals, self.depth_files)

            _logger.info(
                f"Filtered dataset using intervals {self.config.include_intervals}. "
                f"Reduced images from {num_images_rgb} to {len(self.rgb_files)}."
            )

        assert len(self.poses) == 0 or len(self.rgb_files) == len(self.poses), (
            "Number of RGB files and poses must match."
        )
        assert len(self.depth_files) == 0 or len(self.rgb_files) == len(self.depth_files), (
            "Number of depth files and RGB files must match."
        )
        assert len(self.calibrations) == 0 or len(self.rgb_files) == len(self.calibrations), (
            "Number of calibrations and RGB files must match."
        )

        # If no poses are provided (e.g. during the relocalization stage) fill up with dummy identity poses.
        if len(self.poses) == 0:
            _logger.info("No poses provided. Dataset will return identity poses.")
            self.poses = [torch.eye(4, 4)] * len(self.rgb_files)

        # At this stage, number of poses and number of images should match
        if len(self.poses) != len(self.rgb_files):
            raise ValueError(
                f"Number of poses ({len(self.poses)}) does not match number of images ({len(self.rgb_files)})."
            )

        # Apply skip if set.
        if self.config.skip is not None:
            self.rgb_files = self.rgb_files[:: self.config.skip]
            self.poses = self.poses[:: self.config.skip]
            if len(self.calibrations) > 0:
                self.calibrations = self.calibrations[:: self.config.skip]
            if len(self.depth_files) > 0:
                self.depth_files = self.depth_files[:: self.config.skip]

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        transforms = []
        if not self.config.use_color:
            transforms.append(torch_transforms.Grayscale())
        if self.config.use_aug:
            transforms.append(
                torch_transforms.ColorJitter(
                    brightness=self.config.aug_black_white, contrast=self.config.aug_black_white
                )
            )
        transforms.append(torch_transforms.ToTensor())
        transforms.append(torch_transforms.Normalize(mean=[0.4], std=[0.25]))

        self.image_transform = torch_transforms.Compose(transforms)

        # We use this to iterate over all frames.
        self.valid_file_indices = np.arange(len(self.rgb_files)).tolist()

        # Calculate mean camera center (using the valid frames only).
        self.mean_cam_center = self._compute_mean_camera_center()

        # Set subsample factor if specified in config.
        if self.config.subsample_factor is not None:
            self.set_subsample_factor(self.config.subsample_factor)

    def set_external_focal_length(self, focal_length: float) -> None:
        """Set the focal length used when calibration_source is set to "external"."""
        if self.config.calibration_source != "external":
            warnings.warn(utils.YELLOW + "External focal length is set but not enabled." + utils.RESET)
        self.external_focal_length = focal_length

    def set_subsample_factor(self, subsample_factor: Optional[int]) -> None:
        """Set the subsample factor of the regressor."""
        # Create grid of 2D pixel positions used when generating scene coordinates from depth.
        self.subsample_factor = subsample_factor
        if self.use_depth and self.subsample_factor is not None:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

    def _create_prediction_grid(self) -> np.ndarray:
        """Create a downsampled grid whose values are the pixel positions in the image.

        Returns:
            Array if shape (2, 5000/subsample_factor, 5000/subsample_factor) where for each spatial position, the first
            channel is the x position before subsampling and the second channel is the y position before subsampling.
        """
        assert self.subsample_factor is not None, "Subsample factor must be set."

        # Assumes all input images have a resolution smaller than 5000x5000.
        prediction_grid = np.zeros(
            (
                2,
                math.ceil(5000 / self.subsample_factor),
                math.ceil(5000 / self.subsample_factor),
            )
        )

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * self.subsample_factor
                prediction_grid[1, y, x] = y * self.subsample_factor

        return prediction_grid

    def _compute_mean_camera_center(self) -> torch.Tensor:
        mean_cam_center = torch.zeros((3,))
        invalid_poses = 0

        for idx in self.valid_file_indices:
            pose = self.poses[idx].clone()

            if self.config.mirror:
                pose = _mirror_pose(pose)

            if self.config.scale:
                pose[:3, 3] *= self.config.scale

            if torch.any(torch.isnan(pose)) or torch.any(torch.isinf(pose)):
                invalid_poses += 1
                continue

            # Get the translation component.
            mean_cam_center += pose[0:3, 3]

        if invalid_poses > 0:
            _logger.warning(f"Ignored {invalid_poses} poses from mean computation.")

        # Avg.
        mean_cam_center /= len(self) - invalid_poses
        return mean_cam_center

    def _load_image(self, idx: int) -> np.ndarray:
        image = io.imread(self.rgb_files[idx])

        NUM_RGB_DIMENSIONS = 3
        if len(image.shape) < NUM_RGB_DIMENSIONS:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def get_image_size(self, idx: int) -> tuple[int, int]:
        """This method is used to get the size of the image at the given index.

        Opens image in lazy mode to get the size without loading the whole image.

        Args:
            idx: The index of the image for which the size is to be obtained.

        Returns:
            The size of the image in pixels. 2-tuple (width, height).
        """
        with Image.open(self.rgb_files[idx]) as img:
            return img.size

    def get_calibration(self, idx: int) -> float | np.ndarray:
        """This method is used to get the calibration of the camera used to capture the image at the given index.

        The calibration can be obtained in one of three ways:
            1. If external focal length is used, it must be set (principal point = image center).
            2. If the heuristic focal length is enabled, it is calculated based on the image dimensions
               (principal point = image center).
            3. Otherwise, the calibration is taken from pre-loaded calibration files or the pose file.

        Args:
            idx: The index of the image for which the focal length is to be obtained.

        Returns:
            The focal length or intrinsics matrix of the camera used to capture the image.
        """
        if self.config.calibration_source == "external":
            assert self.external_focal_length is not None, "External focal length not set."
            # use external focal length if set
            return self.external_focal_length
        elif self.config.calibration_source == "heuristic":
            # use heuristic focal length derived from image dimensions
            width, height = self.get_image_size(idx)

            # we use 70% of the diagonal as focal length
            return math.sqrt(width**2 + height**2) * 0.7
        elif self.config.calibration_source in {"pose_file", "dataset"}:
            return self.calibrations[idx]
        else:
            raise ValueError("Focal length source not set.")

    def _get_single_item(self, idx: int, image_short_size: int) -> Item:
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image_np = self._load_image(idx)

        # Load intrinsics.
        calibration = self.get_calibration(idx)

        if isinstance(calibration, float):
            focal_length_x = calibration
            focal_length_y = calibration
            cx = image_np.shape[1] / 2
            cy = image_np.shape[0] / 2
        elif isinstance(calibration, np.ndarray):
            focal_length_x = calibration[0, 0]
            focal_length_y = calibration[1, 1]
            cx = calibration[0, 2]
            cy = calibration[1, 2]
        else:
            raise ValueError("Invalid calibration type.")

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_short_size / min(image_np.shape[0], image_np.shape[1])
        focal_length_x *= f_scale_factor
        focal_length_y *= f_scale_factor
        cx *= f_scale_factor
        cy *= f_scale_factor

        # Mirror image
        if self.config.mirror:
            image_np = image_np[:, ::-1, :]

        # Rescale image.
        image_pil = _resize_image(image_np, image_short_size)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image_pil.size[1], image_pil.size[0]))

        # Load ground truth scene coordinates, if needed.
        depth = None
        if self.use_depth:
            if len(self.depth_files) > 0:
                if self.depth_files[idx].endswith(".exr"):
                    # read with OpenCV
                    depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_ANYDEPTH)
                    assert depth is not None, "Depth image could not be read."
                    depth = depth.astype(np.float32)
                else:
                    # read depth map from disk
                    depth = io.imread(self.depth_files[idx])
                    depth = depth.astype(np.float32)
                    depth /= 1000.0  # from millimeters to meters

                if self.config.mirror:
                    depth = depth[:, ::-1]

                depth = resize(
                    depth, (image_pil.size[1], image_pil.size[0]), order=0, preserve_range=True, anti_aliasing=False
                )  # use nearest neighbor interpolation for depth
            else:  # there are no depth files -> estimate depth
                depth = data_io.estimate_depth(self.depth_model, image_pil)

            if self.config.scale is not None:
                depth *= self.config.scale

        # Apply remaining transforms.
        image_torch = self.image_transform(image_pil)

        # Get pose.
        pose = self.poses[idx].clone()
        if self.config.mirror:
            pose = _mirror_pose(pose)
        if self.config.scale:
            pose[:3, 3] *= self.config.scale

        # Apply data augmentation if necessary.
        if self.config.use_aug:
            # Generate a random rotation angle.
            angle = random.uniform(-self.config.aug_rotation, self.config.aug_rotation)

            # Rotate input image and mask.
            image_torch = _rotate_image(image_torch, angle, order=1, mode="reflect")
            image_mask = _rotate_image(image_mask, angle, order=1, mode="constant")

            # If we loaded the GT scene coordinates.
            if self.use_depth:
                # rotate depth maps using TF here
                # >10x faster than skimage when using nearest neighbor interpolation
                # tested with scikit-image 0.23.2 and torchvision 0.19.0
                depth = (
                    TF.rotate(torch.from_numpy(depth).unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST)
                    .squeeze(0)
                    .numpy()
                )

            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.0
            # Create a rotation matrix.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)
        else:
            pose_rot = torch.eye(4)

        # Generate ground truth scene coordinates from depth.
        if self.use_depth and depth is not None:
            assert self.subsample_factor is not None, "Subsample factor must be set."
            assert self.prediction_grid is not None, "Prediction grid must be set."

            # generate initialization targets from depth map
            offsetX = int(self.subsample_factor / 2)
            offsetY = int(self.subsample_factor / 2)

            coords = torch.zeros(
                (
                    3,
                    math.ceil(image_torch.shape[1] / self.subsample_factor),
                    math.ceil(image_torch.shape[2] / self.subsample_factor),
                )
            )

            # subsample to network output size
            depth = depth[
                offsetY :: self.subsample_factor,
                offsetX :: self.subsample_factor,
            ]

            # construct x and y coordinates of camera coordinate
            xy = self.prediction_grid[:, : depth.shape[0], : depth.shape[1]].copy()
            # add random pixel shift
            xy[0] += offsetX
            xy[1] += offsetY
            # subtract principal point
            xy[0] -= cx
            xy[1] -= cy
            # reproject
            xy[0] /= focal_length_x
            xy[1] /= focal_length_y
            xy[0] *= depth
            xy[1] *= depth

            # assemble camera coordinates tensor
            eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
            eye[0:2] = xy
            eye[2] = depth
            eye[3] = 1

            # eye to scene coordinates
            sc = np.matmul(pose.numpy() @ pose_rot.numpy(), eye.reshape(4, -1))
            sc = sc.reshape(4, depth.shape[0], depth.shape[1])

            # mind pixels with invalid depth
            sc[:, depth == 0] = 0
            sc[:, depth > 1000] = 0
            sc = torch.from_numpy(sc[0:3])

            coords[:, : sc.shape[1], : sc.shape[2]] = sc
        else:
            # set coords to all zeros as a default, training loop will catch this case
            assert self.subsample_factor is not None, "Subsample factor must be set."
            coords = torch.zeros(
                (
                    3,
                    math.ceil(image_torch.shape[1] / self.subsample_factor),
                    math.ceil(image_torch.shape[2] / self.subsample_factor),
                )
            )

        # Convert to half if needed.
        if self.config.use_half and torch.cuda.is_available():
            image_torch = image_torch.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose, we need world-to-camera during training.
        pose_inv = pose.inverse()
        pose_rot_inv = pose_rot.inverse()

        # Final check of poses before returning.
        if not data_io.check_pose(pose_inv) or not data_io.check_pose(pose_rot_inv):
            raise ValueError(f"Pose at index {idx} is invalid.")

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = focal_length_x
        intrinsics[1, 1] = focal_length_y
        # Hardcode the principal point to the centre of the image.
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()

        return CamLocDataset.Item(
            image_torch,
            image_mask,
            pose_inv,
            pose_rot_inv,
            intrinsics,
            intrinsics_inv,
            coords,
            str(self.rgb_files[idx]),
            idx,
        )

    def __len__(self) -> int:
        """Return the number of valid items in the dataset."""
        return len(self.valid_file_indices)

    def __getitem__(self, idx: int | list[int]) -> Item | list[Item]:
        """Return the item at the given index."""
        if self.config.use_aug:
            scale_factor = random.uniform(self.config.aug_scale_min, self.config.aug_scale_max)
            # scale_factor = 1 / scale_factor #inverse scale sampling, not used for ACE mapping
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_short_size = int(self.config.image_short_size * scale_factor)

        if isinstance(idx, list):
            # Whole batch.
            tensors = [self._get_single_item(i, image_short_size) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_short_size)

    @property
    def scene_name(self) -> str:
        """Return scene name retrieved from RGB path."""
        dataset_name = pathlib.Path(self.rgb_files[0]).parts[-5]
        return dataset_name + "_" + pathlib.Path(self.rgb_files[0]).parts[-4]

    @property
    def split_name(self) -> str:
        """Return scene name retrieved from RGB path."""
        return pathlib.Path(self.rgb_files[0]).parts[-3]

    def load_calibrations(self) -> list[float | np.ndarray]:
        """Load calibrations from disk."""
        if (self.root_dir / "calibration.txt").is_file():
            calibration = data_io.load_calibration(self.root_dir / "calibration.txt")
            calibrations = [calibration] * len(self.rgb_files)
        elif (self.root_dir / "calibration").is_dir():
            calibration_glob = str(self.root_dir / "calibration" / "*.txt")
            if self.config.calibration_files is not None:
                calibration_glob = self.config.calibration_files
            calibrations = data_io.load_calibration_files(calibration_glob)
        else:
            raise ValueError("No calibration files found in dataset directory.")
        return calibrations


def _mirror_pose(pose: torch.Tensor) -> torch.Tensor:
    """Return mirrored pose.

    When flipping images, poses need to be mirrored to ensure consistency. This functions implements one possible
    mirroring transformation.
    """
    pose = pose.clone()
    # NOTE this is the transform we're doing here: w'Tc' = w'Tw @ wTc @ cTc'
    #  where w' is the mirrored world frame, w is the original world frame,
    #  c' is the mirrored camera frame (after flipping the order of columns) and c is the original camera frame

    # in total this gives a normal SE3 transform
    # left hand multiply with any rotoreflection + no translation
    # for simplicity we use diag(-1, 1, 1, 1)
    pose[0, :] *= -1

    # right hand multiply with diag(-1, 1, 1, 1)
    # this flips the x axis of the camera (i.e., flipping the order of columns)
    pose[:, 0] *= -1
    return pose


def _rotate_image(image: torch.Tensor, angle: float, order: int, mode: str = "constant") -> torch.Tensor:
    # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
    image_np = image.permute(1, 2, 0).numpy()
    # Apply rotation.
    image_np = rotate(image_np, angle, order=order, mode=mode)
    # Back to torch tensor.
    image = torch.from_numpy(image_np).permute(2, 0, 1).float()
    return image


def _resize_image(image: np.ndarray, short_size: int) -> Image.Image:
    # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
    pil_image = TF.to_pil_image(image)
    # Will resize such that shortest side has short_size length in px.
    pil_image: Image.Image = TF.resize(pil_image, short_size)  # type: ignore (TF.resize is incorrectly typed)
    return pil_image
