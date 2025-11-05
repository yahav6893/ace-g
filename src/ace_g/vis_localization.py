# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Visualize registration (i.e., localization) results."""

import argparse
import dataclasses
import logging
import math
import pathlib
import pprint

import cv2
import dacite
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tqdm
import yoco
from skimage import color, io
from skimage.transform import resize

from ace_g import data_io, datasets, encoders, register_images, regressors, scr_heads, utils, vis_utils


@dataclasses.dataclass(kw_only=True)
class VisualizationConfig(register_images.RegistrationConfig):
    """Configuration for visualizing the registration."""

    mapping_intervals: list[float] | None = None
    """Mapping intervals as an even-length list of alternating start and end ratios; only used for visualization."""
    query_intervals: list[float] | None = None
    """Query intervals as an even-length list of alternating start and end ratios; only used for visualization."""
    switch_sequence: str | bool = False
    """Name of other sequence to switch to; or True to switch to other sequence of two .../scene/{seq_1,seq_2}/rgb
    This assumes that two levels above the rgb files there are 2+ directories corresponding to different sequences."""
    filter_depth: float | None = None
    """Filter points based on depth; if set, only points with depth smaller than this value are kept."""
    rrd_path: pathlib.Path | None = None
    """If set, save Rerun's rrd to this path; otherwise, use connect."""
    rerun_spawn: bool = False
    """If True, spawn a new Rerun viewer, otherwise use connect."""


class LocalizationVisualizer:
    """Visualizer for localization results."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize the visualizer."""
        self.config = config
        self._init_regressor()
        self._init_dataset()
        self._init_map_embeddings()
        self.pc_xyz = []
        self.pc_clr = []

    def _init_regressor(self) -> None:
        """Initialize the regressor."""
        encoder = encoders.create_encoder(self.config.encoder)  # type: ignore
        head = scr_heads.create_head(self.config.head_path)
        self.regressor = regressors.Regressor(encoder, head)
        self.regressor = self.regressor.to(self.config.device)
        self.regressor.eval()
        self.pixel_grid = utils.get_pixel_grid(self.regressor.subsample_factor)  # Shape: 2x5000x5000

    def _init_dataset(self) -> None:
        """Initialize the dataset."""
        self.config.dataset.use_color = (
            self.config.use_color if self.config.use_color is not None else self.regressor.encoder.supports_rgb
        )
        if self.config.dataset.use_aug:
            self.config.dataset.use_aug = False
            logging.warning("Data augmentation not supported in visualization script; turning it off.")

        if isinstance(self.config.switch_sequence, str) or self.config.switch_sequence is True:
            if isinstance(self.config.switch_sequence, bool) and self.config.switch_sequence is True:
                sequences_dirs = list(pathlib.Path(self.config.dataset.rgb_files).parent.parent.parent.glob("*"))
                assert len(sequences_dirs) == 2, "Switching sequence only works if there are two sequences."  # noqa: PLR2004
                current_sequence = pathlib.Path(self.config.dataset.rgb_files).parts[-3]
                other_sequence = (
                    sequences_dirs[0] if sequences_dirs[1].name == current_sequence else sequences_dirs[1]
                ).name
            else:
                other_sequence = self.config.switch_sequence

            rgb_parts = list(pathlib.Path(self.config.dataset.rgb_files).parts)
            rgb_parts[-3] = other_sequence
            self.config.dataset.rgb_files = pathlib.Path(*rgb_parts).as_posix()
            if self.config.dataset.depth_files is not None:
                depth_parts = list(pathlib.Path(self.config.dataset.depth_files).parts)
                depth_parts[-3] = other_sequence
                self.config.dataset.depth_files = pathlib.Path(*depth_parts).as_posix()
            if self.config.dataset.pose_files is not None:
                pose_parts = list(pathlib.Path(self.config.dataset.pose_files).parts)
                pose_parts[-3] = other_sequence
                self.config.dataset.pose_files = pathlib.Path(*pose_parts).as_posix()

        self.dataset = datasets.CamLocDataset(self.config.dataset)
        self.dataset.set_subsample_factor(self.regressor.subsample_factor)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, shuffle=False, num_workers=6)

    def _init_map_embeddings(self) -> None:
        """Initialize the map embeddings."""
        if self.config.map_path is not None:
            map_dict = torch.load(self.config.map_path, map_location=self.config.device, weights_only=False)
            self.map_embeddings = map_dict.get("map_embeddings")
            self.mean = map_dict.get("mean")
        else:
            self.map_embeddings = None
            self.mean = None

    def visualize(self) -> None:
        """Visualize the localization results for the entire dataset."""
        vis_utils.rr_init("ace_g_localization", self.config.rrd_path, self.config.rerun_spawn)
        _setup_localization_blueprint()

        config_str = pprint.pformat(dataclasses.asdict(self.config), sort_dicts=False)
        rr.log(
            self.config.rr_prefix + "info",
            rr.TextDocument(
                f"**Head:** {self.config.head_path}\n\n"
                f"**Scene:** {self.config.dataset.rgb_files}\n\n"
                f"**Config:**\n\n```\n{config_str}\n```\n\n",
                # f"**Mapping iterations:** {mapping_iterations}\n\n"
                # f"**Supervision type:** {supervision_type}\n\n"
                # f"**Validation scene:** {validation_scene}\n\n",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

        if self.config.mapping_intervals is not None:
            rr.log(self.config.rr_prefix + "mapping_image", rr.SeriesPoints(), static=True)
        if self.config.query_intervals is not None:
            rr.log(self.config.rr_prefix + "query_image", rr.SeriesPoints(), static=True)

        self.estimates, _ = register_images.register_images(
            config=self.config,
            regressor=self.regressor,
            map_embeddings=self.map_embeddings,
            mean=self.mean,
        )

        self.dataset.set_subsample_factor(self.regressor.subsample_factor)

        with torch.no_grad():
            # iterate over mapping sequence
            for item in tqdm.tqdm(self.data_loader, desc="Processing images", dynamic_ncols=True):
                self.current_ratio = item.idx / len(self.data_loader)
                rr.set_time("iteration", sequence=item.idx)
                rr.set_time("percentage", duration=float(self.current_ratio))
                self._visualize_item(item)

        # merge points
        pc_xyz_np = np.concatenate(self.pc_xyz, axis=1)
        pc_clr_np = np.concatenate(self.pc_clr, axis=1)

        # 3N to N3
        pc_xyz_np = np.transpose(pc_xyz_np)
        pc_clr_np = np.transpose(pc_clr_np)

        # # OpenCV to OpenGL convention
        # pc_xyz[:, 1] = -pc_xyz[:, 1]
        # pc_xyz[:, 2] = -pc_xyz[:, 2]

        # # visualize final pointcloud
        # points_cv = pc_xyz * np.array([1, -1, -1])  # Convert to OpenCV coordinate system
        rr.log(self.config.rr_prefix + "points", rr.Points3D(pc_xyz_np, colors=pc_clr_np / 255.0))

    def _visualize_item(self, item: datasets.CamLocDataset.Item) -> None:
        """Visualize the localization results for a single dataset item."""
        # predict scene coordinate
        image = item.image.to(self.config.device, non_blocking=True)
        w2c_b44 = item.w2c.to(self.config.device, non_blocking=True)
        c2augc_b44 = item.c2augc.to(self.config.device, non_blocking=True)
        intrinsics = item.intrinsics.to(self.config.device, non_blocking=True)

        with torch.autocast("cuda"):
            scene_coords, uncertainties = self.regressor(image, map_embeddings=self.map_embeddings, means=self.mean)
            if uncertainties is not None:
                uncertainties = uncertainties[0, 0].numpy(force=True)  # just (H, W) after this

        b, _, h, w = scene_coords.shape
        scene_coords_hw3 = scene_coords[0].permute(1, 2, 0)
        scene_coords = (scene_coords.view(3, -1)).view(b, 3, h, w)

        if self.config.register_gt_coords:
            scene_coords = item.coords.to(self.config.device, non_blocking=True)
            scene_coords = scene_coords[:, :, :h, :w].contiguous()

        assert b == 1, "Batch size must be 1 for point cloud extraction."

        # scene coordinate to camera coordinates
        pred_scene_coords_B3HW = scene_coords.float()
        pred_scene_coords_B4N = utils.to_homogeneous(pred_scene_coords_B3HW.flatten(2))
        pred_cam_coords_B3N = torch.matmul(c2augc_b44[:, :3, :] @ w2c_b44, pred_scene_coords_B4N)

        # project scene coordinates
        pred_px_B3N = torch.matmul(intrinsics, pred_cam_coords_B3N)
        pred_px_B3N[:, 2].clamp_(min=0.1)  # avoid division by zero
        pred_px_B2N = pred_px_B3N[:, :2] / pred_px_B3N[:, 2, None]

        # measure reprojection error
        pixel_positions_2HW = self.pixel_grid[:, :h, :w].clone()  # Crop to actual size
        pixel_positions_2N = pixel_positions_2HW.view(2, -1)

        reprojection_error_2N = pred_px_B2N.squeeze() - pixel_positions_2N.to(self.config.device)
        reprojection_error_1N = torch.norm(reprojection_error_2N, dim=0, keepdim=True, p=1)

        # filter based on gradient of scene coordinates
        grad_x_BHW = torch.linalg.norm(
            pred_scene_coords_B3HW[:, :, :, 1:] - pred_scene_coords_B3HW[:, :, :, :-1],
            dim=1,
        )
        grad_x_BHW = torch.nn.functional.pad(grad_x_BHW, (1, 0), mode="reflect")
        grad_y_BHW = torch.linalg.norm(
            pred_scene_coords_B3HW[:, :, 1:, :] - pred_scene_coords_B3HW[:, :, :-1, :],
            dim=1,
        )
        grad_y_BHW = torch.nn.functional.pad(grad_y_BHW, (0, 0, 1, 0), mode="reflect")

        grad_BHW = torch.max(grad_x_BHW, grad_y_BHW)
        grad_1N = grad_BHW.view(b, -1)

        # remove points where scene coordinates change more than this threshold from one pixel to the next (in meters)
        # since scene can have vastly different scales, and scales are estimates, we try increasingly relaxed thresholds
        # try different grad thresholds, keep the tightest one that still has enough points

        # total number of points in the point cloud, at least min even with large re-projection errors
        # at most max, even if more points have small re-projection errors
        pc_points_min = 100_000
        pc_points_max = 1_000_000
        pc_points_per_image_min = int(pc_points_min / len(self.data_loader))
        pc_points_per_image_max = int(pc_points_max / len(self.data_loader))
        grad_thresholds = [0.1, 0.5, 1.0, torch.inf]
        for grad_threshold in grad_thresholds:
            sc_grad_mask = grad_1N.squeeze() < grad_threshold
            if sc_grad_mask.sum() > pc_points_per_image_min:
                break

        if self.config.filter_depth is not None:
            # filter predictions based on depth
            sc_depth_mask = pred_cam_coords_B3N[0, 2] < self.config.filter_depth
        else:
            sc_depth_mask = torch.ones_like(sc_grad_mask).bool()

        sc_grad_and_depth_mask = torch.logical_and(sc_grad_mask, sc_depth_mask)

        # if no points survive, keep all
        if sc_grad_and_depth_mask.sum() == 0:
            sc_grad_and_depth_mask[:] = True

        # apply reprojection error
        # remove points with re-projection larger than threshold (in px) as long as we keep a min number of points
        repro_threshold = 100.0
        sc_err_mask = reprojection_error_1N.squeeze() < repro_threshold
        sc_err_mask = torch.logical_and(sc_err_mask, sc_grad_and_depth_mask)

        # check whether enough point survive
        num_valid_points = int(sc_err_mask.sum())

        if num_valid_points < pc_points_per_image_min:
            # take min points with lowest reprojection error
            reprojection_error_within_range_and_smooth_1N = reprojection_error_1N.squeeze()[sc_grad_and_depth_mask]

            sorted_errors, _ = torch.sort(reprojection_error_within_range_and_smooth_1N)
            relaxed_filter_repro_error = sorted_errors[min(pc_points_per_image_min, sorted_errors.shape[0] - 1)]

            sc_err_mask = reprojection_error_1N.squeeze() < relaxed_filter_repro_error
            sc_err_mask = torch.logical_and(sc_grad_and_depth_mask, sc_err_mask)
        elif num_valid_points > pc_points_per_image_max:
            # sub-sample points
            keep_ratio = pc_points_per_image_max / num_valid_points
            sub_sample_mask = torch.randperm(num_valid_points) < int(keep_ratio * num_valid_points)
            sc_err_mask_subsampled = sc_err_mask.clone()
            sc_err_mask_subsampled[sc_err_mask] = sub_sample_mask.to(self.config.device)
            sc_err_mask = sc_err_mask_subsampled

        # load image file to extract colors
        rgb = io.imread(item.file_names[0])
        if self.config.dataset.mirror:
            rgb[:] = rgb[:, ::-1]

        if len(rgb.shape) < 3:
            rgb = color.gray2rgb(rgb)

        # align RGB values with scene coordinate prediction
        rgb = rgb.astype("float64")
        # firstly, resize image to network input resolution
        rgb = resize(rgb, image.shape[2:])
        # secondly, sub-sampling to network output resolution
        # using patch center gives slightly crisper colors than averaging
        nn_stride = self.regressor.subsample_factor
        nn_offset = self.regressor.subsample_factor // 2
        rgb = rgb[nn_offset::nn_stride, nn_offset::nn_stride, :]

        # make sure number of patches and patch centers matches (some encoders include incomplete patches, others
        # don't); so we take the minimum of the two
        # prefix all vars with patch to avoid confusion with previous vars
        h = min(scene_coords.shape[2], rgb.shape[0])
        w = min(scene_coords.shape[3], rgb.shape[1])
        patch_rgb = rgb[:h, :w].transpose(2, 0, 1).reshape(3, -1)
        patch_xyz = pred_scene_coords_B3HW[:, :, :h, :w].reshape(3, -1).numpy(force=True)
        patch_err_mask = (
            sc_err_mask.reshape(scene_coords.shape[2], scene_coords.shape[3])[:h, :w].flatten().numpy(force=True)
        )

        # make sure the resolution fits (catch any striding mismatches)

        # remove invalid map points
        filtered_rgb = patch_rgb[:, patch_err_mask]
        filtered_xyz = patch_xyz[:, patch_err_mask]

        self.pc_xyz.append(filtered_xyz)
        self.pc_clr.append(filtered_rgb)

        is_mapping_image = (
            data_io.in_intervals(self.current_ratio, self.config.mapping_intervals)
            if self.config.mapping_intervals
            else False
        )
        is_query_image = (
            data_io.in_intervals(self.current_ratio, self.config.query_intervals)
            if self.config.query_intervals
            else False
        )

        if is_mapping_image:
            rr.log(self.config.rr_prefix + "mapping_image", rr.Scalars(0))
        if is_query_image:
            rr.log(self.config.rr_prefix + "query_image", rr.Scalars(0))

        rr.log(
            self.config.rr_prefix + "world/camera/image",
            rr.Pinhole(image_from_camera=intrinsics.numpy(force=True), image_plane_distance=0.2, color=[122, 188, 204]),
        )

        w2augc = c2augc_b44 @ w2c_b44
        rr.log(
            self.config.rr_prefix + "world/camera",
            rr.Transform3D(
                translation=w2augc[0, :3, 3].numpy(force=True),
                mat3x3=w2augc[0, :3, :3].numpy(force=True),
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )

        if uncertainties is not None and self.config.uncertainty_filter is not None:
            if self.config.uncertainty_filter == "threshold":
                uncertainty_threshold = torch.tensor(self.config.uncertainty_filter_threshold)
                while True:
                    used_points = (uncertainties < uncertainty_threshold).sum()
                    if used_points > self.config.uncertainty_filter_threshold_min_points:
                        break
                    uncertainty_threshold *= 2
            elif self.config.uncertainty_filter == "quantile":
                uncertainty_threshold = torch.quantile(
                    torch.from_numpy(uncertainties), self.config.uncertainty_filter_quantile
                )
                uncertainty_threshold *= self.config.uncertainty_filter_quantile_factor
            else:
                raise ValueError(f"Invalid uncertainty filter: {self.config.uncertainty_filter}")

            mask = uncertainties < uncertainty_threshold.numpy(force=True)
            fil_patch_rgb = rgb[:h, :w][mask]
            fil_patch_xyz = scene_coords_hw3[mask].numpy(force=True)
            rr.log(
                self.config.rr_prefix + "world/unc_filered_points",
                rr.Points3D(fil_patch_xyz, colors=fil_patch_rgb / 255),
            )

        rr.log(self.config.rr_prefix + "world/all_points", rr.Points3D(patch_xyz.T, colors=patch_rgb.T / 255))
        rr.log(
            self.config.rr_prefix + "world/filtered_points", rr.Points3D(filtered_xyz.T, colors=filtered_rgb.T / 255)
        )

        rgb_uint8_image = ((image[0].permute(1, 2, 0) * 0.25 + 0.4) * 255).byte().cpu().numpy()
        rr.log(self.config.rr_prefix + "world/camera/image", rr.Image(rgb_uint8_image).compress())

        scene_coords_hw3 = scene_coords_hw3.numpy(force=True)
        scene_coords_hw3 = (scene_coords_hw3 - scene_coords_hw3.min(axis=(0, 1))) / (
            np.ptp(scene_coords_hw3, axis=(0, 1))
        )
        rr.log(
            self.config.rr_prefix + "world/camera/scr_image",
            rr.Image((scene_coords_hw3 * 255).astype(np.uint8)).compress(),
        )

        if uncertainties is not None:
            # color map uncertainties
            vmin = 0.0
            vmax = 0.5
            cmap = matplotlib.colormaps["magma"]
            rgb_uncertainties = (cmap((uncertainties - vmin) / (vmax - vmin))[..., :3] * 256).astype(np.uint8)
            rr.log(self.config.rr_prefix + "world/camera/unc", rr.Image(rgb_uncertainties).compress())

        gt_pose = w2augc[0].inverse().cpu().numpy()

        estimate = self.estimates.get(item.file_names[0])
        if estimate is not None:
            scale = self.config.dataset.scale if self.config.dataset.scale is not None else 1.0
            rr.log(self.config.rr_prefix + "score", rr.Scalars(estimate[1]))
            est_pose = estimate[0]

            t_err = float(np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3]))
            rr.log(self.config.rr_prefix + "t_err_cm", rr.Scalars(t_err * 100 / scale))

            r_err = np.linalg.norm(cv2.Rodrigues(np.matmul(gt_pose[:3, :3], est_pose[:3, :3].T))[0]) * 180 / math.pi
            rr.log(self.config.rr_prefix + "r_err_deg", rr.Scalars(r_err))

            rr.log(
                self.config.rr_prefix + "world/est_camera/image",
                rr.Pinhole(
                    image_from_camera=intrinsics.numpy(force=True), image_plane_distance=0.2, color=[152, 175, 117]
                ),
            )
            rr.log(self.config.rr_prefix + "world/est_camera/image", rr.Image(rgb_uint8_image).compress())
            rr.log(
                self.config.rr_prefix + "world/est_camera",
                rr.Transform3D(
                    translation=est_pose[:3, 3],
                    mat3x3=est_pose[:3, :3],
                ),
            )

            # color trajectory based on error
            cmap = plt.get_cmap("cool")
            error = max(t_err * 100, r_err.item())  # cm == deg
            min_error = 5
            max_error = 50
            error_color = cmap((error - min_error) / (max_error - min_error))[:3]

            rr.log(self.config.rr_prefix + "world/camera_center", rr.Points3D(gt_pose[:3, 3], colors=error_color))


def _setup_localization_blueprint() -> None:
    """Setup the blueprint for the mapping visualization."""
    top_left = rrb.Tabs(
        contents=[rrb.TextDocumentView(origin="/info"), rrb.Spatial3DView(name="3D View")], active_tab=1
    )
    top_right = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        name="Image with coords from gt pose",
                        origin="world/camera/image",
                        contents=["$origin/**", "/world/all_points"],
                    ),
                    rrb.Spatial2DView(
                        name="Image with coords from estimated pose",
                        origin="world/est_camera/image",
                        contents=["$origin/**", "/world/all_points"],
                    ),
                ]
            ),
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(name="Scene coordinate map", origin="world/camera/scr_image"),
                    rrb.Spatial2DView(name="Uncertainty map", origin="world/camera/unc"),
                ]
            ),
        ]
    )
    bottom = rrb.TimeSeriesView()
    blueprint = rrb.Blueprint(
        rrb.Vertical(contents=[rrb.Horizontal(contents=[top_left, top_right]), bottom], row_shares=[4, 1])
    )
    rr.send_blueprint(blueprint)


def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Visualize predicted scene coordinates and estimated camera poses.",
    )

    parser.add_argument("--config", nargs="+", help="Config file(s) to load.")

    config_dict = yoco.load_config_from_args(parser, search_paths=[".", "./configs"])

    defaults_dict = {}
    # we overwrite config in order: sst -> reg -> root (root is the last one, so it will overwrite all previous)
    if "sst" in config_dict:  # mapping dataset by default
        defaults_dict = yoco.load_config(config_dict["sst"], defaults_dict)
    if "reg" in config_dict:  # standard registration options by default
        defaults_dict = yoco.load_config(config_dict["reg"], defaults_dict)
    config_dict = yoco.load_config(config_dict, defaults_dict)

    config = dacite.from_dict(
        VisualizationConfig,
        config_dict,
        config=dacite.Config(cast=[pathlib.Path]),  # type: ignore
    )

    visualizer = LocalizationVisualizer(config)
    visualizer.visualize()


if __name__ == "__main__":
    main()
