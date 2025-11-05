# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Script to visualize a dataset."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
from typing import cast

import dacite
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tqdm
import yoco

from ace_g import datasets, vis_utils


@dataclasses.dataclass(kw_only=True)
class DatasetVisualizationConfig(datasets.CamLocDataset.Config):
    """Configuration for visualizing a dataset."""

    rrd_path: pathlib.Path | None = None
    """If set, save Rerun's rrd to this path; otherwise, use connect."""
    rerun_spawn: bool = False
    """If True, spawn a new Rerun viewer, otherwise use connect."""


def _setup_rr_dataset_blueprint() -> None:
    blueprint = rrb.Blueprint(
        rrb.Grid(
            contents=[
                rrb.Spatial3DView(
                    name="3D View",
                ),
                rrb.Spatial2DView(
                    origin="world/camera/image",
                ),
            ],
        )
    )
    rr.send_blueprint(blueprint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Register images to a map given by an SCR network.",
    )
    parser.add_argument("--config", nargs="+", help="Config file(s) to load.", default=[])

    config_dict = yoco.load_config_from_args(parser, search_paths=[".", "./configs"])
    config = dacite.from_dict(
        DatasetVisualizationConfig,
        config_dict,
        config=dacite.Config(cast=[pathlib.Path]),
    )  # type: ignore

    assert config.subsample_factor is not None, "Subsample factor is required"

    vis_utils.rr_init("ace_g_dataset", config.rrd_path, config.rerun_spawn)
    _setup_rr_dataset_blueprint()

    my_dataset = datasets.CamLocDataset(config)

    rr.log("mean", rr.Points3D(my_dataset.mean_cam_center), static=True)

    min_coords = max_coords = None
    for item in tqdm.tqdm(my_dataset):  # type: ignore
        item = cast(datasets.CamLocDataset.Item, item)
        image = item.image
        intrinsics = item.intrinsics
        c2augc = item.c2augc
        w2c = item.w2c
        coords = item.coords
        idx = item.idx
        assert isinstance(idx, int)

        rr.set_time("index", sequence=idx)
        rr.set_time("percentage", duration=idx / len(my_dataset))
        uint8_image = ((image.permute(1, 2, 0) * 0.25 + 0.4) * 255).byte().cpu().numpy()
        rr.log("world/camera/image", rr.Image(uint8_image).compress())
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=intrinsics))
        w2c = c2augc @ w2c
        c2w = w2c.inverse()
        # make sure rotation is a valid rotation matrix
        assert np.isclose(np.linalg.det(w2c[:3, :3]), 1.0)
        rr.log(
            "world/camera",
            rr.Transform3D(translation=w2c[:3, 3], mat3x3=w2c[:3, :3], relation=rr.TransformRelation.ChildFromParent),
        )
        rr.log("world/camera_center", rr.Points3D(c2w[:3, 3].unsqueeze(0)))

        # get RGB color for scene coordinates
        offset = config.subsample_factor // 2
        image_patches_hwc = uint8_image[offset :: config.subsample_factor, offset :: config.subsample_factor]

        height, width = image_patches_hwc.shape[:2]
        num_patches = height * width
        cropped_coords = coords[:, :height, :width]  # skip incomplete patches

        rr.log("world/camera/rgb_image", rr.Image(image_patches_hwc))
        world_coords = cropped_coords.reshape(3, num_patches).permute(1, 0)

        if image_patches_hwc.shape[-1] == 1:
            image_patches_hwc = np.repeat(image_patches_hwc, 3, axis=-1)

        mask = world_coords.any(dim=-1)
        world_coords = world_coords[mask]
        world_coords_color = image_patches_hwc.reshape(num_patches, -1)[mask]
        rr.log("world/coords_1", rr.Points3D(world_coords, colors=world_coords_color))
