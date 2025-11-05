# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Utility functions for visualization."""

import logging
import pathlib
from typing import Any

import matplotlib
import numpy as np
import rerun as rr

_logger = logging.getLogger(__name__)


def rr_init(application_id: str, rrd_path: pathlib.Path | None = None, spawn: bool = False) -> None:
    """Initialize Rerun either via connect or via save."""
    rr.init(application_id, spawn=spawn)
    if rrd_path is not None:
        rr.save(rrd_path)
    elif not spawn:
        rr.connect_grpc()


def rr_log_points_with_scalar(
    entity_path: str,
    points: np.ndarray,
    scalar: np.ndarray,
    colormap: str = "cool",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,  # noqa
) -> None:
    """Log points to Rerun colored based on an additional scalar value."""
    cmap = matplotlib.colormaps[colormap]
    vmin = scalar.min() if vmin is None else vmin
    vmax = scalar.max() if vmax is None else vmax

    assert isinstance(vmin, float) and isinstance(vmax, float)
    rgb_uncertainties = cmap((scalar - vmin) / (vmax - vmin))[..., :3]

    rr.log(entity_path, rr.Points3D(points, colors=rgb_uncertainties, **kwargs))
