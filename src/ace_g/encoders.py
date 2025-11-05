# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Image encoders for camera localization."""

from __future__ import annotations

import logging
import pathlib
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from ace_g import utils

_logger = logging.getLogger(__name__)


class EncoderConfig(TypedDict):
    """Configuration for an encoder network."""

    obj_type: str
    """Type of the encoder; has to be resolvable by ace_g.utils.str_to_object (fully-qualified python name will work
    from anywhere "ace_g.encoders.DINOv2Encoder")"""
    kwargs: dict
    """Keyword arguments passed to the encoder constructor."""
    path: pathlib.Path | None
    """Path to the encoder weights; None if no weights are available."""


class Encoder(nn.Module):
    """Encoder base class."""

    dim_out: int
    """Number of output channels of the encoder."""
    subsample_factor: int
    """Spatial subsampling factor of the encoder."""
    supports_rgb: bool
    """Whether the encoder supports RGB images."""


class DINOv2Encoder(Encoder):
    """DINOv2 encoder, used to extract features from the input images.

    The number of output channels is not modified and depends on the chosen model.

    See https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#model-details.

    Support model names:
        dinov2_vits14
        dinov2_vits14_reg
        dinov2_vitb14
        dinov2_vitb14_reg
        dinov2_vitl14
        dinov2_vitl14_reg
        dinov2_vitg14
        dinov2_vitg14_reg
    """

    supports_rgb = True

    def __init__(self, model_name: str = "dinov2_vits14_reg") -> None:
        """Initialize the DINOv2 encoder.

        Args:
            model_name: Name of the model to load using torch.hub.load. See class docstring for supported models.
        """
        super(DINOv2Encoder, self).__init__()

        hub = "facebookresearch/dinov2"

        # Load the DINOv2 model.
        self.dinov2 = torch.hub.load(
            hub, model_name, verbose=True, trust_repo=True, source="github", skip_validation=True
        )
        self.subsample_factor = self.dinov2.patch_embed.patch_size[0]  # type: ignore
        self.dim_out = self.dinov2.embed_dim  # type: ignore

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute features for images.

        The image is cropped to a size that is a multiple of the patch size (i.e., right and bottom might be cropped).
        Spatial dimensions are reduced by the patch size (e.g., for patches of size 14, H' = H // 14).
        See subsample_factor.

        For grayscsale images the input is broadcasted to 3 channels.

        Args:
            images: Input images. Shape (..., 3 or 1, H, W).

        Returns:
            Features for image patches. Shape (..., dim_feat, H', W').
        """
        leading_dims = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.view(-1, c, h, w)
        if c == 1:  # for grayscale images, pass the same channel three times
            images = images.expand(-1, 3, -1, -1)
        images = images[
            ...,
            : self.subsample_factor * (h // self.subsample_factor),
            : self.subsample_factor * (w // self.subsample_factor),
        ]
        features = self.dinov2.forward_features(images)  # type: ignore
        patch_features = (
            features["x_norm_patchtokens"]
            .permute(0, 2, 1)
            .reshape(*leading_dims, -1, h // self.subsample_factor, w // self.subsample_factor)
        )
        return patch_features


class FCNEncoder(Encoder):
    """FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    supports_rgb = False  # Whether the encoder supports RGB images.

    def __init__(self, dim_out: int = 512) -> None:
        """Initialize the FCN encoder."""
        super().__init__()

        self.dim_out = dim_out

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.dim_out, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.dim_out, 1, 1, 0)

        # con2, conv3, conv4 all downsample by a factor of 2, so the subsample factor is 8.
        self.subsample_factor = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features for images.

        Args:
            images: Gray-scale image. Shape (..., 1, H, W).

        Returns:
            Features for image patches. Shape (..., 512, H', W').
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res += x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


def create_encoder(encoder_config: EncoderConfig) -> Encoder:
    """Create an encoder from a configuration."""
    path = encoder_config.get("path", None)
    if path is not None:
        encoder_state_dict = torch.load(path, map_location="cpu", weights_only=False)
        encoder = encoder_from_state_dict(encoder_state_dict)
        _logger.info(f"Loaded pretrained encoder from: {path}")
    else:
        obj_type = encoder_config["obj_type"]
        kwargs = encoder_config.get("kwargs", {})
        encoder_cls = utils.str_to_object(obj_type)
        encoder = encoder_cls(**kwargs)  # type: ignore
    return encoder


def encoder_from_state_dict(state_dict: dict) -> Encoder:
    """Extract the encoder type and configuration from a state dictionary.

    Args:
        state_dict: State dictionary of the encoder.

    Returns: The encoder object with weights loaded from the state dictionary.
    """
    _logger.info("Creating encoder from state dictionary.")

    # TODO support DINOv2Encoder

    # Number of output channels of the last encoder layer.
    num_encoder_features = state_dict["res2_conv3.weight"].shape[0]

    encoder = FCNEncoder(dim_out=num_encoder_features)

    # Load encoder weights.
    encoder.load_state_dict(state_dict)

    return encoder
