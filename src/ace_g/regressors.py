# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Generic scene coordinate regressor class based on an encoder and a head."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import torch
from torch import nn

from ace_g import encoders, scr_heads


_logger = logging.getLogger(__name__)


class Regressor(nn.Module):
    """Architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled as specified by the subsample_factor property.
    """

    encoder: encoders.Encoder
    head: scr_heads.SCRHead

    def __init__(self, encoder: encoders.Encoder, head: scr_heads.SCRHead) -> None:
        """Constructor.

        Args:
            encoder: Encoder network.
            head: Head network.
        """
        super(Regressor, self).__init__()

        self.encoder = encoder
        self.head = head

    @property
    def feature_dim(self) -> int:
        """Dimension of the feature vector output by the encoder."""
        return self.encoder.dim_out

    @property
    def subsample_factor(self) -> int:
        """Spatial subsampling factor of the encoder+head combination."""
        return self.encoder.subsample_factor

    def load_encoder(self, encoder_dict_file: pathlib.Path) -> None:
        """Load weights into the encoder network."""
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get image features from the encoder."""
        return self.encoder(images)

    def get_scene_coordinates(
        self,
        patch_embeddings: torch.Tensor,
        map_embeddings: Optional[torch.Tensor] = None,
        means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scene coordinates for a set of patch embeddings.

        Args:
            patch_embeddings: Embeddings for image patches. Shape (..., dim_patch_emb, H', W').
            map_embeddings: Embeddings for the map. Shape (..., num_map_embs, dim_map_emb).
            means: See SCRHead.outputs_to_scene_coordinates.

        Returns:
            See SCRHead.outputs_to_scene_coordinates.
        """
        if map_embeddings is None:
            return self.head(patch_embeddings)
        return self.head(patch_embeddings, map_embeddings, means=means)

    def forward(
        self,
        images: torch.Tensor,
        map_embeddings: Optional[torch.Tensor] = None,
        means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scene coordinates for an image.

        Args:
            images: The input image. Shape (..., 1 or 3, H, W).
            map_embeddings:
                Embeddings for the map. Shape (..., num_map_embs, dim_map_emb).
            means:
                Mean scene coordinate for each patch. Unspecified dims must broadcast to patch embedding shape.
                See SCRHead.outputs_to_scene_coordinates.

        Returns:
            See SCRHead.outputs_to_scene_coordinates.
        """
        patch_embeddings = self.get_features(images)
        return self.get_scene_coordinates(
            patch_embeddings, map_embeddings, means
        )

    @property
    def uses_map_embeddings(self) -> bool:
        """Whether the head network uses map embeddings."""
        return self.head.dim_map_emb is not None
