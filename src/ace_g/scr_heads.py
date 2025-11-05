# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Heads for scene coordinate regression.

This module defines a generic SCR head and two concrete implementations: a TransformerHead (used in ACE-G)and the
MLPHead (used in ACE).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
from typing import Callable, Concatenate, Literal, Optional, ParamSpec

import dacite
import einops
import torch
import torch.nn.functional as F
from torch import nn

from ace_g import layers, utils

_logger = logging.getLogger(__name__)

P = ParamSpec("P")


def config_to_state_dict(
    init_fn: Callable[Concatenate[torch.nn.Module, P], None],
) -> Callable[Concatenate[torch.nn.Module, P], None]:
    """Decorator that saves the first argument (assumed to be a config object) to the state dict of a module.

    This allows to reconstruct the object from just the state dict (see create_head function below).
    """

    def get_extra_state(self: torch.nn.Module) -> dict:
        return self.init_args

    def wrapper(self: torch.nn.Module, config) -> None:
        if not hasattr(self, "init_args"):  # So that multiple calls to the wrapper do not overwrite the init_args.
            self.init_args = {}
            self.init_args["_head_config_dict"] = dataclasses.asdict(config)
            self.init_args["_head_config_type"] = f"{config.__class__.__module__}.{config.__class__.__qualname__}"

            setattr(self.__class__, "get_extra_state", get_extra_state)

        return init_fn(self, config)

    return wrapper


class SCRHead(nn.Module):
    """Abstract scene coordinate regression head.

    Provides common functionality for scene coordinate regression heads.
    """

    dim_out: int  # Number of output channels of the network prior to post-processing.

    def __init__(
        self,
        mean: Optional[torch.Tensor],
        use_homogeneous: bool,
        homogeneous_min_scale: float,
        homogeneous_max_scale: float,
        use_uncertainty: bool = False,
    ):
        """Initialize the scene coordinate regression head.

        Args:
            mean:
                mean: Mean scene coordinate stored as part of the network, subtracted from output. 0 if None. Shape (3,).
                use_homogeneous: If True, the network predicts 4D homogeneous coordinates, otherwise 3D coordinates.
                homogeneous_min_scale: Minimum scale for homogeneous coordinates. Only used if use_homogeneous is True.
                homogeneous_max_scale: Maximum scale for homogeneous coordinates. Only used if use_homogeneous is True.
                use_uncertainty: If True, the network predicts an additional channel for uncertainty.
        """
        super(SCRHead, self).__init__()
        self.use_homogeneous = use_homogeneous
        self.use_uncertainty = use_uncertainty
        self.dim_out = 3

        if self.use_homogeneous:
            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1.0 / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1.0 - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1.0 / self.min_scale)
            self.dim_out += 1

        if self.use_uncertainty:
            self.dim_out += 1

        if mean is None:
            self.register_buffer("mean", torch.tensor(0, dtype=torch.float))
        else:
            # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
            self.register_buffer("mean", mean.clone().detach())

    def outputs_to_scene_coordinates(
        self, outputs: torch.Tensor, means: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert the network output to 3D scene coordinates.

        Args:
            outputs: Network outputs. Shape (..., dim_out, H', W').
            means: Mean scene coordinate. Either shape (3,) or any shape broadcastable to (..., 3, H', W').

        Returns:
            Processed scene coordinates. Shape (..., 3, H', W').
            Uncertanties for each scene coordinate if enabled, otherwise None. Shape (..., 1, H', W').
        """
        means = self.mean if means is None else means
        assert means is not None, "No mean scene coordinate provided."

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(outputs[..., 3, :, :].unsqueeze(-3), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            scene_coordinates = outputs[..., :3, :, :] / h_slice
        else:
            scene_coordinates = outputs[..., :3, :, :]

        with torch.autocast("cuda", enabled=False):
            if means.shape == (3,):
                means = means.view(3, 1, 1)

            # Add the mean to the predicted coordinates.
            scene_coordinates = scene_coordinates + means

        uncertainties = None
        if self.use_uncertainty:
            uncertainties = torch.exp(outputs[..., 4, :, :].unsqueeze(-3))

        return scene_coordinates, uncertainties


class TransformerHead(SCRHead):
    """Transformer for map-relative scene coordinate regression.

    Given a set of map embeddings and a patch embedding, the network predicts the 3D scene coordinate of the patch.
    """

    @dataclasses.dataclass(kw_only=True)
    class Config:
        """Configuration for a TransformerHead.

        Attributes:
            obj_type: Type of the head. Used to resolve Config type.
            dim_in: int | None = None  # allow None to allow setting from encoder
            num_blocks: Number of transformer blocks to use.
            num_attention_heads: Number of attention heads used in each block.
            dim_emb:
                Dimension of the transformer embeddings (same for query, key, value). Must be divisible by num_heads.
            mean: See SCRHead.__init__.
            use_homogeneous: See SCRHead.__init__.
            use_uncertainty: See SCRHead.__init__.
            homogeneous_min_scale: See SCRHead.__init__.
            homogeneous_max_scale: See SCRHead.__init__.
            mlp_ratio: Ratio of the hidden layer size to the input size for all MLPs.
            proj_drop: Dropout rate for all but final MLP and final projection of the attention.
            attn_drop: Dropout rate for the attention weights.
            attention_type:
                Type of attention to use.
                "ca": Cross-attention only.
                "sa": Self-attention only (concat patch and map embeddings).
                "alt-reg": Alternating self-attention between patch + register embeddings and cross-attention to map.
                "alt-patch": Alternating self-attention between all image patches and cross-attention to map.
            dim_map_emb: Dimension of the map embeddings. If None, defaults to dim_emb.
            use_final_layer_norm: If True, use layer norm in the final layer.
            final_mlp_layers: Number of layers in the final MLP.
            num_map_code_self_attention_blocks: Number of blocks in the map code encoder. No encoder if 0.
        """

        obj_type: Literal["ace_g.scr_heads.TransformerHead"] = "ace_g.scr_heads.TransformerHead"
        dim_in: int | None = None  # allow None to allow setting from encoder
        num_blocks: int = 6
        num_attention_heads: int = 8
        dim_emb: int = 512
        mean: torch.Tensor | None = None
        use_homogeneous: bool = True
        use_uncertainty: bool = False
        homogeneous_min_scale: float = 0.01
        homogeneous_max_scale: float = 4.0
        mlp_ratio: float = 4.0
        proj_drop: float = 0.0
        attn_drop: float = 0.0
        dim_map_emb: int | None = None
        use_final_layer_norm: bool = True
        final_mlp_layers: int = 2
        final_mlp_ratio: float = 1.0

    dim_map_emb: int  # Dimension of the map embeddings.

    @config_to_state_dict
    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the transformer head."""
        self.config = config
        super(TransformerHead, self).__init__(
            mean=self.config.mean,
            use_homogeneous=self.config.use_homogeneous,
            homogeneous_min_scale=self.config.homogeneous_min_scale,
            homogeneous_max_scale=self.config.homogeneous_max_scale,
            use_uncertainty=self.config.use_uncertainty,
        )

        assert self.config.dim_emb % self.config.num_attention_heads == 0, (
            "dim_emb must be divisible by num_attention_heads."
        )
        assert self.config.dim_in is not None, "dim_in must be provided."

        self.dim_map_emb = self.config.dim_emb if self.config.dim_map_emb is None else self.config.dim_map_emb

        self.in_mlp = layers.Mlp(
            in_features=self.config.dim_in,
            out_features=self.config.dim_emb,
            hidden_features=int(self.config.dim_in * self.config.mlp_ratio),
            norm_layer=torch.nn.LayerNorm,
            drop=self.config.proj_drop,
        )

        self.transformer_blocks = torch.nn.ModuleList()

        for _ in range(self.config.num_blocks):
            kv_dim = self.dim_map_emb

            self.transformer_blocks.append(
                layers.TransformerBlock(
                    dim=self.config.dim_emb,
                    num_heads=self.config.num_attention_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    qkv_bias=True,
                    proj_drop=self.config.proj_drop,
                    attn_drop=self.config.attn_drop,
                    kv_dim=kv_dim,
                )
            )

        self.out_mlp = layers.Mlp(
            in_features=self.config.dim_emb,
            out_features=self.dim_out,
            hidden_features=int(self.config.dim_emb * self.config.final_mlp_ratio),
            norm_layer=torch.nn.LayerNorm if self.config.use_final_layer_norm else None,
            num_layers=self.config.final_mlp_layers,
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        map_embeddings: torch.Tensor,
        means: Optional[torch.Tensor] = None,
        num_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scene coordinates for a patch.

        Note that no self attention between the patches of an image is performed. Hence H' and W' act like per-map batch
        dimensions. I.e., there is only one query with multiple keys and values (coming from the map embedding).

        Args:
            patch_embeddings: Embeddings for image patches. Shape (..., dim_patch_emb, H', W').
            map_embeddings: Embeddings for the map. Shape (..., num_map_embs, dim_map_emb).
            means: See SCRHead.outputs_to_scene_coordinates.
            num_patches:
                Number of valid patches per batch. Used to mask out padding patches. Shape (...).
                This is after flattening the spatial dimensions.

        Returns:
            Scene coordinates for the patches. Shape (..., 3, H', W').
        """
        h, w = patch_embeddings.shape[-2:]

        input_embeddings = einops.rearrange(patch_embeddings, "... dim_patch_emb h w -> ... (h w) 1 dim_patch_emb")
        queries = self.in_mlp(input_embeddings)
        map_embeddings = einops.rearrange(
            map_embeddings, "... num_map_embs dim_map_emb -> ... 1 num_map_embs dim_map_emb"
        )

        for _, block in enumerate(self.transformer_blocks):
            queries = block(queries, map_embeddings, map_embeddings, q_seqlens=num_patches)

        outputs = self.out_mlp(queries)
        outputs = einops.rearrange(outputs, "... (h w) 1 dim_out -> ... dim_out h w", w=w)

        return self.outputs_to_scene_coordinates(outputs, means)


class MLPHead(SCRHead):
    """MLP network predicting per-pixel scene coordinates given a feature vector.

    All layers are 1x1 convolutions.
    """

    @dataclasses.dataclass(kw_only=True)
    class Config:
        """Configuration for an MLPHead.

        Attributes:
            obj_type: Type of the head. Used to resolve Config type.
            dim_in: int | None = None  # allow None to allow setting from encoder
            num_blocks: Number of residual blocks to use. Each block has 3 convolutional layers.
            mean: See SCRHead.__init__.
            use_homogeneous: See SCRHead.__init__.
            homogeneous_min_scale: See SCRHead.__init__.
            homogeneous_max_scale: See SCRHead.__init__.
            use_uncertainty: See SCRHead.__init__.
        """

        obj_type: Literal["ace_g.scr_heads.MLPHead"] = "ace_g.scr_heads.MLPHead"
        dim_in: int | None = None  # allow None to allow setting from encoder
        num_blocks: int = 1
        mean: torch.Tensor | None = None
        use_homogeneous: bool = True
        homogeneous_min_scale: float = 0.01
        homogeneous_max_scale: float = 4.0
        use_uncertainty: bool = False

    dim_map_emb = None  # MLPHead does not use map embeddings.

    @config_to_state_dict
    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initalize the MLPHead.

        Args:
            config: Configuration for the MLPHead.
        """
        self.config = config
        assert self.config.dim_in is not None, "dim_in must be provided."

        super(MLPHead, self).__init__(
            mean=self.config.mean,
            use_homogeneous=self.config.use_homogeneous,
            homogeneous_min_scale=self.config.homogeneous_min_scale,
            homogeneous_max_scale=self.config.homogeneous_max_scale,
            use_uncertainty=self.config.use_uncertainty,
        )

        self.use_homogeneous = self.config.use_homogeneous
        self.dim_in = self.config.dim_in  # Number of encoder features.
        self.dim_hidden = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = (
            nn.Identity() if self.dim_in == self.dim_hidden else nn.Conv2d(self.dim_in, self.dim_hidden, 1, 1, 0)
        )

        self.res3_conv1 = nn.Conv2d(self.dim_in, self.dim_hidden, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0)

        self.res_blocks = []

        for block in range(self.config.num_blocks):
            self.res_blocks.append(
                (
                    nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0),
                    nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0),
                    nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0),
                )
            )

            # TODO why do we need to use super here?
            super(MLPHead, self).add_module(str(block) + "c0", self.res_blocks[block][0])
            super(MLPHead, self).add_module(str(block) + "c1", self.res_blocks[block][1])
            super(MLPHead, self).add_module(str(block) + "c2", self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.dim_hidden, self.dim_hidden, 1, 1, 0)
        self.fc3 = nn.Conv2d(self.dim_hidden, self.dim_out, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        x = F.relu(self.fc1(res))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return self.outputs_to_scene_coordinates(x)


HeadConfig = MLPHead.Config | TransformerHead.Config | pathlib.Path


def create_head(head_config: HeadConfig) -> SCRHead:
    """Create a head from a head configuration."""
    state_dict = None
    if isinstance(head_config, pathlib.Path):
        state_dict = torch.load(head_config, map_location="cpu", weights_only=False)
        extra_state = state_dict.pop("_extra_state")
        head_config_dict = extra_state.pop("_head_config_dict")
        head_config_type_str = extra_state.pop("_head_config_type")
        config = dacite.from_dict(utils.str_to_object(head_config_type_str), head_config_dict)  # type: ignore
    else:
        config = head_config

    # create the model according to the config
    head_cls = utils.str_to_object(config.obj_type)
    head = head_cls(config)  # type: ignore

    # try to load the state dict into the model (if config matches, this should work)
    if state_dict is not None:
        head.load_state_dict(state_dict)
        _logger.info(f"Loaded pretrained head from: {head_config}")

    return head
