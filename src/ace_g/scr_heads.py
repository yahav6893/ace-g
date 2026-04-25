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

    def set_extra_state(self: torch.nn.Module, state: dict) -> None:
        # Enables loading nested modules that also store init config via this decorator.
        self.init_args = state

    def wrapper(self: torch.nn.Module, config) -> None:
        if not hasattr(self, "init_args"):  # So that multiple calls to the wrapper do not overwrite the init_args.
            self.init_args = {}
            self.init_args["_head_config_dict"] = dataclasses.asdict(config)
            self.init_args["_head_config_type"] = f"{config.__class__.__module__}.{config.__class__.__qualname__}"

            setattr(self.__class__, "get_extra_state", get_extra_state)
            setattr(self.__class__, "set_extra_state", set_extra_state)

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

class FusionHead(SCRHead):
    """Cross-attention fusion head for concatenated multi-encoder features.

    Expected input:
        patch_embeddings: Tensor of shape (..., N*C, H', W') where N is the number of encoders.

    Fusion is performed *independently per patch* (no spatial attention):
        - Query: main encoder features -> MLP
        - Keys/Values: all encoder features -> shared linear (K = V)
        - Single-head attention over the encoder axis (sequence length = N)

    The fused per-patch feature has shape (..., C, H', W') and is then passed to an internal MLPHead.
    """

    @dataclasses.dataclass(kw_only=True)
    class Config:
        obj_type: Literal["ace_g.scr_heads.FusionHead"] = "ace_g.scr_heads.FusionHead"
        dim_in: int | None = None  # allow None to be set from encoder
        num_encoders: int
        main_index: int = 0
        q_mlp_ratio: float = 4.0
        proj_drop: float = 0.0
        attn_drop: float = 0.0
        mlp_num_blocks: int = 1

        mean: torch.Tensor | None = None
        use_homogeneous: bool = True
        homogeneous_min_scale: float = 0.01
        homogeneous_max_scale: float = 4.0
        use_uncertainty: bool = False

    dim_map_emb = None

    @config_to_state_dict
    def __init__(self, config: Config) -> None:
        self.config = config
        assert self.config.dim_in is not None, "dim_in must be provided (typically set from encoder.dim_out)."
        assert self.config.num_encoders >= 1, "num_encoders must be >= 1."
        assert 0 <= self.config.main_index < self.config.num_encoders, "main_index out of range."
        assert (
            self.config.dim_in % self.config.num_encoders == 0
        ), f"dim_in ({self.config.dim_in}) must be divisible by num_encoders ({self.config.num_encoders})."

        self.dim_single = self.config.dim_in // self.config.num_encoders

        super(FusionHead, self).__init__(
            mean=self.config.mean,
            use_homogeneous=self.config.use_homogeneous,
            homogeneous_min_scale=self.config.homogeneous_min_scale,
            homogeneous_max_scale=self.config.homogeneous_max_scale,
            use_uncertainty=self.config.use_uncertainty,
        )

        self.q_mlp = layers.Mlp(
            in_features=self.dim_single,
            out_features=self.dim_single,
            hidden_features=int(self.dim_single * self.config.q_mlp_ratio),
            norm_layer=torch.nn.LayerNorm,
            drop=self.config.proj_drop,
        )

        # Shared projection K = V
        self.kv_proj = nn.Linear(self.dim_single, self.dim_single, bias=True)
        self.attn_drop = nn.Dropout(self.config.attn_drop) if self.config.attn_drop > 0 else nn.Identity()

        # Internal ACE MLP head after fusion
        mlp_cfg = MLPHead.Config(
            dim_in=self.dim_single,
            num_blocks=self.config.mlp_num_blocks,
            mean=self.config.mean,
            use_homogeneous=self.config.use_homogeneous,
            homogeneous_min_scale=self.config.homogeneous_min_scale,
            homogeneous_max_scale=self.config.homogeneous_max_scale,
            use_uncertainty=self.config.use_uncertainty,
        )
        self.mlp_head = MLPHead(mlp_cfg)

    def forward(self, patch_embeddings: torch.Tensor):
        leading_dims = patch_embeddings.shape[:-3]
        dim_in, h, w = patch_embeddings.shape[-3:]
        x = patch_embeddings.view(-1, dim_in, h, w)
        b = x.shape[0]

        n = self.config.num_encoders
        if dim_in != n * self.dim_single:
            raise ValueError(
                f"FusionHead got dim_in={dim_in}, but config expects num_encoders*dim_single={n}*{self.dim_single}."
            )

        # (B, N, C, h, w) -> (B*h*w, N, C)
        tokens = x.view(b, n, self.dim_single, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, n, self.dim_single)

        q = tokens[:, self.config.main_index, :]
        q = self.q_mlp(q)

        kv = self.kv_proj(tokens)
        logits = (kv * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.dim_single)

        weights = torch.softmax(logits.float(), dim=1).to(kv.dtype)
        weights = self.attn_drop(weights)

        fused = (weights.unsqueeze(-1) * kv).sum(dim=1)

        fused = fused.reshape(b, h, w, self.dim_single).permute(0, 3, 1, 2)
        fused = fused.view(*leading_dims, self.dim_single, h, w)

        return self.mlp_head(fused)

class LateFusionCoordHead(SCRHead):
    """Late fusion in target (coordinates) domain with optional pretrained expert heads.

    Input: patch_embeddings (..., K*C, H', W') where K=num_experts.

    Each expert i has its own head -> predicts scene coords (and optional uncertainty).
    Gate predicts weights over experts (per patch or global).
    Final coords = sum_i w_i * coords_i.

    New features:
      - expert_head_paths: optional list of paths (length K) to pretrained heads.
      - If a path is given, load that head (must be an MLPHead).
      - Optionally freeze loaded expert heads (default True).
    """

    @dataclasses.dataclass(kw_only=True)
    class Config:
        obj_type: Literal["ace_g.scr_heads.LateFusionCoordHead"] = "ace_g.scr_heads.LateFusionCoordHead"
        dim_in: int | None = None

        # Experts
        num_experts: int
        expert_heads: list[MLPHead.Config] | None = None   # per-expert configs (length K), optional
        expert_head: MLPHead.Config | None = None          # template replicated K times if expert_heads is None

        # Optional pretrained expert loading (paths align with encoder order)
        expert_head_paths: list[pathlib.Path | None] | None = None
        freeze_loaded_experts: bool = True
        force_shared_mean: bool = True   # make all loaded experts share expert[0].mean (safety)

        # New hyperparameters
        l2_reg_weight: float = 0.0
        unfreeze_experts_after_iterations: int = -1
        gate_learning_rate_factor: float = 1.0
        expert_learning_rate_factor: float = 1.0

        # Gating
        gate_input: Literal["main", "concat"] = "concat"
        main_index: int = 0
        gate_hidden_ratio: float = 1.0
        gate_dropout: float = 0.1
        weights_per_patch: bool = True
        temperature: float = 1.0

        # SCRHead base knobs (kept for compatibility; experts do the actual SCR post-processing)
        mean: torch.Tensor | None = None
        use_homogeneous: bool = True
        homogeneous_min_scale: float = 0.01
        homogeneous_max_scale: float = 4.0
        use_uncertainty: bool = False

    dim_map_emb = None

    @config_to_state_dict
    def __init__(self, config: Config) -> None:
        self.config = config
        assert self.config.dim_in is not None, "dim_in must be set (typically encoder.dim_out)."
        assert self.config.num_experts >= 1
        assert 0 <= self.config.main_index < self.config.num_experts
        assert self.config.dim_in % self.config.num_experts == 0, "dim_in must be divisible by num_experts."

        self.c = self.config.dim_in // self.config.num_experts

        super().__init__(
            mean=self.config.mean,
            use_homogeneous=self.config.use_homogeneous,
            homogeneous_min_scale=self.config.homogeneous_min_scale,
            homogeneous_max_scale=self.config.homogeneous_max_scale,
            use_uncertainty=self.config.use_uncertainty,
        )

        # -------------------------
        # Build expert heads (default: K copies of an MLPHead template)
        # -------------------------
        if self.config.expert_heads is not None:
            assert len(self.config.expert_heads) == self.config.num_experts, \
                "expert_heads length must equal num_experts"
            head_cfgs = self.config.expert_heads
        else:
            template = self.config.expert_head
            if template is None:
                template = MLPHead.Config(
                    dim_in=self.c,
                    num_blocks=1,
                    mean=self.config.mean,
                    use_homogeneous=self.config.use_homogeneous,
                    homogeneous_min_scale=self.config.homogeneous_min_scale,
                    homogeneous_max_scale=self.config.homogeneous_max_scale,
                    use_uncertainty=self.config.use_uncertainty,
                )
            head_cfgs = [dataclasses.replace(template, dim_in=self.c) for _ in range(self.config.num_experts)]

        self.expert_heads = nn.ModuleList([MLPHead(cfg) for cfg in head_cfgs])

        # -------------------------
        # Gating MLP (1x1 conv MLP): Linear -> ReLU -> Dropout -> Linear
        # -------------------------
        gate_in = self.c if self.config.gate_input == "main" else self.config.num_experts * self.c
        gate_hidden = max(1, int(gate_in * float(self.config.gate_hidden_ratio)))

        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, gate_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(self.config.gate_dropout)),
            nn.Conv2d(gate_hidden, self.config.num_experts, kernel_size=1, bias=True),
        )

        # -------------------------
        # Optional: load pretrained expert heads + freeze them
        # -------------------------
        if self.config.expert_head_paths is not None:
            paths = list(self.config.expert_head_paths)
            assert len(paths) == self.config.num_experts, \
                "expert_head_paths must have length == num_experts (use None for missing entries)."

            for i, p in enumerate(paths):
                if p is None:
                    continue

                # Use create_head to reconstruct exactly as saved (handles _extra_state properly).
                loaded = create_head(pathlib.Path(p))

                if not isinstance(loaded, MLPHead):
                    raise TypeError(
                        f"expert_head_paths[{i}] must point to an MLPHead checkpoint, got {type(loaded)}"
                    )

                # Sanity: input dim must match this expert's C
                if getattr(loaded.config, "dim_in", None) is not None and loaded.config.dim_in != self.c:
                    raise ValueError(
                        f"Loaded expert[{i}] dim_in={loaded.config.dim_in} but expected {self.c} "
                        f"(dim_in={self.config.dim_in}, num_experts={self.config.num_experts})."
                    )

                # Replace the randomly initialized expert with the loaded one.
                self.expert_heads[i] = loaded

                # Freeze loaded expert if requested
                if self.config.freeze_loaded_experts:
                    for prm in self.expert_heads[i].parameters():
                        prm.requires_grad_(False)

            # Optional safety: force shared mean across experts (prevents mixing incompatible coordinate frames)
            if self.config.force_shared_mean and len(self.expert_heads) >= 2:
                ref = self.expert_heads[0].mean.detach().clone()
                for i in range(1, self.config.num_experts):
                    if not torch.allclose(self.expert_heads[i].mean, ref, atol=1e-6, rtol=0.0):
                        _logger.warning(
                            f"LateFusionCoordHead: expert[{i}] mean differs from expert[0]; "
                            "forcing expert[i].mean = expert[0].mean for safe mixing."
                        )
                        self.expert_heads[i].mean.copy_(ref)

    def forward(self, patch_embeddings: torch.Tensor):
        leading_dims = patch_embeddings.shape[:-3]
        dim_in, h, w = patch_embeddings.shape[-3:]

        x = patch_embeddings.view(-1, dim_in, h, w)  # (B, K*C, h, w)
        b = x.shape[0]

        k = self.config.num_experts
        c = self.c
        if dim_in != k * c:
            raise ValueError(f"LateFusionCoordHead: expected dim_in={k*c}, got {dim_in}")

        # Split expert features: (B, K, C, h, w)
        feats = x.view(b, k, c, h, w)

        # --- expert predictions ---
        preds = []
        uncs = []
        any_unc = False
        for i in range(k):
            yi, ui = self.expert_heads[i](feats[:, i])  # yi: (B, 3, h, w), ui: (B,1,h,w) or None
            preds.append(yi)
            uncs.append(ui)
            any_unc = any_unc or (ui is not None)

        # Stack -> (B, K, D, h, w)
        y = torch.stack(preds, dim=1)
        d = y.shape[2]

        # --- gating input ---
        if self.config.gate_input == "main":
            g_in = feats[:, self.config.main_index]  # (B, C, h, w)
        else:
            g_in = x                                  # (B, K*C, h, w)

        logits = self.gate(g_in)  # (B, K, h, w)

        # --- weights ---
        T = float(self.config.temperature)
        if self.config.weights_per_patch:
            wts = torch.softmax((logits / T).float(), dim=1)  # keep float32 for stability
        else:
            lg = logits.float().mean(dim=(2, 3))              # (B,K)
            wg = torch.softmax(lg / T, dim=1)                 # (B,K)
            wts = wg[:, :, None, None].expand(-1, -1, h, w)

        # Weighted sum in coordinate domain
        y_hat = (wts[:, :, None, :, :] * y.float()).sum(dim=1)  # (B, D, h, w) float32

        # Uncertainty combine (only if all experts provide it)
        u_hat = None
        if any_unc:
            if any(u is None for u in uncs):
                raise RuntimeError("LateFusionCoordHead: some experts returned uncertainty and some did not.")
            u = torch.stack([u for u in uncs], dim=1)         # (B, K, 1, h, w)
            u_hat = (wts[:, :, None, :, :] * u.float()).sum(dim=1)  # (B,1,h,w) float32

        # Reshape back
        y_hat = y_hat.view(*leading_dims, d, h, w)
        if u_hat is not None:
            u_hat = u_hat.view(*leading_dims, 1, h, w)

        if getattr(self.config, 'l2_reg_weight', 0.0) > 0.0:
            self.last_l2_reg_loss = (logits ** 2).mean() * self.config.l2_reg_weight
        else:
            self.last_l2_reg_loss = None

        return y_hat, u_hat

    def unfreeze_experts_if_needed(self, iteration: int):
        target_iter = getattr(self.config, 'unfreeze_experts_after_iterations', -1)
        if target_iter > 0 and iteration == target_iter:
            _logger.info(f"Unfreezing expert heads at iteration {iteration}")
            for prm in self.expert_heads.parameters():
                prm.requires_grad_(True)

    def get_param_groups(self, min_lr: float, max_lr: float) -> list[dict]:
        gate_factor = getattr(self.config, 'gate_learning_rate_factor', 1.0)
        expert_factor = getattr(self.config, 'expert_learning_rate_factor', 1.0)

        groups = []
        gate_params = list(self.gate.parameters())
        if gate_params:
            groups.append({
                "name": "gate",
                "params": gate_params,
                "lr": max_lr * gate_factor,
                "max_lr": max_lr * gate_factor,
                "min_lr": min_lr * gate_factor,
            })

        expert_params = list(self.expert_heads.parameters())
        if expert_params:
            groups.append({
                "name": "experts",
                "params": expert_params,
                "lr": max_lr * expert_factor,
                "max_lr": max_lr * expert_factor,
                "min_lr": min_lr * expert_factor,
            })

        handled = set(id(p) for p in gate_params + expert_params)
        other_params = [p for p in self.parameters() if id(p) not in handled]
        if other_params:
            groups.append({
                "name": "other",
                "params": other_params,
                "lr": max_lr,
                "max_lr": max_lr,
                "min_lr": min_lr,
            })
            
        return groups

# HeadConfig = MLPHead.Config | TransformerHead.Config | FusionHead.Config | pathlib.Path
HeadConfig = MLPHead.Config | TransformerHead.Config | FusionHead.Config | LateFusionCoordHead.Config | pathlib.Path

def create_head(head_config: HeadConfig) -> SCRHead:
    """Create a head from a head configuration."""
    state_dict = None
    if isinstance(head_config, pathlib.Path):
        state_dict = torch.load(head_config, map_location="cpu", weights_only=False)
        extra_state = dict(state_dict["_extra_state"])
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
