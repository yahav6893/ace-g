# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Common layers."""
import itertools
from typing import Final, Optional

import einops
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath, LayerScale, use_fused_attn

try:
    import warnings

    warnings.filterwarnings("ignore", message="`torch.library.impl_abstract` was renamed to")
    warnings.filterwarnings("ignore", message="xFormers is available")
    import xformers.ops as xops
except ImportError:
    xops = None


class Mlp(torch.nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        norm_layer=None,
        bias: bool = True,
        drop: float = 0.0,
        num_layers: int = 2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.act = act_layer()
        self.norms = torch.nn.ModuleList(
            [
                norm_layer(hidden_features) if norm_layer is not None else torch.nn.Identity()
                for _ in range(num_layers - 1)
            ]
        )
        self.drops = torch.nn.ModuleList([torch.nn.Dropout(drop) for _ in range(num_layers)])
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_features, hidden_features, bias=bias))
        for _ in range(num_layers - 2):
            self.fcs.append(torch.nn.Linear(hidden_features, hidden_features, bias=bias))
        self.fcs.append(torch.nn.Linear(hidden_features, out_features, bias=bias))

    def forward(self, x):
        for fc, norm, drop in zip(self.fcs, self.norms, self.drops):
            x = fc(x)
            x = self.act(x)
            x = drop(x)
            x = norm(x)

        x = self.fcs[-1](x)
        x = self.drops[-1](x)
        return x


class Attention(torch.nn.Module):
    """Attention layer with separate queries, keys, and values.

    Can be used for cross and self-attention. Based on timm.models.vision_transformer.Attention.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        kv_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        if kv_dim is None:
            kv_dim = dim

        self.q_proj = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = torch.nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.v_proj = torch.nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_seqlens: torch.Tensor | None = None,
        kv_seqlens: torch.Tensor | None = None,
        diag_attn_bias: bool = False,
    ) -> torch.Tensor:
        """Forward function.

        Note that the benefits of faster attention implementations mainly apply when both sequences are reasonably long.

        Args:
            q: Queries. Shape (..., num_queries, dim_emb).
            k: Keys. Shape (..., num_keys, dim_emb).
            v: Values. Shape (..., num_keys, dim_emb).
            q_seqlens: Sequence lengths of queries. Shape (...).
            kv_seqlens: Sequence lengths of keys and values. Shape (...).
            diag_attn_bias: Use diagonal attention bias (i.e., each query can only attend to the key at the same position).

        Returns:
            Output embeddings. Shape (..., num_queries, dim_emb).
        """
        final_resize = None
        if q.shape[-2] == 1:
            # NOTE if there is only one query per-batch, we can combine all leading dimensions in which only q varies
            # into the sequence dimension; this can be significantly faster with fast attention implementations that are
            # mainly optimized for long sequences, not for large batches (this is mathematically equivalent)

            # find all leading dimensions in which only q varies (i.e., where k, v are either 1 or shorter)
            movable_dim_names = []
            movable_dim_names_dict = {}
            q_dim_names = [f"d{i}" for i in range(len(q.shape))]
            from_einops_dims = ["1", q_dim_names[-1]]
            to_einops_dims = ["placeholder", q_dim_names[-1]]
            for q_dim, q_dim_name, k_dim, v_dim in itertools.zip_longest(
                q.shape[-3::-1], q_dim_names[-3::-1], k.shape[-3::-1], v.shape[-3::-1]
            ):
                if (
                    (q_dim != 1 and q_dim is not None)
                    and (k_dim == 1 or k_dim is None)
                    and (v_dim == 1 or v_dim is None)
                ):
                    movable_dim_names.insert(0, q_dim_name)
                    movable_dim_names_dict[q_dim_name] = q_dim
                    from_einops_dims.insert(0, q_dim_name)
                    to_einops_dims.insert(0, "1")
                else:
                    from_einops_dims.insert(0, q_dim_name)
                    to_einops_dims.insert(0, q_dim_name)

            if movable_dim_names:
                to_einops_dims[-2] = f"({' '.join(movable_dim_names)})"  # replace placeholder with moved dims
                from_einops = " ".join(from_einops_dims)
                to_einops = " ".join(to_einops_dims)
                q = einops.rearrange(q, f"{from_einops} -> {to_einops}")

                def final_resize(q):
                    return einops.rearrange(q, f"{to_einops} -> {from_einops}", **movable_dim_names_dict)

        if not (q.shape[:-2] == k.shape[:-2] == v.shape[:-2]):
            # NOTE unclear if torch's scaled_dot_product_attention supports broadcasting properly, so we do it manually
            #  (I observed some slowdowns when all that was needed was an unsqueeze)
            broadcasted_leading_dims = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
            q = q.broadcast_to(broadcasted_leading_dims + q.shape[-2:])
            k = k.broadcast_to(broadcasted_leading_dims + k.shape[-2:])
            v = v.broadcast_to(broadcasted_leading_dims + v.shape[-2:])

        dim_emb = q.shape[-1]
        if xops is not None:  # use xformers attention if available
            # NOTE added this for testing, but torch already includes this and other fast implementations
            #  according to its documentatation; and it seems minimally slower than the option below
            q = einops.rearrange(self.q_proj(q), "... n (h d) -> ... n h d", h=self.num_heads)
            k = einops.rearrange(self.k_proj(k), "... n (h d) -> ... n h d", h=self.num_heads)
            v = einops.rearrange(self.v_proj(v), "... n (h d) -> ... n h d", h=self.num_heads)

            q, k = self.q_norm(q), self.k_norm(k)

            broadcasted_leading_dims = torch.broadcast_shapes(q.shape[:-3], k.shape[:-3], v.shape[:-3])
            q = q.broadcast_to(broadcasted_leading_dims + q.shape[-3:]).view(-1, *q.shape[-3:])
            k = k.broadcast_to(broadcasted_leading_dims + k.shape[-3:]).view(-1, *k.shape[-3:])
            v = v.broadcast_to(broadcasted_leading_dims + v.shape[-3:]).view(-1, *v.shape[-3:])
            attn_bias = None
            if q_seqlens is not None or kv_seqlens is not None:
                # NOTE xformers does not support batch size != 1 with a BlockDiagonalMask
                #  so we create the mask manually (likely slower than xformers' implementation)

                # NOTE need multiple of 8 for sequence lenghts for the attention bias
                #  (limitation of xformers, slicing it afterwards)
                num_queries_8 = (q.shape[1] // 8 + 1) * 8
                num_keys_8 = (k.shape[1] // 8 + 1) * 8
                attn_bias = torch.zeros(
                    (q.shape[0], self.num_heads, num_queries_8, num_keys_8), device=q.device, dtype=q.dtype
                )
                if q_seqlens is not None:
                    seq_indices = torch.arange(num_queries_8, device=q_seqlens.device).reshape(1, 1, -1, 1)
                    invalid_queries = seq_indices >= q_seqlens.view(-1, 1, 1, 1)
                    attn_bias[invalid_queries.broadcast_to(attn_bias.shape)] = float("-inf")
                if kv_seqlens is not None:
                    seq_indices = torch.arange(num_keys_8, device=kv_seqlens.device).reshape(1, 1, 1, -1)
                    invalid_keys = seq_indices >= kv_seqlens.view(-1, 1, 1, 1)
                    attn_bias[invalid_keys.broadcast_to(attn_bias.shape)] = float("-inf")
                attn_bias = attn_bias[..., : q.shape[1], : k.shape[1]]
            elif diag_attn_bias:
                num_queries_8 = (q.shape[1] // 8 + 1) * 8
                num_keys_8 = (k.shape[1] // 8 + 1) * 8
                attn_bias = torch.full(
                    (q.shape[0], self.num_heads, num_queries_8, num_keys_8),
                    float("-inf"),
                    device=q.device,
                    dtype=q.dtype,
                )
                attn_bias[..., torch.arange(num_queries_8), torch.arange(num_keys_8)] = 0.0
                attn_bias = attn_bias[..., : q.shape[1], : k.shape[1]]

            x = xops.memory_efficient_attention(
                q, k, v, p=self.attn_drop.p if self.training else 0.0, attn_bias=attn_bias
            )

            x = x.reshape(*broadcasted_leading_dims, -1, dim_emb)
        else:
            assert q_seqlens is None and kv_seqlens is None, "q_seqlens and kv_seqlens are not implemented with torch"

            # project qkv and split into heads as extra batch dim, so we have (..., num_heads, num_{queries,keys}, dim_emb // num_heads)
            # also flatten all batch dims together; this way we get memory efficient attention from torch
            out_shape = q.shape[:-1] + (v.shape[-1],)
            q = einops.rearrange(self.q_proj(q), "... n (h d) -> (...) h n d", h=self.num_heads)
            k = einops.rearrange(self.k_proj(k), "... n (h d) -> (...) h n d", h=self.num_heads)
            v = einops.rearrange(self.v_proj(v), "... n (h d) -> (...) h n d", h=self.num_heads)

            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = einops.rearrange(x, "... h n d -> ... n (h d)")  # combine heads back
            x = x.view(*out_shape)  # reshape back to original shape

        if final_resize:
            x = final_resize(x)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(torch.nn.Module):
    """Transformer block inspired by ViT but supporting separate queries, key, and values.

    Based on timm.models.vision_transformer.Block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        mlp_layer: torch.nn.Module = Mlp,
        kv_dim: int | None = None,
    ) -> None:
        super().__init__()

        if kv_dim is None:
            kv_dim = dim

        self.q_norm = norm_layer(dim)
        self.k_norm = norm_layer(kv_dim)
        self.v_norm = norm_layer(kv_dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            kv_dim=kv_dim,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_seqlens: torch.Tensor | None = None,
        kv_seqlens: torch.Tensor | None = None,
        diag_attn_bias: bool = False,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            q: Queries. Shape (..., num_queries, dim_emb).
            k: Keys. Shape (..., num_keys, dim_emb).
            v: Values. Shape (..., num_keys, dim_emb).
            q_seqlens: Sequence lengths of queries. Shape (...).
            kv_seqlens: Sequence lengths of keys and values. Shape (...).
            diag_attn_bias: Use diagonal attention bias (i.e., each query can only attend to the key at the same position).

        Returns:
            Output embeddings. Shape (..., num_queries, dim_emb).
        """
        x = self.drop_path1(
            self.ls1(self.attn(self.q_norm(q), self.k_norm(k), self.v_norm(v), q_seqlens, kv_seqlens, diag_attn_bias))
        )
        x = x + q
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
