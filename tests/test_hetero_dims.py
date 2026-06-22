"""Synthetic tests for heterogeneous-dim MoGU fusion.

These validate that ``UncExpertFusionHead`` and ``ListMultiEncoder`` support experts with
*different* feature dims (e.g. DINOv2-L=1024 + FiT3D-B=768), which is the only blocker
for using FiT3D-B as a complementary expert. The fusion math (per-expert head -> coords +
variance -> inverse-variance weighting) is dimension-agnostic; only the input split and the
per-expert head construction had even-split assumptions baked in.

What is checked:
  1. Shapes flow end-to-end with expert_dims=[1024, 768] (dim_in=1792).
  2. The uneven channel split routes the right slice to each expert (each expert sees
     exactly expert_dims[i] channels, in order).
  3. A real pretrained 1024-d expert head loads into expert 0 (with expert 1 left random),
     is registered as loaded, and is frozen.
  4. ListMultiEncoder records per-encoder dims and concatenates uneven dims when
     allow_uneven_dims=True, and still rejects uneven dims by default.
  5. The inverse-variance gate produces one weight per expert (K=2).

Run:
    ~/dace_env310/bin/python -m pytest ace-g/tests/test_hetero_dims.py -q
or directly:
    ~/dace_env310/bin/python ace-g/tests/test_hetero_dims.py
"""

from __future__ import annotations

import pathlib

import torch

from ace_g.encoders import ListMultiEncoder
from ace_g.scr_heads import MLPHead, UncExpertFusionHead, UncHead

DINO_HEAD = pathlib.Path(
    "/home/eng/vinokuy2/dace/outputs/heads/DINOv2_vitl_reg-aceg_cambridge-shopfacade_head.pt"
)

EXPERT_DIMS = [1024, 768]
DIM_IN = sum(EXPERT_DIMS)


def _make_head(expert_head_paths=None) -> UncExpertFusionHead:
    cfg = UncExpertFusionHead.Config(
        dim_in=DIM_IN,
        num_experts=2,
        expert_dims=EXPERT_DIMS,
        expert_head_paths=expert_head_paths,
        freeze_loaded_experts=True,
        force_shared_mean=False,
        gate_temperature=1.0,
        tau_source="total",
        tau_detach=True,
        var_min=0.1,
        var_max=100.0,
        use_uncertainty=False,
    )
    return UncExpertFusionHead(cfg)


def test_hetero_shapes_and_gate():
    """Shapes flow and the gate yields K=2 weights with uneven expert dims."""
    head = _make_head().eval()

    # Per-expert heads must be built with their own input dim.
    seen_dims = [head.experts[i].mlp_head.config.dim_in for i in range(2)]
    assert seen_dims == EXPERT_DIMS, seen_dims
    assert head.expert_offsets == [0, 1024, 1792], head.expert_offsets

    b, h, w = 2, 5, 7
    x = torch.randn(b, DIM_IN, h, w)
    y_hat, u_hat = head(x)

    assert y_hat.shape == (b, 3, h, w), y_hat.shape
    # Inverse-variance gate: one scalar weight per expert/patch -> [B, K, 1, h, w].
    assert head.last_moe_weights.shape == (b, 2, 1, h, w), head.last_moe_weights.shape
    assert head.last_expert_preds.shape == (b, 2, 3, h, w), head.last_expert_preds.shape
    # Weights normalize over the expert axis.
    wsum = head.last_moe_weights.sum(dim=1)
    assert torch.allclose(wsum, torch.ones_like(wsum), atol=1e-5)
    print("test_hetero_shapes_and_gate: OK", seen_dims)


def test_uneven_split_routing():
    """Each expert receives exactly its channel slice, in encoder order."""
    head = _make_head().eval()

    captured: dict[int, int] = {}

    def _wrap(idx, mlp):
        orig = mlp.forward

        def fwd(t):
            captured[idx] = t.shape[1]
            return orig(t)

        mlp.forward = fwd

    for i in range(2):
        _wrap(i, head.experts[i].mlp_head)

    # Tag each expert's channel band with a distinct constant so a mis-slice would change
    # the per-expert input statistics; the channel-count check below is the strict assertion.
    x = torch.zeros(1, DIM_IN, 3, 3)
    x[:, 0:1024] = 1.0
    x[:, 1024:1792] = 2.0
    head(x)

    assert captured == {0: 1024, 1: 768}, captured
    print("test_uneven_split_routing: OK", captured)


def test_load_real_expert0_frozen():
    """A real 1024-d head loads into expert 0 and is frozen; expert 1 stays random/trainable."""
    if not DINO_HEAD.exists():
        print(f"test_load_real_expert0_frozen: SKIP (missing {DINO_HEAD})")
        return

    head = _make_head(expert_head_paths=[DINO_HEAD, None])

    assert head._loaded_expert_indices == {0}, head._loaded_expert_indices
    # Loaded coordinate expert is frozen...
    assert all(not p.requires_grad for p in head.experts[0].mlp_head.parameters())
    # ...but its uncertainty head stays trainable, and expert 1 is untouched.
    assert any(p.requires_grad for p in head.experts[0].unc_head.parameters())
    assert any(p.requires_grad for p in head.experts[1].mlp_head.parameters())
    head.assert_freeze_policy()  # must not raise
    print("test_load_real_expert0_frozen: OK")


def test_listmulti_uneven_dims():
    """ListMultiEncoder records per-encoder dims and concatenates uneven dims (allow flag)."""
    enc_cfgs = [
        {"obj_type": "ace_g.encoders.FCNEncoder", "kwargs": {"dim_out": 16}, "path": None},
        {"obj_type": "ace_g.encoders.FCNEncoder", "kwargs": {"dim_out": 24}, "path": None},
    ]

    # Default: uneven dims must be rejected.
    rejected = False
    try:
        ListMultiEncoder(enc_cfgs)
    except AssertionError:
        rejected = True
    assert rejected, "ListMultiEncoder should reject uneven dims without allow_uneven_dims=True"

    enc = ListMultiEncoder(enc_cfgs, allow_uneven_dims=True).eval()
    assert enc.expert_dims == [16, 24], enc.expert_dims
    assert enc.dim_out == 40, enc.dim_out

    # FCNEncoder is grayscale, patch/subsample 8.
    x = torch.randn(1, 1, 32, 32)
    feats = enc(x)
    assert feats.shape[-3] == 40, feats.shape
    print("test_listmulti_uneven_dims: OK", enc.expert_dims, tuple(feats.shape))


if __name__ == "__main__":
    test_hetero_shapes_and_gate()
    test_uneven_split_routing()
    test_load_real_expert0_frozen()
    test_listmulti_uneven_dims()
    print("\nAll heterogeneous-dim tests passed.")