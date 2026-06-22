"""Synthetic gate-selection tests for UncExpertFusionHead.

These tests validate the *learning mechanism* of the uncertainty-based MoE fusion in
isolation from the encoders, data, and reprojection geometry. The question they answer:

    Given two frozen experts where expert 0 is exactly correct and expert 1 is wrong,
    can the inverse-variance gate (driven only by the trainable UncHeads) learn to put
    (almost) all weight on the correct expert, so the fused prediction recovers it?

If this fails, no amount of data/hyperparameters will make the real fusion beat the best
single expert, because the gradient pathway that is supposed to select the better expert
is broken or degenerate. If it passes, the mechanism *can* select, and the real-data
shortfall (fused worse than best single, see outputs/summaries/...expert_eval_summary.json)
is a signal-strength / loss-shaping problem, not a structural one.

Run:
    ~/dace_env310/bin/python -m pytest ace-g/tests/test_gate_selection.py -q
or directly:
    ~/dace_env310/bin/python ace-g/tests/test_gate_selection.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ace_g.scr_heads import MLPHead, UncExpertFusionHead, UncHead


class _ConstPred(nn.Module):
    """Stand-in for a frozen expert MLPHead: ignores input, returns a fixed coordinate."""

    def __init__(self, coord_xyz: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coord", coord_xyz.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        # Mirror MLPHead.forward, which returns (scene_coords, uncertainty_or_None).
        b, _, h, w = x.shape
        return self.coord.expand(b, 3, h, w).clone(), None


def _build_head(num_experts: int = 2, c: int = 8, var_min: float = 1e-4, var_max: float = 1e4) -> UncExpertFusionHead:
    cfg = UncExpertFusionHead.Config(
        dim_in=c * num_experts,
        num_experts=num_experts,
        expert_head=MLPHead.Config(dim_in=c, num_blocks=1, use_homogeneous=True, use_uncertainty=False),
        unc_head=UncHead.Config(dim_in=c, out_dim=1, arc_type="mlp", hidden_ratio=1.0, head_dropout=0.0, var_eps=1e-6),
        expert_head_paths=None,
        freeze_loaded_experts=False,
        force_shared_mean=False,
        var_min=var_min,
        var_max=var_max,
        tau_source="none",
    )
    return UncExpertFusionHead(cfg)


def _stub_experts(head: UncExpertFusionHead, coords: list[torch.Tensor]) -> None:
    """Replace each expert's coordinate MLPHead with a constant; keep UncHeads trainable."""
    for i, coord in enumerate(coords):
        head.experts[i].mlp_head = _ConstPred(coord)


def test_equal_experts_give_uniform_weights() -> None:
    """Forward sanity: identical experts -> ~0.5/0.5 fusion weights (no spurious bias)."""
    torch.manual_seed(0)
    head = _build_head().eval()
    same = torch.tensor([1.0, 2.0, 3.0])
    _stub_experts(head, [same.clone(), same.clone()])

    feats = torch.randn(4, head.config.dim_in, 2, 2)
    with torch.no_grad():
        head(feats)
    w = head.last_moe_weights  # [B, K, 1, h, w]
    w0 = w[:, 0].mean().item()
    # UncHeads are randomly initialized; with identical inputs per expert the gate should
    # not be wildly lopsided. Allow a generous band; the point is "no structural bias".
    assert 0.2 < w0 < 0.8, f"expected roughly balanced weights for equal experts, got w0={w0:.3f}"


def test_gate_learns_to_select_correct_expert() -> None:
    """Core test: train only the UncHeads to minimize fused MSE; the gate must pick expert 0."""
    torch.manual_seed(0)
    head = _build_head()

    gt = torch.tensor([1.0, 2.0, 3.0])
    wrong = torch.tensor([-5.0, 5.0, -5.0])
    _stub_experts(head, [gt.clone(), wrong.clone()])

    # Only the UncHeads carry parameters (experts are constant buffers, no gate MLP here).
    trainable = [p for p in head.parameters() if p.requires_grad]
    assert trainable, "expected trainable UncHead parameters"

    feats = torch.randn(8, head.config.dim_in, 2, 2)  # fixed 'patch' features
    target = gt.view(1, 3, 1, 1).expand(8, 3, 2, 2)

    opt = torch.optim.Adam(trainable, lr=1e-2)
    head.train()
    init_mse = None
    for step in range(800):
        opt.zero_grad()
        y_hat, _ = head(feats)
        loss = torch.mean((y_hat - target) ** 2)
        if init_mse is None:
            init_mse = loss.item()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        y_hat, _ = head(feats)
        final_mse = torch.mean((y_hat - target) ** 2).item()
        w0 = head.last_moe_weights[:, 0].mean().item()

    assert final_mse < 0.05 * init_mse, f"fused MSE did not collapse: init={init_mse:.4f} final={final_mse:.4f}"
    assert w0 > 0.9, f"gate failed to select the correct expert: w0={w0:.3f} (want >0.9)"


if __name__ == "__main__":
    test_equal_experts_give_uniform_weights()
    test_gate_learns_to_select_correct_expert()
    print("OK: gate-selection mechanism tests passed.")
