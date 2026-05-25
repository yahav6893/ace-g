#!/usr/bin/env python3
"""
Train K=2 alpha gating classifier.

Main changes vs the first V0 script:
  - default loss uses soft targets derived from error_curve
  - evaluates train + eval split every epoch
  - reports hard and soft alpha metrics
  - stratified overfit sanity test
  - optional class weighting / balanced sampler diagnostics

Expected dataset keys:
  z:            FloatTensor [N, 2, 1024]
  target_class: LongTensor  [N], values 0..10
  alpha:        FloatTensor [N], optional; if missing, target_class/10 is used
  error_curve:  FloatTensor [N, 11], required for soft losses
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)


# -----------------------------
# Repro / utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_bins(device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 11, device=device)


def make_class_weights(targets: torch.Tensor, mode: str, device: torch.device) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = torch.bincount(targets.cpu().long(), minlength=11).float().clamp_min(1.0)
    if mode == "inv_sqrt":
        w = 1.0 / counts.sqrt()
    elif mode == "inv":
        w = 1.0 / counts
    else:
        raise ValueError(f"Unknown class_weight mode: {mode}")
    w = w / w.mean()
    return w.to(device)


def build_balanced_sampler(targets: torch.Tensor) -> WeightedRandomSampler:
    counts = torch.bincount(targets.cpu().long(), minlength=11).float().clamp_min(1.0)
    sample_weights = 1.0 / counts[targets.cpu().long()]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# -----------------------------
# Dataset
# -----------------------------

class GatingDataset(Dataset):
    def __init__(self, data_path: Path):
        _logger.info(f"Loading dataset from {data_path}")
        self.data = torch.load(data_path, map_location="cpu")

        self.z = self.data["z"].detach().float()
        self.target_class = self.data["target_class"].detach().long()
        self.alpha = self.data.get("alpha", self.target_class.float() / 10.0).detach().float()
        self.error_curve = self.data.get("error_curve", None)
        if self.error_curve is not None:
            self.error_curve = self.error_curve.detach().float()

        self.validate()
        _logger.info(f"Loaded {len(self)} samples from {data_path.name}")

    def validate(self) -> None:
        if self.z.ndim != 3 or self.z.shape[1:] != (2, 1024):
            raise ValueError(f"Expected z [N,2,1024], got {tuple(self.z.shape)}")
        if self.target_class.ndim != 1 or self.target_class.shape[0] != self.z.shape[0]:
            raise ValueError("target_class must be [N]")
        if self.target_class.min() < 0 or self.target_class.max() > 10:
            raise ValueError("target_class must be in [0,10]")
        if not torch.isfinite(self.z).all():
            raise ValueError("z contains NaN/Inf")
        if not torch.isfinite(self.alpha).all():
            raise ValueError("alpha contains NaN/Inf")
        if self.error_curve is not None:
            if self.error_curve.shape != (self.z.shape[0], 11):
                raise ValueError(f"Expected error_curve [N,11], got {tuple(self.error_curve.shape)}")
            if not torch.isfinite(self.error_curve).all():
                raise ValueError("error_curve contains NaN/Inf")

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "z": self.z[idx],
            "target_class": self.target_class[idx],
            "alpha_gt": self.alpha[idx],
        }
        if self.error_curve is not None:
            item["error_curve"] = self.error_curve[idx]
        return item


# -----------------------------
# Model
# -----------------------------

class GatingClassifier(nn.Module):
    """K=2 classifier mapping two encoder embeddings to 11 alpha-bin logits."""

    def __init__(self, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_ln = nn.LayerNorm(1024)
        self.shared_proj = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder_id_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.encoder_id_embed, std=0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 11),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 3 or z.shape[1:] != (2, 1024):
            raise ValueError(f"Expected z [B,2,1024], got {tuple(z.shape)}")

        b = z.shape[0]
        x = self.input_ln(z)
        x = self.shared_proj(x)                         # [B,2,D]
        x = x + self.encoder_id_embed[None, :, :]        # [B,2,D]
        x = x.reshape(b, -1)                             # [B,2D]

        logits = self.classifier(x)                      # [B,11]
        probs = F.softmax(logits, dim=-1)
        bins = alpha_bins(z.device)

        pred_class = logits.argmax(dim=-1)
        alpha_hard = pred_class.float() / 10.0
        alpha_soft = probs @ bins

        weights_hard = torch.stack([alpha_hard, 1.0 - alpha_hard], dim=-1)
        weights_soft = torch.stack([alpha_soft, 1.0 - alpha_soft], dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "pred_class": pred_class,
            "alpha_hard": alpha_hard,
            "alpha_soft": alpha_soft,
            "weights_hard": weights_hard,
            "weights_soft": weights_soft,
        }


# -----------------------------
# Losses
# -----------------------------

def errors_to_soft_targets(
    error_curve: torch.Tensor,
    tau: float,
    error_norm: str = "range",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert lower-is-better error curves [B,11] to soft labels [B,11]."""
    err = error_curve.float()
    err = err - err.min(dim=1, keepdim=True).values

    if error_norm == "range":
        denom = err.max(dim=1, keepdim=True).values.clamp_min(eps)
        err = err / denom
    elif error_norm == "std":
        denom = err.std(dim=1, keepdim=True).clamp_min(eps)
        err = err / denom
    elif error_norm == "none":
        pass
    else:
        raise ValueError(f"Unknown error_norm: {error_norm}")

    return F.softmax(-err / tau, dim=1)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_mode: str,
    tau: float,
    error_norm: str,
    class_weights: Optional[torch.Tensor],
    alpha_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    target_class = batch["target_class"]
    alpha_gt = batch["alpha_gt"]

    logs: Dict[str, float] = {}

    if loss_mode == "hard_ce":
        loss_main = F.cross_entropy(logits, target_class, weight=class_weights)

    elif loss_mode in {"soft_ce", "soft_ce_alpha"}:
        if "error_curve" not in batch:
            raise ValueError(f"loss_mode={loss_mode} requires error_curve in dataset")
        soft_target = errors_to_soft_targets(batch["error_curve"], tau=tau, error_norm=error_norm)
        log_probs = F.log_softmax(logits, dim=1)
        per_class_loss = -soft_target * log_probs
        if class_weights is not None:
            per_class_loss = per_class_loss * class_weights[None, :]
        loss_main = per_class_loss.sum(dim=1).mean()
        logs["target_entropy"] = float((-(soft_target * (soft_target + 1e-8).log()).sum(dim=1).mean()).detach())

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    loss = loss_main
    logs["loss_main"] = float(loss_main.detach())

    if loss_mode == "soft_ce_alpha" or alpha_loss_weight > 0.0:
        loss_alpha = F.smooth_l1_loss(outputs["alpha_soft"], alpha_gt)
        loss = loss + alpha_loss_weight * loss_alpha
        logs["loss_alpha"] = float(loss_alpha.detach())

    logs["loss_total"] = float(loss.detach())
    return loss, logs


# -----------------------------
# Metrics / evaluation
# -----------------------------

def print_histogram(title: str, classes: torch.Tensor) -> None:
    counts = torch.bincount(classes.cpu().long(), minlength=11).numpy()
    total = max(int(counts.sum()), 1)
    _logger.info(f"  {title}:")
    for i, count in enumerate(counts):
        _logger.info(f"    class {i:2d} alpha={i/10.0:.1f}: {count:7d} ({100.0 * count / total:6.2f}%)")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    loss_mode: str,
    tau: float,
    error_norm: str,
    class_weights: Optional[torch.Tensor],
    alpha_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_pred = []
    all_target = []
    all_alpha_hard = []
    all_alpha_soft = []
    all_alpha_gt = []
    all_entropy = []

    with torch.no_grad():
        for batch_cpu in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items()}
            outputs = model(batch["z"])
            loss, _ = compute_loss(
                outputs, batch, loss_mode, tau, error_norm, class_weights, alpha_loss_weight
            )
            b = batch["z"].shape[0]
            total_loss += float(loss.detach()) * b
            total_samples += b

            probs = outputs["probs"]
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)

            all_pred.append(outputs["pred_class"].detach().cpu())
            all_target.append(batch["target_class"].detach().cpu())
            all_alpha_hard.append(outputs["alpha_hard"].detach().cpu())
            all_alpha_soft.append(outputs["alpha_soft"].detach().cpu())
            all_alpha_gt.append(batch["alpha_gt"].detach().cpu())
            all_entropy.append(entropy.detach().cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    alpha_hard = torch.cat(all_alpha_hard)
    alpha_soft = torch.cat(all_alpha_soft)
    alpha_gt = torch.cat(all_alpha_gt)
    entropy = torch.cat(all_entropy)

    exact_acc = (pred == target).float().mean().item()
    off_by_1_acc = ((pred - target).abs() <= 1).float().mean().item()
    hard_alpha_mae = (alpha_hard - alpha_gt).abs().mean().item()
    soft_alpha_mae = (alpha_soft - alpha_gt).abs().mean().item()
    mean_loss = total_loss / max(total_samples, 1)

    _logger.info(f"\nEvaluation: {split_name}")
    _logger.info(f"  loss:               {mean_loss:.6f}")
    _logger.info(f"  exact_acc_hard:     {100.0 * exact_acc:6.2f}%")
    _logger.info(f"  off_by_1_acc_hard:  {100.0 * off_by_1_acc:6.2f}%")
    _logger.info(f"  hard_alpha_mae:     {hard_alpha_mae:.6f}")
    _logger.info(f"  soft_alpha_mae:     {soft_alpha_mae:.6f}")
    _logger.info(f"  mean_alpha_gt:      {alpha_gt.mean().item():.6f}")
    _logger.info(f"  mean_alpha_hard:    {alpha_hard.mean().item():.6f}")
    _logger.info(f"  mean_alpha_soft:    {alpha_soft.mean().item():.6f}")
    _logger.info(f"  mean_pred_entropy:  {entropy.mean().item():.6f}")
    print_histogram("target histogram", target)
    print_histogram("hard prediction histogram", pred)
    _logger.info("-" * 60)

    return {
        "loss": mean_loss,
        "exact_acc": exact_acc,
        "off_by_1_acc": off_by_1_acc,
        "hard_alpha_mae": hard_alpha_mae,
        "soft_alpha_mae": soft_alpha_mae,
    }


# -----------------------------
# Sanity overfit
# -----------------------------

def make_stratified_indices(targets: torch.Tensor, samples_per_class: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    idxs = []
    for c in range(11):
        cls_idx = torch.where(targets.cpu().long() == c)[0]
        if len(cls_idx) == 0:
            continue
        perm = cls_idx[torch.randperm(len(cls_idx), generator=gen)]
        idxs.append(perm[: min(samples_per_class, len(perm))])
    if not idxs:
        raise ValueError("Could not build stratified subset")
    out = torch.cat(idxs)
    return out[torch.randperm(len(out), generator=gen)]


def run_stratified_overfit(
    train_dataset: GatingDataset,
    device: torch.device,
    d_model: int,
    samples_per_class: int,
    steps: int,
    seed: int,
) -> None:
    _logger.info("\n=======================================================")
    _logger.info("Stratified overfit sanity test")
    _logger.info("=======================================================")

    idx = make_stratified_indices(train_dataset.target_class, samples_per_class, seed)
    subset_z = train_dataset.z[idx].to(device)
    subset_target = train_dataset.target_class[idx].to(device)

    _logger.info(f"Subset size: {len(idx)}")
    print_histogram("subset target histogram", subset_target.cpu())

    model = GatingClassifier(d_model=d_model, dropout=0.0).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(subset_z)
        loss = F.cross_entropy(outputs["logits"], subset_target)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                pred = model(subset_z)["pred_class"]
                acc = (pred == subset_target).float().mean().item()
                off1 = ((pred - subset_target).abs() <= 1).float().mean().item()
            _logger.info(f"  step {step:4d} | loss {loss.item():.6f} | exact {100*acc:6.2f}% | off1 {100*off1:6.2f}%")

    with torch.no_grad():
        pred = model(subset_z)["pred_class"]
        final_acc = (pred == subset_target).float().mean().item()

    if final_acc < 0.95:
        raise ValueError(f"SANITY FAIL: stratified overfit exact acc={100*final_acc:.2f}%")
    _logger.info(f"SANITY PASS: stratified overfit exact acc={100*final_acc:.2f}%")
    _logger.info("=======================================================\n")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train K=2 alpha gating classifier")

    parser.add_argument("--dataset_root", type=str, default="outputs/alpha_test/shopfacade")
    parser.add_argument("--scene", type=str, default="shopfacade")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None, help="Validation file. If omitted, uses *_dataset_test.pt")

    parser.add_argument("--loss_mode", type=str, default="hard_ce", choices=["hard_ce", "soft_ce", "soft_ce_alpha"])
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--error_norm", type=str, default="range", choices=["range", "std", "none"])
    parser.add_argument("--alpha_loss_weight", type=float, default=0.0)
    parser.add_argument("--class_weight", type=str, default="inv_sqrt", choices=["none", "inv_sqrt", "inv"])
    parser.add_argument("--balanced_sampler", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--run_overfit", action="store_true")
    parser.add_argument("--overfit_samples_per_class", type=int, default=16)
    parser.add_argument("--overfit_steps", type=int, default=1000)

    parser.add_argument("--save_by", type=str, default="soft_alpha_mae", choices=["loss", "exact_acc", "soft_alpha_mae", "hard_alpha_mae"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--wandb-entity", type=str, default="yahav6893")
    parser.add_argument("--wandb-project", type=str, default="DACE_gating")
    parser.add_argument("--disable-wandb", action="store_true")

    return parser.parse_args()


def metric_is_better(name: str, value: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    if name == "exact_acc":
        return value > best
    return value < best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    _logger.info(f"Args: {vars(args)}")

    if not args.disable_wandb:
        try:
            import wandb
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"gating_{args.scene}_{timestamp}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=run_name,
                tags=[args.scene, "gating_classifier"],
            )
        except Exception as e:
            _logger.warning(f"Failed to initialize W&B: {e}. Proceeding without W&B.")
            args.disable_wandb = True

    dataset_dir = Path(args.dataset_root)
    train_path = Path(args.train_file) if args.train_file else dataset_dir / f"{args.scene}_dataset_train.pt"
    eval_path = Path(args.eval_file) if args.eval_file else dataset_dir / f"{args.scene}_dataset_test.pt"

    train_dataset = GatingDataset(train_path)
    eval_dataset = GatingDataset(eval_path)

    if args.run_overfit:
        run_stratified_overfit(
            train_dataset=train_dataset,
            device=device,
            d_model=args.d_model,
            samples_per_class=args.overfit_samples_per_class,
            steps=args.overfit_steps,
            seed=args.seed,
        )

    class_weights = make_class_weights(train_dataset.target_class, args.class_weight, device)
    if class_weights is not None:
        _logger.info(f"Class weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")

    pin_memory = device.type == "cuda"
    if args.balanced_sampler:
        sampler = build_balanced_sampler(train_dataset.target_class)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = GatingClassifier(d_model=args.d_model, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_path = dataset_dir / f"{args.scene}_gating_head_best.pt"
    best_value: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch_cpu in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["z"])
            loss, _ = compute_loss(
                outputs=outputs,
                batch=batch,
                loss_mode=args.loss_mode,
                tau=args.tau,
                error_norm=args.error_norm,
                class_weights=class_weights,
                alpha_loss_weight=args.alpha_loss_weight,
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            b = batch["z"].shape[0]
            running_loss += float(loss.detach()) * b
            running_count += b

        train_step_loss = running_loss / max(running_count, 1)
        _logger.info(f"\nEpoch {epoch:03d}/{args.epochs:03d} | train_step_loss={train_step_loss:.6f}")

        train_metrics = evaluate(
            model, train_loader, device, "train", args.loss_mode, args.tau, args.error_norm, class_weights, args.alpha_loss_weight
        )
        eval_metrics = evaluate(
            model, eval_loader, device, "eval", args.loss_mode, args.tau, args.error_norm, class_weights, args.alpha_loss_weight
        )

        if not args.disable_wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_step_loss": train_step_loss,
                    "train/loss": train_metrics["loss"],
                    "train/exact_acc": train_metrics["exact_acc"],
                    "train/off_by_1_acc": train_metrics["off_by_1_acc"],
                    "train/hard_alpha_mae": train_metrics["hard_alpha_mae"],
                    "train/soft_alpha_mae": train_metrics["soft_alpha_mae"],
                    "eval/loss": eval_metrics["loss"],
                    "eval/exact_acc": eval_metrics["exact_acc"],
                    "eval/off_by_1_acc": eval_metrics["off_by_1_acc"],
                    "eval/hard_alpha_mae": eval_metrics["hard_alpha_mae"],
                    "eval/soft_alpha_mae": eval_metrics["soft_alpha_mae"],
                })
            except Exception as e:
                _logger.warning(f"Failed to log to W&B: {e}")

        current = eval_metrics[args.save_by]
        if metric_is_better(args.save_by, current, best_value):
            best_value = current
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "save_by": args.save_by,
                    "best_value": best_value,
                },
                save_path,
            )
            _logger.info(f"Saved best checkpoint to {save_path} | {args.save_by}={best_value:.6f}")
            if not args.disable_wandb:
                try:
                    import wandb
                    wandb.run.summary[f"best_eval_{args.save_by}"] = best_value
                except Exception:
                    pass

    _logger.info("\n=======================================================")
    _logger.info("Training complete")
    _logger.info(f"Best eval {args.save_by}: {best_value}")
    _logger.info(f"Checkpoint: {save_path}")
    _logger.info("=======================================================")

    if not args.disable_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
