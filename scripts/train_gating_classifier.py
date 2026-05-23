#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger(__name__)


class GatingDataset(Dataset):
    """PyTorch Dataset wrapper for the saved alpha fusion samples."""
    def __init__(self, data_path: Path):
        _logger.info(f"Loading dataset from {data_path}...")
        self.data = torch.load(data_path)
        self.z = self.data["z"]                    # [N, 2, 1024]
        self.target_class = self.data["target_class"]  # [N]
        self.alpha = self.data["alpha"]            # [N]
        self.error_curve = self.data["error_curve"]  # [N, 11]
        
        _logger.info(f"Loaded {self.z.shape[0]} samples.")

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return {
            "z": self.z[idx],
            "target_class": self.target_class[idx],
            "alpha_gt": self.alpha[idx],
            "error_curve": self.error_curve[idx]
        }


class GatingClassifier(nn.Module):
    """Gating network classifier mapping two head embeddings to an 11-class alpha distribution."""
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # 1. Input LayerNorm on the embedding dimension
        self.input_ln = nn.LayerNorm(1024)
        
        # 2. Shared Linear Projection
        self.shared_proj = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Learned encoder-id embedding
        self.encoder_id_embed = nn.Parameter(torch.zeros(2, d_model))
        # Initialize with standard normal scaled down
        nn.init.normal_(self.encoder_id_embed, std=0.02)
        
        # 4. Classifier Head on Flattened Embeddings [B, 512]
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 11)
        )

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: FloatTensor of shape [B, 2, 1024]
        Returns:
            Dictionary with:
                - logits: [B, 11]
                - pred_class: [B]
                - alpha: [B]
                - weights: [B, 2]
                - alpha_soft: [B] (for debug / soft blending)
        """
        B = z.shape[0]
        
        # Apply Input LayerNorm: [B, 2, 1024] -> [B, 2, 1024]
        z = self.input_ln(z)
        
        # Shared Projection: [B, 2, 1024] -> [B, 2, 256]
        z = self.shared_proj(z)
        
        # Add learned encoder-id embedding: [B, 2, 256] + [2, 256] (broadcasted to [B, 2, 256])
        z = z + self.encoder_id_embed
        
        # Flatten: [B, 2, 256] -> [B, 512]
        z_flat = z.view(B, -1)
        
        # Predict logits: [B, 11]
        logits = self.classifier(z_flat)
        
        # Compute predictions following the exact specification
        pred_class = logits.argmax(dim=-1)                                         # [B]
        alpha = pred_class.float() / 10.0                                          # [B]
        weights = torch.stack([alpha, 1.0 - alpha], dim=-1)                        # [B, 2]
        
        # Optional soft blending for debug
        probs = logits.softmax(dim=-1)
        alpha_soft = probs @ torch.linspace(0, 1, 11, device=z.device)             # [B]
        
        return {
            "logits": logits,
            "pred_class": pred_class,
            "alpha": alpha,
            "weights": weights,
            "alpha_soft": alpha_soft
        }


def print_histogram(title: str, classes: torch.Tensor, total_samples: int):
    """Print class frequency histogram to logs."""
    counts = torch.bincount(classes, minlength=11).cpu().numpy()
    _logger.info(f"  {title}:")
    for i, count in enumerate(counts):
        pct = (count / total_samples) * 100.0
        _logger.info(f"    Class {i:2d} (alpha={i/10.0:.1f}): {count:6d} ({pct:6.2f}%)")


def run_sanity_overfit(train_dataset: GatingDataset, device: torch.device):
    """Sanity test: overfit 32 samples to near-100% training accuracy."""
    _logger.info("\n=======================================================")
    _logger.info(" Starting Sanity Overfitting Test (32 samples)")
    _logger.info("=======================================================")
    
    # Extract the first 32 samples
    subset_z = train_dataset.z[:32].to(device)
    subset_targets = train_dataset.target_class[:32].to(device)
    
    # Initialize model, optimizer
    model = GatingClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # slightly higher LR for faster overfitting
    
    model.train()
    for step in range(1, 301):
        optimizer.zero_grad()
        outputs = model(subset_z)
        loss = F.cross_entropy(outputs["logits"], subset_targets)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == 1 or step == 300:
            pred = outputs["pred_class"]
            exact_acc = (pred == subset_targets).float().mean().item() * 100.0
            off_by_1 = ((pred - subset_targets).abs() <= 1).float().mean().item() * 100.0
            _logger.info(f"  Step {step:3d} | Loss: {loss.item():.4f} | Exact Acc: {exact_acc:6.2f}% | Off-by-1 Acc: {off_by_1:6.2f}%")
            
    # Final check
    final_pred = model(subset_z)["pred_class"]
    final_acc = (final_pred == subset_targets).float().mean().item() * 100.0
    _logger.info("=======================================================")
    if final_acc >= 95.0:
        _logger.info(f" SANITY PASS: Achieved final exact accuracy of {final_acc:.2f}%!")
    else:
        raise ValueError(f"SANITY FAIL: Expected near-100% exact accuracy, got {final_acc:.2f}%")
    _logger.info("=======================================================\n")


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, split_name: str):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            z = batch["z"].to(device)
            target_class = batch["target_class"].to(device)
            
            outputs = model(z)
            loss = F.cross_entropy(outputs["logits"], target_class, reduction="sum")
            
            total_loss += loss.item()
            total_samples += z.shape[0]
            
            all_preds.append(outputs["pred_class"])
            all_targets.append(target_class)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    exact_acc = (all_preds == all_targets).float().mean().item()
    off_by_1_acc = ((all_preds - all_targets).abs() <= 1).float().mean().item()
    alpha_mae = (all_preds.float() / 10.0 - all_targets.float() / 10.0).abs().mean().item()
    mean_loss = total_loss / total_samples
    
    _logger.info(f"\nEvaluation Results on Split: {split_name}")
    _logger.info(f"  Mean Loss:        {mean_loss:.4f}")
    _logger.info(f"  Exact Accuracy:   {exact_acc * 100.0:6.2f}%")
    _logger.info(f"  Off-by-1 Acc:     {off_by_1_acc * 100.0:6.2f}%")
    _logger.info(f"  Alpha MAE:        {alpha_mae:.4f}")
    
    # Print prediction and label distributions
    print_histogram("True Label Class Histogram", all_targets, total_samples)
    print_histogram("Predicted Class Histogram", all_preds, total_samples)
    _logger.info("-" * 60)
    
    return exact_acc, alpha_mae


def main():
    parser = argparse.ArgumentParser(description="Train Gating Classifier for Multi-Expert Scene Localization")
    parser.add_argument("--dataset_root", type=str, default="outputs/alpha_test/shopfacade", help="Path containing dataset .pt files")
    parser.add_argument("--scene", type=str, default="shopfacade", help="Name of the scene")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    
    dataset_dir = Path(args.dataset_root)
    train_path = dataset_dir / f"{args.scene}_dataset_train.pt"
    test_path = dataset_dir / f"{args.scene}_dataset_test.pt"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train or test dataset in {dataset_dir}")
        
    train_dataset = GatingDataset(train_path)
    test_dataset = GatingDataset(test_path)
    
    # 1. Run Sanity Overfit Test
    run_sanity_overfit(train_dataset, device)
    
    # 2. Main DataLoader Setup
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Main Model, Optimizer, and Loss
    model = GatingClassifier(d_model=args.d_model, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    _logger.info(f"Starting Main Training Loop for {args.epochs} epochs...")
    best_exact_acc = 0.0
    best_mae = 999.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        
        for batch in train_loader:
            z = batch["z"].to(device)
            target_class = batch["target_class"].to(device)
            
            optimizer.zero_grad()
            outputs = model(z)
            loss = F.cross_entropy(outputs["logits"], target_class)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_batches += 1
            
        mean_epoch_loss = epoch_loss / total_batches
        _logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {mean_epoch_loss:.4f}")
        
        # Evaluate on test set at the end of each epoch
        test_acc, test_mae = evaluate(model, test_loader, device, f"test_epoch_{epoch}")
        
        if test_acc > best_exact_acc:
            best_exact_acc = test_acc
            best_mae = test_mae
            
            # Save the best gating head
            model_save_path = dataset_dir / f"{args.scene}_gating_head_best.pt"
            torch.save(model.state_dict(), model_save_path)
            _logger.info(f"Saved new best gating head to {model_save_path}")
            
    _logger.info("\n=======================================================")
    _logger.info(" Training Complete!")
    _logger.info(f"  Best Test Exact Accuracy: {best_exact_acc * 100.0:.2f}%")
    _logger.info(f"  Best Test Alpha MAE:      {best_mae:.4f}")
    _logger.info("=======================================================")


if __name__ == "__main__":
    main()
