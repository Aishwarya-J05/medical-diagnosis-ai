# src/training/train_classifier.py
"""
Training loop for chest X-ray binary classifier.
Includes: mixed precision training, early stopping,
checkpoint saving, and MLflow experiment tracking.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import get_dataloaders
from src.models.classifier import build_classifier


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "data_dir":       "data/raw/chest_xray",
    "num_classes":    2,
    "image_size":     (224, 224),
    "batch_size":     16,          # 16 for 4GB VRAM — increase to 32 if no OOM
    "num_epochs":     20,
    "learning_rate":  1e-4,
    "weight_decay":   1e-4,
    "dropout_rate":   0.5,
    "freeze_epochs":  5,           # train head only for first 5 epochs
    "patience":       5,           # early stopping patience
    "checkpoint_dir": "checkpoints",
    "experiment_name": "chest-xray-classifier",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when val loss stops improving.
    Saves best model checkpoint automatically.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_auc: float,
    path: str
) -> None:
    """Save model checkpoint with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":   val_loss,
        "val_auc":    val_auc,
    }, path)
    print(f"Checkpoint saved → {path}")


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """
    Single training epoch with mixed precision.
    Returns (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass — cuts VRAM usage ~40%
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluation loop — no gradients, no augmentation.
    Returns (avg_loss, accuracy, auc_roc).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # AUC-ROC — primary metric for medical imaging (not accuracy)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"]
    )

    # Model — start with frozen backbone
    model = build_classifier(
        num_classes=CONFIG["num_classes"],
        freeze_backbone=True,
        device=device
    )

    # Loss — CrossEntropy for binary classification
    # Note: dataset is imbalanced (3875 pneumonia vs 1341 normal)
    # We handle this with class weights
    class_weights = torch.tensor([3875 / 1341, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer — only head params initially (backbone frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # LR scheduler — cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"]
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Early stopping
    early_stopper = EarlyStopping(patience=CONFIG["patience"])

    best_auc = 0.0

    # MLflow tracking
    mlflow.set_experiment(CONFIG["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        for epoch in range(1, CONFIG["num_epochs"] + 1):
            print(f"\nEpoch [{epoch}/{CONFIG['num_epochs']}]")
            start = time.time()

            # Unfreeze backbone after freeze_epochs
            if epoch == CONFIG["freeze_epochs"] + 1:
                model.unfreeze_backbone()
                # Reset optimizer to include all params
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=CONFIG["learning_rate"] / 10,  # lower LR for fine-tuning
                    weight_decay=CONFIG["weight_decay"]
                )
                print("Switched to full fine-tuning mode")

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )

            # Validate
            val_loss, val_acc, val_auc = evaluate(
                model, val_loader, criterion, device
            )

            scheduler.step()
            elapsed = time.time() - start

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            print(f"Time: {elapsed:.1f}s")

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_auc":    val_auc,
            }, step=epoch)

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_auc,
                    path=f"{CONFIG['checkpoint_dir']}/best_model.pth"
                )

            # Early stopping
            if early_stopper(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_auc = evaluate(
            model, test_loader, criterion, device
        )
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc":  test_acc,
            "test_auc":  test_auc,
        })

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    train()