# src/data/dataset.py
"""
PyTorch Dataset for chest X-ray classification.
Supports binary classification (NORMAL vs PNEUMONIA).
Handles train/val/test splits with augmentation for training only.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.data.dicom_loader import preprocess_xray


# Label mapping — extend this when moving to multi-label NIH dataset
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}


def get_transforms(mode: str = "train") -> A.Compose:
    """
    Augmentation pipeline.
    Train: random flips, rotation, brightness/contrast shifts.
    Val/Test: no augmentation — deterministic preprocessing only.

    Why these augmentations:
    - HorizontalFlip: X-rays can be mirrored without clinical meaning change
    - Rotation ±10°: handles slight patient positioning variation
    - Brightness/Contrast: simulates different scanner settings
    - NO vertical flip — lungs have anatomical orientation
    """
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ])
    else:
        # No augmentation for val/test — pure evaluation
        return A.Compose([])


class ChestXrayDataset(Dataset):
    """
    Dataset for binary chest X-ray classification.

    Directory structure expected:
        root/
            NORMAL/   ← label 0
            PNEUMONIA/ ← label 1

    Args:
        root_dir: Path to split directory (e.g., data/raw/chest_xray/train)
        mode: 'train', 'val', or 'test' — controls augmentation
        image_size: Target (H, W) for resize
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_size = image_size
        self.transforms = get_transforms(mode)

        # Build flat list of (image_path, label) pairs
        self.samples = self._load_samples()
        print(f"[{mode}] Loaded {len(self.samples)} samples | "
              f"NORMAL: {sum(1 for _, l in self.samples if l == 0)} | "
              f"PNEUMONIA: {sum(1 for _, l in self.samples if l == 1)}")

    def _load_samples(self) -> list[Tuple[Path, int]]:
        """Scan directory and build (path, label) list."""
        samples = []
        for label_name, label_idx in LABELS.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                raise FileNotFoundError(f"Label directory not found: {label_dir}")

            for img_path in sorted(label_dir.glob("*")):
                if img_path.suffix.lower() in {".jpeg", ".jpg", ".png", ".dcm"}:
                    samples.append((img_path, label_idx))

        if len(samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Step 1: Preprocess (normalize, CLAHE, resize)
        tensor = preprocess_xray(img_path, size=self.image_size)

        # Step 2: Apply augmentation (train only)
        # Convert back to numpy for albumentations, then back to tensor
        if self.mode == "train":
            img_np = tensor.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
            augmented = self.transforms(image=img_np)
            img_np = augmented["image"]
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H,W,C) → (C,H,W)

        return tensor, torch.tensor(label, dtype=torch.long)


def get_dataloaders(
    data_dir: str | Path = "data/raw/chest_xray",
    batch_size: int = 32,
    num_workers: int = 0,  # 0 for Windows — avoids multiprocessing issues
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Args:
        data_dir: Root directory containing train/val/test folders
        batch_size: Images per batch — reduce to 16 if OOM on 4GB VRAM
        num_workers: Parallel workers — keep 0 on Windows
        image_size: Target image size

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_dataset = ChestXrayDataset(data_dir / "train", mode="train", image_size=image_size)
    val_dataset   = ChestXrayDataset(data_dir / "val",   mode="val",   image_size=image_size)
    test_dataset  = ChestXrayDataset(data_dir / "test",  mode="test",  image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle train only
        num_workers=num_workers,
        pin_memory=True         # faster CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Sanity check — verifies dataset loads correctly
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")        # Expected: torch.Size([8, 3, 224, 224])
    print(f"Labels: {labels}")                   # Expected: tensor of 0s and 1s
    print(f"Image dtype: {images.dtype}")        # Expected: torch.float32
    print(f"Min/Max: {images.min():.3f} / {images.max():.3f}")
    print("Dataset pipeline OK")