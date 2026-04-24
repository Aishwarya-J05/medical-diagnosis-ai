# src/data/dicom_loader.py
"""
Image loading and preprocessing pipeline for chest X-rays.
Handles both DICOM (.dcm) and standard formats (.jpeg, .png).
All images are normalized and converted to 3-channel tensors.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import torch


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load image from disk — handles both DICOM and standard formats.
    Returns raw pixel array as numpy array (H, W).
    """
    path = Path(image_path)

    if path.suffix.lower() == ".dcm":
        import pydicom
        dcm = pydicom.dcmread(str(path))
        image = dcm.pixel_array.astype(np.float32)
    else:
        # Standard formats: JPEG, PNG
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        image = image.astype(np.float32)

    return image


def apply_windowing(
    image: np.ndarray,
    window_width: float = 1500,
    window_center: float = -600
) -> np.ndarray:
    """
    Apply radiological windowing to control contrast.
    Chest window default: WW=1500, WC=-600.
    Brain window: WW=80, WC=40.
    Clips pixel values to the window range then normalizes to [0, 1].
    """
    lower = window_center - (window_width / 2)
    upper = window_center + (window_width / 2)
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image.astype(np.float32)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Min-max normalize image to [0, 1].
    Used for standard JPEG/PNG X-rays that don't need windowing.
    """
    min_val = image.min()
    max_val = image.max()

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)

    return ((image - min_val) / (max_val - min_val)).astype(np.float32)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Critical for X-rays — enhances local contrast to reveal subtle abnormalities.
    clip_limit=2.0, tile_size=8x8 are standard for chest X-rays.
    """
    # CLAHE works on uint8
    image_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_uint8)
    return (enhanced / 255.0).astype(np.float32)


def to_3channel(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale (H, W) to 3-channel (H, W, 3).
    Required because ImageNet-pretrained backbones expect RGB input.
    """
    return np.stack([image, image, image], axis=-1)


def resize(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target size using area interpolation (best for downscaling)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_xray(
    image_path: str | Path,
    size: Tuple[int, int] = (224, 224),
    is_dicom: bool = False
) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single chest X-ray.

    Pipeline:
        load → normalize/window → CLAHE → resize → 3-channel → tensor

    Args:
        image_path: Path to image file
        size: Target (H, W) for resize
        is_dicom: If True, applies radiological windowing instead of min-max norm

    Returns:
        torch.Tensor of shape (3, H, W), float32, values in [0, 1]
    """
    image = load_image(image_path)

    if is_dicom:
        image = apply_windowing(image)
    else:
        image = normalize(image)

    image = apply_clahe(image)
    image = resize(image, size)
    image = to_3channel(image)

    # (H, W, 3) → (3, H, W) for PyTorch
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor


if __name__ == "__main__":
    # Quick sanity check — run: python src/data/dicom_loader.py
    import sys
    from pathlib import Path

    test_image = Path("data/raw/chest_xray/train/NORMAL").glob("*.jpeg").__next__()
    print(f"Testing with: {test_image}")

    tensor = preprocess_xray(test_image)
    print(f"Output shape: {tensor.shape}")       # Expected: torch.Size([3, 224, 224])
    print(f"dtype: {tensor.dtype}")              # Expected: torch.float32
    print(f"Min: {tensor.min():.4f}, Max: {tensor.max():.4f}")  # Expected: ~0.0, ~1.0
    print("Preprocessing pipeline OK")