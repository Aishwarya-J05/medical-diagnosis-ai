# src/models/classifier.py
"""
ResNet50-based chest X-ray classifier.
Uses ImageNet pretrained weights with a custom classification head.
Supports binary classification (NORMAL vs PNEUMONIA) and 
multi-label (for NIH ChestX-ray14 upgrade path).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ChestXrayClassifier(nn.Module):
    """
    ResNet50 backbone with custom head for chest X-ray classification.

    Architecture:
        ResNet50 (pretrained) → GlobalAvgPool → Dropout → FC → Output

    Args:
        num_classes: Number of output classes
                     2 for binary, 14 for NIH multi-label
        pretrained: Use ImageNet weights (always True in production)
        dropout_rate: Regularization — 0.5 is standard for medical imaging
        freeze_backbone: If True, only train the head (faster, less data needed)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC layer — we replace it with our own head
        # backbone.fc was: Linear(2048, 1000) for ImageNet
        in_features = backbone.fc.in_features  # 2048

        # Keep everything except the final FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output of backbone: (batch, 2048, 1, 1)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (batch, 2048)
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
            # No softmax here — handled by loss function
            # BCEWithLogitsLoss or CrossEntropyLoss expects raw logits
        )

        # Optionally freeze backbone — useful when dataset is small
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, 3, H, W)
        Returns:
            logits: (batch, num_classes) — raw scores before activation
        """
        features = self.backbone(x)      # (batch, 2048, 1, 1)
        logits = self.classifier(features)  # (batch, num_classes)
        return logits

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze backbone for full fine-tuning.
        Call this after initial head training converges.
        Standard practice: train head for 5 epochs, then unfreeze all.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — full fine-tuning enabled")

    def get_feature_extractor(self) -> nn.Sequential:
        """Returns backbone only — used by Grad-CAM for feature extraction."""
        return self.backbone


def build_classifier(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> ChestXrayClassifier:
    """
    Factory function — builds and moves model to correct device.

    Args:
        num_classes: 2 for binary, 14 for multi-label NIH
        freeze_backbone: Freeze ResNet layers, train head only
        device: Target device — auto-detects GPU if None

    Returns:
        Model on correct device, ready for training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChestXrayClassifier(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.5,
        freeze_backbone=freeze_backbone
    )

    model = model.to(device)
    print(f"Model loaded on: {device}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    return model


if __name__ == "__main__":
    # Sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_classifier(num_classes=2, freeze_backbone=False)

    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")   # (4, 3, 224, 224)
    print(f"Output shape: {output.shape}")         # (4, 2)
    print(f"Output (logits): {output}")
    print("Classifier OK")