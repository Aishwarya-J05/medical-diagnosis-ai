# src/utils/gradcam.py
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Produces heatmaps showing which regions of the X-ray influenced the model's decision.
Critical for clinical trust — radiologists need to see WHY the model predicted what it did.

Paper: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for ResNet50.

    How it works:
    1. Forward pass → get predictions
    2. Backward pass on target class score → get gradients at target layer
    3. Global average pool the gradients → importance weights per channel
    4. Weighted sum of feature maps → raw heatmap
    5. ReLU + normalize → final heatmap overlaid on original image

    Args:
        model: Trained ChestXrayClassifier
        target_layer: Layer to hook into — last conv layer of ResNet
                      'backbone.7' = layer4 of ResNet50 (best for Grad-CAM)
    """

    def __init__(self, model: nn.Module, target_layer: str = "backbone.7"):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks on target layer
        target = self._get_layer(target_layer)
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _get_layer(self, layer_name: str) -> nn.Module:
        """Navigate model to get target layer by dot-notation name."""
        layer = self.model
        for part in layer_name.split("."):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _save_activation(self, module, input, output) -> None:
        """Forward hook — saves feature maps at target layer."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Backward hook — saves gradients at target layer."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class to explain. If None, uses predicted class.

        Returns:
            heatmap: np.ndarray (H, W), values in [0, 1]
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward on target class score only
        score = logits[0, target_class]
        score.backward()

        # Global average pool gradients → (channels,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)  # only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            original_image: (H, W, 3) uint8 RGB image
            heatmap: (H, W) float32 in [0, 1]
            alpha: Heatmap transparency — 0.4 is standard

        Returns:
            overlaid: (H, W, 3) uint8 image
        """
        # Resize heatmap to match image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap (jet: blue=low, red=high activation)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # (H, W, 3), float
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend with original
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid


def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    save_path: Optional[str] = None,
    class_names: list = ["NORMAL", "PNEUMONIA"]
) -> None:
    """
    Full Grad-CAM visualization pipeline for a single image.
    Saves or displays: original | heatmap | overlay side by side.

    Args:
        model: Trained classifier
        image_path: Path to input X-ray image
        save_path: Where to save the visualization. None = display only.
        class_names: Label names for display
    """
    from src.data.dicom_loader import preprocess_xray

    # Preprocess
    tensor = preprocess_xray(image_path)

    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(tensor)

    # Get prediction
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    # Load original image for display
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    # Generate overlay
    overlay = gradcam.overlay_on_image(original, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPred: {class_names[pred_class]} ({confidence:.2%})"
    )
    axes[2].axis("off")

    plt.suptitle(f"Grad-CAM Explanation — {Path(image_path).name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from src.models.classifier import build_classifier

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(num_classes=2, device=device)

    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint — Val AUC: {checkpoint['val_auc']:.4f}")

    # Test on one PNEUMONIA image
    # test_image = Path("data/raw/chest_xray/test/PNEUMONIA/person1660_virus_2869.jpeg")
    # test_image = Path("data/raw/chest_xray/test/NORMAL/IM-0023-0001.jpeg")
    test_image = next(Path("data/raw/chest_xray/test/PNEUMONIA").glob("*.jpeg"))
    print(f"Testing on: {test_image}")

    visualize_gradcam(
        model=model,
        image_path=str(test_image),
        save_path="outputs/gradcam_sample.png"
    )