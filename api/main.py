# api/main.py
"""
FastAPI backend for chest X-ray diagnosis.
Endpoints:
    POST /analyze  — upload X-ray, get prediction + Grad-CAM heatmap
    GET  /health   — health check
    GET  /metrics  — basic inference stats
"""

import io
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.classifier import build_classifier
from src.data.dicom_loader import preprocess_xray
from src.utils.gradcam import GradCAM

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Global model state ────────────────────────────────────────────────────────

class ModelRegistry:
    """Holds loaded model + device. Loaded once at startup."""
    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    gradcam: Optional[GradCAM] = None
    inference_count: int = 0
    total_latency_ms: float = 0.0


registry = ModelRegistry()


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_classifier(num_classes=2, device=device)
    
    checkpoint_path = Path("checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    registry.model = model
    registry.device = device
    registry.gradcam = GradCAM(model)
    
    logger.info(f"Model loaded on {device} | Val AUC: {checkpoint['val_auc']:.4f}")
    
    yield  # App runs here
    
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Diagnosis AI",
    description="Chest X-ray analysis with explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str            # "NORMAL" or "PNEUMONIA"
    confidence: float          # 0.0 - 1.0
    probabilities: dict        # {"NORMAL": 0.3, "PNEUMONIA": 0.7}
    heatmap_base64: str        # base64 encoded PNG of Grad-CAM overlay
    inference_time_ms: float   # latency for this request


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_inferences: int
    avg_latency_ms: float


# ── Utilities ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def decode_image(file_bytes: bytes) -> str:
    """
    Save uploaded bytes to a temp file and return path.
    Handles JPEG, PNG. DICOM support can be added later.
    """
    temp_path = Path("outputs/temp_upload.jpg")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    return str(temp_path)


def generate_heatmap_base64(
    image_path: str,
    heatmap: np.ndarray
) -> str:
    """
    Overlay heatmap on original image and encode as base64 PNG.
    Frontend can render directly: <img src="data:image/png;base64,...">
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Load and resize original
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    # Encode to base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — used by load balancer and monitoring."""
    return HealthResponse(
        status="ok",
        device=str(registry.device),
        model_loaded=registry.model is not None
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Basic inference statistics."""
    avg_latency = (
        registry.total_latency_ms / registry.inference_count
        if registry.inference_count > 0 else 0.0
    )
    return MetricsResponse(
        total_inferences=registry.inference_count,
        avg_latency_ms=round(avg_latency, 2)
    )


@app.post("/analyze", response_model=PredictionResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Main inference endpoint.
    
    Accepts: JPEG or PNG chest X-ray image
    Returns: prediction, confidence, probabilities, Grad-CAM heatmap (base64)
    """
    # Validate file type
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Read uploaded file
        file_bytes = await file.read()
        image_path = decode_image(file_bytes)

        # Preprocess
        tensor = preprocess_xray(image_path)

        # Inference
        with torch.no_grad():
            logits = registry.model(
                tensor.unsqueeze(0).to(registry.device)
            )
            probs = torch.softmax(logits, dim=1)[0]

        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

        # Grad-CAM
        heatmap = registry.gradcam.generate(tensor, target_class=pred_class)
        heatmap_b64 = generate_heatmap_base64(image_path, heatmap)

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        registry.inference_count += 1
        registry.total_latency_ms += latency_ms

        logger.info(
            f"Inference | pred={CLASS_NAMES[pred_class]} | "
            f"conf={confidence:.3f} | latency={latency_ms:.1f}ms"
        )

        return PredictionResponse(
            prediction=CLASS_NAMES[pred_class],
            confidence=round(confidence, 4),
            probabilities={
                "NORMAL": round(probs[0].item(), 4),
                "PNEUMONIA": round(probs[1].item(), 4)
            },
            heatmap_base64=heatmap_b64,
            inference_time_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # never use reload=True with GPU models
    )