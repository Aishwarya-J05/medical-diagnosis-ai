# 🫁 Medical Diagnosis AI

> A production-grade AI-powered chest X-ray diagnostic system with explainable AI

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://aishwaryanj-medical-diagnosis-ai.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Aishwarya-J05/medical-diagnosis-ai)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?logo=supabase&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This system assists radiologists by automatically analyzing chest X-rays, detecting pathologies, and generating explainable visual heatmaps showing which regions of the image influenced the model's decision.

Built as a full-stack MLOps project — from raw DICOM preprocessing to cloud deployment.

---

## 🚀 Live Demo

**[Launch App → aishwaryanj-medical-diagnosis-ai.hf.space](https://aishwaryanj-medical-diagnosis-ai.hf.space)**

---

## ✨ Features

- **Binary Classification** — Detects Normal vs Pneumonia with 88.6% accuracy and 0.943 AUC-ROC
- **Grad-CAM Explainability** — Visual heatmaps highlighting regions that influenced the prediction
- **Diagnosis History** — All analyses stored in Supabase with full audit trail
- **Statistics Dashboard** — Aggregate metrics with interactive bar charts
- **Production API** — FastAPI backend with health checks, audit logging, and inference metrics
- **Glassmorphism UI** — Professional dark teal themed React frontend with Framer Motion animations

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | ResNet50 (ImageNet pretrained, fine-tuned) |
| Explainability | Grad-CAM |
| Training | PyTorch 2.6, MONAI, Albumentations |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, Framer Motion, Recharts, Lucide |
| Database | Supabase (PostgreSQL) |
| Deployment | HuggingFace Docker Space |
| Experiment Tracking | MLflow |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 88.6% |
| Test AUC-ROC | 0.943 |
| Avg Inference Time (CPU) | ~400ms |
| Training Epochs | 6 (early stopped) |
| Dataset | Chest X-Ray Pneumonia — 5,216 train / 624 test |

---

## 🏗️ Architecture

```
X-Ray Image
    ↓
DICOM Preprocessing (windowing, CLAHE, normalization)
    ↓
ResNet50 Backbone (ImageNet pretrained)
    ↓
Classification Head → Prediction + Confidence Score
    ↓
Grad-CAM → Activation Heatmap Overlay
    ↓
FastAPI Backend (port 7860)
    ↓
React Frontend → Supabase (Diagnosis History)
```

---

## 📁 Project Structure

```
medical-diagnosis-ai/
│
├── src/                              # Core ML source code
│   ├── __init__.py
│   ├── data/
│   │   ├── dicom_loader.py           # DICOM preprocessing pipeline
│   │   └── dataset.py               # PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── classifier.py            # ResNet50 classifier
│   ├── training/
│   │   └── train_classifier.py      # Training loop with MLflow tracking
│   └── utils/
│       ├── gradcam.py               # Grad-CAM explainability
│       ├── metrics.py               # AUC-ROC metrics
│       └── windowing.py             # DICOM windowing utilities
│
├── api/
│   └── main.py                      # FastAPI backend
│
├── frontend/
│   └── src/
│       ├── App.jsx                  # Main React application
│       ├── App.css                  # Glassmorphism teal theme
│       ├── main.jsx                 # React entry point
│       ├── index.css                # Global styles
│       └── supabaseClient.js        # Supabase client
│
├── data/                            # Dataset (gitignored)
│   ├── raw/                         # Raw chest X-ray images
│   ├── processed/                   # Preprocessed tensors
│   └── splits/                      # Train/val/test CSVs
│
├── checkpoints/                     # Model weights (LFS tracked)
│   └── best_model.pth
│
├── outputs/                         # Generated outputs (gitignored)
│   └── gradcam_sample.png
│
├── Dockerfile                       # HuggingFace deployment
├── requirements_hf.txt              # Linux-compatible dependencies
├── requirements.txt                 # Full development dependencies
└── README.md
```

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.11+
- Node.js 22+
- NVIDIA GPU (optional — CPU works for inference)

### 1. Clone & Install Backend

```bash
git clone https://github.com/Aishwarya-J05/medical-diagnosis-ai.git
cd medical-diagnosis-ai

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip -p data/raw
```

### 3. Train the Model

```bash
set PYTHONPATH=.
python src/training/train_classifier.py
```

### 4. Run the API

```bash
set PYTHONPATH=.
python api/main.py
```

API runs at **http://localhost:8000**

### 5. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns model status and device |
| GET | `/metrics` | Inference statistics — total runs, avg latency |
| POST | `/analyze` | Main endpoint — upload X-ray, get prediction + heatmap |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@chest_xray.jpeg"
```

### Example Response

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.573,
  "probabilities": {
    "NORMAL": 0.427,
    "PNEUMONIA": 0.573
  },
  "heatmap_base64": "...",
  "inference_time_ms": 371.2
}
```

---

## 🗄️ Database Schema

Supabase table: `diagnoses`

| Column | Type | Description |
|---|---|---|
| id | uuid | Primary key |
| created_at | timestamptz | Timestamp |
| filename | text | Uploaded filename |
| prediction | text | NORMAL or PNEUMONIA |
| confidence | float4 | Model confidence 0-1 |
| normal_prob | float4 | Normal class probability |
| pneumonia_prob | float4 | Pneumonia class probability |
| heatmap_base64 | text | Base64 Grad-CAM overlay |
| inference_time_ms | float4 | Latency in milliseconds |

---

## 🗺️ Roadmap

- [x] Binary classification (Normal vs Pneumonia)
- [x] Grad-CAM explainability heatmaps
- [x] FastAPI backend with audit logging
- [x] React frontend with glassmorphism design
- [x] Supabase diagnosis history
- [x] HuggingFace Docker deployment

---

## 🏥 Industry Context

Similar systems are deployed in production by:

- **Siemens Healthineers** — AI-Rad Companion for chest CT analysis and Deep Resolve AI for MRI
- **Mayo Clinic** — Multimodal AI foundation models integrating text and X-ray images
- **The Queen's Health Systems** — AI-powered MRI, CT, PET, and X-ray diagnostic workflows

---

## 📄 License

MIT License — free for research and educational use.

---

## 👩‍💻 Author

**Aishwarya Joshi**
- GitHub: [@Aishwarya-J05](https://github.com/Aishwarya-J05)
- HuggingFace: [@AishwaryaNJ](https://huggingface.co/AishwaryaNJ)
