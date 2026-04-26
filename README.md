# Medical Diagnosis AI

A production-grade AI-powered chest X-ray diagnostic system with explainable AI, built with ResNet50, Grad-CAM, FastAPI, and React.

## Live Demo

**[Launch App on HuggingFace](https://aishwaryanj-medical-diagnosis-ai.hf.space)**

---

## Overview

This system assists radiologists by automatically analyzing chest X-rays, detecting pathologies, and generating explainable visual heatmaps showing which regions of the image influenced the model's decision.

---

## Features

- **Multi-class Classification** — Detects Normal vs Pneumonia with 88.6% accuracy and 0.943 AUC-ROC
- **Grad-CAM Explainability** — Visual heatmaps highlighting regions that influenced the prediction
- **Diagnosis History** — All analyses stored in Supabase with full audit trail
- **Statistics Dashboard** — Aggregate metrics across all diagnoses with interactive charts
- **Production API** — FastAPI backend with health checks, audit logging, and inference metrics
- **Glassmorphism UI** — Professional dark teal themed React frontend

---

## Tech Stack

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

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 88.6% |
| Test AUC-ROC | 0.943 |
| Avg Inference Time (CPU) | ~400ms |
| Training Epochs | 6 (early stopped) |

Trained on the Chest X-Ray Pneumonia dataset (5,216 train / 624 test images).

---

## Architecture

X-Ray Image → DICOM Preprocessing → ResNet50 Backbone
↓
Multi-label Head → Prediction + Confidence
↓
Grad-CAM → Heatmap Overlay
↓
FastAPI Backend → React Frontend
↓
Supabase (Diagnosis History)

---

## Project Structure

\```
medical-diagnosis-ai/
├── src/
│   ├── data/
│   │   ├── dicom_loader.py      # DICOM preprocessing pipeline
│   │   └── dataset.py           # PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── classifier.py        # ResNet50 classifier
│   ├── training/
│   │   └── train_classifier.py  # Training loop with MLflow tracking
│   └── utils/
│       └── gradcam.py           # Grad-CAM explainability
├── api/
│   └── main.py                  # FastAPI backend
├── frontend/
│   └── src/
│       ├── App.jsx              # Main React app
│       └── supabaseClient.js    # Supabase client
├── Dockerfile                   # HuggingFace deployment
└── requirements_hf.txt          # Linux-compatible dependencies
\```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 22+
- NVIDIA GPU (optional, CPU works)

### Backend

```bash
# Clone repo
git clone https://github.com/Aishwarya-J05/medical-diagnosis-ai.git
cd medical-diagnosis-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_hf.txt

# Run API
set PYTHONPATH=.
python api/main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Dataset

- **Chest X-Ray Pneumonia** (Kaggle) — 5,216 training images, 624 test images
- Classes: NORMAL, PNEUMONIA (bacterial + viral)

---

## Roadmap

- [ ] Multi-label classification on NIH ChestX-ray14 (14 pathologies)
- [ ] MRI brain tumor segmentation with Mask R-CNN
- [ ] Automated clinical report generation with BioGPT
- [ ] DICOM viewer integration with Cornerstone.js
- [ ] LLM-as-a-judge report quality evaluation

---

## Industry Context

Similar systems are deployed by:
- **Siemens Healthineers** — AI-Rad Companion for chest CT analysis
- **Mayo Clinic** — Multimodal AI foundation models for radiology
- **The Queen's Health Systems** — AI-powered MRI, CT, and X-ray workflows

---

## License

MIT License — free for research and educational use.

---

Built by [Aishwarya Joshi](https://github.com/Aishwarya-J05)
