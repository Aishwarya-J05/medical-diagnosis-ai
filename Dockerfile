FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_hf.txt .

# Install PyTorch CPU first, then rest
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY src/ ./src/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/
COPY frontend/dist/ ./static/

RUN mkdir -p outputs

EXPOSE 7860

ENV PYTHONPATH=/app
ENV PORT=7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]