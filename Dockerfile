# DPI-600 Drug Logo Recognition - Training Docker Image
# =====================================================
# Multi-stage build for optimized image size

# Stage 1: Base with CUDA support
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Stage 2: Training environment
FROM base AS training

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train_model.py .
COPY inference.py .
COPY mock_dataset_generator.py .
COPY azure_ml_pipeline.py .

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs

# Default command
CMD ["python", "train_model.py", "--help"]

# =====================================================
# Stage 3: Inference-only (smaller image)
FROM base AS inference

WORKDIR /app

# Install minimal packages for inference
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
        Pillow \
        flask \
        gunicorn \
        onnxruntime-gpu

# Copy only inference scripts
COPY inference.py .
COPY train_model.py .

# Create model directory
RUN mkdir -p /app/models

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run API server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "inference:create_api('/app/models/best_model.pth')"]

# =====================================================
# Usage:
# 
# Build training image:
#   docker build --target training -t dpi600-training .
#
# Build inference image:
#   docker build --target inference -t dpi600-inference .
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#       dpi600-training python train_model.py -d /app/data -o /app/models
#
# Run inference API:
#   docker run --gpus all -p 5000:5000 -v $(pwd)/models:/app/models \
#       dpi600-inference
# =====================================================
