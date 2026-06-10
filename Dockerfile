# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Runtime system libraries needed by OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model artifacts
COPY src/ src/
COPY api/ api/
COPY model_artifacts/ model_artifacts/

# App Runner expects the container to listen on PORT (default 8080).
# Torchvision needs HOME/TORCH_HOME for pretrained ResNet weight caching.
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    MODEL_DEVICE=cpu \
    HOME=/app \
    TORCH_HOME=/app/.cache/torch

EXPOSE 8080

# Cache pretrained ResNet weights during the image build so cold starts do not
# depend on runtime downloads, then run as a non-root user.
RUN mkdir -p /app/.cache/torch \
    && python -c "from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.DEFAULT)" \
    && useradd --no-create-home --home-dir /app --shell /bin/false appuser \
    && chown -R appuser:appuser /app/.cache
USER appuser

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
