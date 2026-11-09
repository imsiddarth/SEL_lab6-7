# Multi-stage/lean production Dockerfile for ML Inference API
FROM python:3.11-slim

# Avoid writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and trained model artifact
COPY configs/ ./configs/
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Health check instruction inside container
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

