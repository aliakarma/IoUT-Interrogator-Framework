FROM python:3.9-slim

LABEL maintainer="akarma@iu.edu.sa"
LABEL description="IoUT Interrogator Framework — reproducible research environment"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository contents
COPY . .

# Create output directories
RUN mkdir -p data/raw data/processed data/sample \
             simulation/outputs \
             model/checkpoints \
             analysis/plots \
             analysis/stats

# Set environment variables
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Default command: run the full reproducibility pipeline
CMD ["python", "scripts/run_full_pipeline.py", "--seed", "42", "--runs", "30"]
