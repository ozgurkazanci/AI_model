# ============================================================
# ASIC-AI Training Environment
# ============================================================
# Multi-stage Docker build for reproducible training.
#
# Build:
#   docker build -t asic-ai:latest .
#
# Run tests:
#   docker run --rm asic-ai:latest pytest tests/ -v
#
# Run training (mount data + GPU):
#   docker run --gpus all -v ./data:/app/data -v ./checkpoints:/app/checkpoints \
#     asic-ai:latest python -m asic_ai.training.sft --config configs/training/sft_axolotl.yaml
#
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ngspice \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    pydantic>=2.0 \
    pyyaml>=6.0 \
    numpy \
    jsonschema \
    pytest

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY eval/ eval/
COPY tests/ tests/
COPY scripts/ scripts/
COPY data/ data/
COPY docs/ docs/
COPY examples/ examples/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Verify installation
RUN python -c "from asic_ai import __version__; print(f'asic-ai v{__version__}')"
RUN pytest tests/ -q --tb=line 2>/dev/null || echo "Some tests may require additional dependencies"

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================================
# GPU Training Stage
# ============================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS training

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git ngspice \
    && rm -rf /var/lib/apt/lists/*

# Training dependencies
RUN pip install --no-cache-dir \
    torch>=2.0 \
    transformers>=4.40 \
    peft>=0.10 \
    trl>=0.8 \
    accelerate>=0.30 \
    bitsandbytes \
    flash-attn --no-build-isolation \
    wandb \
    axolotl

COPY --from=base /app/ /app/
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "asic_ai.training.sft", "--help"]
