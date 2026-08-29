#!/bin/bash
# ASIC-AI: 35B Model Cloud Training — Quick Start
#
# Run on A100 (80GB) or H100:
#   MODEL=Qwen/Qwen3.6-35B-A3B EPOCHS=2 bash scripts/cloud/train_35b.sh
#
# Cost estimate:
#   - A100 80GB: ~$1.50/hr x 8-12hr = $12-18 (SFT)
#   - H100: ~$3.00/hr x 4-6hr = $12-18 (SFT)
#
# Verified locally with 0.5B model:
#   - 267 steps, loss 2.18 -> 0.0046 (99.8% reduction)
#   - Model learned tool-calling format (3/5 validation pass)
#
# Training data: 1032 examples (929 train + 103 val)
#   15 JSONL files covering full IC design flow

set -e

# Configuration
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
OUTPUT_DIR="outputs/cloud_sft_35b"
REPO="https://github.com/ozgurkazanci/AI_model.git"

echo "============================================================"
echo "   ASIC-AI 35B Cloud SFT Training"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  Data:       1032 examples (929 train + 103 val)"
echo "  Epochs:     $EPOCHS"
echo "  Batch:      $BATCH_SIZE x $GRAD_ACCUM = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "  LR:         $LR"
echo "  LoRA:       r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Max seq:    $MAX_SEQ_LEN"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# Step 1: Setup
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi || { echo "ERROR: No GPU found!"; exit 1; }
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU VRAM: ${VRAM}MB"
if [ "$VRAM" -lt 40000 ]; then
    echo "WARNING: 35B model needs >= 40GB VRAM. Consider quantized training."
fi

# Step 2: Clone repo
echo ""
echo "[2/6] Setting up repository..."
if [ ! -d "AI_model" ]; then
    git clone "$REPO"
fi
cd AI_model
git pull origin main

# Step 3: Install dependencies
echo ""
echo "[3/6] Installing dependencies..."
pip install -r requirements.txt
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not available, using default attention"
pip install bitsandbytes  # For QLoRA if needed

# Step 4: Verify data
echo ""
echo "[4/6] Verifying training data..."
PYTHONPATH=src python -c "
import json
from pathlib import Path
train = list(Path('data/sft/train_final.jsonl').open())
val = list(Path('data/sft/val_final.jsonl').open())
print(f'  Train: {len(train)} examples')
print(f'  Val:   {len(val)} examples')
print(f'  Total: {len(train) + len(val)} examples')
"

# Step 5: Run training
echo ""
echo "[5/6] Starting SFT training..."
echo "  Estimated time: 8-12 hours on A100, 4-6 hours on H100"
echo ""

PYTHONPATH=src python scripts/finetune_local.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --save-steps 100 \
    --logging-steps 20 \
    --output-dir "$OUTPUT_DIR"

# Step 6: Validate and push
echo ""
echo "[6/6] Post-training validation..."
PYTHONPATH=src python scripts/benchmark_model.py --model "$OUTPUT_DIR/final"
PYTHONPATH=src python scripts/validate_trained_model.py --model "$OUTPUT_DIR/final"

echo ""
echo "============================================================"
echo "   Training Complete!"
echo "============================================================"
echo "  Model: $OUTPUT_DIR/final"
echo ""
echo "  Next steps:"
echo "  1. Check benchmark results: eval_results/benchmark.json"
echo "  2. Run GRPO: PYTHONPATH=src python scripts/grpo_ngspice.py --model $OUTPUT_DIR/final"
echo "  3. Merge LoRA: PYTHONPATH=src python scripts/merge_lora.py --model $OUTPUT_DIR/final"
echo "  4. Export GGUF: PYTHONPATH=src python scripts/export_gguf.py --model $OUTPUT_DIR/merged"
echo "  5. Push to HuggingFace"
echo "============================================================"

# Push results
git add -A
git commit -m "Cloud training results: 35B SFT complete" || true
git push origin main || true
