#!/bin/bash
# ASIC-AI Cloud Training Script
# Config: SFT - Large Model (35B MoE, 3B active)
# Provider: lambda
# GPU: a100_80gb (min 45GB VRAM)
# Estimated time: 8.0 hours
# Estimated cost: ~$14.00

set -e

echo "=== ASIC-AI Training Setup ==="

# 1. Clone repository
git clone https://github.com/ozgurkazanci/AI_model.git
cd AI_model

# 2. Install dependencies
pip install -e ".[train,dev]"
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn skipped"

# 3. Verify environment
echo "=== Environment Check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 4. Run tests
PYTHONPATH=src pytest tests/ -q --tb=short

# 5. Validate training data
echo "=== Training Data ==="
PYTHONPATH=src python scripts/analyze_sft_data.py

# 6. Launch training
echo "=== Starting Training ==="

# SFT Training with Axolotl
pip install axolotl

# Merge all SFT data
cat data/sft/mock_analog_v1.jsonl data/sft/mock_digital_v1.jsonl data/sft/augmented_v1.jsonl > data/sft/combined_train.jsonl
echo "Combined training data: $(wc -l < data/sft/combined_train.jsonl) examples"

# Run SFT
accelerate launch -m axolotl.cli.train configs/training/sft_axolotl.yaml \
    --base_model=Qwen/Qwen3.6-35B-A3B \
    --datasets.0.path=data/sft/combined_train.jsonl \
    --output_dir=outputs/sft \
    --num_epochs=3 \
    --micro_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --sequence_len=8192 \
    --lora_r=32

echo "=== SFT Training Complete ==="
echo "Output: outputs/sft/"

# 7. Upload results
echo "=== Uploading Results ==="
# Option A: Push to HuggingFace
# huggingface-cli upload ozgurkazanci/asic-ai-v1 outputs/sft/

# Option B: Copy to persistent storage
# cp -r outputs/ /workspace/outputs/

echo "=== All Done ==="
