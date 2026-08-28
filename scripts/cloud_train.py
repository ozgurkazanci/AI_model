#!/usr/bin/env python3
"""Cloud GPU training launcher and cost estimator.

Generates ready-to-run training scripts for cloud GPU providers.

Supported providers:
- RunPod (A100/H100)
- Lambda Labs (A100/H100)
- Google Colab Pro+ (A100)

Usage:
    PYTHONPATH=src python scripts/cloud_train.py --estimate
    PYTHONPATH=src python scripts/cloud_train.py --provider runpod --generate-script
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_train")

# GPU specs for cost estimation
GPU_SPECS = {
    "a100_40gb": {"vram_gb": 40, "tflops_bf16": 312, "cost_hr": {"runpod": 1.64, "lambda": 1.25}},
    "a100_80gb": {"vram_gb": 80, "tflops_bf16": 312, "cost_hr": {"runpod": 2.09, "lambda": 1.75}},
    "h100_80gb": {"vram_gb": 80, "tflops_bf16": 990, "cost_hr": {"runpod": 3.49, "lambda": 2.49}},
    "a10_24gb":  {"vram_gb": 24, "tflops_bf16": 125, "cost_hr": {"runpod": 0.69, "lambda": 0.60}},
    "l40s_48gb": {"vram_gb": 48, "tflops_bf16": 362, "cost_hr": {"runpod": 1.14, "lambda": 0.99}},
}

# Training configs
TRAINING_CONFIGS = {
    "sft_small": {
        "name": "SFT - Small Model (3B)",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "model_size_gb": 6,
        "lora_r": 16,
        "batch_size": 4,
        "grad_accum": 4,
        "epochs": 3,
        "seq_len": 4096,
        "est_vram_gb": 12,
        "est_hours": 1.0,
        "min_gpu": "a10_24gb",
    },
    "sft_large": {
        "name": "SFT - Large Model (35B MoE, 3B active)",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "model_size_gb": 70,
        "lora_r": 32,
        "batch_size": 1,
        "grad_accum": 8,
        "epochs": 3,
        "seq_len": 8192,
        "est_vram_gb": 45,
        "est_hours": 8.0,
        "min_gpu": "a100_80gb",
    },
    "rl_grpo": {
        "name": "RL/GRPO - Large Model",
        "model": "outputs/sft/checkpoint-final",
        "model_size_gb": 70,
        "lora_r": 32,
        "batch_size": 1,
        "grad_accum": 4,
        "epochs": 1,
        "seq_len": 8192,
        "est_vram_gb": 55,
        "est_hours": 24.0,
        "min_gpu": "a100_80gb",
    },
}


def estimate_costs():
    """Print cost estimates for all training configurations."""
    print("\n" + "=" * 75)
    print("   ASIC-AI Cloud Training Cost Estimates")
    print("=" * 75)

    for cfg_id, cfg in TRAINING_CONFIGS.items():
        print(f"\n  {cfg['name']}")
        print(f"  Model: {cfg['model']}")
        print(f"  VRAM needed: ~{cfg['est_vram_gb']} GB")
        print(f"  Estimated time: ~{cfg['est_hours']:.1f} hours")
        print()

        print(f"  {'GPU':<20s} {'VRAM':>6s} {'Fits?':>6s} {'RunPod':>10s} {'Lambda':>10s}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")

        for gpu_id, gpu in GPU_SPECS.items():
            fits = "Yes" if gpu["vram_gb"] >= cfg["est_vram_gb"] else "No"
            if fits == "Yes":
                rp_cost = gpu["cost_hr"]["runpod"] * cfg["est_hours"]
                lm_cost = gpu["cost_hr"]["lambda"] * cfg["est_hours"]
                print(f"  {gpu_id:<20s} {gpu['vram_gb']:>4d}GB {fits:>6s} ${rp_cost:>8.2f} ${lm_cost:>8.2f}")
            else:
                print(f"  {gpu_id:<20s} {gpu['vram_gb']:>4d}GB {fits:>6s}       ---        ---")

    print(f"\n  Recommended setup:")
    print(f"  - SFT (3B test):  A10 24GB on Lambda    -> ~$0.60")
    print(f"  - SFT (35B):      A100 80GB on Lambda   -> ~$14.00")
    print(f"  - RL/GRPO (35B):  A100 80GB on Lambda   -> ~$42.00")
    print(f"  - Total (full pipeline):                 -> ~$57.00")
    print(f"\n{'=' * 75}\n")


def generate_training_script(config_id: str, provider: str = "runpod"):
    """Generate a ready-to-run cloud training script."""
    if config_id not in TRAINING_CONFIGS:
        log.error(f"Unknown config: {config_id}. Available: {list(TRAINING_CONFIGS.keys())}")
        return

    cfg = TRAINING_CONFIGS[config_id]
    gpu = cfg["min_gpu"]

    script = f"""#!/bin/bash
# ASIC-AI Cloud Training Script
# Config: {cfg['name']}
# Provider: {provider}
# GPU: {gpu} (min {cfg['est_vram_gb']}GB VRAM)
# Estimated time: {cfg['est_hours']:.1f} hours
# Estimated cost: ~${GPU_SPECS[gpu]['cost_hr'][provider] * cfg['est_hours']:.2f}

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
print(f'PyTorch: {{torch.__version__}}')
print(f'CUDA: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'VRAM: {{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}} GB')
"

# 4. Run tests
PYTHONPATH=src pytest tests/ -q --tb=short

# 5. Validate training data
echo "=== Training Data ==="
PYTHONPATH=src python scripts/analyze_sft_data.py

# 6. Launch training
echo "=== Starting Training ==="
"""

    if config_id.startswith("sft"):
        script += f"""
# SFT Training with Axolotl
pip install axolotl

# Merge all SFT data
cat data/sft/mock_analog_v1.jsonl data/sft/mock_digital_v1.jsonl data/sft/augmented_v1.jsonl > data/sft/combined_train.jsonl
echo "Combined training data: $(wc -l < data/sft/combined_train.jsonl) examples"

# Run SFT
accelerate launch -m axolotl.cli.train configs/training/sft_axolotl.yaml \\
    --base_model={cfg['model']} \\
    --datasets.0.path=data/sft/combined_train.jsonl \\
    --output_dir=outputs/sft \\
    --num_epochs={cfg['epochs']} \\
    --micro_batch_size={cfg['batch_size']} \\
    --gradient_accumulation_steps={cfg['grad_accum']} \\
    --sequence_len={cfg['seq_len']} \\
    --lora_r={cfg['lora_r']}

echo "=== SFT Training Complete ==="
echo "Output: outputs/sft/"
"""
    elif config_id == "rl_grpo":
        script += f"""
# RL/GRPO Training
PYTHONPATH=src python -m asic_ai.training.rl_grpo \\
    --config configs/training/rl_grpo.yaml

echo "=== RL/GRPO Training Complete ==="
echo "Output: outputs/rl/"
"""

    script += """
# 7. Upload results
echo "=== Uploading Results ==="
# Option A: Push to HuggingFace
# huggingface-cli upload ozgurkazanci/asic-ai-v1 outputs/sft/

# Option B: Copy to persistent storage
# cp -r outputs/ /workspace/outputs/

echo "=== All Done ==="
"""

    output_name = f"train_{config_id}_{provider}.sh"
    output_path = Path("scripts/cloud") / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    print(f"Generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Cloud GPU training launcher")
    parser.add_argument("--estimate", action="store_true", help="Show cost estimates")
    parser.add_argument("--generate-script", action="store_true", help="Generate training script")
    parser.add_argument("--config", default="sft_large", choices=TRAINING_CONFIGS.keys())
    parser.add_argument("--provider", default="lambda", choices=["runpod", "lambda"])
    args = parser.parse_args()

    if args.estimate:
        estimate_costs()
    elif args.generate_script:
        generate_training_script(args.config, args.provider)
        if args.config != "sft_small":
            generate_training_script("sft_small", args.provider)
    else:
        estimate_costs()


if __name__ == "__main__":
    main()
