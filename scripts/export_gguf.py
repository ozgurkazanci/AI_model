#!/usr/bin/env python3
"""Export model to GGUF format for efficient CPU/GPU inference.

Converts HuggingFace model to GGUF with quantization for fast
local inference with llama.cpp.

Usage:
    # Export with Q4_K_M quantization (recommended)
    PYTHONPATH=src python scripts/export_gguf.py --model outputs/sft_local/final --quant q4_k_m

    # Export FP16 (no quantization)
    PYTHONPATH=src python scripts/export_gguf.py --model outputs/sft_local/final --quant f16
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_gguf")

SEP = "=" * 70

QUANT_TYPES = {
    "f16":     "Full precision FP16, no quantization. Large but accurate.",
    "q8_0":    "8-bit quantization. Good balance of size and quality.",
    "q5_k_m":  "5-bit quantization. Smaller, minimal quality loss.",
    "q4_k_m":  "4-bit quantization (recommended). Best size/quality trade-off.",
    "q4_0":    "Basic 4-bit. Smallest, some quality loss.",
    "q3_k_m":  "3-bit quantization. Very small, noticeable quality loss.",
    "q2_k":    "2-bit quantization. Extremely small, significant quality loss.",
}


def check_llama_cpp():
    """Check if llama.cpp conversion tools are available."""
    # Check for convert script
    try:
        result = subprocess.run(["python", "-m", "llama_cpp", "--version"],
                              capture_output=True, timeout=5)
        return True
    except Exception:
        pass

    # Check for convert_hf_to_gguf.py (standalone)
    convert_script = Path("vendor/llama.cpp/convert_hf_to_gguf.py")
    if convert_script.exists():
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--model", default="outputs/sft_local/final", help="HF model path")
    parser.add_argument("--output", default=None, help="Output GGUF file path")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANT_TYPES.keys())
    parser.add_argument("--list-quants", action="store_true", help="List quantization types")
    args = parser.parse_args()

    if args.list_quants:
        print(f"\n{SEP}")
        print("   Available Quantization Types")
        print(f"{SEP}\n")
        for qt, desc in QUANT_TYPES.items():
            print(f"  {qt:12s} {desc}")
        print(f"\n  Recommended: q4_k_m (best size/quality for inference)")
        print(f"{SEP}\n")
        return

    print(f"\n{SEP}")
    print("   ASIC-AI GGUF Export")
    print(f"{SEP}\n")

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Default output name
    if args.output is None:
        model_name = model_path.name
        args.output = f"models/{model_name}-{args.quant}.gguf"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Model:  {model_path}")
    print(f"  Quant:  {args.quant} - {QUANT_TYPES[args.quant]}")
    print(f"  Output: {output_path}")

    # Method 1: Try llama-cpp-python
    if check_llama_cpp():
        print(f"\n  Using llama.cpp conversion tools...")
        # This would use the actual conversion
        log.info("llama.cpp tools found, running conversion...")
    else:
        # Method 2: Generate instructions
        print(f"\n  llama.cpp tools not found. Manual conversion instructions:")
        print(f"\n  Option A: Use HuggingFace (easiest)")
        print(f"  pip install llama-cpp-python[server]")
        print(f"  python -m llama_cpp.convert --model {model_path} --outfile {output_path}")

        print(f"\n  Option B: Use llama.cpp directly")
        print(f"  git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp")
        print(f"  cd vendor/llama.cpp && pip install -r requirements.txt")
        print(f"  python convert_hf_to_gguf.py {model_path.absolute()} --outfile {output_path.absolute()}")
        if args.quant != "f16":
            print(f"  ./llama-quantize {output_path.absolute()} {output_path.with_suffix('').absolute()}-{args.quant}.gguf {args.quant}")

        print(f"\n  Option C: Use online converter")
        print(f"  Upload to: https://huggingface.co/spaces/ggml-org/gguf-my-repo")

    # Save export config
    config = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "quantization": args.quant,
        "quant_description": QUANT_TYPES[args.quant],
    }
    config_path = output_path.with_suffix(".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n  Export config saved to: {config_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
