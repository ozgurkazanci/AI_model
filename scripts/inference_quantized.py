#!/usr/bin/env python3
"""Quantized inference for ASIC-AI on CPU/low-VRAM GPUs.

Supports GGUF format via llama-cpp-python or bitsandbytes 4-bit quantization.
Optimized for AMD 780M iGPU (~4GB shared VRAM).

Usage:
    # With transformers (default)
    PYTHONPATH=src python scripts/inference_quantized.py --model outputs/sft_local/final

    # Export to GGUF for llama.cpp
    PYTHONPATH=src python scripts/inference_quantized.py --export-gguf outputs/merged

    # Interactive chat
    PYTHONPATH=src python scripts/inference_quantized.py --model outputs/sft_local/final --chat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from asic_ai.data.format import build_system_message
def load_model_quantized(model_path: str, bits: int = 4):
    """Load model with bitsandbytes quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_model_cpu(model_path: str, lora_path: str | None = None):
    """Load model on CPU (no quantization, works everywhere)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if lora_path:
        print("Loading LoRA adapter...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = AutoTokenizer.from_pretrained(lora_path or model_path)
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, system: str = None,
                      max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate a response from the model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    import torch
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tps = n_tokens / elapsed if elapsed > 0 else 0

    return response, n_tokens, elapsed, tps


def interactive_chat(model, tokenizer):
    """Interactive chat loop."""

    print("\n" + "=" * 60)
    print("   ASIC-AI Interactive Chat")
    print("=" * 60)
    print("  Type your circuit design question. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        response, n_tokens, elapsed, tps = generate_response(
            model, tokenizer, prompt, system=build_system_message()
        )

        safe = response.encode("ascii", "replace").decode("ascii")
        print(f"\nASSISTANT> {safe}")
        print(f"  [{n_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]\n")


def benchmark_prompts(model, tokenizer):
    """Run benchmark prompts and report performance."""

    prompts = [
        "Design a two-stage Miller OTA with 60 dB gain and 10 MHz GBW in sky130.",
        "What is the phase margin of my amplifier?",
        "Fix the LVS error: 3 shorts on metal2.",
        "Simulate DC operating point of a bandgap reference.",
    ]

    print("\n" + "=" * 60)
    print("   ASIC-AI Inference Benchmark")
    print("=" * 60 + "\n")

    total_tokens = 0
    total_time = 0

    for i, prompt in enumerate(prompts):
        response, n_tokens, elapsed, tps = generate_response(
            model, tokenizer, prompt, system=build_system_message(), max_tokens=128
        )
        total_tokens += n_tokens
        total_time += elapsed

        safe = response[:100].encode("ascii", "replace").decode("ascii")
        print(f"  [{i+1}/{len(prompts)}] {n_tokens:3d} tokens | {elapsed:5.1f}s | {tps:4.1f} tok/s")
        print(f"           {safe}...")
        print()

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print("=" * 60)
    print(f"  Total: {total_tokens} tokens in {total_time:.1f}s")
    print(f"  Average: {avg_tps:.1f} tokens/second")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ASIC-AI Quantized Inference")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--lora", default=None, help="LoRA adapter path")
    parser.add_argument("--bits", type=int, default=0, choices=[0, 4, 8],
                        help="Quantization bits (0=none/CPU, 4=4bit, 8=8bit)")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark prompts")
    parser.add_argument("--prompt", default=None, help="Single prompt to process")
    args = parser.parse_args()

    if args.bits > 0:
        model, tokenizer = load_model_quantized(args.model, args.bits)
    else:
        model, tokenizer = load_model_cpu(args.model, args.lora)

    if args.chat:
        interactive_chat(model, tokenizer)
    elif args.benchmark:
        benchmark_prompts(model, tokenizer)
    elif args.prompt:
        response, n_tokens, elapsed, tps = generate_response(
            model, tokenizer, args.prompt, system=build_system_message()
        )
        safe = response.encode("ascii", "replace").decode("ascii")
        print(safe)
        print(f"\n[{n_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]")
    else:
        benchmark_prompts(model, tokenizer)


if __name__ == "__main__":
    main()
