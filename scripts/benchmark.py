#!/usr/bin/env python3
"""Benchmark inference speed across different configurations.

Measures tokens/sec for various model sizes, quantizations, and devices.

Usage:
    PYTHONPATH=src python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct
    PYTHONPATH=src python scripts/benchmark.py --model outputs/sft_local/final --runs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import SYSTEM_PROMPT

SEP = "=" * 60

BENCHMARK_PROMPTS = [
    "Design a two-stage OTA with 60dB gain for sky130.",
    "What is the threshold voltage of sky130 NMOS?",
    "Run AC simulation on my bandgap reference circuit.",
    "Debug: my LDO has poor PSRR. How to fix it?",
    "Compare folded cascode vs telescopic OTA for low power.",
]


def benchmark_model(model, tokenizer, prompts, max_tokens=64, runs=3):
    """Run benchmark and return results."""
    import torch

    results = []
    for run_idx in range(runs):
        for prompt in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_len = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            gen_time = time.time() - t0
            gen_tokens = outputs.shape[1] - input_len

            results.append({
                "run": run_idx + 1,
                "prompt_tokens": input_len,
                "gen_tokens": gen_tokens,
                "time_s": round(gen_time, 3),
                "tok_per_s": round(gen_tokens / gen_time, 1) if gen_time > 0 else 0,
                "ttft_s": round(gen_time / max(1, gen_tokens), 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ASIC-AI inference")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="eval_results/benchmark.json")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Inference Benchmark")
    print(f"{SEP}\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Model: {args.model}")
    print(f"  Device: CPU (PyTorch {torch.__version__})")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs: {args.runs}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.time() - t0

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.0f}M")
    print(f"  Load time: {load_time:.1f}s")
    print(f"\n  Running {args.runs} x {len(BENCHMARK_PROMPTS)} prompts...\n")

    results = benchmark_model(model, tokenizer, BENCHMARK_PROMPTS, args.max_tokens, args.runs)

    # Summary
    avg_tok_s = sum(r["tok_per_s"] for r in results) / len(results)
    avg_time = sum(r["time_s"] for r in results) / len(results)
    total_tokens = sum(r["gen_tokens"] for r in results)
    total_time = sum(r["time_s"] for r in results)

    print(f"{SEP}")
    print(f"   Results")
    print(f"{SEP}")
    print(f"  Model:           {args.model}")
    print(f"  Params:          {params:.0f}M")
    print(f"  Load time:       {load_time:.1f}s")
    print(f"  Avg tok/s:       {avg_tok_s:.1f}")
    print(f"  Avg latency:     {avg_time:.2f}s per prompt")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Throughput:      {total_tokens/total_time:.1f} tok/s overall")
    print(f"{SEP}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model, "params_m": round(params),
        "device": "cpu", "pytorch": torch.__version__,
        "avg_tok_per_s": round(avg_tok_s, 1),
        "avg_latency_s": round(avg_time, 2),
        "load_time_s": round(load_time, 1),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "runs": args.runs, "prompts": len(BENCHMARK_PROMPTS),
        "details": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
