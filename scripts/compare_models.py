#!/usr/bin/env python3
"""Compare base model vs fine-tuned model responses.

Side-by-side comparison showing the effect of SFT training
on circuit design responses.

Usage:
    PYTHONPATH=src python scripts/compare_models.py \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --finetuned outputs/sft_local/final
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("compare")

SEP = "=" * 70

TEST_PROMPTS = [
    {
        "id": "ota_design",
        "prompt": "Design a two-stage OTA with dc_gain > 60dB, UGB > 30MHz, PM > 60deg for sky130 PDK.",
    },
    {
        "id": "pdk_query",
        "prompt": "Query the sky130 PDK for available NMOS devices and their threshold voltages.",
    },
    {
        "id": "debug_gain",
        "prompt": "My OTA only achieves 45dB gain. The spec requires 60dB. How should I diagnose and fix this?",
    },
    {
        "id": "tool_use",
        "prompt": "I need to verify my bandgap reference circuit. What simulation should I run first?",
    },
]


def load_and_generate(model_path: str, prompts: list[dict], max_tokens: int = 128) -> list[dict]:
    """Load model and generate responses for all prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for p in prompts:
        messages = [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": p["prompt"]},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"<|im_start|>system\n{build_system_message()}<|im_end|>\n<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=3072)
        t0 = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, do_sample=True, top_p=0.9,
                repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - t0
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        results.append({
            "id": p["id"],
            "response": response,
            "tokens": len(new_tokens),
            "time_s": round(gen_time, 1),
            "tok_per_s": round(len(new_tokens) / gen_time, 1) if gen_time > 0 else 0,
        })

    return results


def score_response(response: str) -> dict:
    """Score a response on domain-relevant criteria."""
    text = response.lower()
    scores = {
        "mentions_tools": any(t in text for t in ["sim.", "pdk.", "spec.", "netlist.", "tool_call"]),
        "mentions_devices": any(d in text for d in ["nfet", "pfet", "nmos", "pmos", "mosfet"]),
        "mentions_params": any(p in text for p in ["gm", "vth", "w/l", "width", "length", "vgs"]),
        "mentions_specs": any(s in text for s in ["gain", "ugb", "bandwidth", "phase margin", "pm"]),
        "mentions_pdk": any(p in text for p in ["sky130", "pdk", "process"]),
        "mentions_sim": any(s in text for s in ["simulat", "spice", "ac ", "dc ", "transient"]),
    }
    scores["total"] = sum(scores.values())
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--finetuned", default="outputs/sft_local/final")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Model Comparison: Base vs Fine-Tuned")
    print(f"{SEP}\n")

    # Generate with base model
    print(f"[1/3] Generating with BASE model: {args.base}")
    base_results = load_and_generate(args.base, TEST_PROMPTS, args.max_tokens)

    # Generate with fine-tuned model
    ft_path = Path(args.finetuned)
    if ft_path.exists():
        print(f"\n[2/3] Generating with FINE-TUNED model: {args.finetuned}")
        ft_results = load_and_generate(args.finetuned, TEST_PROMPTS, args.max_tokens)
    else:
        print(f"\n[2/3] Fine-tuned model not found at {args.finetuned}")
        print(f"  Run: python scripts/finetune_local.py --quick-test")
        ft_results = None

    # Compare
    print(f"\n[3/3] Comparison Results")
    print(f"{SEP}\n")

    base_total = 0
    ft_total = 0

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Prompt [{prompt['id']}]: {prompt['prompt'][:80]}")
        print()

        base = base_results[i]
        base_score = score_response(base["response"])
        base_total += base_score["total"]
        print(f"  BASE ({base['tokens']} tok, {base['time_s']}s):")
        print(f"    {base['response'][:150]}...")
        print(f"    Score: {base_score['total']}/6 {dict((k,v) for k,v in base_score.items() if v and k != 'total')}")

        if ft_results:
            ft = ft_results[i]
            ft_score = score_response(ft["response"])
            ft_total += ft_score["total"]
            print(f"  FINE-TUNED ({ft['tokens']} tok, {ft['time_s']}s):")
            print(f"    {ft['response'][:150]}...")
            print(f"    Score: {ft_score['total']}/6 {dict((k,v) for k,v in ft_score.items() if v and k != 'total')}")

            diff = ft_score["total"] - base_score["total"]
            if diff > 0:
                print(f"    >> Fine-tuned is BETTER (+{diff})")
            elif diff < 0:
                print(f"    >> Base is better ({diff})")
            else:
                print(f"    >> Same score")

        print()

    # Summary
    print(f"{SEP}")
    print(f"   Summary")
    print(f"{SEP}")
    print(f"  Base model total score:       {base_total}/{len(TEST_PROMPTS)*6}")
    if ft_results:
        print(f"  Fine-tuned model total score: {ft_total}/{len(TEST_PROMPTS)*6}")
        improvement = ft_total - base_total
        print(f"  Improvement:                  {'+' if improvement >= 0 else ''}{improvement} points")
    print(f"{SEP}\n")

    # Save comparison
    out = {
        "base_model": args.base,
        "finetuned_model": args.finetuned,
        "base_score": base_total,
        "ft_score": ft_total if ft_results else None,
        "prompts": len(TEST_PROMPTS),
    }
    out_path = Path("eval_results/model_comparison.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
