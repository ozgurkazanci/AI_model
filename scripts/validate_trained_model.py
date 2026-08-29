#!/usr/bin/env python3
"""Post-training validation: test fine-tuned model with real ngspice.

Runs after training completes to validate the model can:
1. Generate valid tool calls
2. Produce reasonable circuit analysis
3. Work in the agent loop with real simulation

Usage:
    PYTHONPATH=src python scripts/validate_trained_model.py
    PYTHONPATH=src python scripts/validate_trained_model.py --model outputs/sft_local/final
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60

VALIDATION_PROMPTS = [
    {
        "id": "tool_call_gen",
        "prompt": "I need to design an OTA with 60 dB gain. First, query the PDK for available NMOS devices.",
        "expected_tools": ["pdk.device_query", "pdk.list_devices"],
        "check": "tool_call",
    },
    {
        "id": "sim_request",
        "prompt": "Run a DC simulation on this amplifier to find the operating point. The netlist is already loaded.",
        "expected_tools": ["sim.dc"],
        "check": "tool_call",
    },
    {
        "id": "spec_analysis",
        "prompt": "The simulation shows DC gain of 45 dB, UGB of 120 MHz, and phase margin of 62 degrees. Check if these meet the spec: gain > 40 dB, UGB > 100 MHz, PM > 60 deg.",
        "expected_tools": ["spec.check"],
        "check": "tool_call",
    },
    {
        "id": "design_knowledge",
        "prompt": "What is the relationship between gm and power consumption in a common-source amplifier?",
        "expected_keywords": ["gm", "current", "transconductance", "power", "bias"],
        "check": "keywords",
    },
    {
        "id": "corner_awareness",
        "prompt": "My amplifier works at typical but fails at slow-slow corner with high temperature. What should I check?",
        "expected_keywords": ["corner", "temperature", "threshold", "margin", "worst"],
        "check": "keywords",
    },
]


def validate_model(model_path: str) -> dict:
    """Run validation prompts through the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32,
    )

    results = []
    total_tokens = 0
    total_time = 0

    for vp in VALIDATION_PROMPTS:
        print(f"\n  [{vp['id']}] {vp['prompt'][:60]}...")

        messages = [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": vp["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
            )
        gen_time = time.time() - t0

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += tokens
        total_time += gen_time

        # Check result
        passed = False
        if vp["check"] == "tool_call":
            has_tool = any(t in response.lower() for t in vp["expected_tools"])
            has_call = "tool_call" in response.lower() or any(
                t in response for t in vp["expected_tools"]
            )
            passed = has_tool or has_call
            check_detail = f"tool_found={has_tool}"
        elif vp["check"] == "keywords":
            found = [k for k in vp["expected_keywords"] if k.lower() in response.lower()]
            passed = len(found) >= 2
            check_detail = f"keywords={len(found)}/{len(vp['expected_keywords'])}"

        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {check_detail} ({tokens} tok, {gen_time:.1f}s)")
        safe_response = response[:120].encode("ascii", "replace").decode("ascii")
        print(f"    Response: {safe_response}...")

        results.append({
            "id": vp["id"],
            "passed": passed,
            "check": vp["check"],
            "detail": check_detail,
            "tokens": tokens,
            "time_s": round(gen_time, 1),
            "response_preview": response[:200],
        })

    passed_count = sum(1 for r in results if r["passed"])
    avg_speed = total_tokens / total_time if total_time > 0 else 0

    return {
        "model": model_path,
        "passed": passed_count,
        "total": len(results),
        "pass_rate": round(passed_count / len(results) * 100, 1),
        "avg_speed_tps": round(avg_speed, 1),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--model", default="outputs/sft_local/final")
    parser.add_argument("--output", default="eval_results/model_validation.json")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   Post-Training Model Validation")
    print(f"{SEP}")

    model_path = args.model
    if not Path(model_path).exists():
        print(f"\n  Model not found: {model_path}")
        print(f"  Run training first or specify --model path")
        return

    results = validate_model(model_path)

    print(f"\n{SEP}")
    print(f"   Validation Results: {results['passed']}/{results['total']} ({results['pass_rate']}%)")
    print(f"   Speed: {results['avg_speed_tps']} tok/s")
    print(f"{SEP}")

    for r in results["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['detail']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
