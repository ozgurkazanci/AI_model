#!/usr/bin/env python3
"""Comprehensive model benchmark across all design domains.

Tests the model's capability across:
1. Analog circuit design (CS amp, OTA, bandgap)
2. Digital RTL design (FSM, FIFO)
3. Spectre-specific (STB, PSS)
4. Physical verification (DRC, LVS)
5. Layout knowledge (matching, floorplan)
6. Multi-step reasoning (iteration, error recovery)

Usage:
    PYTHONPATH=src python scripts/benchmark_model.py --model outputs/sft_local/final
    PYTHONPATH=src python scripts/benchmark_model.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60

# Benchmark prompts covering all domains
BENCHMARK_PROMPTS = [
    # Category 1: Analog Design
    {
        "id": "analog_cs_amp",
        "category": "analog",
        "difficulty": "easy",
        "prompt": "Design a common-source amplifier with gain > 20 dB in sky130 1.8V.",
        "expected_tools": ["sim.dc", "sim.ac"],
        "expected_keywords": ["nmos", "gain", "gm", "RD", "bandwidth"],
    },
    {
        "id": "analog_ota_design",
        "category": "analog",
        "difficulty": "medium",
        "prompt": "Design a folded cascode OTA with GBW > 50 MHz and PM > 60 degrees.",
        "expected_tools": ["sim.ac", "sim.stb", "spec.check"],
        "expected_keywords": ["folded", "cascode", "compensation", "phase margin", "gm"],
    },
    {
        "id": "analog_bandgap",
        "category": "analog",
        "difficulty": "hard",
        "prompt": "Design a bandgap voltage reference with TC < 10 ppm/C across -40 to 125C.",
        "expected_tools": ["sim.dc", "sim.corners"],
        "expected_keywords": ["bandgap", "PTAT", "CTAT", "1.2V", "temperature"],
    },

    # Category 2: Digital Design
    {
        "id": "digital_counter",
        "category": "digital",
        "difficulty": "easy",
        "prompt": "Design a 4-bit synchronous up/down counter in Verilog.",
        "expected_tools": [],
        "expected_keywords": ["counter", "always", "posedge", "clk", "rst"],
    },
    {
        "id": "digital_fsm",
        "category": "digital",
        "difficulty": "medium",
        "prompt": "Design a UART receiver FSM with configurable baud rate.",
        "expected_tools": [],
        "expected_keywords": ["UART", "state", "baud", "start bit", "stop bit"],
    },

    # Category 3: Spectre-Specific
    {
        "id": "spectre_stb",
        "category": "spectre",
        "difficulty": "medium",
        "prompt": "How do I use Spectre STB analysis to check loop stability of an amplifier?",
        "expected_tools": ["sim.stb"],
        "expected_keywords": ["STB", "phase margin", "gain margin", "probe", "loop gain"],
    },
    {
        "id": "spectre_pss",
        "category": "spectre",
        "difficulty": "hard",
        "prompt": "Set up PSS and PNoise analysis for a VCO in Spectre format.",
        "expected_tools": [],
        "expected_keywords": ["PSS", "PNoise", "periodic", "harmonics", "phase noise"],
    },

    # Category 4: Physical Verification
    {
        "id": "pvs_drc",
        "category": "signoff",
        "difficulty": "medium",
        "prompt": "My layout has minimum spacing DRC errors near the guard ring. How do I fix them?",
        "expected_tools": ["lint.check"],
        "expected_keywords": ["DRC", "spacing", "guard ring", "enclosure", "fix"],
    },
    {
        "id": "pvs_lvs",
        "category": "signoff",
        "difficulty": "medium",
        "prompt": "LVS shows unmatched nets in my OTA layout. Debug strategy?",
        "expected_tools": ["lint.check"],
        "expected_keywords": ["LVS", "unmatched", "via", "short", "open"],
    },

    # Category 5: Layout
    {
        "id": "layout_matching",
        "category": "layout",
        "difficulty": "medium",
        "prompt": "How should I layout a differential pair for minimum offset in sky130?",
        "expected_tools": [],
        "expected_keywords": ["common-centroid", "matching", "interdigit", "dummy", "orientation"],
    },

    # Category 6: Multi-step Reasoning
    {
        "id": "reason_tradeoff",
        "category": "reasoning",
        "difficulty": "hard",
        "prompt": "Is it feasible to achieve GBW > 1 GHz with power < 1 mW in sky130? Analyze the trade-offs.",
        "expected_tools": [],
        "expected_keywords": ["gm", "GBW", "power", "trade-off", "feasib"],
    },
    {
        "id": "reason_corner_fail",
        "category": "reasoning",
        "difficulty": "hard",
        "prompt": "My amplifier meets specs at TT corner but fails at SS,125C. Root cause and fix?",
        "expected_tools": ["sim.corners"],
        "expected_keywords": ["SS", "Vth", "temperature", "mobility", "slow"],
    },
]


def score_response(response: str, prompt_info: dict) -> dict:
    """Score a model response against expected criteria."""
    response_lower = response.lower()

    # Keyword hits
    keywords = prompt_info["expected_keywords"]
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    keyword_score = hits / len(keywords) if keywords else 1.0

    # Tool call detection
    has_tool_call = "<tool_call>" in response
    expected_tools = prompt_info["expected_tools"]
    tool_hits = sum(1 for t in expected_tools if t in response)
    tool_score = tool_hits / len(expected_tools) if expected_tools else (1.0 if not has_tool_call else 0.5)

    # Length score (prefer detailed responses)
    word_count = len(response.split())
    length_score = min(1.0, word_count / 100)

    # Structure score (has explanation + code/netlist)
    has_code = "```" in response or ".model" in response or "module" in response
    has_explanation = any(w in response_lower for w in ["because", "since", "therefore", "strategy", "approach"])
    structure_score = (0.5 if has_code else 0.0) + (0.5 if has_explanation else 0.0)

    # Overall score
    overall = 0.4 * keyword_score + 0.2 * tool_score + 0.2 * length_score + 0.2 * structure_score

    return {
        "keyword_score": round(keyword_score, 3),
        "keyword_hits": f"{hits}/{len(keywords)}",
        "tool_score": round(tool_score, 3),
        "length_score": round(length_score, 3),
        "word_count": word_count,
        "structure_score": round(structure_score, 3),
        "overall": round(overall, 3),
    }


def run_benchmark(model_path: str, use_model: bool = False) -> dict:
    """Run benchmark. If use_model=False, just validate the benchmark structure."""
    results = []

    if use_model:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"  Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"  [WARN] Could not load model: {e}")
            print(f"  Running in validation mode (no model inference)")
            use_model = False

    for p in BENCHMARK_PROMPTS:
        if use_model:
            try:
                messages = [
                    {"role": "system", "content": "You are an expert ASIC circuit design AI assistant."},
                    {"role": "user", "content": p["prompt"]},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt")
                with __import__("torch").no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=512, temperature=0.7,
                        do_sample=True, top_p=0.9,
                    )
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:
                response = f"[ERROR: {e}]"
        else:
            response = "[VALIDATION MODE - no model loaded]"

        scores = score_response(response, p)
        results.append({
            "id": p["id"],
            "category": p["category"],
            "difficulty": p["difficulty"],
            "prompt": p["prompt"][:60] + "...",
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "scores": scores,
        })

        status = "PASS" if scores["overall"] >= 0.5 else "FAIL" if use_model else "SKIP"
        print(f"  [{status}] {p['id']:25s} | {p['category']:8s} | {p['difficulty']:6s} | {scores['overall']:.3f}")

    return {"prompts": len(BENCHMARK_PROMPTS), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Benchmark")
    parser.add_argument("--model", default="outputs/sft_local/final")
    parser.add_argument("--no-model", action="store_true", help="Validate benchmark without model")
    parser.add_argument("--output", default="eval_results/benchmark.json")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Comprehensive Model Benchmark")
    print(f"{SEP}\n")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {len(BENCHMARK_PROMPTS)} across 6 categories\n")

    t0 = time.time()
    use_model = not args.no_model and Path(args.model).exists()
    results = run_benchmark(args.model, use_model=use_model)
    elapsed = time.time() - t0

    # Category breakdown
    print(f"\n{SEP}")
    print("   Category Breakdown")
    print(f"{SEP}")
    categories = {}
    for r in results["results"]:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["scores"]["overall"])

    for cat, scores in sorted(categories.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat:12s}: {avg:.3f} avg ({len(scores)} prompts)")

    overall_avg = sum(r["scores"]["overall"] for r in results["results"]) / len(results["results"])
    print(f"\n  {'OVERALL':12s}: {overall_avg:.3f}")
    print(f"  Duration: {elapsed:.1f}s")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "model": args.model,
        "overall_score": overall_avg,
        "category_scores": {k: sum(v)/len(v) for k, v in categories.items()},
        "duration_s": elapsed,
        "results": results["results"],
    }, indent=2, default=str), encoding="utf-8")

    print(f"  Results: {args.output}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
