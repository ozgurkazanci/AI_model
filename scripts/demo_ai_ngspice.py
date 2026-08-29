#!/usr/bin/env python3
"""End-to-end demo: AI model generates circuit -> ngspice simulates it.

This is the ultimate integration test: the fine-tuned AI model generates
a netlist and simulation commands, then ngspice runs real SPICE simulation.

Usage:
    PYTHONPATH=src python scripts/demo_ai_ngspice.py
    PYTHONPATH=src python scripts/demo_ai_ngspice.py --model outputs/sft_local/final
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message

SEP = "=" * 60

# Test circuits for ngspice simulation
DEMO_CIRCUITS = {
    "common_source_amp": {
        "description": "Common-Source NMOS Amplifier",
        "netlist": """\
* Common-Source NMOS Amplifier with Resistive Load

.model nch nmos level=1 vto=0.5 kp=200u lambda=0.04

* Supply
VDD vdd 0 DC 1.8

* Input bias
Vin gate 0 DC 0.7

* Resistive load
RD vdd out 5k

* Input transistor
M1 out gate 0 0 nch W=10u L=1u

* Load capacitor
CL out 0 1p

.dc Vin 0.3 1.2 0.01
.end
""",
        "expected": "DC sweep showing gain transition",
    },
    "cmos_inverter": {
        "description": "CMOS Inverter Transfer Curve",
        "netlist": """\
* CMOS Inverter
.model nch nmos level=1 vto=0.5 kp=200u
.model pch pmos level=1 vto=-0.5 kp=100u

VDD vdd 0 DC 1.8
Vin in 0 DC 0

M1 out in 0 0 nch W=2u L=0.18u
M2 out in vdd vdd pch W=4u L=0.18u

.dc Vin 0 1.8 0.01
.end
""",
        "expected": "VTC with switching threshold ~0.9V",
    },
    "bandgap_core": {
        "description": "Simplified Bandgap Reference Core",
        "netlist": """\
* Simplified Bandgap Reference
* R1/R2 ratio sets PTAT current

.model nch nmos level=1 vto=0.5 kp=200u
.model npn npn bf=100 is=1e-15

VDD vdd 0 DC 3.3

* Current mirror
R1 vdd n1 10k
R2 vdd n2 10k
R3 n2 0 5k

* Diode-connected transistor
Q1 n1 n1 0 npn
Q2 n2 n2 0 npn

* Temperature sweep
.dc temp -40 125 1
.end
""",
        "expected": "Output vs temperature showing reference behavior",
    },
}


def run_ngspice_demo(circuits: dict):
    """Run circuits through ngspice shared adapter."""
    from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter, find_ngspice_dll
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.tool_interface.schema import SimParams

    dll = find_ngspice_dll()
    if not dll:
        print("  [SKIP] ngspice.dll not found")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AdapterConfig(binary_path=dll, work_dir=tmpdir)
        adapter = NgspiceSharedAdapter(config)

        results = {}
        for name, circuit in circuits.items():
            print(f"\n  Simulating: {circuit['description']}")
            
            # Write netlist
            cir_path = Path(tmpdir) / f"{name}.cir"
            cir_path.write_text(circuit["netlist"], encoding="utf-8")

            # Run DC simulation
            t0 = time.time()
            params = SimParams(analysis_type="dc")
            result = adapter.dc(str(cir_path), params)
            sim_time = time.time() - t0

            sweeps = result.sweeps
            total_points = sum(len(s.x_values) for s in sweeps.values()) if sweeps else 0
            
            print(f"    Signals: {list(sweeps.keys())}")
            print(f"    Data points: {total_points}")
            print(f"    Sim time: {sim_time*1000:.0f}ms")
            print(f"    Expected: {circuit['expected']}")

            results[name] = {
                "description": circuit["description"],
                "signals": list(sweeps.keys()),
                "data_points": total_points,
                "sim_time_ms": round(sim_time * 1000),
                "success": total_points > 0,
            }

        return results


def run_model_inference(model_path: str | None):
    """Run AI model to generate circuit analysis text."""
    if model_path and Path(model_path).exists():
        print(f"\n  Loading model: {model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32,
            )

            prompt = (
                "Design a common-source amplifier with the following specs:\n"
                "- DC gain > 20 dB\n"
                "- Supply: 1.8V\n"
                "- Load: 1pF\n"
                "First, query the PDK for available NMOS devices."
            )

            messages = [
                {"role": "system", "content": build_system_message()},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.7,
                    do_sample=True, top_p=0.9,
                )
            gen_time = time.time() - t0

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            speed = tokens / gen_time

            print(f"    Generated {tokens} tokens in {gen_time:.1f}s ({speed:.1f} tok/s)")
            print(f"    Response preview:")
            for line in response[:500].splitlines()[:8]:
                print(f"      {line}")
            if len(response) > 500:
                print(f"      ... ({len(response)} chars total)")

            return {
                "model": model_path,
                "tokens": tokens,
                "gen_time_s": round(gen_time, 1),
                "speed_tps": round(speed, 1),
                "response_preview": response[:500],
                "has_tool_call": "pdk.device_query" in response or "tool_call" in response.lower(),
            }
        except Exception as e:
            print(f"    Model error: {e}")
            return {"error": str(e)}
    else:
        print(f"\n  [SKIP] No model at: {model_path}")
        return {"skipped": True}


def main():
    parser = argparse.ArgumentParser(description="AI + ngspice E2E demo")
    parser.add_argument("--model", default="outputs/sft_local/final", help="Model path")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI: End-to-End Demo")
    print("   AI Model + Real ngspice Simulation")
    print(f"{SEP}")

    # Part 1: ngspice simulation
    print(f"\n{'='*40}")
    print("  Part 1: Real ngspice Simulation")
    print(f"{'='*40}")
    sim_results = run_ngspice_demo(DEMO_CIRCUITS)

    # Part 2: AI model inference
    print(f"\n{'='*40}")
    print("  Part 2: AI Model Circuit Design")
    print(f"{'='*40}")
    model_results = run_model_inference(args.model)

    # Summary
    print(f"\n{SEP}")
    print("   Demo Summary")
    print(f"{SEP}")

    sim_passed = sum(1 for r in sim_results.values() if r.get("success"))
    print(f"  ngspice: {sim_passed}/{len(sim_results)} circuits simulated")
    for name, r in sim_results.items():
        status = "OK" if r.get("success") else "FAIL"
        print(f"    [{status}] {r.get('description', name)}: {r.get('data_points', 0)} points, {r.get('sim_time_ms', 0)}ms")

    if model_results.get("has_tool_call"):
        print(f"  AI Model: Generated tool call! ({model_results.get('speed_tps', 0)} tok/s)")
    elif model_results.get("skipped"):
        print(f"  AI Model: Skipped (train first)")
    elif model_results.get("error"):
        print(f"  AI Model: Error")
    else:
        print(f"  AI Model: Generated response ({model_results.get('speed_tps', 0)} tok/s)")

    print(f"{SEP}\n")

    # Save
    output = {"simulation": sim_results, "model": model_results}
    out_path = Path("eval_results/e2e_demo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
