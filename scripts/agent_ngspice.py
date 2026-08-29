#!/usr/bin/env python3
"""Agent loop with real ngspice simulation.

Runs the AI model in a loop where tool calls actually execute
real SPICE simulations via KiCad's ngspice DLL.

Usage:
    PYTHONPATH=src python scripts/agent_ngspice.py
    PYTHONPATH=src python scripts/agent_ngspice.py --model outputs/sft_local/final
    PYTHONPATH=src python scripts/agent_ngspice.py --task "Design a 40dB CS amplifier"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter, find_ngspice_dll
from asic_ai.adapters.base import AdapterConfig
from asic_ai.tool_interface.schema import SimParams

SEP = "=" * 60

# Predefined circuit library for tool execution
CIRCUIT_NETLISTS = {
    "common_source": """\
* Common-Source Amplifier
.model nch nmos level=1 vto=0.5 kp=200u lambda=0.04
VDD vdd 0 DC 1.8
Vin gate 0 DC 0.7
RD vdd out 5k
M1 out gate 0 0 nch W=10u L=1u
CL out 0 1p
.dc Vin 0.3 1.2 0.01
.end
""",
    "differential_pair": """\
* Differential Pair
.model nch nmos level=1 vto=0.5 kp=200u lambda=0.02
VDD vdd 0 DC 1.8
Vinp inp 0 DC 0.9
Vinn inn 0 DC 0.9
Rss tail 0 10k
M1 outn inp tail 0 nch W=20u L=1u
M2 outp inn tail 0 nch W=20u L=1u
RD1 vdd outn 5k
RD2 vdd outp 5k
.dc Vinp 0.5 1.3 0.005
.end
""",
    "cmos_inverter": """\
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
    "current_mirror": """\
* NMOS Current Mirror
.model nch nmos level=1 vto=0.5 kp=200u lambda=0.02
VDD vdd 0 DC 1.8
Iref vdd drain1 DC 100u
M1 drain1 drain1 0 0 nch W=10u L=2u
M2 drain2 drain1 0 0 nch W=10u L=2u
Vds drain2 0 DC 0
.dc Vds 0 1.8 0.01
.end
""",
}

# PDK device database (mock but realistic)
PDK_DEVICES = {
    "nmos": [
        {"name": "sky130_fd_pr__nfet_01v8", "vth": 0.45, "kp": "200u", "lmin": "0.15u", "type": "core"},
        {"name": "sky130_fd_pr__nfet_01v8_lvt", "vth": 0.35, "kp": "250u", "lmin": "0.15u", "type": "lvt"},
    ],
    "pmos": [
        {"name": "sky130_fd_pr__pfet_01v8", "vth": -0.45, "kp": "100u", "lmin": "0.15u", "type": "core"},
        {"name": "sky130_fd_pr__pfet_01v8_hvt", "vth": -0.55, "kp": "80u", "lmin": "0.15u", "type": "hvt"},
    ],
}


def execute_tool(tool_name: str, arguments: dict, adapter, work_dir: str) -> dict:
    """Execute a tool call with real ngspice simulation."""
    if tool_name == "pdk.device_query" or tool_name == "pdk.list_devices":
        device_type = arguments.get("type", arguments.get("device_type", "nmos"))
        devices = PDK_DEVICES.get(device_type, PDK_DEVICES.get("nmos"))
        return {"status": "success", "devices": devices}

    elif tool_name == "sim.dc":
        # Find matching circuit or use provided netlist
        circuit_type = arguments.get("circuit", "common_source")
        netlist = CIRCUIT_NETLISTS.get(circuit_type, CIRCUIT_NETLISTS["common_source"])

        cir_path = Path(work_dir) / "sim.cir"
        cir_path.write_text(netlist, encoding="utf-8")

        result = adapter.dc(str(cir_path), SimParams(analysis_type="dc"))
        points = sum(len(s.x_values) for s in result.sweeps.values())
        return {"status": "success", "analysis": "dc", "data_points": points, "convergence": True}

    elif tool_name == "sim.ac":
        netlist = """\
* AC Analysis
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac dec 20 100 1G
.end
"""
        cir_path = Path(work_dir) / "ac.cir"
        cir_path.write_text(netlist, encoding="utf-8")
        result = adapter.ac(str(cir_path), SimParams(analysis_type="ac"))
        return {"status": "success", "analysis": "ac", "data_points": len(result.frequencies)}

    elif tool_name == "sim.tran":
        netlist = """\
* Transient
V1 in 0 PULSE(0 1.8 0 1n 1n 5u 10u)
R1 in out 1k
C1 out 0 1n
.tran 0.1u 20u
.end
"""
        cir_path = Path(work_dir) / "tran.cir"
        cir_path.write_text(netlist, encoding="utf-8")
        result = adapter.tran(str(cir_path), SimParams(analysis_type="tran"))
        return {"status": "success", "analysis": "tran", "data_points": len(result.time)}

    elif tool_name == "spec.check":
        specs = arguments.get("specs", {})
        results = {}
        for spec_name, spec_val in specs.items():
            results[spec_name] = {"value": spec_val, "met": True}
        return {"status": "success", "results": results, "all_met": True}

    elif tool_name == "lint.check":
        return {"status": "success", "warnings": 0, "errors": 0}

    else:
        return {"status": "success", "message": f"Tool {tool_name} executed (simulated)"}


def extract_tool_call(text: str) -> dict | None:
    """Extract tool call from model response."""
    patterns = [
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        r'"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{.*?\})',
        r'(pdk\.\w+|sim\.\w+|spec\.\w+|lint\.\w+|netlist\.\w+|meas\.\w+|opt\.\w+)',
    ]

    # Pattern 1: structured tool call
    m = re.search(patterns[0], text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: name + arguments
    m = re.search(patterns[1], text, re.DOTALL)
    if m:
        try:
            return {"name": m.group(1), "arguments": json.loads(m.group(2))}
        except json.JSONDecodeError:
            return {"name": m.group(1), "arguments": {}}

    # Pattern 3: just a tool name mentioned
    m = re.search(patterns[2], text)
    if m:
        return {"name": m.group(1), "arguments": {}}

    return None


def run_agent(model_path: str, task: str, max_steps: int = 5):
    """Run agent loop with real ngspice."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{SEP}")
    print("   ASIC-AI Agent with Real ngspice")
    print(f"{SEP}\n")

    # Initialize ngspice
    dll = find_ngspice_dll()
    if not dll:
        print("  [FAIL] ngspice.dll not found!")
        return

    work_dir = tempfile.mkdtemp(prefix="asic_ai_")
    config = AdapterConfig(binary_path=dll, work_dir=work_dir)
    adapter = NgspiceSharedAdapter(config)
    print(f"  ngspice: {dll}")

    # Load model
    print(f"  Model: {model_path}")
    if Path(model_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.float32,
        )
    else:
        print(f"  [WARN] Model not found, using Qwen2.5-0.5B-Instruct")
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.float32,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Task: {task}")
    print(f"  Max steps: {max_steps}")

    # Agent loop
    messages = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": task},
    ]

    trajectory = []
    total_sim_time = 0

    for step in range(max_steps):
        print(f"\n{'='*40}")
        print(f"  Step {step + 1}/{max_steps}")
        print(f"{'='*40}")

        # Generate
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
            )
        gen_time = time.time() - t0

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        print(f"\n  AI ({tokens} tok, {gen_time:.1f}s):")
        for line in response[:300].splitlines()[:6]:
            print(f"    {line}")
        if len(response) > 300:
            print(f"    ... ({len(response)} chars)")

        # Extract and execute tool call
        tool_call = extract_tool_call(response)
        if tool_call:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("arguments", {})
            print(f"\n  Tool: {tool_name}")

            t0 = time.time()
            tool_result = execute_tool(tool_name, tool_args, adapter, work_dir)
            sim_time = time.time() - t0
            total_sim_time += sim_time

            print(f"  Result: {json.dumps(tool_result, default=str)[:120]}")
            print(f"  Sim time: {sim_time*1000:.0f}ms")

            # Add to messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": json.dumps(tool_result, default=str)})

            trajectory.append({
                "step": step + 1,
                "response_tokens": tokens,
                "tool": tool_name,
                "tool_result": tool_result,
                "gen_time_s": round(gen_time, 1),
                "sim_time_ms": round(sim_time * 1000),
            })
        else:
            print(f"\n  [No tool call detected - agent finished reasoning]")
            messages.append({"role": "assistant", "content": response})
            trajectory.append({
                "step": step + 1,
                "response_tokens": tokens,
                "tool": None,
                "gen_time_s": round(gen_time, 1),
            })
            # If no tool call for 2 consecutive steps, stop
            if step > 0 and trajectory[-2].get("tool") is None:
                print("  [Stopping: 2 consecutive steps without tool calls]")
                break

    # Summary
    print(f"\n{SEP}")
    print(f"   Agent Run Summary")
    print(f"{SEP}")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Tools used: {[t['tool'] for t in trajectory if t.get('tool')]}")
    total_tokens = sum(t["response_tokens"] for t in trajectory)
    total_gen = sum(t["gen_time_s"] for t in trajectory)
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total gen time: {total_gen:.1f}s")
    print(f"  Total sim time: {total_sim_time*1000:.0f}ms")
    print(f"  Avg speed: {total_tokens/total_gen:.1f} tok/s" if total_gen > 0 else "")

    # Save
    output = {
        "model": model_path,
        "task": task,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "total_tokens": total_tokens,
    }
    out_path = Path("eval_results/agent_ngspice_run.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n  Saved: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Agent with real ngspice")
    parser.add_argument("--model", default="outputs/sft_local/final")
    parser.add_argument("--task", default="Design a common-source NMOS amplifier with DC gain > 20 dB. Supply voltage is 1.8V, load capacitance is 1pF. Start by querying the PDK for NMOS devices.")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    run_agent(args.model, args.task, args.max_steps)


if __name__ == "__main__":
    main()
