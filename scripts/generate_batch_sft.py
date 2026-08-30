#!/usr/bin/env python3
"""Batch SFT data generator - produces 500+ examples programmatically.

Generates diverse training examples by combining:
- Circuit topologies (20+)
- Component values (randomized within realistic ranges)
- Simulation types (DC, AC, tran, corners, MC)
- Spec targets (gain, BW, power, noise, etc.)

Usage:
    PYTHONPATH=src python scripts/generate_batch_sft.py
    PYTHONPATH=src python scripts/generate_batch_sft.py --count 1000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message
SEP = "=" * 60

# Circuit topology templates
TOPOLOGIES = [
    {"name": "common_source", "type": "analog", "complexity": "easy",
     "desc": "Common-source amplifier", "key_specs": ["gain", "bandwidth", "power"]},
    {"name": "common_gate", "type": "analog", "complexity": "easy",
     "desc": "Common-gate amplifier", "key_specs": ["gain", "input_impedance", "bandwidth"]},
    {"name": "source_follower", "type": "analog", "complexity": "easy",
     "desc": "Source follower (common drain)", "key_specs": ["gain", "output_impedance", "bandwidth"]},
    {"name": "cascode", "type": "analog", "complexity": "medium",
     "desc": "Cascode amplifier", "key_specs": ["gain", "bandwidth", "output_impedance"]},
    {"name": "diff_pair", "type": "analog", "complexity": "medium",
     "desc": "Differential pair", "key_specs": ["gain", "cmrr", "offset"]},
    {"name": "folded_cascode", "type": "analog", "complexity": "hard",
     "desc": "Folded cascode OTA", "key_specs": ["gain", "gbw", "phase_margin", "slew_rate"]},
    {"name": "two_stage_ota", "type": "analog", "complexity": "hard",
     "desc": "Two-stage Miller OTA", "key_specs": ["gain", "gbw", "phase_margin", "power"]},
    {"name": "telescopic_ota", "type": "analog", "complexity": "hard",
     "desc": "Telescopic cascode OTA", "key_specs": ["gain", "gbw", "swing"]},
    {"name": "current_mirror", "type": "analog", "complexity": "easy",
     "desc": "Current mirror", "key_specs": ["ratio_accuracy", "output_impedance", "compliance"]},
    {"name": "bandgap", "type": "analog", "complexity": "hard",
     "desc": "Bandgap voltage reference", "key_specs": ["tc", "psrr", "accuracy"]},
    {"name": "ldo", "type": "analog", "complexity": "hard",
     "desc": "LDO voltage regulator", "key_specs": ["dropout", "load_regulation", "psrr"]},
    {"name": "comparator", "type": "analog", "complexity": "medium",
     "desc": "Voltage comparator", "key_specs": ["delay", "offset", "resolution"]},
    {"name": "schmitt_trigger", "type": "analog", "complexity": "easy",
     "desc": "Schmitt trigger", "key_specs": ["hysteresis", "threshold", "delay"]},
    {"name": "ring_oscillator", "type": "analog", "complexity": "medium",
     "desc": "Ring oscillator", "key_specs": ["frequency", "phase_noise", "power"]},
    {"name": "charge_pump", "type": "analog", "complexity": "medium",
     "desc": "Charge pump", "key_specs": ["current_matching", "clock_feedthrough", "output_impedance"]},
    {"name": "inverter", "type": "digital", "complexity": "easy",
     "desc": "CMOS inverter", "key_specs": ["threshold", "delay", "power"]},
    {"name": "nand_gate", "type": "digital", "complexity": "easy",
     "desc": "CMOS NAND gate", "key_specs": ["delay", "noise_margin", "fanout"]},
    {"name": "dff", "type": "digital", "complexity": "medium",
     "desc": "D flip-flop", "key_specs": ["setup_time", "hold_time", "clk_to_q"]},
    {"name": "sram_cell", "type": "digital", "complexity": "hard",
     "desc": "6T SRAM cell", "key_specs": ["snm", "read_current", "write_margin"]},
    {"name": "level_shifter", "type": "digital", "complexity": "medium",
     "desc": "Voltage level shifter", "key_specs": ["delay", "power", "voltage_range"]},
]

# Simulation analysis types
ANALYSES = [
    {"type": "dc", "tool": "sim.dc", "desc": "DC operating point / sweep"},
    {"type": "ac", "tool": "sim.ac", "desc": "AC frequency response"},
    {"type": "tran", "tool": "sim.tran", "desc": "Transient simulation"},
    {"type": "noise", "tool": "sim.noise", "desc": "Noise analysis"},
    {"type": "stb", "tool": "sim.stb", "desc": "Stability (loop gain) analysis"},
    {"type": "corners", "tool": "sim.corners", "desc": "PVT corner sweep"},
    {"type": "mc", "tool": "sim.mc", "desc": "Monte Carlo yield analysis"},
]

# PDK options
PDKS = ["sky130", "gf180mcu", "tsmc28nm", "tsmc65lp"]
SUPPLIES = {"sky130": 1.8, "gf180mcu": 3.3, "tsmc28nm": 0.9, "tsmc65lp": 1.2}

# Spec ranges for random generation
SPEC_RANGES = {
    "gain": {"min": 10, "max": 80, "unit": "dB"},
    "bandwidth": {"min": 1e6, "max": 1e9, "unit": "Hz"},
    "gbw": {"min": 1e6, "max": 5e9, "unit": "Hz"},
    "phase_margin": {"min": 45, "max": 75, "unit": "degrees"},
    "power": {"min": 10e-6, "max": 10e-3, "unit": "W"},
    "cmrr": {"min": 40, "max": 100, "unit": "dB"},
    "offset": {"min": 0.1e-3, "max": 10e-3, "unit": "V"},
    "slew_rate": {"min": 1e6, "max": 1e9, "unit": "V/s"},
    "tc": {"min": 1, "max": 50, "unit": "ppm/C"},
    "psrr": {"min": 30, "max": 80, "unit": "dB"},
    "dropout": {"min": 0.1, "max": 0.5, "unit": "V"},
    "frequency": {"min": 100e6, "max": 10e9, "unit": "Hz"},
    "delay": {"min": 10e-12, "max": 10e-9, "unit": "s"},
    "noise_margin": {"min": 0.3, "max": 0.8, "unit": "V"},
    "threshold": {"min": 0.3, "max": 0.6, "unit": "V"},
    "snm": {"min": 100e-3, "max": 400e-3, "unit": "V"},
}


def format_value(val, unit):
    """Format a value with appropriate engineering prefix."""
    prefixes = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""),
                (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")]
    for threshold, prefix in prefixes:
        if abs(val) >= threshold:
            return f"{val/threshold:.1f} {prefix}{unit}"
    return f"{val:.3e} {unit}"


def generate_spec_string(topology):
    """Generate random but realistic specs for a topology."""
    specs = []
    for spec_name in topology["key_specs"]:
        if spec_name in SPEC_RANGES:
            sr = SPEC_RANGES[spec_name]
            if sr["max"] / sr["min"] > 100:  # Log scale
                val = math.exp(random.uniform(math.log(sr["min"]), math.log(sr["max"])))
            else:
                val = random.uniform(sr["min"], sr["max"])
            specs.append(f"{spec_name.replace('_', ' ')} > {format_value(val, sr['unit'])}")
    return ", ".join(specs[:3])


def generate_design_response(topology, analysis, pdk, specs_str):
    """Generate a design response with tool calls."""
    supply = SUPPLIES.get(pdk, 1.8)
    
    # Opening analysis
    response = f"I'll design a {topology['desc'].lower()} in {pdk} ({supply}V supply).\n"
    response += f"Target specs: {specs_str}\n\n"
    
    # Design approach
    if topology["complexity"] == "easy":
        response += f"This is a straightforward {topology['desc'].lower()} design.\n"
    elif topology["complexity"] == "medium":
        response += f"Key design considerations for {topology['desc'].lower()}:\n"
        response += "- Need to balance gain vs bandwidth\n"
        response += "- Bias point selection critical for linearity\n"
    else:
        response += f"Complex design requiring careful optimization:\n"
        response += "- Multiple trade-offs between specs\n"
        response += "- May need iterative refinement\n"
    
    # Tool call
    response += f"\nLet me run a {analysis['desc'].lower()}:\n\n"
    response += "<tool_call>\n"
    response += f'  {{"name": "{analysis["tool"]}", "arguments": {{"analysis_type": "{analysis["type"]}"}}}}\n'
    response += "</tool_call>"
    
    return response


def generate_tool_response(analysis, topology):
    """Generate realistic tool response data."""
    if analysis["type"] == "dc":
        return json.dumps({
            "op_points": {
                "vout": round(random.uniform(0.5, 1.5), 3),
                "id_m1": round(random.uniform(10e-6, 500e-6), 6),
            },
            "sweeps": {"sweep": {
                "name": "sweep",
                "x_values": [round(x*0.1, 1) for x in range(5, 15)],
                "y_values": [round(random.uniform(0.3, 1.7), 3) for _ in range(10)],
            }}
        })
    elif analysis["type"] == "ac":
        freqs = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
        gain_dc = random.uniform(20, 60)
        gains = [round(gain_dc * max(0.01, 1 - (i*0.15)**2), 1) for i in range(7)]
        return json.dumps({
            "frequencies": freqs,
            "signals": {"vout": {
                "name": "vout",
                "x_values": freqs,
                "y_values": gains,
            }}
        })
    elif analysis["type"] == "tran":
        times = [round(i * 1e-9, 10) for i in range(11)]
        vals = [round(0.9 + 0.5 * math.sin(2*math.pi*1e8*t) + random.gauss(0, 0.01), 4) for t in times]
        return json.dumps({
            "time": times,
            "signals": {"vout": {"name": "vout", "x_values": times, "y_values": vals}}
        })
    else:
        return json.dumps({"status": "success", "data": {"metric": round(random.uniform(0.5, 50), 2)}})


def generate_analysis_response(analysis, topology, specs_str):
    """Generate the analysis of results."""
    response = f"Results look "
    if random.random() > 0.3:
        response += "good. All target specs are within range.\n\n"
        response += f"The {topology['desc'].lower()} meets the requirements: {specs_str}.\n"
        response += "Key design decisions that enabled this:\n"
        if topology["type"] == "analog":
            response += "- Bias current optimized for gm/Id target\n"
            response += "- Device sizing balances gain and bandwidth\n"
        else:
            response += "- Transistor sizing optimized for speed/power balance\n"
            response += "- Logic thresholds well-centered\n"
    else:
        response += "marginal. Some specs need improvement.\n\n"
        response += "I'll need to iterate on the design by adjusting device sizing.\n\n"
        response += "<tool_call>\n"
        response += '  {"name": "netlist.patch", "arguments": {"changes": "Adjusted W/L ratios for better performance"}}\n'
        response += "</tool_call>"
    
    return response


def generate_example(idx):
    """Generate one SFT training example."""
    topology = random.choice(TOPOLOGIES)
    analysis = random.choice(ANALYSES[:5])  # Stick to common analyses
    pdk = random.choice(PDKS)
    specs_str = generate_spec_string(topology)
    supply = SUPPLIES.get(pdk, 1.8)
    
    user_prompt = f"Design a {topology['desc'].lower()} with {specs_str} in {pdk} {supply}V."
    
    design_response = generate_design_response(topology, analysis, pdk, specs_str)
    tool_response = generate_tool_response(analysis, topology)
    analysis_response = generate_analysis_response(analysis, topology, specs_str)
    
    messages = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": design_response},
        {"role": "tool", "content": tool_response},
        {"role": "assistant", "content": analysis_response},
    ]
    
    return {
        "messages": messages,
        "source": "batch_v1",
        "circuit_id": f"batch_{topology['name']}_{idx:04d}",
        "domain": topology["type"],
        "complexity": topology["complexity"],
    }


def main():
    parser = argparse.ArgumentParser(description="Batch SFT Data Generator")
    parser.add_argument("--count", type=int, default=500, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data/sft/batch_v1.jsonl", help="Output file")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = args.output

    print(f"\n{SEP}")
    print(f"   Batch SFT Data Generator ({args.count} examples)")
    print(f"{SEP}\n")

    examples = []
    topology_counts = {}
    complexity_counts = {"easy": 0, "medium": 0, "hard": 0}
    domain_counts = {"analog": 0, "digital": 0}

    for i in range(args.count):
        ex = generate_example(i)
        examples.append(ex)
        
        topo = ex["circuit_id"].split("_")[1]
        topology_counts[topo] = topology_counts.get(topo, 0) + 1
        complexity_counts[ex["complexity"]] += 1
        domain_counts[ex["domain"]] += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Generated: {len(examples)} examples")
    print(f"  Topologies: {len(topology_counts)}")
    for name, count in sorted(topology_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:20s}: {count}")
    print(f"  Complexity: easy={complexity_counts['easy']}, medium={complexity_counts['medium']}, hard={complexity_counts['hard']}")
    print(f"  Domain: analog={domain_counts['analog']}, digital={domain_counts['digital']}")
    print(f"  Saved: {output_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
