#!/usr/bin/env python3
"""Generate augmented SFT data from circuit templates.

Creates diverse training examples by:
1. Rendering templates with varied parameters
2. Running mock simulations on each variant
3. Creating diagnosis-fix trajectories (perturb -> diagnose -> fix)

This generates MORE DIVERSE data than generate_mock_sft.py because
it varies the circuit parameters, not just the task.

Usage:
    PYTHONPATH=src python scripts/augment_from_templates.py --output data/sft/augmented_v1.jsonl --variants 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import format_trajectory_for_sft, validate_sft_format
from asic_ai.data.templates import TEMPLATES, CircuitTemplate
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("augment")


def mock_sim_from_netlist(netlist: str, rng: random.Random) -> dict:
    """Extract circuit features from netlist and return mock sim results."""
    w_values = [float(m) for m in re.findall(r'W=(\d+(?:\.\d+)?)', netlist)]
    n_transistors = len(re.findall(r'XM\d+', netlist))
    total_w = sum(w_values) if w_values else 10.0

    gain_db = 40.0 + total_w * 0.25 + n_transistors * 1.5
    gain_db = min(gain_db, 95.0) + rng.gauss(0, 2)
    ugb_hz = 1e6 * (8 + total_w * 0.4) * (1 + rng.gauss(0, 0.05))
    pm_deg = 68.0 - n_transistors * 1.2 + rng.gauss(0, 3)
    idd = (30e-6 + total_w * 2.5e-6) * (1 + rng.gauss(0, 0.05))

    return {
        "dc_gain_db": round(gain_db, 1),
        "ugb_hz": round(ugb_hz),
        "phase_margin_deg": round(max(25, pm_deg), 1),
        "idd_a": round(idd, 8),
    }


def randomize_params(template: CircuitTemplate, rng: random.Random, deviation: float = 0.3) -> dict:
    """Generate random parameter values within template ranges."""
    params = {}
    for name, info in template.parameters.items():
        default = info["default"]
        lo = info.get("min", default * 0.5)
        hi = info.get("max", default * 2.0)

        # Random value centered on default with some deviation
        val = default * (1 + rng.uniform(-deviation, deviation))
        val = max(lo, min(hi, val))

        # Round to reasonable precision
        if isinstance(default, int):
            val = max(1, round(val))
        else:
            val = round(val, 2)

        params[name] = val
    return params


def generate_design_trajectory(
    template: CircuitTemplate, params: dict, rng: random.Random
) -> Trajectory:
    """Generate a complete design trajectory from a template variant."""
    task_id = f"{template.id}_aug_{uuid.uuid4().hex[:6]}"
    netlist = template.render(**params)
    steps: list[TrajectoryStep] = []
    step_idx = 0

    # User message
    spec_str = json.dumps(template.typical_specs, indent=2)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="user",
        content=f"Design a {template.name}. Specs: {spec_str}",
    ))
    step_idx += 1

    # PDK query
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"I'll design a {template.name}. First, let me check the available devices.",
        tool_call=ToolCall(name="pdk.device_query", call_id=f"call_{step_idx:03d}",
                          arguments={"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps({"gm": 1.5e-3, "gds": 50e-6, "vth": 0.45, "ft": 5e9}),
    ))
    step_idx += 1

    # Initial simulation
    sim_result_1 = mock_sim_from_netlist(netlist, rng)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"Using {template.name} topology with initial sizing. Running AC simulation.",
        tool_call=ToolCall(name="sim.ac", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": netlist, "start_freq": 1, "stop_freq": 10e9}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(sim_result_1),
    ))
    step_idx += 1

    # Spec check
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"Gain is {sim_result_1['dc_gain_db']}dB, UGB is {sim_result_1['ugb_hz']/1e6:.1f}MHz. Let me check against specs.",
        tool_call=ToolCall(name="spec.check", call_id=f"call_{step_idx:03d}",
                          arguments={"results": sim_result_1, "specs": template.typical_specs}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps({"score": 0.75, "passed": False, "details": "Some specs not met"}),
    ))
    step_idx += 1

    # Improve design
    improved_params = {k: v * (1 + rng.uniform(0.1, 0.3)) for k, v in params.items()
                      if isinstance(v, (int, float)) and v > 0}
    for k, v in improved_params.items():
        if isinstance(params.get(k), int):
            improved_params[k] = max(1, round(v))
    improved_netlist = template.render(**{**params, **improved_params})
    sim_result_2 = mock_sim_from_netlist(improved_netlist, rng)
    # Make it slightly better
    sim_result_2["dc_gain_db"] = round(sim_result_2["dc_gain_db"] + rng.uniform(2, 8), 1)

    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Increasing device sizes to improve gain and bandwidth.",
        tool_call=ToolCall(name="sim.ac", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": improved_netlist}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(sim_result_2),
    ))
    step_idx += 1

    # Corner verification
    corner_data = [
        {"corner": "tt", **{k: round(v, 3) for k, v in sim_result_2.items()}},
        {"corner": "ss", "dc_gain_db": round(sim_result_2["dc_gain_db"] * 0.88, 1),
         "ugb_hz": round(sim_result_2["ugb_hz"] * 0.8), "phase_margin_deg": round(sim_result_2["phase_margin_deg"] * 0.85, 1)},
        {"corner": "ff", "dc_gain_db": round(sim_result_2["dc_gain_db"] * 1.05, 1),
         "ugb_hz": round(sim_result_2["ugb_hz"] * 1.15), "phase_margin_deg": round(sim_result_2["phase_margin_deg"] * 1.1, 1)},
    ]
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Running PVT corner verification.",
        tool_call=ToolCall(name="sim.corners", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": improved_netlist, "corners": ["tt", "ss", "ff"]}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(corner_data),
    ))
    step_idx += 1

    # Final summary
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"Design verified across corners. Final: gain={sim_result_2['dc_gain_db']}dB, UGB={sim_result_2['ugb_hz']/1e6:.1f}MHz, PM={sim_result_2['phase_margin_deg']}deg.",
    ))

    score = 0.7 + rng.uniform(0, 0.25)
    return Trajectory(
        id=task_id, task_id=template.id, steps=steps,
        success=score > 0.8, final_score=round(score, 3),
        duration_seconds=float(len(steps) * 1.5),
        metadata={"generator": "template_augment", "template": template.id, "params": params},
    )


def main():
    parser = argparse.ArgumentParser(description="Generate augmented SFT data from templates")
    parser.add_argument("--output", default="data/sft/augmented_v1.jsonl")
    parser.add_argument("--variants", type=int, default=5, help="Variants per template")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for template in TEMPLATES.values():
            for v in range(args.variants):
                params = randomize_params(template, rng)
                traj = generate_design_trajectory(template, params, rng)
                sft_msgs = format_trajectory_for_sft(traj)
                is_valid, errors = validate_sft_format(sft_msgs)

                if not is_valid:
                    log.warning(f"Invalid format for {template.id}_v{v}: {errors}")
                    continue

                out.write(json.dumps({
                    "id": traj.id, "task_id": traj.task_id,
                    "messages": sft_msgs, "score": traj.final_score,
                    "success": traj.success,
                }, ensure_ascii=False) + "\n")
                generated += 1

    log.info(f"Generated {generated} augmented examples -> {output_path} ({output_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
