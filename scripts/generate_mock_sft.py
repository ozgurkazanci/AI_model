#!/usr/bin/env python3
"""Generate SFT training data using the mock simulator.

This script produces real training-ready JSONL files without needing
ngspice or an API key. Uses MockSimulatorAdapter for deterministic results.

Usage:
    PYTHONPATH=src python scripts/generate_mock_sft.py --output data/sft/mock_v1.jsonl --n-per-task 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from asic_ai.data.format import SYSTEM_PROMPT, TOOL_DEFINITIONS, format_trajectory_for_sft, validate_sft_format
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall
from asic_ai.reward.reward import RewardFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mock_sft")


def mock_ac(netlist: str, params: dict, rng: random.Random) -> dict:
    """Simple inline mock AC simulation based on netlist content."""
    # Count transistors and extract W values for realistic results
    import re
    w_values = [float(m) for m in re.findall(r'W=(\d+(?:\.\d+)?)', netlist)]
    n_transistors = len(re.findall(r'XM\d+', netlist))
    total_w = sum(w_values) if w_values else 10.0

    # More W/transistors = more gain and bandwidth
    gain_db = 40.0 + total_w * 0.3 + n_transistors * 2
    gain_db = min(gain_db, 90.0)
    ugb_hz = 1e6 * (10 + total_w * 0.5)
    pm_deg = 65.0 - n_transistors * 1.5 + rng.gauss(0, 2)
    idd = 50e-6 + total_w * 3e-6

    return {
        "gain_db": round(gain_db + rng.gauss(0, 1), 1),
        "ugb_hz": round(ugb_hz * (1 + rng.gauss(0, 0.05))),
        "phase_margin_deg": round(max(30, pm_deg), 1),
        "idd_a": round(idd * (1 + rng.gauss(0, 0.03)), 7),
    }


def mock_corners(netlist: str, pvt_list: list, rng: random.Random) -> list[dict]:
    """Mock corner simulation with realistic PVT derating."""
    nominal = mock_ac(netlist, {}, rng)
    results = []
    derating = {
        "tt": {"gain": 1.0, "speed": 1.0, "current": 1.0},
        "ss": {"gain": 0.88, "speed": 0.80, "current": 0.85},
        "ff": {"gain": 1.05, "speed": 1.15, "current": 1.15},
        "sf": {"gain": 0.95, "speed": 0.90, "current": 0.95},
        "fs": {"gain": 0.97, "speed": 1.05, "current": 1.02},
    }
    for pvt in pvt_list:
        proc = pvt.get("process", "tt") if isinstance(pvt, dict) else "tt"
        d = derating.get(proc, derating["tt"])
        results.append({
            "corner": proc,
            "gain_db": round(nominal["gain_db"] * d["gain"], 1),
            "ugb_hz": round(nominal["ugb_hz"] * d["speed"]),
            "phase_margin_deg": round(nominal["phase_margin_deg"] * d["speed"], 1),
            "idd_a": round(nominal["idd_a"] * d["current"], 7),
        })
    return results


# Template netlists for different topologies
NETLIST_TEMPLATES = {
    "ota": """\
.subckt ota VDD VSS INP INM OUT
XM1 net1 INM net3 VSS nfet_01v8 W={w1}u L=180n m=4
XM2 net2 INP net3 VSS nfet_01v8 W={w1}u L=180n m=4
XM3 net1 net1 VDD VDD pfet_01v8 W={w3}u L=180n m=4
XM4 net2 net1 VDD VDD pfet_01v8 W={w3}u L=180n m=4
XM5 net3 Vbn VSS VSS nfet_01v8 W={w5}u L=500n m=2
Cc net2 OUT {cc}p
XM6 OUT net2 VDD VDD pfet_01v8 W={w6}u L=180n m=8
XM7 OUT Vbn VSS VSS nfet_01v8 W={w7}u L=180n m=8
Ibias VDD Vbn {ibias}u
.ends""",

    "current_mirror": """\
.subckt cm VDD VSS IN OUT
XM1 IN IN VSS VSS nfet_01v8 W={w1}u L={l1}n m={m1}
XM2 OUT IN VSS VSS nfet_01v8 W={w1}u L={l1}n m={m2}
.ends""",

    "source_follower": """\
.subckt sf VDD VSS IN OUT
XM1 VDD IN OUT VSS nfet_01v8 W={w1}u L=180n m=4
Ibias OUT VSS {ibias}u
.ends""",
}


def generate_ota_trajectory(task: dict, rng: random.Random, attempt: int) -> Trajectory:
    """Generate a complete OTA design trajectory using mock simulator."""
    task_id = task["id"]
    steps: list[TrajectoryStep] = []
    step_idx = 0

    # Step 0: User provides the task
    specs = task.get("specs", {})
    spec_summary = ", ".join(f"{k}: {v}" for k, v in specs.items())
    steps.append(TrajectoryStep(
        step_index=step_idx, role="user",
        content=f"Design task: {task.get('description', task_id)}. Specs: {spec_summary}. PDK: {task.get('pdk', 'sky130')}, VDD: {task.get('supply', 1.8)}V, Load: {task.get('load', '2p')}.",
    ))
    step_idx += 1

    # Step 1: Query PDK
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Let me start by querying the PDK for nfet_01v8 parameters to determine initial sizing using gm/ID methodology.",
        tool_call=ToolCall(name="pdk.device_query", call_id=f"call_{step_idx:03d}",
                          arguments={"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9}),
    ))
    step_idx += 1
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps({"gm": 1.5e-3, "gds": 50e-6, "id": 200e-6, "ft": 5e9, "vth": 0.45, "cgs": 20e-15, "cgd": 5e-15}),
    ))
    step_idx += 1

    # Step 2: Initial design with small W
    w1_init = 8 + attempt * 2
    w3_init = 16 + attempt * 4
    netlist_v1 = NETLIST_TEMPLATES["ota"].format(
        w1=w1_init, w3=w3_init, w5=5, w6=30, w7=15, cc=2, ibias=100,
    )
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"gm/ID = 7.5 V^-1, reasonable for moderate inversion. Initial sizing: M1/M2 W={w1_init}u, M3/M4 W={w3_init}u. Let me simulate AC response.",
        tool_call=ToolCall(name="sim.ac", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": netlist_v1, "start_freq": 1, "stop_freq": 10e9}),
    ))
    step_idx += 1

    # Mock AC result - initial (intentionally below spec)
    ac_result_1 = mock_ac(netlist_v1, {"start": 1, "stop": 10e9}, rng)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(ac_result_1),
    ))
    step_idx += 1

    # Step 3: Diagnose and improve
    w1_v2 = w1_init + 5
    w3_v2 = w3_init + 8
    netlist_v2 = NETLIST_TEMPLATES["ota"].format(
        w1=w1_v2, w3=w3_v2, w5=6, w6=40, w7=20, cc=2.5, ibias=120,
    )
    gain_1 = ac_result_1.get("gain_db", 55)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"Gain is {gain_1:.1f}dB, need improvement. Increasing M1/M2 to {w1_v2}u for higher gm, M3/M4 to {w3_v2}u for lower gds. Also increasing tail current to 120uA.",
        tool_call=ToolCall(name="sim.ac", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": netlist_v2, "start_freq": 1, "stop_freq": 10e9}),
    ))
    step_idx += 1

    ac_result_2 = mock_ac(netlist_v2, {"start": 1, "stop": 10e9}, rng)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(ac_result_2),
    ))
    step_idx += 1

    # Step 4: Fine-tune compensation
    netlist_v3 = NETLIST_TEMPLATES["ota"].format(
        w1=w1_v2 + 3, w3=w3_v2 + 4, w5=7, w6=50, w7=25, cc=3, ibias=130,
    )
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Getting closer. Adjusting compensation capacitor to 3pF for better phase margin, and fine-tuning second stage sizing.",
        tool_call=ToolCall(name="sim.ac", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": netlist_v3, "start_freq": 1, "stop_freq": 10e9}),
    ))
    step_idx += 1

    ac_result_3 = mock_ac(netlist_v3, {"start": 1, "stop": 10e9}, rng)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(ac_result_3),
    ))
    step_idx += 1

    # Step 5: Corner verification
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Nominal specs look good. Running corner verification across tt, ss, ff, sf, fs.",
        tool_call=ToolCall(name="sim.corners", call_id=f"call_{step_idx:03d}",
                          arguments={"netlist": netlist_v3, "corners": ["tt", "ss", "ff", "sf", "fs"]}),
    ))
    step_idx += 1

    corner_results = mock_corners(netlist_v3, [
        {"process": "tt", "voltage": 1.8, "temperature": 27},
        {"process": "ss", "voltage": 1.62, "temperature": 125},
        {"process": "ff", "voltage": 1.98, "temperature": -40},
    ], rng)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="tool",
        content=json.dumps(corner_results),
    ))
    step_idx += 1

    # Step 6: Final summary
    gain_final = ac_result_3.get("gain_db", 65)
    ugb_final = ac_result_3.get("ugb_hz", 60e6)
    pm_final = ac_result_3.get("phase_margin_deg", 62)
    steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content=f"Design complete. Final performance at nominal: gain={gain_final:.1f}dB, UGB={ugb_final/1e6:.1f}MHz, PM={pm_final:.1f}deg. All corners verified.",
    ))

    # Score with reward
    score = 0.85 + attempt * 0.03  # Slightly better each attempt
    success = score > 0.8

    return Trajectory(
        id=f"mock_{task_id}_{attempt:03d}_{uuid.uuid4().hex[:8]}",
        task_id=task_id,
        steps=steps,
        success=success,
        final_score=min(score, 1.0),
        duration_seconds=float(len(steps) * 2),
        metadata={"generator": "mock_sft", "attempt": attempt, "model": "mock"},
    )


def generate_simple_trajectory(task: dict, rng: random.Random, attempt: int) -> Trajectory:
    """Generate a simplified trajectory for non-OTA tasks."""
    task_id = task["id"]
    steps = [
        TrajectoryStep(step_index=0, role="user",
                      content=f"Design task: {task.get('description', task_id)}"),
        TrajectoryStep(step_index=1, role="assistant",
                      content="Let me analyze the requirements and query the PDK.",
                      tool_call=ToolCall(name="pdk.device_query", call_id="call_001",
                                        arguments={"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9})),
        TrajectoryStep(step_index=2, role="tool",
                      content=json.dumps({"gm": 1.5e-3, "gds": 50e-6, "id": 200e-6, "vth": 0.45})),
        TrajectoryStep(step_index=3, role="assistant",
                      content="Based on device parameters, I'll proceed with initial sizing and simulate.",
                      tool_call=ToolCall(name="sim.dc", call_id="call_002",
                                        arguments={"netlist": ".subckt test VDD VSS\n.ends"})),
        TrajectoryStep(step_index=4, role="tool",
                      content=json.dumps({"v_out": 0.9, "i_total": 150e-6})),
        TrajectoryStep(step_index=5, role="assistant",
                      content="Initial results look reasonable. Design complete for this prototype."),
    ]

    return Trajectory(
        id=f"mock_{task_id}_{attempt:03d}_{uuid.uuid4().hex[:8]}",
        task_id=task_id, steps=steps,
        success=True, final_score=0.7 + attempt * 0.05,
        duration_seconds=12.0,
        metadata={"generator": "mock_sft_simple", "attempt": attempt},
    )


def main():
    parser = argparse.ArgumentParser(description="Generate mock SFT training data")
    parser.add_argument("--tasks", default="eval/tasks/analog", help="Tasks directory")
    parser.add_argument("--output", default="data/sft/mock_v1.jsonl", help="Output JSONL")
    parser.add_argument("--n-per-task", type=int, default=3, help="Trajectories per task")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks (0=all)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    # Load tasks
    tasks = []
    for f in sorted(Path(args.tasks).rglob("*.yaml")):
        with open(f) as fh:
            tasks.append(yaml.safe_load(fh))

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    log.info(f"Generating {args.n_per_task} trajectories x {len(tasks)} tasks = {args.n_per_task * len(tasks)} total")

    generated = 0
    successful = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for task in tasks:
            task_id = task.get("id", "unknown")
            is_ota = "ota" in task_id.lower()

            for attempt in range(args.n_per_task):
                try:
                    if is_ota:
                        traj = generate_ota_trajectory(task, rng, attempt)
                    else:
                        traj = generate_simple_trajectory(task, rng, attempt)

                    # Format for SFT
                    sft_messages = format_trajectory_for_sft(traj)
                    is_valid, errors = validate_sft_format(sft_messages)

                    if not is_valid:
                        log.warning(f"Format validation failed for {task_id}_{attempt}: {errors}")
                        continue

                    # Write as JSONL (each line = one training example)
                    out.write(json.dumps({
                        "id": traj.id,
                        "task_id": traj.task_id,
                        "messages": sft_messages,
                        "score": traj.final_score,
                        "success": traj.success,
                    }, ensure_ascii=False) + "\n")

                    generated += 1
                    if traj.success:
                        successful += 1

                except Exception as e:
                    log.error(f"Error generating {task_id}_{attempt}: {e}")

    log.info(f"Done! Generated: {generated}, Successful: {successful}")
    log.info(f"Output: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
