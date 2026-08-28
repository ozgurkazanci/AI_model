#!/usr/bin/env python3
"""Generate SFT data covering underrepresented tools.

Current data is heavy on sim.ac, pdk.device_query but light on:
- sim.tran, sim.noise, sim.stb, sim.mc
- netlist.patch, lint.check, opt.suggest, meas.eval

This script generates targeted examples for each missing tool.

Usage:
    PYTHONPATH=src python scripts/generate_diverse_sft.py --output data/sft/diverse_tools_v1.jsonl
"""
from __future__ import annotations

import json
import logging
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import format_trajectory_for_sft, validate_sft_format
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diverse_sft")


def gen_tran_trajectory(rng: random.Random) -> Trajectory:
    """Transient simulation trajectory — settling time, slew rate analysis."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Design an OTA with settling time < 100ns and slew rate > 50V/us for sky130 PDK."),
        TrajectoryStep(step_index=1, role="assistant",
            content="I'll start by checking available devices, then run transient simulation to measure settling time.",
            tool_call=ToolCall(name="pdk.device_query", call_id="call_001",
                              arguments={"model": "nfet_01v8", "W": 10e-6, "L": 180e-9})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({"gm": 1.5e-3, "gds": 50e-6, "vth": 0.45, "ft": 5e9})),
        TrajectoryStep(step_index=3, role="assistant",
            content="Good device parameters. Now running transient simulation with a step input to measure settling.",
            tool_call=ToolCall(name="sim.tran", call_id="call_002",
                              arguments={"netlist": ".subckt ota VDD VSS VIN VOUT\n.ends",
                                        "tstep": 1e-9, "tstop": 500e-9, "stimulus": "step 0.9 1.0 10n"})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({
                "settling_time_ns": round(80 + rng.gauss(0, 15), 1),
                "slew_rate_vus": round(55 + rng.gauss(0, 10), 1),
                "overshoot_pct": round(5 + rng.gauss(0, 2), 1),
                "rise_time_ns": round(15 + rng.gauss(0, 3), 1),
            })),
        TrajectoryStep(step_index=5, role="assistant",
            content="Settling time and slew rate meet specs. Let me also check with a larger step."),
    ]
    return Trajectory(id=f"tran_{uuid.uuid4().hex[:6]}", task_id="tran_analysis",
                     steps=steps, success=True, final_score=round(0.7 + rng.uniform(0, 0.25), 3),
                     duration_seconds=8.5)


def gen_noise_trajectory(rng: random.Random) -> Trajectory:
    """Noise simulation trajectory — input-referred noise analysis."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Analyze input-referred noise of my LNA. Target: noise figure < 3dB at 1GHz."),
        TrajectoryStep(step_index=1, role="assistant",
            content="Running noise analysis to measure input-referred noise density across frequency.",
            tool_call=ToolCall(name="sim.noise", call_id="call_001",
                              arguments={"netlist": ".subckt lna VDD VSS VIN VOUT\n.ends",
                                        "output_node": "VOUT", "input_source": "VIN",
                                        "fstart": 1e6, "fstop": 10e9})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({
                "input_noise_density_1ghz": round(2.5e-9 + rng.gauss(0, 0.3e-9), 11),
                "integrated_noise_1mhz_1ghz": round(80e-6 + rng.gauss(0, 10e-6), 8),
                "noise_figure_1ghz_db": round(2.5 + rng.gauss(0, 0.5), 2),
                "dominant_source": "M1_thermal",
            })),
        TrajectoryStep(step_index=3, role="assistant",
            content="Noise figure is 2.5dB, within spec. Dominant noise source is M1 thermal noise. To reduce it further, increase W/L of M1.",
            tool_call=ToolCall(name="meas.eval", call_id="call_002",
                              arguments={"expression": "noise_figure_1ghz_db < 3.0",
                                        "value": 2.5})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({"result": True, "margin": 0.5})),
        TrajectoryStep(step_index=5, role="assistant",
            content="Noise specification met with 0.5dB margin. Design is complete."),
    ]
    return Trajectory(id=f"noise_{uuid.uuid4().hex[:6]}", task_id="noise_analysis",
                     steps=steps, success=True, final_score=round(0.8 + rng.uniform(0, 0.15), 3),
                     duration_seconds=7.2)


def gen_stb_trajectory(rng: random.Random) -> Trajectory:
    """Stability analysis trajectory — loop gain, phase margin."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Verify stability of my feedback amplifier. Need PM > 60deg."),
        TrajectoryStep(step_index=1, role="assistant",
            content="Running STB (stability) analysis to measure loop gain and phase margin.",
            tool_call=ToolCall(name="sim.stb", call_id="call_001",
                              arguments={"netlist": ".subckt fb_amp VDD VSS VIN VOUT\n.ends",
                                        "probe_element": "L0", "fstart": 1, "fstop": 10e9})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({
                "loop_gain_db": round(65 + rng.gauss(0, 5), 1),
                "phase_margin_deg": round(55 + rng.gauss(0, 8), 1),
                "gain_margin_db": round(12 + rng.gauss(0, 3), 1),
                "ugb_hz": round(30e6 + rng.gauss(0, 5e6)),
            })),
        TrajectoryStep(step_index=3, role="assistant",
            content="Phase margin is marginal. Adding Miller compensation capacitor to improve stability.",
            tool_call=ToolCall(name="netlist.patch", call_id="call_002",
                              arguments={"netlist": ".subckt fb_amp VDD VSS VIN VOUT\n.ends",
                                        "patch": "CC VOUT1 VOUT 2p"})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({"success": True, "modified_netlist": ".subckt fb_amp VDD VSS VIN VOUT\nCC VOUT1 VOUT 2p\n.ends"})),
        TrajectoryStep(step_index=5, role="assistant",
            content="Compensation added. Re-running stability analysis.",
            tool_call=ToolCall(name="sim.stb", call_id="call_003",
                              arguments={"netlist": ".subckt fb_amp VDD VSS VIN VOUT\nCC VOUT1 VOUT 2p\n.ends",
                                        "probe_element": "L0"})),
        TrajectoryStep(step_index=6, role="tool",
            content=json.dumps({
                "loop_gain_db": round(62 + rng.gauss(0, 3), 1),
                "phase_margin_deg": round(68 + rng.gauss(0, 4), 1),
                "gain_margin_db": round(15 + rng.gauss(0, 2), 1),
            })),
        TrajectoryStep(step_index=7, role="assistant",
            content="Phase margin improved to 68deg with compensation. Design is stable."),
    ]
    return Trajectory(id=f"stb_{uuid.uuid4().hex[:6]}", task_id="stability_analysis",
                     steps=steps, success=True, final_score=round(0.85 + rng.uniform(0, 0.1), 3),
                     duration_seconds=12.0)


def gen_mc_trajectory(rng: random.Random) -> Trajectory:
    """Monte Carlo trajectory — yield analysis."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Run Monte Carlo analysis to verify yield > 95% for my OTA gain spec."),
        TrajectoryStep(step_index=1, role="assistant",
            content="Running 100-point Monte Carlo simulation to evaluate yield.",
            tool_call=ToolCall(name="sim.mc", call_id="call_001",
                              arguments={"netlist": ".subckt ota VDD VSS VIN VOUT\n.ends",
                                        "num_runs": 100, "param_variation": "process+mismatch",
                                        "measure": "dc_gain_db"})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({
                "mean": round(62 + rng.gauss(0, 2), 1),
                "std": round(3.5 + rng.gauss(0, 0.5), 2),
                "min": round(53 + rng.gauss(0, 2), 1),
                "max": round(71 + rng.gauss(0, 2), 1),
                "yield_above_60db": round(82 + rng.uniform(0, 15), 1),
                "num_runs": 100,
            })),
        TrajectoryStep(step_index=3, role="assistant",
            content="Yield is only 82%, below 95% target. Need to increase gain margin. Increasing W/L ratio.",
            tool_call=ToolCall(name="netlist.patch", call_id="call_002",
                              arguments={"netlist": ".subckt ota VDD VSS VIN VOUT\n.ends",
                                        "patch": "XM1 ... W=20u L=0.5u"})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({"success": True})),
        TrajectoryStep(step_index=5, role="assistant",
            content="Patched netlist. Running Monte Carlo again.",
            tool_call=ToolCall(name="sim.mc", call_id="call_003",
                              arguments={"netlist": ".subckt ota_v2 VDD VSS VIN VOUT\n.ends",
                                        "num_runs": 100})),
        TrajectoryStep(step_index=6, role="tool",
            content=json.dumps({
                "mean": round(68 + rng.gauss(0, 2), 1),
                "std": round(2.8 + rng.gauss(0, 0.3), 2),
                "yield_above_60db": round(96 + rng.uniform(0, 3), 1),
            })),
        TrajectoryStep(step_index=7, role="assistant",
            content="Yield improved to 97%. Monte Carlo verification passed."),
    ]
    return Trajectory(id=f"mc_{uuid.uuid4().hex[:6]}", task_id="monte_carlo_yield",
                     steps=steps, success=True, final_score=round(0.9 + rng.uniform(0, 0.08), 3),
                     duration_seconds=15.0)


def gen_lint_trajectory(rng: random.Random) -> Trajectory:
    """Lint check + fix trajectory."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Check my netlist for common errors before simulation."),
        TrajectoryStep(step_index=1, role="assistant",
            content="Running lint check to find potential issues.",
            tool_call=ToolCall(name="lint.check", call_id="call_001",
                              arguments={"netlist": ".subckt amp VDD VSS VIN VOUT\nXM1 VOUT VIN n1 VSS nfet W=10u\n.ends"})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({
                "errors": [
                    {"severity": "warning", "message": "Node 'n1' has only one connection", "line": 3},
                    {"severity": "error", "message": "Missing L parameter for XM1", "line": 3},
                ],
                "total_errors": 1, "total_warnings": 1,
            })),
        TrajectoryStep(step_index=3, role="assistant",
            content="Found 1 error (missing L parameter) and 1 warning (floating node). Fixing both.",
            tool_call=ToolCall(name="netlist.patch", call_id="call_002",
                              arguments={"netlist": ".subckt amp VDD VSS VIN VOUT\nXM1 VOUT VIN n1 VSS nfet W=10u\n.ends",
                                        "patch": "XM1 VOUT VIN VSS VSS nfet W=10u L=0.18u"})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({"success": True})),
        TrajectoryStep(step_index=5, role="assistant",
            content="Fixed floating node and added L parameter. Running lint again.",
            tool_call=ToolCall(name="lint.check", call_id="call_003",
                              arguments={"netlist": ".subckt amp VDD VSS VIN VOUT\nXM1 VOUT VIN VSS VSS nfet W=10u L=0.18u\n.ends"})),
        TrajectoryStep(step_index=6, role="tool",
            content=json.dumps({"errors": [], "total_errors": 0, "total_warnings": 0})),
        TrajectoryStep(step_index=7, role="assistant",
            content="Lint clean. Netlist is ready for simulation."),
    ]
    return Trajectory(id=f"lint_{uuid.uuid4().hex[:6]}", task_id="lint_fix",
                     steps=steps, success=True, final_score=round(0.85 + rng.uniform(0, 0.1), 3),
                     duration_seconds=5.0)


def gen_opt_trajectory(rng: random.Random) -> Trajectory:
    """Optimizer-guided design trajectory."""
    steps = [
        TrajectoryStep(step_index=0, role="user",
            content="Optimize my OTA for maximum gain-bandwidth product under 200uA power budget."),
        TrajectoryStep(step_index=1, role="assistant",
            content="Using Bayesian optimizer to explore the design space.",
            tool_call=ToolCall(name="opt.suggest", call_id="call_001",
                              arguments={"objective": "maximize GBW",
                                        "constraints": {"idd_max": 200e-6},
                                        "parameters": {"W1": [2e-6, 50e-6], "W2": [5e-6, 100e-6], "Cc": [0.5e-12, 5e-12]}})),
        TrajectoryStep(step_index=2, role="tool",
            content=json.dumps({
                "suggested": {"W1": round(15e-6 + rng.gauss(0, 3e-6), 7),
                             "W2": round(30e-6 + rng.gauss(0, 5e-6), 7),
                             "Cc": round(2e-12 + rng.gauss(0, 0.5e-12), 13)},
                "iteration": 1, "acquisition_value": 0.85,
            })),
        TrajectoryStep(step_index=3, role="assistant",
            content="Optimizer suggests W1=15u, W2=30u, Cc=2p. Simulating.",
            tool_call=ToolCall(name="sim.ac", call_id="call_002",
                              arguments={"netlist": ".subckt ota_opt VDD VSS VIN VOUT\n.ends"})),
        TrajectoryStep(step_index=4, role="tool",
            content=json.dumps({
                "dc_gain_db": round(65 + rng.gauss(0, 3), 1),
                "ugb_hz": round(45e6 + rng.gauss(0, 5e6)),
                "gbw_product": round(45e6 * 10**(65/20), 0),
            })),
        TrajectoryStep(step_index=5, role="assistant",
            content="GBW product is excellent. Reporting back to optimizer for next iteration.",
            tool_call=ToolCall(name="spec.check", call_id="call_003",
                              arguments={"results": {"gbw": 45e6, "idd": 180e-6},
                                        "specs": {"gbw": {"min": 30e6}, "idd": {"max": 200e-6}}})),
        TrajectoryStep(step_index=6, role="tool",
            content=json.dumps({"score": 0.92, "passed": True})),
        TrajectoryStep(step_index=7, role="assistant",
            content="Optimization complete. All specs met with margin."),
    ]
    return Trajectory(id=f"opt_{uuid.uuid4().hex[:6]}", task_id="optimization",
                     steps=steps, success=True, final_score=round(0.88 + rng.uniform(0, 0.1), 3),
                     duration_seconds=18.0)


GENERATORS = [
    ("sim.tran", gen_tran_trajectory, 15),
    ("sim.noise", gen_noise_trajectory, 12),
    ("sim.stb", gen_stb_trajectory, 12),
    ("sim.mc", gen_mc_trajectory, 12),
    ("lint.check", gen_lint_trajectory, 10),
    ("opt.suggest", gen_opt_trajectory, 10),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate diverse tool-usage SFT data")
    parser.add_argument("--output", default="data/sft/diverse_tools_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for tool_name, gen_fn, count in GENERATORS:
            for i in range(count):
                traj = gen_fn(rng)
                sft_msgs = format_trajectory_for_sft(traj)
                is_valid, errors = validate_sft_format(sft_msgs)

                if not is_valid:
                    log.warning(f"Invalid: {tool_name}_{i}: {errors}")
                    continue

                out.write(json.dumps({
                    "id": traj.id, "task_id": traj.task_id,
                    "messages": sft_msgs, "score": traj.final_score,
                    "success": traj.success, "primary_tool": tool_name,
                }, ensure_ascii=False) + "\n")
                generated += 1

    log.info(f"Generated {generated} diverse examples -> {output_path} ({output_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
