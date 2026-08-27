#!/usr/bin/env python3
"""End-to-end integration example: Load eval task → compute reward → report.

This script demonstrates the complete pipeline without requiring
a simulator or trained model. It validates that all components
work together correctly.

Usage:
    PYTHONPATH=src python examples/e2e_integration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.reward.reward import RewardFunction, RewardMode
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall, TrajectoryDataset
from asic_ai.data.perturbation import (
    PerturbationPipeline,
    BiasShift,
    ScaleWL,
    RemoveComponent,
)
from asic_ai.data.validator import validate_trajectory, validate_dataset
from asic_ai.tokenizer.extend import get_new_tokens, TokenExtensionConfig
from asic_ai.tool_interface.schema import (
    SimParams,
    PVTCorner,
    AgentAction,
    ActionType,
    SpecCheckResult,
    SpecCheckDetail,
    AgentObservation,
)


def demo_reward_function() -> None:
    """Demo: Load eval task YAML and compute reward."""
    print("=" * 60)
    print("DEMO 1: Reward Function with Eval Task")
    print("=" * 60)

    # Load a real eval task
    task_path = Path("eval/tasks/analog/ota_2stage_001.yaml")
    with open(task_path) as f:
        task = yaml.safe_load(f)

    print(f"\nTask: {task['id']}")
    print(f"Description: {task.get('description', 'N/A')}")
    print(f"Difficulty: {task.get('difficulty', 'N/A')}")

    # Create reward function from eval task
    rf = RewardFunction.from_eval_task(task)
    print(f"Specs: {list(rf.specs.keys())}")
    print(f"Mode: {rf.mode.value}")

    # Simulate different design outcomes
    scenarios = {
        "Perfect design": {
            "dc_gain_db": 75.0,
            "ugb_hz": 80e6,
            "phase_margin_deg": 70.0,
            "current_a": 150e-6,
        },
        "Marginal design": {
            "dc_gain_db": 62.0,
            "ugb_hz": 52e6,
            "phase_margin_deg": 61.0,
            "current_a": 195e-6,
        },
        "Failed design": {
            "dc_gain_db": 45.0,
            "ugb_hz": 30e6,
            "phase_margin_deg": 40.0,
            "current_a": 350e-6,
        },
        "Convergence failure": {},
    }

    for name, results in scenarios.items():
        convergence_failed = name == "Convergence failure"
        reward = rf.compute(
            results=results,
            convergence_failed=convergence_failed,
        )
        print(f"\n  {name}:")
        print(f"    Total reward: {reward.total_reward:.4f}")
        print(f"    All specs met: {reward.all_specs_met}")
        if not convergence_failed:
            for spec_score in reward.spec_scores:
                status = "[OK]" if spec_score.met else "[FAIL]"
                print(f"    {status} {spec_score.name}: {spec_score.actual:.4g} (score: {spec_score.score:.3f})")


def demo_trajectory_pipeline() -> None:
    """Demo: Create, validate, and serialize a trajectory."""
    print("\n" + "=" * 60)
    print("DEMO 2: Trajectory Pipeline")
    print("=" * 60)

    # Create a sample trajectory
    traj = Trajectory(
        id="demo_traj_001",
        task_id="ota_2stage_001",
        steps=[
            TrajectoryStep(
                step_index=0,
                role="user",
                content="Design a two-stage OTA with 60dB gain, 50MHz UGB, 60° PM, ≤200µA on sky130.",
            ),
            TrajectoryStep(
                step_index=1,
                role="assistant",
                content="I'll use a two-stage Miller-compensated OTA. Let me start with initial sizing using gm/ID methodology.",
                tool_call=ToolCall(
                    name="pdk.device_query",
                    arguments={"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9},
                    call_id="call_001",
                ),
            ),
            TrajectoryStep(
                step_index=2,
                role="tool",
                content='{"gm": 1.5e-3, "gds": 50e-6, "id": 200e-6, "ft": 5e9, "vth": 0.45}',
                tool_result={"gm": 1.5e-3, "gds": 50e-6, "id": 200e-6, "ft": 5e9, "vth": 0.45},
            ),
            TrajectoryStep(
                step_index=3,
                role="assistant",
                content="gm/ID = 7.5 V⁻¹, which is reasonable for moderate inversion. Let me generate the initial netlist and simulate AC response.",
                tool_call=ToolCall(
                    name="sim.ac",
                    arguments={
                        "netlist": ".subckt ota VDD VSS INP INM OUT\n...\n.ends",
                        "params": {"analysis_type": "ac", "start": 1, "stop": 10e9, "points": 100},
                    },
                    call_id="call_002",
                ),
            ),
            TrajectoryStep(
                step_index=4,
                role="tool",
                content='{"gain_db": 55.0, "ugb_hz": 42000000, "phase_margin_deg": 52.0}',
                tool_result={"gain_db": 55.0, "ugb_hz": 42e6, "phase_margin_deg": 52.0},
            ),
            TrajectoryStep(
                step_index=5,
                role="assistant",
                content="Gain is 55dB (need 60dB) and UGB is 42MHz (need 50MHz). I need to increase first stage gain by widening M1/M2 and increase tail current for higher gm.",
                tool_call=ToolCall(
                    name="netlist.patch",
                    arguments={"diff": {"operations": [
                        {"op": "modify_param", "target": "XM1", "value": "W=15u"},
                        {"op": "modify_param", "target": "XM2", "value": "W=15u"},
                    ]}},
                    call_id="call_003",
                ),
            ),
        ],
        success=True,
        final_score=0.92,
        metadata={"model": "demo", "timestamp": "2026-08-28"},
        duration_seconds=45.0,
    )

    # Validate
    result = validate_trajectory(traj)
    print(f"\nTrajectory validation: {'[PASS]' if result.is_valid else '[FAIL]'}")
    if result.errors:
        for err in result.errors:
            print(f"  Error: {err}")

    # Serialize and deserialize
    jsonl = traj.to_jsonl()
    restored = Trajectory.from_jsonl(jsonl)
    print(f"Serialization round-trip: {'[PASS]' if restored.id == traj.id else '[FAIL]'}")

    # Convert to chat format (SFT training input)
    chat = traj.to_chat_format()
    print(f"Chat format messages: {len(chat)}")

    # Dataset operations
    dataset = TrajectoryDataset(trajectories=[traj])
    stats = dataset.statistics()
    print(f"Dataset stats: {json.dumps(stats, indent=2)}")


def demo_perturbation() -> None:
    """Demo: Synthetic perturbation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Synthetic Perturbation Pipeline")
    print("=" * 60)

    netlist = """\
.subckt two_stage_ota VDD VSS INP INM OUT VBIAS
XM1 net1 INM net3 VSS sky130_fd_pr__nfet_01v8 W=10u L=180n m=4
XM2 net2 INP net3 VSS sky130_fd_pr__nfet_01v8 W=10u L=180n m=4
XM3 net1 net1 VDD VDD sky130_fd_pr__pfet_01v8 W=20u L=180n m=4
XM4 net2 net1 VDD VDD sky130_fd_pr__pfet_01v8 W=20u L=180n m=4
XM5 net3 VBIAS VSS VSS sky130_fd_pr__nfet_01v8 W=5u L=500n m=2
Cc net2 OUT 2p
XM6 OUT net2 VDD VDD sky130_fd_pr__pfet_01v8 W=40u L=180n m=8
XM7 OUT VBIAS VSS VSS sky130_fd_pr__nfet_01v8 W=20u L=180n m=8
Ibias VDD VBIAS 100u
.ends"""

    pipeline = PerturbationPipeline(perturbations=[BiasShift(), ScaleWL(), RemoveComponent()])

    results = [pipeline.generate(netlist, seed=i) for i in range(5)]
    print(f"\nGenerated {len(results)} perturbations:")
    for i, result in enumerate(results):
        print(f"\n  [{i+1}] Types: {result.perturbations_applied}")
        print(f"      Description: {result.perturbations_applied}")
        # Show first difference
        orig_lines = result.original_netlist.strip().split("\n")
        pert_lines = result.perturbed_netlist.strip().split("\n")
        for j, (o, p) in enumerate(zip(orig_lines, pert_lines)):
            if o != p:
                print(f"      Changed line {j+1}:")
                print(f"        - {o.strip()}")
                print(f"        + {p.strip()}")
                break


def demo_tokenizer() -> None:
    """Demo: Tokenizer extension tokens."""
    print("\n" + "=" * 60)
    print("DEMO 4: Tokenizer Extension")
    print("=" * 60)

    tokens = get_new_tokens()
    print(f"\nTotal new tokens: {len(tokens)}")

    config = TokenExtensionConfig()
    print(f"SI prefixes: {config.si_prefixes}")
    print(f"sky130 devices: {len(config.sky130_devices)}")
    print(f"Netlist keywords: {len(config.netlist_keywords)}")
    print(f"Circuit terms: {len(config.circuit_terms)}")


def demo_schema() -> None:
    """Demo: Tool interface schema."""
    print("\n" + "=" * 60)
    print("DEMO 5: Tool Interface Schema")
    print("=" * 60)

    # Create a complete agent action
    action = AgentAction(
        action_type=ActionType.SIMULATE,
        arguments={
            "netlist": ".subckt test...",
            "params": {"analysis_type": "ac", "start": 1, "stop": 10e9},
        },
    )
    print(f"\nAgent action: {action.action_type.value}")

    # Create a spec check result
    spec_result = SpecCheckResult(
        score=0.85,
        breakdown={
            "gain_db": SpecCheckDetail(min_value=60.0, actual=65.0, met=True, score=0.9),
            "ugb_hz": SpecCheckDetail(min_value=50e6, actual=48e6, met=False, score=-0.1),
        },
    )
    print(f"Spec check score: {spec_result.score}")
    for name, detail in spec_result.breakdown.items():
        status = "[OK]" if detail.met else "[FAIL]"
        print(f"  {status} {name}: actual={detail.actual:.4g}, score={detail.score:.3f}")

    # PVT corners
    corners = [
        PVTCorner(process="tt", voltage=1.8, temperature=27.0),
        PVTCorner(process="ss", voltage=1.62, temperature=125.0),
        PVTCorner(process="ff", voltage=1.98, temperature=-40.0),
        PVTCorner(process="sf", voltage=1.8, temperature=27.0),
        PVTCorner(process="fs", voltage=1.8, temperature=27.0),
    ]
    print(f"\nPVT corners: {len(corners)}")
    for c in corners:
        print(f"  {c.process}: {c.voltage}V, {c.temperature}°C")


def main() -> None:
    """Run all demos."""
    print("ASIC-AI End-to-End Integration Demo")
    print("=" * 60)

    # Count eval tasks
    analog_tasks = list(Path("eval/tasks/analog").glob("*.yaml"))
    digital_tasks = list(Path("eval/tasks/digital").glob("*.yaml"))
    print(f"\nEval tasks: {len(analog_tasks)} analog + {len(digital_tasks)} digital = {len(analog_tasks) + len(digital_tasks)} total")

    demo_reward_function()
    demo_trajectory_pipeline()
    demo_perturbation()
    demo_tokenizer()
    demo_schema()

    print("\n" + "=" * 60)
    print("[OK] All integration demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
