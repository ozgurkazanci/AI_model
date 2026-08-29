#!/usr/bin/env python3
"""Full pipeline demonstration — runs everything end-to-end with mock.

This script demonstrates the COMPLETE ASIC-AI workflow without any external
dependencies (no GPU, no ngspice, no API key):

1. Load eval task
2. Create mock simulator
3. Run agent loop (scripted) through RL environment
4. Compute reward
5. Record trajectory
6. Format for SFT training
7. Validate format
8. Generate eval report

Usage:
    PYTHONPATH=src python scripts/demo_full_pipeline.py
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from asic_ai.data.format import (
    TOOL_DEFINITIONS,
    build_system_message,
    format_trajectory_for_sft,
    validate_sft_format,
)
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall
from asic_ai.training.rl_env import CircuitDesignEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demo")

SEPARATOR = "=" * 70


def load_sample_task() -> dict:
    """Load a sample OTA eval task."""
    task_path = Path("eval/tasks/analog/ota_2stage_001.yaml")
    if task_path.exists():
        with open(task_path) as f:
            return yaml.safe_load(f)

    # Fallback inline task
    return {
        "id": "ota_2stage_001",
        "description": "Two-stage CMOS OTA with Miller compensation",
        "category": "analog",
        "difficulty": "medium",
        "pdk": "sky130",
        "supply": 1.8,
        "load": "2pF",
        "specs": {
            "dc_gain": {"min": 60, "unit": "dB"},
            "ugb": {"min": 30e6, "unit": "Hz"},
            "phase_margin": {"min": 60, "unit": "deg"},
            "idd": {"max": 500e-6, "unit": "A"},
        },
    }


def run_demo():
    """Run the full pipeline demo."""
    print(f"\n{SEPARATOR}")
    print("   ASIC-AI Full Pipeline Demo")
    print(f"{SEPARATOR}\n")

    # =============================================
    # Step 1: Load eval task
    # =============================================
    print("[1/7] Loading eval task...")
    task = load_sample_task()
    print(f"  Task: {task['id']} ({task.get('category', '?')}/{task.get('difficulty', '?')})")
    print(f"  Description: {task.get('description', '?')}")
    specs = task.get("specs", {})
    for name, spec in specs.items():
        parts = []
        if "min" in spec:
            parts.append(f"min={spec['min']}")
        if "max" in spec:
            parts.append(f"max={spec['max']}")
        parts.append(spec.get("unit", ""))
        print(f"  Spec: {name}: {', '.join(parts)}")

    # =============================================
    # Step 2: Create RL environment with mock
    # =============================================
    print(f"\n[2/7] Creating RL environment with mock simulator...")

    def simple_reward_fn(specs, results):
        """Simple reward based on spec coverage."""
        measurements = results.get("measurements", {})
        if not measurements:
            return 0.0
        return min(1.0, len(measurements) / max(1, len(specs)) * 0.8)

    env = CircuitDesignEnv(
        adapter=None,
        reward_fn=simple_reward_fn,
        max_steps=15,
        step_penalty=0.005,
    )
    obs = env.reset(task)
    print(f"  Environment ready. Max steps: 15")
    print(f"  Initial observation: {obs[:100]}...")

    # =============================================
    # Step 3: Run scripted agent loop
    # =============================================
    print(f"\n[3/7] Running agent loop (scripted design actions)...")

    # Scripted design actions (simulating what the model would do)
    actions = [
        {
            "name": "pdk.list_devices",
            "arguments": {},
            "reasoning": "Query available devices in sky130 PDK",
        },
        {
            "name": "pdk.device_query",
            "arguments": {"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9},
            "reasoning": "Get NFET parameters for initial gm/ID estimation",
        },
        {
            "name": "lint.check",
            "arguments": {"netlist": ".subckt ota VDD VSS INP INM OUT\nXM1 net1 INM net3 VSS nfet_01v8 W=10u L=180n\n.ends"},
            "reasoning": "Check initial netlist structure",
        },
        {
            "name": "sim.ac",
            "arguments": {"netlist": ".subckt ota VDD VSS\n.ends", "start_freq": 1, "stop_freq": 10e9},
            "reasoning": "Run AC analysis to check gain and bandwidth",
        },
        {
            "name": "spec.check",
            "arguments": {
                "results": {"dc_gain": 55, "ugb": 25e6, "phase_margin": 58, "idd": 200e-6},
                "specs": specs,
            },
            "reasoning": "Check if current design meets specs",
        },
        {
            "name": "netlist.patch",
            "arguments": {
                "operations": [{"op": "modify_param", "target": "XM1", "value": "W=15u"}],
            },
            "reasoning": "Increase M1 width for higher gm -> better gain",
        },
        {
            "name": "sim.ac",
            "arguments": {"netlist": ".subckt ota VDD VSS\nXM1 W=15u\n.ends"},
            "reasoning": "Re-simulate after sizing change",
        },
        {
            "name": "spec.check",
            "arguments": {
                "results": {"dc_gain": 63, "ugb": 35e6, "phase_margin": 62, "idd": 280e-6},
                "specs": specs,
            },
            "reasoning": "Verify improved design meets specs",
        },
    ]

    trajectory_steps: list[TrajectoryStep] = []
    step_idx = 0

    # Initial user message
    trajectory_steps.append(TrajectoryStep(
        step_index=step_idx, role="user",
        content=f"Design: {task.get('description', task['id'])}. Specs: {json.dumps(specs)}",
    ))
    step_idx += 1

    for i, action in enumerate(actions):
        reasoning = action.pop("reasoning", "")

        # Record assistant step (thinking + tool call)
        trajectory_steps.append(TrajectoryStep(
            step_index=step_idx, role="assistant",
            content=reasoning,
            tool_call=ToolCall(
                name=action["name"],
                call_id=f"call_{step_idx:03d}",
                arguments=action["arguments"],
            ),
        ))
        step_idx += 1

        # Execute in environment
        result = env.step(action)
        print(f"  Step {i+1}/{len(actions)}: {action['name']} -> reward={result.reward:+.3f} (total={result.info['total_reward']:.3f})")

        # Record tool result
        trajectory_steps.append(TrajectoryStep(
            step_index=step_idx, role="tool",
            content=result.observation[:500],
        ))
        step_idx += 1

    # Final assistant summary
    trajectory_steps.append(TrajectoryStep(
        step_index=step_idx, role="assistant",
        content="Design complete. Two-stage OTA meets all specifications after sizing optimization.",
    ))

    summary = env.get_episode_summary()
    print(f"\n  Episode: {summary['steps']} steps, reward={summary['total_reward']:.3f}, success={summary['success']}")

    # =============================================
    # Step 4: Record trajectory
    # =============================================
    print(f"\n[4/7] Recording trajectory...")

    trajectory = Trajectory(
        id=f"demo_{uuid.uuid4().hex[:8]}",
        task_id=task["id"],
        steps=trajectory_steps,
        success=True,
        final_score=0.85,
        duration_seconds=summary.get("duration_sec", 5.0),
        metadata={"generator": "demo", "model": "scripted"},
    )
    print(f"  Trajectory ID: {trajectory.id}")
    print(f"  Steps: {len(trajectory.steps)}")
    print(f"  Score: {trajectory.final_score}")

    # =============================================
    # Step 5: Format for SFT training
    # =============================================
    print(f"\n[5/7] Formatting for SFT training...")

    sft_messages = format_trajectory_for_sft(trajectory)
    print(f"  Messages: {len(sft_messages)}")
    print(f"  Roles: {[m['role'] for m in sft_messages]}")
    print(f"  System prompt: {len(sft_messages[0]['content'])} chars")

    # =============================================
    # Step 6: Validate format
    # =============================================
    print(f"\n[6/7] Validating SFT format...")

    is_valid, errors = validate_sft_format(sft_messages)
    if is_valid:
        print(f"  Format: VALID")
    else:
        print(f"  Format: INVALID")
        for err in errors:
            print(f"    Error: {err}")

    # =============================================
    # Step 7: Save and summarize
    # =============================================
    print(f"\n[7/7] Summary...")

    # Save demo output
    output = {
        "id": trajectory.id,
        "task_id": trajectory.task_id,
        "messages": sft_messages,
        "score": trajectory.final_score,
        "success": trajectory.success,
    }

    output_path = Path("data/sft/demo_output.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"  Saved to: {output_path}")

    # Print pipeline summary
    print(f"\n{SEPARATOR}")
    print("   Pipeline Summary")
    print(f"{SEPARATOR}")
    print(f"  Task:            {task['id']}")
    print(f"  Steps:           {len(actions)}")
    print(f"  Reward:          {summary['total_reward']:.3f}")
    print(f"  SFT messages:    {len(sft_messages)}")
    print(f"  Format valid:    {is_valid}")
    print(f"  System prompt:   {len(build_system_message())} chars")
    print(f"  Tool count:      {len(TOOL_DEFINITIONS)}")
    print(f"  Output file:     {output_path}")
    print(f"\n  Pipeline: Task -> RL Env -> Agent Loop -> Reward -> Trajectory -> SFT Format -> Validated")
    print(f"\n{SEPARATOR}")
    print("  All components working! Ready for real model + simulator.")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    run_demo()
