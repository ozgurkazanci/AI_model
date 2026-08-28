#!/usr/bin/env python3
"""Demo: RL environment with real ngspice simulation.

Shows how the RL training loop works with real SPICE results.
This is the foundation for GRPO training with simulator rewards.

Usage:
    PYTHONPATH=src python scripts/demo_rl_ngspice.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter, find_ngspice_dll
from asic_ai.adapters.base import AdapterConfig
from asic_ai.training.rl_env import CircuitDesignEnv
from asic_ai.reward import RewardFunction, SpecTarget

SEP = "=" * 60


def main():
    print(f"\n{SEP}")
    print("   RL Environment + Real ngspice Demo")
    print(f"{SEP}\n")

    dll = find_ngspice_dll()
    if not dll:
        print("  [FAIL] ngspice.dll not found!")
        return

    # Set up ngspice adapter
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        config = AdapterConfig(binary_path=dll, work_dir=td)
        adapter = NgspiceSharedAdapter(config)

        # Define spec targets for reward
        specs = [
            SpecTarget(name="dc_gain", min_val=20.0, weight=1.0, unit="dB"),
            SpecTarget(name="idd", max_val=500e-6, weight=0.5, unit="A"),
        ]
        reward_fn = RewardFunction(specs=specs)

        # Create RL environment with real simulator
        env = CircuitDesignEnv(
            adapter=adapter,
            reward_fn=reward_fn,
            max_steps=5,
        )

        # Define a design task
        task = {
            "id": "cs_amp_20db",
            "description": "Design a common-source amplifier with DC gain > 20 dB",
            "pdk": "sky130",
            "supply": 1.8,
            "load": "1pF",
            "specs": {
                "dc_gain": {"min": 20, "unit": "dB"},
                "supply": {"value": 1.8, "unit": "V"},
                "idd": {"max": 500e-6, "unit": "A"},
            },
        }

        # Episode starts
        print("  1. Starting episode...")
        obs = env.reset(task)
        print(f"     Task: {task['description']}")
        print(f"     Specs: {json.dumps(task['specs'], indent=2)[:200]}")

        # Write a circuit netlist (simulating what the model would generate)
        netlist = """\
* Common-Source Amplifier
.model nch nmos level=1 vto=0.5 kp=200u lambda=0.04
VDD vdd 0 DC 1.8
Vin gate 0 DC 0.7
RD vdd out 5k
M1 out gate 0 0 nch W=10u L=1u
CL out 0 1p
.dc Vin 0.3 1.2 0.01
.end
"""
        cir_path = Path(td) / "cs_amp.cir"
        cir_path.write_text(netlist, encoding="utf-8")

        # Step 1: Simulate DC
        print("\n  2. Agent action: sim.dc")
        action = {"name": "sim.dc", "arguments": {"netlist": str(cir_path)}}
        result = env.step(action)
        print(f"     Reward: {result.reward:.3f}")
        print(f"     Done: {result.done}")
        print(f"     Observation: {result.observation[:150]}...")

        # Step 2: Check specs
        print("\n  3. Agent action: spec.check")
        action = {
            "name": "spec.check",
            "arguments": {
                "specs": task["specs"],
                "results": {"dc_gain": 12, "idd": 200e-6},
            },
        }
        result = env.step(action)
        print(f"     Reward: {result.reward:.3f}")
        print(f"     Done: {result.done}")
        print(f"     Observation: {result.observation[:150]}...")

        # Step 3: Patch netlist (increase gain)
        print("\n  4. Agent action: netlist.patch (increase RD)")
        action = {
            "name": "netlist.patch",
            "arguments": {
                "netlist": netlist.replace("RD vdd out 5k", "RD vdd out 15k"),
                "changes": "Increased RD from 5k to 15k for higher gain",
            },
        }
        result = env.step(action)
        print(f"     Reward: {result.reward:.3f}")
        print(f"     Done: {result.done}")

        # Step 4: Re-simulate with improved design
        print("\n  5. Agent action: sim.dc (re-simulate)")
        improved_netlist = netlist.replace("RD vdd out 5k", "RD vdd out 15k")
        cir2_path = Path(td) / "cs_amp_v2.cir"
        cir2_path.write_text(improved_netlist, encoding="utf-8")
        action = {"name": "sim.dc", "arguments": {"netlist": str(cir2_path)}}
        result = env.step(action)
        print(f"     Reward: {result.reward:.3f}")
        print(f"     Done: {result.done}")
        print(f"     Observation: {result.observation[:150]}...")

        # Summary
        state = env.state
        print(f"\n{SEP}")
        print(f"   Episode Summary")
        print(f"{SEP}")
        print(f"  Steps:        {state.step}")
        print(f"  Total Reward: {state.total_reward:.3f}")
        print(f"  Success:      {state.success}")
        print(f"  Done:         {state.done}")
        print(f"\n  This demonstrates the RL loop:")
        print(f"    Task -> Model generates action -> ngspice simulates")
        print(f"    -> Reward computed -> Model improves -> Repeat")
        print(f"\n  For GRPO training, replace manual actions with LLM output.")
        print(f"{SEP}\n")

        # Save results
        output = {
            "task": task,
            "steps": state.step,
            "total_reward": state.total_reward,
            "history": state.history,
        }
        out_path = Path("eval_results/rl_ngspice_demo.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
