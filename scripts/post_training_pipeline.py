#!/usr/bin/env python3
"""Post-training pipeline: runs all validation after training completes.

Automatically:
1. Check training metrics (loss curve)
2. Validate model with 5 prompts
3. Run E2E ngspice demo
4. Save comprehensive report

Usage:
    PYTHONPATH=src python scripts/post_training_pipeline.py
    PYTHONPATH=src python scripts/post_training_pipeline.py --model outputs/sft_local/final
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60


def run_script(name: str, args: list[str] = None) -> tuple[int, str]:
    """Run a script and capture output."""
    cmd = [sys.executable, f"scripts/{name}"] + (args or [])
    env_vars = {"PYTHONPATH": "src"}
    import os
    env = {**os.environ, **env_vars}

    print(f"\n  Running: {name}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env,
            cwd=str(Path(__file__).parent.parent),
        )
        output = result.stdout + result.stderr
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"  [{status}] Exit code: {result.returncode}")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}")
        return -1, "timeout"
    except Exception as e:
        print(f"  [ERROR] {e}")
        return -1, str(e)


def check_training_complete(model_path: str) -> dict:
    """Check if training is complete and gather metrics."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        return {"complete": False, "reason": f"Model not found: {model_path}"}

    # Check for adapter_config (LoRA) or config.json (merged)
    has_config = (model_dir / "adapter_config.json").exists() or (model_dir / "config.json").exists()
    if not has_config:
        return {"complete": False, "reason": "No model config found"}

    # Gather loss data from checkpoints
    loss_data = []
    parent = model_dir.parent
    for cp_dir in sorted(parent.glob("checkpoint-*")):
        state_file = cp_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            logs = [l for l in state.get("log_history", []) if "loss" in l]
            for l in logs:
                if l not in loss_data:
                    loss_data.append(l)

    return {
        "complete": True,
        "model_path": str(model_dir),
        "loss_data": loss_data,
        "initial_loss": loss_data[0]["loss"] if loss_data else None,
        "final_loss": loss_data[-1]["loss"] if loss_data else None,
        "best_loss": min(l["loss"] for l in loss_data) if loss_data else None,
        "total_steps": loss_data[-1]["step"] if loss_data else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-training pipeline")
    parser.add_argument("--model", default="outputs/sft_local/final")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Post-Training Pipeline")
    print(f"{SEP}")

    # Step 1: Check training
    print(f"\n{'='*40}")
    print(f"  Step 1: Training Status")
    print(f"{'='*40}")
    train_info = check_training_complete(args.model)

    if not train_info["complete"]:
        print(f"  Training not complete: {train_info['reason']}")
        print(f"  Waiting for training to finish...")
        return

    print(f"  Model: {train_info['model_path']}")
    print(f"  Loss: {train_info['initial_loss']:.4f} -> {train_info['final_loss']:.4f}")
    print(f"  Best: {train_info['best_loss']:.4f}")
    print(f"  Steps: {train_info['total_steps']}")
    reduction = (1 - train_info["best_loss"] / train_info["initial_loss"]) * 100
    print(f"  Reduction: {reduction:.1f}%")

    # Step 2: Model validation
    print(f"\n{'='*40}")
    print(f"  Step 2: Model Validation (5 prompts)")
    print(f"{'='*40}")
    rc2, out2 = run_script("validate_trained_model.py", ["--model", args.model])

    # Step 3: E2E ngspice demo
    print(f"\n{'='*40}")
    print(f"  Step 3: E2E AI + ngspice Demo")
    print(f"{'='*40}")
    rc3, out3 = run_script("demo_ai_ngspice.py", ["--model", args.model])

    # Step 4: Training monitor
    print(f"\n{'='*40}")
    print(f"  Step 4: Training Monitor Report")
    print(f"{'='*40}")
    rc4, out4 = run_script("training_monitor.py")

    # Summary
    print(f"\n{SEP}")
    print(f"   Post-Training Summary")
    print(f"{SEP}")
    print(f"  Training:   {'COMPLETE' if train_info['complete'] else 'INCOMPLETE'}")
    print(f"  Loss:       {train_info['initial_loss']:.4f} -> {train_info['best_loss']:.4f} ({reduction:.1f}% reduction)")
    print(f"  Validation: {'PASS' if rc2 == 0 else 'FAIL'}")
    print(f"  E2E Demo:   {'PASS' if rc3 == 0 else 'FAIL'}")
    print(f"  Monitor:    {'PASS' if rc4 == 0 else 'FAIL'}")

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "training": train_info,
        "validation_exit": rc2,
        "demo_exit": rc3,
        "monitor_exit": rc4,
    }
    out_path = Path("eval_results/post_training_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report: {out_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
