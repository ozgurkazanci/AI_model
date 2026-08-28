#!/usr/bin/env python3
"""Monitor and visualize training progress.

Reads training logs and generates a text-based progress report.

Usage:
    PYTHONPATH=src python scripts/training_monitor.py --output-dir outputs/sft_local
    PYTHONPATH=src python scripts/training_monitor.py --log-file path/to/trainer_state.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEP = "=" * 60


def read_trainer_state(output_dir: str) -> dict | None:
    """Read trainer_state.json from output directory."""
    for p in Path(output_dir).rglob("trainer_state.json"):
        return json.loads(p.read_text())
    return None


def format_loss_chart(log_history: list[dict], width: int = 50) -> str:
    """Create ASCII loss chart."""
    losses = [(entry.get("step", 0), entry["loss"]) for entry in log_history if "loss" in entry]
    if not losses:
        return "  No loss data available yet."

    min_loss = min(l for _, l in losses)
    max_loss = max(l for _, l in losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    lines = ["  Loss over training steps:", ""]
    for step, loss in losses:
        bar_len = int((loss - min_loss) / loss_range * width)
        bar = "#" * max(1, bar_len)
        lines.append(f"  Step {step:4d} | {loss:.4f} |{bar}")

    lines.append(f"\n  Min loss: {min_loss:.4f} | Max loss: {max_loss:.4f}")
    if len(losses) >= 2:
        improvement = losses[0][1] - losses[-1][1]
        pct = improvement / losses[0][1] * 100 if losses[0][1] > 0 else 0
        lines.append(f"  Improvement: {improvement:.4f} ({pct:.1f}%)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--output-dir", default="outputs/sft_local")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Training Monitor")
    print(f"{SEP}\n")

    state = read_trainer_state(args.output_dir)
    if not state:
        print(f"  No trainer_state.json found in {args.output_dir}")
        print(f"  Training may still be in progress.")
        print(f"  Check: ls {args.output_dir}/*/trainer_state.json")
        return

    # Basic info
    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    max_steps = state.get("max_steps", 0)
    best_metric = state.get("best_metric")
    log_history = state.get("log_history", [])

    pct = global_step / max_steps * 100 if max_steps else 0

    print(f"  Step:     {global_step}/{max_steps} ({pct:.1f}%)")
    print(f"  Epoch:    {epoch:.2f}")
    if best_metric is not None:
        print(f"  Best:     {best_metric:.4f}")

    # Checkpoints
    checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        print(f"  Checkpoints: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            print(f"    {cp.name}")

    # Final model
    final = Path(args.output_dir) / "final"
    if final.exists():
        info_path = final / "training_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            print(f"\n  Final model: {final}")
            print(f"    Base: {info.get('base_model', '?')}")
            print(f"    LoRA: r={info.get('lora_r', '?')}, alpha={info.get('lora_alpha', '?')}")
            print(f"    Trainable: {info.get('trainable_params_m', '?')}M / {info.get('total_params_m', '?')}M")
            print(f"    Epochs: {info.get('epochs', '?')}")
            print(f"    Examples: {info.get('examples', '?')}")

    # Loss chart
    if log_history:
        print(f"\n{format_loss_chart(log_history)}")

    # Training loss summary
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    if train_entries:
        first_loss = train_entries[0]["loss"]
        last_loss = train_entries[-1]["loss"]
        print(f"\n  First loss: {first_loss:.4f}")
        print(f"  Last loss:  {last_loss:.4f}")
        print(f"  Reduction:  {first_loss - last_loss:.4f}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
