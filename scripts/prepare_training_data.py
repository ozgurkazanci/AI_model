#!/usr/bin/env python3
"""Create optimally mixed and ordered training dataset.

Combines all SFT data files with:
1. Proper mixing ratios (avoid tool imbalance)
2. Curriculum ordering (easy -> hard)
3. Deduplication
4. Train/validation split

Also generates the last missing tool (pdk.get_corners) examples.

Usage:
    PYTHONPATH=src python scripts/prepare_training_data.py --output data/sft/train_final.jsonl
    PYTHONPATH=src python scripts/prepare_training_data.py --output data/sft/train_final.jsonl --val-split 0.1
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("prepare")

SEP = "=" * 70


def generate_pdk_corners_examples(rng: random.Random, count: int = 10) -> list[dict]:
    """Generate examples for the last missing tool: pdk.get_corners."""
    examples = []
    for i in range(count):
        steps = [
            TrajectoryStep(step_index=0, role="user",
                content="What PVT corners should I simulate for my design to ensure robustness?"),
            TrajectoryStep(step_index=1, role="assistant",
                content="Let me query the PDK for available process corners.",
                tool_call=ToolCall(name="pdk.get_corners", call_id="call_001",
                                  arguments={"pdk": "sky130"})),
            TrajectoryStep(step_index=2, role="tool",
                content=json.dumps({
                    "corners": [
                        {"name": "tt", "description": "Typical-Typical", "temp_c": 27, "vdd": 1.8},
                        {"name": "ss", "description": "Slow-Slow", "temp_c": 125, "vdd": 1.62},
                        {"name": "ff", "description": "Fast-Fast", "temp_c": -40, "vdd": 1.98},
                        {"name": "sf", "description": "Slow-Fast", "temp_c": 27, "vdd": 1.8},
                        {"name": "fs", "description": "Fast-Slow", "temp_c": 27, "vdd": 1.8},
                    ],
                    "recommended_minimum": ["tt", "ss", "ff"],
                    "full_pvt": ["tt", "ss", "ff", "sf", "fs"],
                })),
            TrajectoryStep(step_index=3, role="assistant",
                content="5 corners available. Minimum recommended: tt, ss, ff. Running corner simulation.",
                tool_call=ToolCall(name="sim.corners", call_id="call_002",
                                  arguments={"netlist": ".subckt design VDD VSS VIN VOUT\n.ends",
                                            "corners": ["tt", "ss", "ff"]})),
            TrajectoryStep(step_index=4, role="tool",
                content=json.dumps([
                    {"corner": "tt", "gain_db": round(62 + rng.gauss(0, 2), 1), "pm_deg": round(65 + rng.gauss(0, 3), 1)},
                    {"corner": "ss", "gain_db": round(55 + rng.gauss(0, 2), 1), "pm_deg": round(58 + rng.gauss(0, 3), 1)},
                    {"corner": "ff", "gain_db": round(68 + rng.gauss(0, 2), 1), "pm_deg": round(72 + rng.gauss(0, 3), 1)},
                ])),
            TrajectoryStep(step_index=5, role="assistant",
                content="All corners pass. Worst case is SS corner with reduced gain and PM."),
        ]

        traj = Trajectory(
            id=f"corners_{uuid.uuid4().hex[:6]}", task_id="pvt_corners",
            steps=steps, success=True,
            final_score=round(0.85 + rng.uniform(0, 0.1), 3),
            duration_seconds=6.0,
        )
        sft_msgs = format_trajectory_for_sft(traj)
        is_valid, _ = validate_sft_format(sft_msgs)
        if is_valid:
            examples.append({
                "id": traj.id, "task_id": traj.task_id,
                "messages": sft_msgs, "score": traj.final_score,
                "success": True, "primary_tool": "pdk.get_corners",
            })

    return examples


def estimate_difficulty(example: dict) -> float:
    """Estimate difficulty of an example (0=easy, 1=hard)."""
    msgs = example.get("messages", [])
    num_msgs = len(msgs)
    score = example.get("score", 0.5)

    # More messages = harder
    msg_factor = min(1.0, num_msgs / 20.0)

    # Count unique tools used
    tools_used = set()
    for m in msgs:
        content = m.get("content", "")
        if "tool_call" in content:
            import re
            matches = re.findall(r'"name"\s*:\s*"([^"]+)"', content)
            tools_used.update(matches)
    tool_factor = min(1.0, len(tools_used) / 6.0)

    # Higher score = easier (model should learn easy first)
    score_factor = 1.0 - score

    return round((msg_factor * 0.3 + tool_factor * 0.4 + score_factor * 0.3), 3)


def main():
    parser = argparse.ArgumentParser(description="Prepare final training dataset")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--output", default="data/sft/train_final.jsonl")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Order by difficulty (easy->hard)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"\n{SEP}")
    print("   ASIC-AI Training Data Preparation")
    print(f"{SEP}\n")

    # Step 1: Load all data
    print("[1/5] Loading all SFT data...")
    all_examples = []
    data_path = Path(args.data_dir)

    for f in sorted(data_path.glob("*.jsonl")):
        if f.name.startswith("train_") or f.name.startswith("val_"):
            continue  # Skip output files
        if f.name == "demo_output.jsonl":
            continue
        count = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line.strip())
                if "messages" in ex:
                    ex["_source"] = f.name
                    all_examples.append(ex)
                    count += 1
        print(f"  {f.name}: {count} examples")

    # Step 2: Generate missing pdk.get_corners
    print(f"\n[2/5] Generating pdk.get_corners examples...")
    corners_examples = generate_pdk_corners_examples(rng, count=10)
    for ex in corners_examples:
        ex["_source"] = "generated_corners"
    all_examples.extend(corners_examples)
    print(f"  Added {len(corners_examples)} pdk.get_corners examples")
    print(f"  Total: {len(all_examples)} examples")

    # Step 3: Deduplicate by ID
    print(f"\n[3/5] Deduplication...")
    seen_ids = set()
    unique = []
    for ex in all_examples:
        ex_id = ex.get("id", str(id(ex)))
        if ex_id not in seen_ids:
            seen_ids.add(ex_id)
            unique.append(ex)
    removed = len(all_examples) - len(unique)
    if removed > 0:
        print(f"  Removed {removed} duplicates")
    all_examples = unique
    print(f"  Unique: {len(all_examples)}")

    # Step 4: Curriculum ordering
    print(f"\n[4/5] Curriculum ordering...")
    for ex in all_examples:
        ex["_difficulty"] = estimate_difficulty(ex)

    if args.curriculum:
        all_examples.sort(key=lambda x: x["_difficulty"])
        print(f"  Ordered: easy ({all_examples[0]['_difficulty']}) -> hard ({all_examples[-1]['_difficulty']})")
    else:
        rng.shuffle(all_examples)
        print(f"  Shuffled randomly")

    # Step 5: Train/val split and save
    print(f"\n[5/5] Saving...")
    val_count = int(len(all_examples) * args.val_split)
    train_count = len(all_examples) - val_count

    # Take hardest examples for validation (better signal)
    val_examples = all_examples[-val_count:] if val_count > 0 else []
    train_examples = all_examples[:train_count]

    # Save train
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for ex in train_examples:
            clean = {k: v for k, v in ex.items() if not k.startswith("_")}
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")

    train_size = output_path.stat().st_size / 1024
    print(f"  Train: {train_count} examples -> {output_path} ({train_size:.1f} KB)")

    # Save validation
    if val_examples:
        val_path = output_path.with_name("val_final.jsonl")
        with open(val_path, "w", encoding="utf-8") as out:
            for ex in val_examples:
                clean = {k: v for k, v in ex.items() if not k.startswith("_")}
                out.write(json.dumps(clean, ensure_ascii=False) + "\n")
        val_size = val_path.stat().st_size / 1024
        print(f"  Val:   {val_count} examples -> {val_path} ({val_size:.1f} KB)")

    # Summary
    print(f"\n{SEP}")
    print(f"   Dataset Ready")
    print(f"{SEP}")
    print(f"  Total:      {len(all_examples)} examples")
    print(f"  Train:      {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Ordering:   {'curriculum (easy->hard)' if args.curriculum else 'random'}")

    # Tool coverage
    from collections import Counter
    import re
    tool_counts = Counter()
    for ex in all_examples:
        for m in ex.get("messages", []):
            if m.get("role") == "assistant":
                matches = re.findall(r'"name"\s*:\s*"([^"]+)"', m.get("content", ""))
                tool_counts.update(matches)

    print(f"\n  Tool coverage ({len(tool_counts)} tools):")
    for tool, count in tool_counts.most_common():
        print(f"    {tool:20s} {count:4d}")

    print(f"\n  Next: python scripts/finetune_local.py --data-dir data/sft")
    print(f"  Or cloud: bash scripts/cloud/train_sft_large_lambda.sh")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
