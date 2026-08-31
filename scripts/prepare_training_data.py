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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare")

SEP = "=" * 70


# Files whose examples taught the exact behaviours the 945ex eval measured
# failing, excluded at the source so a re-run cannot resurrect them:
#   batch_v1/v2      600 sim.* calls with NO netlist argument, answered with
#                    fabricated random "success" data -- the model passed a
#                    real netlist in 5 of 357 eval sim calls because 65 pct of
#                    its training calls carried none.
#   augmented_v1/v2  numbers from a regex-over-W formula, not a simulator,
#                    and a byte-identical opener + pdk.device_query in every
#                    example -- the f32 A/B probe found that exact opener
#                    memorised and emitted for every task, digital included.
# Their replacement is grounded_v1.jsonl (generate_grounded_sft.py), whose
# every observation came out of the real env. This list is the "fix the
# generator, not the data" rule applied to the MIX: the files stay on disk
# for archaeology, but no future train_final can include them.
EXCLUDED_SOURCES = {
    "batch_v1.jsonl", "batch_v2.jsonl",
    "augmented_v1.jsonl", "augmented_v2.jsonl",
    "demo_output.jsonl",
}

# This module used to inline-generate 10 pdk.get_corners examples whose
# sim.corners call carried the placeholder ".subckt design VDD VSS VIN VOUT"
# -- the eval model reproduced that exact empty shell in 258 sim calls.
# pdk.get_corners coverage now comes from grounded_v1's PDK-preamble examples,
# with the same real deck the rest of the trajectory simulates.


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
        if f.name in EXCLUDED_SOURCES:
            print(f"  {f.name}: EXCLUDED (see EXCLUDED_SOURCES)")
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

    print(f"\n[2/5] Source mix...")
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

    # Step 4: Train/val split, STRATIFIED BY SOURCE.
    #
    # The split used to take the hardest tail of the curriculum ordering as
    # validation. Multi-turn examples with real netlists score as "hard", so
    # that split sent 312 real-netlist sim calls to validation and left the
    # training set with 211 -- the model TRAINED mostly on placeholder calls
    # and was VALIDATED mostly on real ones. Sampling the val fraction from
    # every source keeps both sides the same mixture.
    print(f"\n[4/5] Stratified train/val split...")
    by_source: dict[str, list] = {}
    for ex in all_examples:
        by_source.setdefault(ex["_source"], []).append(ex)

    train_examples, val_examples = [], []
    for source in sorted(by_source):
        group = by_source[source]
        rng.shuffle(group)
        n_val = round(len(group) * args.val_split)
        val_examples.extend(group[:n_val])
        train_examples.extend(group[n_val:])
    print(f"  {len(train_examples)} train / {len(val_examples)} val, "
          f"val drawn {args.val_split:.0%} from each of {len(by_source)} sources")

    # Step 5: Curriculum ordering (train only -- val order is irrelevant).
    print(f"\n[5/5] Curriculum ordering and saving...")
    for ex in train_examples:
        ex["_difficulty"] = estimate_difficulty(ex)
    if args.curriculum:
        train_examples.sort(key=lambda x: x["_difficulty"])
        print(f"  Ordered: easy ({train_examples[0]['_difficulty']}) -> "
              f"hard ({train_examples[-1]['_difficulty']})")
    else:
        rng.shuffle(train_examples)
        print(f"  Shuffled randomly")
    val_count = len(val_examples)
    train_count = len(train_examples)

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
