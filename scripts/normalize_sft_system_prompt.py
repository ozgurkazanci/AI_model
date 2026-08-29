#!/usr/bin/env python3
"""Normalize the system message across all SFT source files.

Every training example must carry the byte-identical system message produced by
asic_ai.data.format.build_system_message(). Mixed system prompts are the #1
cause of a fine-tuned model that will not emit tool calls at inference time.

The generator scripts historically wrote a bare SYSTEM_PROMPT (no tool list)
while the format_trajectory_for_sft() path wrote SYSTEM_PROMPT + tool list.
That left two variants in the corpus. This script rewrites all of them to the
canonical one.

Source files only -- train_final.jsonl / val_final.jsonl are regenerated from
these by scripts/prepare_training_data.py.

Usage:
    PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --check
    PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message

SEP = "=" * 70

# Regenerated downstream, not edited here.
DERIVED = {"train_final.jsonl", "val_final.jsonl"}


def normalize_example(example: dict, canonical: str) -> str:
    """Rewrite an example's system message in place. Returns the action taken."""
    key = "messages" if "messages" in example else "conversations"
    messages = example.get(key)
    if not messages:
        return "no_messages"

    first = messages[0]
    if first.get("role") != "system":
        messages.insert(0, {"role": "system", "content": canonical})
        return "inserted"

    if first.get("content") == canonical:
        return "already_canonical"

    first["content"] = canonical
    return "rewritten"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize SFT system prompts")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--write", action="store_true", help="Apply changes in place")
    parser.add_argument("--check", action="store_true",
                        help="Report only; exit 1 if any file deviates")
    args = parser.parse_args()

    if not args.write and not args.check:
        args.check = True

    canonical = build_system_message()

    print(f"\n{SEP}")
    print("   SFT System Prompt Normalization")
    print(f"{SEP}\n")
    print(f"Canonical system message: {len(canonical)} chars\n")

    data_path = Path(args.data_dir)
    totals: Counter[str] = Counter()
    dirty_files = 0

    for path in sorted(data_path.glob("*.jsonl")):
        if path.name in DERIVED:
            continue

        examples = []
        actions: Counter[str] = Counter()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                actions[normalize_example(ex, canonical)] += 1
                examples.append(ex)

        totals.update(actions)
        changed = actions["rewritten"] + actions["inserted"]

        if changed:
            dirty_files += 1
            detail = ", ".join(f"{k}={v}" for k, v in sorted(actions.items()))
            print(f"  {path.name:28s} {changed:4d} to fix   ({detail})")
            if args.write:
                with open(path, "w", encoding="utf-8") as out:
                    for ex in examples:
                        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        else:
            print(f"  {path.name:28s}    ok       ({actions['already_canonical']} canonical)")

    print(f"\n{SEP}")
    total_changed = totals["rewritten"] + totals["inserted"]
    print(f"  Already canonical: {totals['already_canonical']}")
    print(f"  Rewritten:         {totals['rewritten']}")
    print(f"  Inserted:          {totals['inserted']}")
    if totals["no_messages"]:
        print(f"  Skipped (no msgs): {totals['no_messages']}")
    print(f"  Files affected:    {dirty_files}")

    if args.write:
        print(f"\n  Written. Next: regenerate the derived split:")
        print(f"    PYTHONPATH=src python scripts/prepare_training_data.py")
        print(f"{SEP}\n")
        return 0

    print(f"{SEP}\n")
    if total_changed:
        print(f"  {total_changed} examples deviate. Run with --write to fix.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
