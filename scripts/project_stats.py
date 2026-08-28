#!/usr/bin/env python3
"""Show project statistics dashboard.

Usage:
    PYTHONPATH=src python scripts/project_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return 0


def main():
    root = Path(__file__).parent.parent

    print(f"\n{SEP}")
    print("   ASIC-AI Project Statistics")
    print(f"{SEP}\n")

    # Source code
    src_files = list((root / "src").rglob("*.py"))
    src_lines = sum(count_lines(f) for f in src_files)
    print(f"  Source Code:")
    print(f"    Python modules: {len(src_files)}")
    print(f"    Total lines:    {src_lines:,}")

    # Scripts
    scripts = list((root / "scripts").glob("*.py"))
    script_lines = sum(count_lines(f) for f in scripts)
    print(f"\n  Scripts:")
    print(f"    CLI tools:      {len(scripts)}")
    print(f"    Total lines:    {script_lines:,}")

    # Tests
    test_files = list((root / "tests").rglob("*.py"))
    test_lines = sum(count_lines(f) for f in test_files)
    print(f"\n  Tests:")
    print(f"    Test files:     {len(test_files)}")
    print(f"    Total lines:    {test_lines:,}")

    # Eval tasks
    analog_tasks = list((root / "eval/tasks/analog").glob("*.yaml"))
    digital_tasks = list((root / "eval/tasks/digital").glob("*.yaml"))
    print(f"\n  Eval Tasks:")
    print(f"    Analog:         {len(analog_tasks)}")
    print(f"    Digital:        {len(digital_tasks)}")
    print(f"    Total:          {len(analog_tasks) + len(digital_tasks)}")

    # SFT data
    sft_dir = root / "data/sft"
    if sft_dir.exists():
        sft_files = list(sft_dir.glob("*.jsonl"))
        total_examples = 0
        total_size = 0
        for f in sft_files:
            with open(f, encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
            total_examples += count
            total_size += f.stat().st_size

        train_path = sft_dir / "train_final.jsonl"
        val_path = sft_dir / "val_final.jsonl"
        train_count = sum(1 for _ in open(train_path, encoding="utf-8")) if train_path.exists() else 0
        val_count = sum(1 for _ in open(val_path, encoding="utf-8")) if val_path.exists() else 0

        print(f"\n  Training Data:")
        print(f"    Data files:     {len(sft_files)}")
        print(f"    Train examples: {train_count}")
        print(f"    Val examples:   {val_count}")
        print(f"    Total size:     {total_size/1024:.0f} KB")

    # Templates
    from asic_ai.data.templates import list_templates
    templates = list_templates()
    analog_t = [t for t in templates if t.category == "analog"]
    digital_t = [t for t in templates if t.category == "digital"]
    print(f"\n  Circuit Templates:")
    print(f"    Analog:         {len(analog_t)}")
    print(f"    Digital:        {len(digital_t)}")
    print(f"    Total:          {len(templates)}")

    # Domain tokens
    try:
        from asic_ai.tokenizer.extend import get_new_tokens
        domain_tokens = get_new_tokens()
        print(f"\n  Tokenizer:")
        print(f"    Domain tokens:  {len(domain_tokens)}")
    except Exception:
        print(f"\n  Tokenizer:  (import error)")

    # Tool interface
    from asic_ai.data.format import TOOL_DEFINITIONS
    print(f"\n  Tool Interface:")
    print(f"    Tools:          {len(TOOL_DEFINITIONS)}")
    for t in TOOL_DEFINITIONS:
        print(f"      {t['function']['name']}")

    # Git
    import subprocess
    try:
        result = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True, cwd=str(root))
        commits = len(result.stdout.strip().splitlines())
        print(f"\n  Git:")
        print(f"    Commits:        {commits}")
    except Exception:
        pass

    # Training outputs
    outputs_dir = root / "outputs/sft_local"
    if outputs_dir.exists():
        checkpoints = list(outputs_dir.glob("checkpoint-*"))
        final = outputs_dir / "final"
        print(f"\n  Training Outputs:")
        print(f"    Checkpoints:    {len(checkpoints)}")
        print(f"    Final model:    {'Yes' if final.exists() else 'No'}")
        if final.exists():
            info_path = final / "training_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                print(f"    Base model:     {info.get('base_model', '?')}")
                print(f"    LoRA r/alpha:   {info.get('lora_r', '?')}/{info.get('lora_alpha', '?')}")

    # Grand total
    total_py = len(src_files) + len(scripts) + len(test_files)
    total_lines = src_lines + script_lines + test_lines
    print(f"\n{SEP}")
    print(f"  TOTALS")
    print(f"    Python files:   {total_py}")
    print(f"    Python lines:   {total_lines:,}")
    print(f"    Eval tasks:     {len(analog_tasks) + len(digital_tasks)}")
    print(f"    Templates:      {len(templates)}")
    print(f"    Train examples: {train_count}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
