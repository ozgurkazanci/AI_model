#!/usr/bin/env python3
"""Analyze SFT training data quality and distribution.

Reports on:
- Total examples, message counts, token estimates
- Tool usage distribution
- Role sequence patterns
- Score distribution
- Dataset comparison

Usage:
    PYTHONPATH=src python scripts/analyze_sft_data.py data/sft/mock_analog_v1.jsonl data/sft/mock_digital_v1.jsonl data/sft/augmented_v1.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def analyze_file(path: Path) -> dict:
    """Analyze a single JSONL file."""
    lines = path.read_text(encoding="utf-8").strip().split("\n")

    scores = []
    msg_counts = []
    tool_calls = Counter()
    role_patterns = Counter()
    total_chars = 0
    successes = 0

    for line in lines:
        data = json.loads(line)
        msgs = data.get("messages", [])
        score = data.get("score", 0)
        success = data.get("success", False)

        scores.append(score)
        msg_counts.append(len(msgs))
        if success:
            successes += 1

        # Role pattern
        roles = tuple(m["role"] for m in msgs)
        role_patterns[roles] += 1

        # Tool usage + char count
        for msg in msgs:
            total_chars += len(msg.get("content", ""))
            content = msg.get("content", "")
            if msg["role"] == "assistant" and "<tool_call>" in content:
                # Extract tool name
                import re
                matches = re.findall(r'"name"\s*:\s*"([^"]+)"', content)
                for m in matches:
                    tool_calls[m] += 1

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_msgs = sum(msg_counts) / len(msg_counts) if msg_counts else 0
    est_tokens = total_chars // 4  # Rough estimate

    return {
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024, 1),
        "examples": len(lines),
        "successes": successes,
        "success_rate": round(successes / len(lines) * 100, 1) if lines else 0,
        "avg_score": round(avg_score, 3),
        "avg_messages": round(avg_msgs, 1),
        "total_chars": total_chars,
        "est_tokens": est_tokens,
        "tool_usage": dict(tool_calls.most_common()),
        "unique_role_patterns": len(role_patterns),
    }


def main():
    if len(sys.argv) < 2:
        # Auto-discover
        files = sorted(Path("data/sft").glob("*.jsonl"))
    else:
        files = [Path(f) for f in sys.argv[1:]]

    if not files:
        print("No JSONL files found.")
        return

    print("\n" + "=" * 72)
    print("   ASIC-AI SFT Training Data Analysis")
    print("=" * 72)

    all_stats = []
    total_examples = 0
    total_tokens = 0
    combined_tools = Counter()

    for f in files:
        if not f.exists():
            print(f"\n  MISSING: {f}")
            continue

        stats = analyze_file(f)
        all_stats.append(stats)
        total_examples += stats["examples"]
        total_tokens += stats["est_tokens"]
        combined_tools.update(stats["tool_usage"])

        print(f"\n  {stats['file']} ({stats['size_kb']} KB)")
        print(f"    Examples:     {stats['examples']}")
        print(f"    Successes:    {stats['successes']} ({stats['success_rate']}%)")
        print(f"    Avg score:    {stats['avg_score']}")
        print(f"    Avg messages: {stats['avg_messages']}")
        print(f"    Est. tokens:  {stats['est_tokens']:,}")
        if stats["tool_usage"]:
            print(f"    Top tools:    {', '.join(f'{k}({v})' for k, v in list(stats['tool_usage'].items())[:5])}")

    # Summary
    print(f"\n{'=' * 72}")
    print(f"   Combined Summary")
    print(f"{'=' * 72}")
    print(f"  Total files:       {len(all_stats)}")
    print(f"  Total examples:    {total_examples}")
    print(f"  Total est. tokens: {total_tokens:,}")
    print(f"  Combined size:     {sum(s['size_kb'] for s in all_stats):.1f} KB")

    if combined_tools:
        print(f"\n  Tool Usage Distribution:")
        for tool, count in combined_tools.most_common():
            bar = "#" * min(40, count)
            print(f"    {tool:20s} {count:4d} {bar}")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
