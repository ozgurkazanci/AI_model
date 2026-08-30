#!/usr/bin/env python3
"""Compare two eval runs: did the model actually get better?

    PYTHONPATH=src python scripts/compare_baselines.py \\
        eval_results/baseline_local_0.5b.json eval_results/after_945.json

Exists because "the loss went down" is not the question. The old 354-example
run reported loss 2.18 -> 0.005 -- a 99.8 pct reduction -- and the model it
produced passed 0 of 77 tasks. Loss measures fit to the data; this compares
what the reward actually cares about, on identical tasks, with the identical
harness.

Beyond pass rate it diffs the BEHAVIOUR the baseline showed to be broken, so a
change in either direction is visible even while the pass rate sits at zero:

  - hallucinated tool names (baseline: 8 calls to tools that do not exist)
  - unparseable tool calls   (baseline: 25 turns)
  - repetition: tasks that burned every step on the same failing call
    (baseline: 51 of 77 never varied their approach)
  - whether spec.check ever measured anything (baseline: never)

A comparison is only valid between runs of the same task set; anything else is
refused rather than averaged over.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 72


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def known_tools() -> set:
    from asic_ai.data.format import TOOL_DEFINITIONS
    return {t["function"]["name"] for t in TOOL_DEFINITIONS}


def profile(run: Dict[str, Any]) -> Dict[str, Any]:
    """Everything worth comparing, from one run's results."""
    contract = known_tools()
    rs = run["results"]

    tools: Counter = Counter()
    hallucinated: Counter = Counter()
    parse_error_turns = 0
    stuck_tasks = 0
    measured_checks = 0
    per_task: Dict[str, Dict[str, Any]] = {}

    for r in rs:
        sigs: List[tuple] = []
        for t in r.get("trajectory", []):
            names = t.get("tool_calls", [])
            sigs.append(tuple(names))
            for n in names:
                tools[n] += 1
                if n not in contract:
                    hallucinated[n] += 1
            if t.get("parse_errors"):
                parse_error_turns += 1
            for o in (t.get("observations") or []):
                if '"measured"' in o and '"measured": {}' not in o.replace(" ", ""):
                    measured_checks += 1
        # Stuck: every step made the same call(s) and there were several steps.
        non_empty = [s for s in sigs if s]
        if len(non_empty) >= 3 and len(set(non_empty)) == 1:
            stuck_tasks += 1
        per_task[r["task_id"]] = {
            "passed": bool(r.get("passed")),
            "score": float(r.get("final_score", 0.0)),
            "steps": int(r.get("steps", 0)),
            "error": r.get("error"),
        }

    return {
        "n": len(rs),
        "passed": sum(1 for r in rs if r.get("passed")),
        "avg_score": (sum(float(r.get("final_score", 0.0)) for r in rs) / len(rs))
                     if rs else 0.0,
        "tool_calls": sum(tools.values()),
        "tools": tools,
        "hallucinated": hallucinated,
        "parse_error_turns": parse_error_turns,
        "stuck_tasks": stuck_tasks,
        "measured_checks": measured_checks,
        "per_task": per_task,
        "model": run.get("model", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two eval runs")
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()

    a = profile(load(args.before))
    b = profile(load(args.after))

    tasks_a, tasks_b = set(a["per_task"]), set(b["per_task"])
    if tasks_a != tasks_b:
        only_a, only_b = tasks_a - tasks_b, tasks_b - tasks_a
        print("REFUSED: the runs cover different task sets, so their numbers "
              "are not comparable.")
        if only_a:
            print(f"  only in {args.before}: {sorted(only_a)[:5]}...")
        if only_b:
            print(f"  only in {args.after}: {sorted(only_b)[:5]}...")
        return 1

    def row(label: str, va, vb, better: str = "up") -> None:
        arrow = ""
        if va != vb:
            improved = (vb > va) if better == "up" else (vb < va)
            arrow = "  IYILESME" if improved else "  GERILEME"
        print(f"  {label:34s} {va:>10}  ->  {vb:>10}{arrow}")

    print(f"\n{SEP}")
    print(f"  ONCE : {a['model']}")
    print(f"  SONRA: {b['model']}")
    print(f"{SEP}\n")
    row("gecen gorev", f"{a['passed']}/{a['n']}", f"{b['passed']}/{b['n']}")
    row("ortalama skor", f"{a['avg_score']:+.4f}", f"{b['avg_score']:+.4f}")
    row("spec.check ile olculen", a["measured_checks"], b["measured_checks"])
    print()
    row("halusinasyon arac cagrisi",
        sum(a["hallucinated"].values()), sum(b["hallucinated"].values()), "down")
    row("ayristirilamayan tur", a["parse_error_turns"], b["parse_error_turns"], "down")
    row("takilan gorev (ayni cagri x3+)", a["stuck_tasks"], b["stuck_tasks"], "down")
    row("toplam arac cagrisi", a["tool_calls"], b["tool_calls"])

    if a["hallucinated"] or b["hallucinated"]:
        print("\n  uydurulan arac adlari:")
        names = set(a["hallucinated"]) | set(b["hallucinated"])
        for n in sorted(names):
            print(f"    {n:24s} {a['hallucinated'].get(n, 0):>3}  ->  "
                  f"{b['hallucinated'].get(n, 0):>3}")

    flips_up = [t for t in tasks_a
                if not a["per_task"][t]["passed"] and b["per_task"][t]["passed"]]
    flips_down = [t for t in tasks_a
                  if a["per_task"][t]["passed"] and not b["per_task"][t]["passed"]]
    if flips_up:
        print(f"\n  gecmeye baslayan ({len(flips_up)}): {sorted(flips_up)}")
    if flips_down:
        print(f"\n  GECEMEZ OLAN ({len(flips_down)}): {sorted(flips_down)}")

    moved = sorted(
        ((t, b["per_task"][t]["score"] - a["per_task"][t]["score"])
         for t in tasks_a),
        key=lambda kv: abs(kv[1]), reverse=True)
    moved = [(t, d) for t, d in moved if abs(d) > 1e-9][:8]
    if moved:
        print("\n  skoru en cok degisen gorevler:")
        for t, d in moved:
            print(f"    {t:32s} {d:+.4f}")

    print(f"\n{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
