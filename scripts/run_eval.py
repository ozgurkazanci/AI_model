#!/usr/bin/env python3
"""Run all eval tasks with a model and generate a report.

Runs each eval task through the agent, records results, and
generates an HTML report with scores, pass rates, and analysis.

Usage:
    # Run all analog tasks
    PYTHONPATH=src python scripts/run_eval.py --model outputs/sft_local/final --category analog

    # Run all tasks with max 3 steps each
    PYTHONPATH=src python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3

    # Quick smoke test (first 5 tasks only)
    PYTHONPATH=src python scripts/run_eval.py --model outputs/sft_local/final --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

SEP = "=" * 70


def discover_tasks(category: str | None = None) -> list[dict]:
    """Discover all eval tasks."""
    tasks = []
    for f in sorted(Path("eval/tasks").rglob("*.yaml")):
        with open(f) as fh:
            task = yaml.safe_load(fh)
            if task and isinstance(task, dict) and "id" in task:
                task["_path"] = str(f)
                if category:
                    if category in str(f):
                        tasks.append(task)
                else:
                    tasks.append(task)
    return tasks


def run_single_task(model, tokenizer, task: dict, max_steps: int, max_tokens: int) -> dict:
    """Run a single task and return results."""
    from scripts.run_agent import run_agent

    try:
        traj = run_agent(model, tokenizer, task, max_steps, max_tokens)
        return {
            "task_id": task["id"],
            "success": traj.success,
            "score": traj.final_score,
            "steps": len(traj.steps),
            "duration_s": traj.duration_seconds,
            "error": None,
        }
    except Exception as e:
        log.error(f"Task {task['id']} failed: {e}")
        return {
            "task_id": task["id"],
            "success": False,
            "score": 0.0,
            "steps": 0,
            "duration_s": 0,
            "error": str(e),
        }


def generate_report(results: list[dict], model_name: str, output_path: Path):
    """Generate HTML evaluation report."""
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    avg_score = sum(r["score"] for r in results) / total if total else 0
    avg_steps = sum(r["steps"] for r in results) / total if total else 0
    total_time = sum(r["duration_s"] for r in results)
    errors = sum(1 for r in results if r["error"])

    # JSON report
    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total,
        "successes": successes,
        "success_rate": round(successes / total * 100, 1) if total else 0,
        "avg_score": round(avg_score, 3),
        "avg_steps": round(avg_steps, 1),
        "total_time_s": round(total_time, 1),
        "errors": errors,
        "results": results,
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # HTML report
    rows = ""
    for r in results:
        status = "[OK]" if r["success"] else "[FAIL]"
        err = f' <span style="color:red">{r["error"][:50]}</span>' if r["error"] else ""
        rows += f"""<tr>
            <td>{r['task_id']}</td>
            <td>{status}</td>
            <td>{r['score']:.3f}</td>
            <td>{r['steps']}</td>
            <td>{r['duration_s']:.1f}s</td>
            <td>{err}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html><head><title>ASIC-AI Eval Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #2563eb; color: white; }}
tr:nth-child(even) {{ background: #f8fafc; }}
.summary {{ background: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
</style></head><body>
<h1>ASIC-AI Evaluation Report</h1>
<div class="summary">
<p><strong>Model:</strong> {model_name}</p>
<p><strong>Date:</strong> {time.strftime("%Y-%m-%d %H:%M")}</p>
<p><strong>Tasks:</strong> {total} | <strong>Passed:</strong> {successes} ({report['success_rate']}%)</p>
<p><strong>Avg Score:</strong> {avg_score:.3f} | <strong>Avg Steps:</strong> {avg_steps:.1f}</p>
<p><strong>Total Time:</strong> {total_time:.0f}s | <strong>Errors:</strong> {errors}</p>
</div>
<table>
<tr><th>Task</th><th>Status</th><th>Score</th><th>Steps</th><th>Time</th><th>Error</th></tr>
{rows}
</table>
</body></html>"""

    html_path = output_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    print(f"\n  Reports saved:")
    print(f"    JSON: {json_path}")
    print(f"    HTML: {html_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run eval tasks with model")
    parser.add_argument("--model", default="outputs/sft_local/final")
    parser.add_argument("--category", default=None, choices=["analog", "digital"])
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--output", default="eval_results/eval_report")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Evaluation Harness")
    print(f"{SEP}\n")

    # Discover tasks
    tasks = discover_tasks(args.category)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"  Tasks: {len(tasks)}")
    print(f"  Model: {args.model}")
    print(f"  Max steps: {args.max_steps}")

    # Load model
    print(f"\n  Loading model...")
    from scripts.run_agent import load_model
    model, tokenizer = load_model(args.model)

    # Run tasks
    results = []
    for i, task in enumerate(tasks):
        print(f"\n{'='*40} Task {i+1}/{len(tasks)}: {task['id']} {'='*40}")
        result = run_single_task(model, tokenizer, task, args.max_steps, args.max_tokens)
        results.append(result)
        print(f"  -> score={result['score']:.3f}, success={result['success']}")

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.model, output_path)

    print(f"\n{SEP}")
    print(f"   Evaluation Complete")
    print(f"{SEP}")
    print(f"  Tasks:     {report['total_tasks']}")
    print(f"  Passed:    {report['successes']} ({report['success_rate']}%)")
    print(f"  Avg Score: {report['avg_score']:.3f}")
    print(f"  Time:      {report['total_time_s']:.0f}s")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
