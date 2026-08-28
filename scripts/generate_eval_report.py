#!/usr/bin/env python3
"""Generate HTML eval report from baseline measurement results.

Usage:
    python scripts/generate_eval_report.py --input eval_results/baseline.json --output eval_results/report.html
    python scripts/generate_eval_report.py --compare eval_results/baseline.json eval_results/finetuned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASIC-AI Eval Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
  .card .value {{ font-size: 2.5em; font-weight: bold; color: #0f3460; }}
  .card .label {{ color: #666; margin-top: 5px; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
  th {{ background: #16213e; color: white; padding: 12px 15px; text-align: left; font-weight: 500; }}
  td {{ padding: 10px 15px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f4ff; }}
  .pass {{ color: #27ae60; font-weight: bold; }}
  .fail {{ color: #e74c3c; font-weight: bold; }}
  .bar {{ display: inline-block; height: 18px; background: #3498db; border-radius: 3px; min-width: 2px; }}
  .bar-bg {{ display: inline-block; height: 18px; background: #ecf0f1; border-radius: 3px; width: 100px; }}
  .meta {{ color: #888; font-size: 0.85em; margin: 10px 0; }}
  .difficulty-easy {{ color: #27ae60; }} .difficulty-medium {{ color: #f39c12; }} .difficulty-hard {{ color: #e74c3c; }}
</style>
</head>
<body>
<h1>ASIC-AI Eval Report</h1>
<p class="meta">Model: <strong>{model}</strong> | Generated: {timestamp} | Tasks: {total_tasks}</p>

<div class="summary">
  <div class="card"><div class="value">{pass_rate_pct}%</div><div class="label">Overall Pass Rate</div></div>
  <div class="card"><div class="value">{avg_score}</div><div class="label">Avg Score</div></div>
  <div class="card"><div class="value">{tasks_passed}/{total_tasks}</div><div class="label">Tasks Passed</div></div>
  <div class="card"><div class="value">{avg_steps}</div><div class="label">Avg Steps</div></div>
</div>

<h2>By Category</h2>
<div class="summary">
  <div class="card"><div class="value">{analog_rate}%</div><div class="label">Analog ({analog_count})</div></div>
  <div class="card"><div class="value">{digital_rate}%</div><div class="label">Digital ({digital_count})</div></div>
</div>

<h2>By Difficulty</h2>
<div class="summary">
  <div class="card"><div class="value difficulty-easy">{easy_rate}%</div><div class="label">Easy ({easy_count})</div></div>
  <div class="card"><div class="value difficulty-medium">{medium_rate}%</div><div class="label">Medium ({medium_count})</div></div>
  <div class="card"><div class="value difficulty-hard">{hard_rate}%</div><div class="label">Hard ({hard_count})</div></div>
</div>

<h2>Task Results</h2>
<table>
<tr><th>Task ID</th><th>Category</th><th>Difficulty</th><th>Score</th><th>Steps</th><th>Status</th><th>Time</th></tr>
{task_rows}
</table>

</body>
</html>
"""


def generate_report(data: dict) -> str:
    results = data.get("results", [])

    # Count by category and difficulty
    analog = [r for r in results if r.get("category") == "analog"]
    digital = [r for r in results if r.get("category") == "digital"]
    easy = [r for r in results if r.get("difficulty") == "easy"]
    medium = [r for r in results if r.get("difficulty") == "medium"]
    hard = [r for r in results if r.get("difficulty") == "hard"]

    def rate(items): 
        passed = sum(1 for r in items if r.get("passed"))
        return f"{passed / len(items) * 100:.0f}" if items else "0"

    # Task rows
    rows = []
    for r in sorted(results, key=lambda x: (x.get("category", ""), x.get("difficulty", ""), x.get("task_id", ""))):
        status = '<span class="pass">PASS</span>' if r.get("passed") else '<span class="fail">FAIL</span>'
        score = r.get("final_score", 0)
        bar_width = max(0, min(100, int(score * 100)))
        bar_html = f'<div class="bar-bg"><div class="bar" style="width:{bar_width}px"></div></div> {score:.3f}'
        diff_class = f'difficulty-{r.get("difficulty", "")}'
        rows.append(
            f'<tr><td>{r.get("task_id", "?")}</td>'
            f'<td>{r.get("category", "?")}</td>'
            f'<td><span class="{diff_class}">{r.get("difficulty", "?")}</span></td>'
            f'<td>{bar_html}</td>'
            f'<td>{r.get("steps", 0)}</td>'
            f'<td>{status}</td>'
            f'<td>{r.get("wall_time_sec", 0):.1f}s</td></tr>'
        )

    return HTML_TEMPLATE.format(
        model=data.get("model", "unknown"),
        timestamp=data.get("timestamp", "?"),
        total_tasks=data.get("total_tasks", len(results)),
        pass_rate_pct=f"{data.get('pass_rate', 0) * 100:.0f}",
        avg_score=f"{data.get('avg_score', 0):.3f}",
        tasks_passed=data.get("tasks_passed", 0),
        avg_steps=f"{data.get('avg_steps', 0):.1f}",
        analog_rate=rate(analog), analog_count=len(analog),
        digital_rate=rate(digital), digital_count=len(digital),
        easy_rate=rate(easy), easy_count=len(easy),
        medium_rate=rate(medium), medium_count=len(medium),
        hard_rate=rate(hard), hard_count=len(hard),
        task_rows="\n".join(rows),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate HTML eval report")
    parser.add_argument("--input", required=True, help="Baseline JSON result file")
    parser.add_argument("--output", default=None, help="Output HTML file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    html = generate_report(data)
    output = args.output or args.input.replace(".json", ".html")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
