"""Record a baseline so later runs can be compared against it.

The design document's build order puts "baseline measurement" before any
training, for the obvious reason: without a number from before, an improvement
afterwards cannot be told from a change in the harness. This module was
previously a bare `pass`, so no baseline was ever recorded and every later
claim of improvement rested on nothing.

Usage:
    PYTHONPATH=src python -m eval.baseline --tasks eval/tasks \\
        --model asic-ai-0.5b-q4_k_m --output eval_results/baseline_local.json

    # then, after a change
    PYTHONPATH=src python -m eval.metrics_compare baseline.json after.json
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from eval.metrics import compute_metrics
from eval.runner import DEFAULT_MAX_STEPS, _default_engine, run_task

log = logging.getLogger("eval.baseline")


def _environment() -> Dict[str, Any]:
    """What the numbers were produced on. A baseline without this is not one."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        from asic_ai.inference import llama_server
        env["llama_cpp_devices"] = llama_server.list_devices()
        cfg = llama_server.ServerConfig.from_config()
        env["gguf_model"] = Path(cfg.model).name if cfg.model else None
        env["n_gpu_layers"] = cfg.n_gpu_layers
    except Exception:
        pass
    try:
        from asic_ai.adapters.ngspice_shared import find_ngspice_dll
        env["ngspice_dll"] = find_ngspice_dll()
    except Exception:
        pass
    return env


def run_baseline(tasks_dir: str, model_id: str, output: str,
                 limit: int | None = None,
                 max_steps: int = DEFAULT_MAX_STEPS) -> int:
    """Run the eval set and write a baseline record. Returns a process exit code."""
    task_files = sorted(Path(tasks_dir).rglob("*.yaml"))
    if limit:
        task_files = task_files[:limit]
    if not task_files:
        log.error("no tasks found under %s", tasks_dir)
        return 1

    engine = _default_engine()
    if engine is None:
        # Emitting a baseline of zeros would be worse than emitting nothing:
        # every later run would look like an improvement.
        log.error("no inference engine is reachable; refusing to record a "
                  "baseline that would make any later run look better")
        log.error("start one with: PYTHONPATH=src python scripts/serve_local.py")
        return 1

    log.info("baseline: %d task(s), model %s", len(task_files), model_id)
    results = []
    for path in task_files:
        r = run_task(path, model_id, engine=engine, max_steps=max_steps)
        mark = "PASS" if r.passed else "fail"
        note = f"  [{r.error}]" if r.error else ""
        log.info("  %s  %-28s score %+.4f  %d steps  %.1fs%s",
                 mark, r.task_id, r.final_score, r.steps, r.wall_time_sec, note)
        results.append(r.model_dump())

    payload = {
        "model": model_id,
        "tasks_dir": str(tasks_dir),
        "max_steps": max_steps,
        "environment": _environment(),
        "results": results,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metrics = compute_metrics(str(out))
    log.info("pass_rate %.3f  avg_score %+.4f  -> %s",
             metrics.get("pass_rate", 0.0), metrics.get("avg_score", 0.0), out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an eval baseline")
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    return run_baseline(args.tasks, args.model, args.output,
                        limit=args.limit, max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
