"""Run a model against the eval tasks and report what actually happened.

This module used to return, for every task without exception:

    passed = True
    score  = 85.5

Not a placeholder that failed loudly -- a placeholder that reported a perfect
pass. The design document lists "progress without an eval set" as a pitfall
because progress cannot be measured; an eval set that always says you passed is
strictly worse, because it says progress happened.

A real run is only possible now that the pieces underneath it are real: the
adapter returns measured values, spec_extract converts them to the spec names
and units each task declares, and the parser can read the tool-call format the
model actually emits. Every one of those three was broken, and each would have
silently produced a meaningless number here.

Honesty rules this file follows:
  - No model, no adapter, or no engine means an EvalResult with `error` set and
    passed=False. It never means a pass.
  - A task whose specs cannot be measured is reported as such, not scored.
  - The trajectory records what the model emitted and what came back, so a
    result can be audited rather than trusted.

Usage:
    PYTHONPATH=src python -m eval.runner --tasks eval/tasks --model local \\
        --output eval_results/run.json --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

log = logging.getLogger("eval.runner")

DEFAULT_MAX_STEPS = 12


class _Discover:
    """Sentinel: 'find one for me', as distinct from 'there is none'.

    `x = arg or discover()` cannot express "explicitly no engine", so a test
    that passes None to prove the no-engine path silently exercises the
    discovered one instead. That mistake has now appeared three times in this
    codebase (here, LlamaServer(binary=...), and ASIC_AI_LLAMA_CPP_DIR), each
    time hiding the very path the caller was trying to test.
    """

    def __repr__(self) -> str:
        return "<discover>"


DISCOVER = _Discover()


class EvalResult(BaseModel):
    task_id: str
    passed: bool
    final_score: float
    steps: int
    wall_time_sec: float
    trajectory: List[Dict[str, Any]]
    error: str | None = None


def load_task(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_task_prompt(task: Dict[str, Any]) -> str:
    """The user turn: the specification, in the shape the SFT data uses."""
    specs = json.dumps(task.get("specs", {}), indent=2)
    lines = [
        f"Design: {task.get('description', task.get('id', 'unknown'))}",
        f"PDK: {task.get('pdk', 'sky130')}",
        f"Supply: {task.get('supply', 1.8)}V",
    ]
    if task.get("topology_hint"):
        lines.append(f"Topology hint: {task['topology_hint']}")
    if task.get("load"):
        lines.append(f"Load: {task['load']}")
    lines.append(f"Corners: {task.get('corners', ['tt'])}")
    lines.append(f"\nSpecifications:\n{specs}")
    lines.append("\nDesign a circuit meeting ALL specifications. "
                 "Use the available tools to simulate and verify.")
    return "\n".join(lines)


def _default_engine():
    """A local llama.cpp server engine, or None when one is not available."""
    try:
        from asic_ai.inference import llama_server
    except ImportError:
        return None
    if not llama_server.available():
        return None
    cfg = llama_server.ServerConfig.from_config()
    if not cfg.model or not Path(cfg.model).exists():
        return None
    engine = llama_server.LlamaServerEngine(cfg.base_url)
    return engine if engine.healthy() else None


def _default_adapter():
    """A real ngspice adapter, or a mock, or None."""
    import tempfile
    from asic_ai.adapters import get_adapter
    for backend in ("ngspice_shared", "mock"):
        try:
            return get_adapter(backend, binary_path="", work_dir=tempfile.mkdtemp())
        except Exception:  # backend genuinely unavailable on this machine
            continue
    return None


def run_task(task_path: Path, model_id: str = "local", *,
             engine: Any = DISCOVER, adapter: Any = DISCOVER,
             max_steps: int = DEFAULT_MAX_STEPS,
             temperature: float = 0.2) -> EvalResult:
    """Run one eval task end to end and report the real outcome."""
    start = time.time()
    task_id = Path(task_path).stem
    trajectory: List[Dict[str, Any]] = []

    try:
        task = load_task(Path(task_path))
        task_id = task.get("id", task_id)
    except Exception as exc:
        return EvalResult(task_id=task_id, passed=False, final_score=0.0, steps=0,
                          wall_time_sec=time.time() - start, trajectory=[],
                          error=f"could not load task: {exc}")

    engine = _default_engine() if engine is DISCOVER else engine
    if engine is None:
        # No model. This is NOT a pass, and it is not a design failure either.
        return EvalResult(
            task_id=task_id, passed=False, final_score=0.0, steps=0,
            wall_time_sec=time.time() - start, trajectory=[],
            error=("no inference engine available -- start one with "
                   "scripts/serve_local.py, or pass engine=..."))

    adapter = _default_adapter() if adapter is DISCOVER else adapter
    if adapter is None:
        return EvalResult(task_id=task_id, passed=False, final_score=0.0, steps=0,
                          wall_time_sec=time.time() - start, trajectory=[],
                          error="no simulator adapter available")

    from asic_ai.inference.runner import run_agent_loop
    from asic_ai.reward.reward import RewardFunction
    from asic_ai.training.rl_env import CircuitDesignEnv

    try:
        reward_fn = RewardFunction.from_eval_task(task)
    except Exception as exc:
        return EvalResult(task_id=task_id, passed=False, final_score=0.0, steps=0,
                          wall_time_sec=time.time() - start, trajectory=[],
                          error=f"could not build a reward function: {exc}")

    env = CircuitDesignEnv(adapter, reward_fn, max_steps=max_steps)
    env.reset(task)

    # One implementation of the agent loop, shared with InferenceRunner.
    # Three copies of it existed before (here, inference/runner.py and
    # agent/loop.py); two of the three produced nothing at all.
    run = run_agent_loop(task, engine, env, max_steps=max_steps,
                         temperature=temperature,
                         user_prompt=format_task_prompt(task))

    return EvalResult(
        task_id=run.task_id if run.task_id != "unknown" else task_id,
        passed=run.passed,
        final_score=run.final_score,
        steps=run.steps,
        wall_time_sec=time.time() - start,
        trajectory=run.trajectory,
        error=run.error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eval set against a model")
    parser.add_argument("--tasks", type=str, required=True, help="tasks directory")
    parser.add_argument("--model", type=str, default="local", help="model identifier")
    parser.add_argument("--output", type=str, required=True, help="output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--parallel", action="store_true",
                        help="run tasks concurrently (the simulator is a shared "
                             "process; leave this off unless you know it is safe)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    task_files = sorted(Path(args.tasks).rglob("*.yaml"))
    if args.limit:
        task_files = task_files[:args.limit]
    if not task_files:
        print(f"no tasks found under {args.tasks}")
        return 1

    engine = _default_engine()
    if engine is None:
        print("No inference engine is reachable.")
        print("Start one with:  PYTHONPATH=src python scripts/serve_local.py")
        print("Refusing to emit results rather than reporting fabricated passes.")
        return 1

    print(f"running {len(task_files)} task(s)")
    results: List[Dict[str, Any]] = []

    if args.parallel:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_task, f, args.model, max_steps=args.max_steps)
                       for f in task_files]
            for fut in futures:
                results.append(fut.result().model_dump())
    else:
        for f in task_files:
            r = run_task(f, args.model, engine=engine, max_steps=args.max_steps)
            mark = "PASS" if r.passed else "fail"
            note = f"  [{r.error}]" if r.error else ""
            print(f"  {mark}  {r.task_id:28s} score {r.final_score:+.4f}  "
                  f"{r.steps} steps  {r.wall_time_sec:.1f}s{note}")
            results.append(r.model_dump())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)

    n_pass = sum(1 for r in results if r["passed"])
    n_err = sum(1 for r in results if r.get("error"))
    print(f"\n{n_pass}/{len(results)} passed, {n_err} with errors -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
