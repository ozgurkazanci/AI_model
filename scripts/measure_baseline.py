#!/usr/bin/env python3
"""Baseline measurement script.

Measures existing LLMs (via API) on the eval set WITHOUT fine-tuning.
This establishes the baseline that our fine-tuned model must beat.

Usage:
    PYTHONPATH=src python scripts/measure_baseline.py \
        --model openai/gpt-4o \
        --tasks eval/tasks/ \
        --output eval_results/baseline_gpt4o.json \
        --max-steps 20

Supported model backends:
    - openai/gpt-4o, openai/gpt-4o-mini
    - anthropic/claude-3-opus, anthropic/claude-3-sonnet
    - together/Qwen/Qwen3.6-35B-A3B
    - local/<path-to-model>  (via vLLM or transformers)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.reward.reward import RewardFunction, RewardMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("baseline")


@dataclass
class BaselineConfig:
    """Configuration for baseline measurement."""
    model: str = "openai/gpt-4o-mini"
    tasks_dir: str = "eval/tasks"
    output_path: str = "eval_results/baseline.json"
    max_steps: int = 20
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_per_step: float = 60.0
    n_attempts: int = 3  # attempts per task (best-of-n)
    simulator: str = "ngspice"
    dry_run: bool = False  # if True, just count tasks without running


@dataclass
class TaskResult:
    """Result from running a single task."""
    task_id: str
    category: str
    difficulty: str
    passed: bool = False
    final_score: float = 0.0
    steps: int = 0
    wall_time_sec: float = 0.0
    error: str | None = None
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class BaselineReport:
    """Full baseline measurement report."""
    model: str
    timestamp: str
    config: dict[str, Any]
    # Per-task results
    results: list[dict[str, Any]] = field(default_factory=list)
    # Aggregate metrics
    total_tasks: int = 0
    tasks_passed: int = 0
    pass_rate: float = 0.0
    avg_score: float = 0.0
    avg_steps: float = 0.0
    # By category
    analog_pass_rate: float = 0.0
    digital_pass_rate: float = 0.0
    # By difficulty
    easy_pass_rate: float = 0.0
    medium_pass_rate: float = 0.0
    hard_pass_rate: float = 0.0


def load_all_tasks(tasks_dir: str) -> list[dict[str, Any]]:
    """Load all YAML eval tasks from directory tree."""
    tasks = []
    for yaml_file in sorted(Path(tasks_dir).rglob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            task = yaml.safe_load(f)
        task["_file"] = str(yaml_file)
        tasks.append(task)
    return tasks


def compute_aggregate_metrics(results: list[TaskResult]) -> dict[str, Any]:
    """Compute aggregate metrics from task results."""
    if not results:
        return {}

    total = len(results)
    passed = [r for r in results if r.passed]

    # By category
    analog = [r for r in results if r.category == "analog"]
    digital = [r for r in results if r.category == "digital"]
    analog_passed = [r for r in analog if r.passed]
    digital_passed = [r for r in digital if r.passed]

    # By difficulty
    easy = [r for r in results if r.difficulty == "easy"]
    medium = [r for r in results if r.difficulty == "medium"]
    hard = [r for r in results if r.difficulty == "hard"]
    easy_passed = [r for r in easy if r.passed]
    medium_passed = [r for r in medium if r.passed]
    hard_passed = [r for r in hard if r.passed]

    return {
        "total_tasks": total,
        "tasks_passed": len(passed),
        "pass_rate": len(passed) / total,
        "avg_score": sum(r.final_score for r in results) / total,
        "avg_steps": sum(r.steps for r in passed) / len(passed) if passed else 0,
        "analog_pass_rate": len(analog_passed) / len(analog) if analog else 0,
        "digital_pass_rate": len(digital_passed) / len(digital) if digital else 0,
        "easy_pass_rate": len(easy_passed) / len(easy) if easy else 0,
        "medium_pass_rate": len(medium_passed) / len(medium) if medium else 0,
        "hard_pass_rate": len(hard_passed) / len(hard) if hard else 0,
    }


def dry_run_report(tasks: list[dict]) -> None:
    """Print task summary without running anything."""
    categories = {}
    difficulties = {}
    for t in tasks:
        cat = t.get("category", "unknown")
        diff = t.get("difficulty", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Categories: {categories}")
    print(f"Difficulties: {difficulties}")
    print("\nTasks:")
    for t in tasks:
        specs = list(t.get("specs", {}).keys())
        print(f"  [{t.get('difficulty', '?'):6s}] {t.get('category', '?'):7s} | {t.get('id', '?'):30s} | specs: {specs}")


async def run_task_with_model(
    task: dict,
    config: BaselineConfig,
) -> TaskResult:
    """Run a single task with the specified model.

    This is a SKELETON — the actual model interaction requires:
    1. An API client (OpenAI, Anthropic, etc.) or local model
    2. A running simulator (ngspice)
    3. The system prompt from data.format

    Drives the shared agent loop. With no engine reachable it still returns
    passed=False with a reason rather than a score -- a baseline of zeros
    would make every later run look like an improvement.
    """
    task_id = task.get("id", "unknown")
    category = task.get("category", "unknown")
    difficulty = task.get("difficulty", "unknown")

    log.info(f"Running task: {task_id} ({category}/{difficulty})")

    start = time.time()

    try:
        from asic_ai.inference.runner import run_agent_loop
        from eval.runner import _default_adapter, _default_engine

        engine = _default_engine()
        if engine is None:
            # Still the honest placeholder: no model means no score, never a
            # pass. A baseline of zeros would make every later run look like an
            # improvement.
            return TaskResult(
                task_id=task_id,
                category=category,
                difficulty=difficulty,
                passed=False,
                final_score=0.0,
                steps=0,
                wall_time_sec=time.time() - start,
                error=("no inference engine reachable -- start one with "
                       "scripts/serve_local.py, or point ASIC_AI_LLAMA_SERVER_URL "
                       "at an OpenAI-compatible endpoint"),
            )

        adapter = _default_adapter()
        if adapter is None:
            return TaskResult(
                task_id=task_id, category=category, difficulty=difficulty,
                passed=False, final_score=0.0, steps=0,
                wall_time_sec=time.time() - start,
                error="no simulator adapter available",
            )

        from asic_ai.reward.reward import RewardFunction
        from asic_ai.training.rl_env import CircuitDesignEnv

        reward_fn = RewardFunction.from_eval_task(task)
        env = CircuitDesignEnv(adapter, reward_fn, max_steps=config.max_steps)
        env.reset(task)

        # The single implementation of the agent loop, shared with
        # eval/runner.py, agent/loop.py and InferenceRunner.
        run = run_agent_loop(task, engine, env, max_steps=config.max_steps,
                             temperature=config.temperature,
                             max_new_tokens=config.max_tokens)

        return TaskResult(
            task_id=task_id,
            category=category,
            difficulty=difficulty,
            passed=run.passed,
            final_score=run.final_score,
            steps=run.steps,
            wall_time_sec=time.time() - start,
            error=run.error,
            reward_breakdown=run.reward_breakdown,
            token_usage={"prompt": run.prompt_tokens,
                         "completion": run.completion_tokens},
        )
    except Exception as e:
        return TaskResult(
            task_id=task_id,
            category=category,
            difficulty=difficulty,
            error=str(e),
            wall_time_sec=time.time() - start,
        )


async def main_async(config: BaselineConfig) -> None:
    """Main async entry point."""
    tasks = load_all_tasks(config.tasks_dir)
    log.info(f"Loaded {len(tasks)} eval tasks from {config.tasks_dir}")

    if config.dry_run:
        dry_run_report(tasks)
        return

    results: list[TaskResult] = []
    for task in tasks:
        result = await run_task_with_model(task, config)
        results.append(result)

    # Compute metrics
    metrics = compute_aggregate_metrics(results)

    # Build report
    report = BaselineReport(
        model=config.model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config=asdict(config),
        results=[asdict(r) for r in results],
        **metrics,
    )

    # Save
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    log.info(f"Report saved to {output_path}")
    log.info(f"Pass rate: {metrics.get('pass_rate', 0):.1%}")
    log.info(f"Avg score: {metrics.get('avg_score', 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure baseline LLM performance on eval set")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model identifier")
    parser.add_argument("--tasks", default="eval/tasks", help="Tasks directory")
    parser.add_argument("--output", default="eval_results/baseline.json", help="Output file")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per task")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n-attempts", type=int, default=3, help="Best-of-N attempts per task")
    parser.add_argument("--simulator", default="ngspice", help="Simulator backend")
    parser.add_argument("--dry-run", action="store_true", help="Just list tasks, don't run")
    args = parser.parse_args()

    config = BaselineConfig(
        model=args.model,
        tasks_dir=args.tasks,
        output_path=args.output,
        max_steps=args.max_steps,
        temperature=args.temperature,
        n_attempts=args.n_attempts,
        simulator=args.simulator,
        dry_run=args.dry_run,
    )

    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
