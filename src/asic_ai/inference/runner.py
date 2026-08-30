"""The agent loop. One implementation, used by everything that runs the model.

    spec -> generate -> parse tool calls -> validate -> execute -> observe
              ^                                                       |
              +-------------------------------------------------------+

This module previously set `self.engine = None` and, when it was None, used the
literal string "Dummy response" as the model's output -- so run_task produced a
trajectory of nothing and returned passed=False, final_score=0.0 unconditionally.
The same shape as the adapter returning zeros and the eval runner reporting a
constant pass.

There were three would-be homes for this loop: here, `agent/loop.py` (a while
loop whose body was comments), and `eval/runner.py`. Three implementations is
worse than one, so the loop lives here and the other two call it. `eval/runner`
adds task-file loading and result aggregation; `agent/loop` is a thin façade
kept for its existing callers.

Honesty rules, matching the rest of the stack:
  - No engine is an error, never a run of dummy text.
  - A tool outside the frozen contract is rejected before it reaches the
    adapter, and the rejection is fed back to the model as an observation, which
    is how it learns to recover.
  - The score comes from a real spec.check, never a constant.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from asic_ai.inference.parser import ToolCallParser

log = logging.getLogger(__name__)

__all__ = [
    "InferenceConfig", "InferenceResult", "EvalReport", "InferenceRunner",
    "SimulatorAdapter", "run_agent_loop",
]


class InferenceConfig(BaseModel):
    """Configuration for inference."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048
    max_steps: int = 20
    beam_search: bool = False
    tool_parse_strategy: str = "chatml"
    retry_on_parse_failure: bool = True
    timeout_per_step: int = 300


class InferenceResult(BaseModel):
    """Result of an inference task."""
    task_id: str
    passed: bool
    final_score: float
    steps: int
    wall_time: float
    trajectory: List[Dict[str, Any]]
    reward_breakdown: Dict[str, float]
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


class EvalReport(BaseModel):
    """Report for a full evaluation run."""
    total_tasks: int
    passed_tasks: int
    success_rate: float
    average_score: float
    task_results: List[InferenceResult]


class SimulatorAdapter(BaseModel):
    """Minimal in-memory adapter, kept for existing callers and tests."""

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        return "Tool executed successfully."


def run_agent_loop(task: Dict[str, Any],
                   engine: Any,
                   env: Any,
                   *,
                   max_steps: int = 12,
                   temperature: float = 0.2,
                   max_new_tokens: int = 1024,
                   user_prompt: Optional[str] = None) -> InferenceResult:
    """Drive one design episode. The single implementation of the loop.

    Args:
        task: eval task dict, with at least `id` and `specs`.
        engine: a ModelEngine.
        env: a CircuitDesignEnv, already reset onto this task.
        user_prompt: the task turn. A default is built when omitted.

    Returns:
        InferenceResult. `error` is set for a harness failure -- a broken run
        must be distinguishable from a design that failed on its merits.
    """
    from asic_ai.data.format import build_system_message

    start = time.time()
    parser = ToolCallParser()
    task_id = task.get("id", "unknown")

    if user_prompt is None:
        specs = json.dumps(task.get("specs", {}), indent=2)
        user_prompt = (
            f"Design: {task.get('description', task_id)}\n"
            f"PDK: {task.get('pdk', 'sky130')}\n"
            f"Supply: {task.get('supply', 1.8)}V\n\n"
            f"Specifications:\n{specs}\n\n"
            "Design a circuit meeting ALL specifications. Use the available "
            "tools to simulate and verify.")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": user_prompt},
    ]

    trajectory: List[Dict[str, Any]] = []
    breakdown: Dict[str, float] = {}
    score, passed, steps = 0.0, False, 0
    prompt_tokens = completion_tokens = 0
    error: Optional[str] = None

    try:
        for step in range(max_steps):
            gen = engine.generate(messages, temperature=temperature,
                                  max_new_tokens=max_new_tokens)
            text = gen.text or ""
            prompt_tokens += int(getattr(gen, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(gen, "completion_tokens", 0) or 0)
            messages.append({"role": "assistant", "content": text})

            calls = parser.parse(text)
            parse_errors = parser.parse_errors(text)
            record: Dict[str, Any] = {
                "step": step,
                "assistant": text[:2000],
                "tool_calls": [c.name for c in calls],
                "parse_errors": parse_errors,
            }
            trajectory.append(record)

            if not calls:
                if parse_errors:
                    error = f"unparseable tool call at step {step}: {parse_errors[0]}"
                break

            for call in calls:
                ok, why = parser.validate_tool_call(call)
                if not ok:
                    # Feed the rejection back rather than aborting: recovering
                    # from a bad call is the behaviour worth measuring.
                    observation = json.dumps({"error": why})
                else:
                    result = env.step({"name": call.name,
                                       "arguments": call.arguments})
                    observation = result.observation
                    if call.name == "spec.check":
                        try:
                            payload = json.loads(observation)
                            score = float(payload.get("score", score))
                            passed = bool(payload.get("passed", passed))
                            measured = payload.get("measured") or {}
                            breakdown = {k: float(v) for k, v in measured.items()
                                         if isinstance(v, (int, float))}
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                messages.append({"role": "tool", "content": observation[:4000]})
                record.setdefault("observations", []).append(observation[:1000])

            steps = step + 1
            if passed:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.warning("task %s aborted: %s", task_id, error)

    return InferenceResult(
        task_id=task_id, passed=passed, final_score=score, steps=steps,
        wall_time=time.time() - start, trajectory=trajectory,
        reward_breakdown=breakdown, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, error=error,
    )


class InferenceRunner:
    """Run the model on eval tasks.

    Accepts a ready ModelEngine, or a model path to build one from.
    """

    def __init__(self, model_path: str = "", adapter: Any = None,
                 config: Optional[InferenceConfig] = None, *, engine: Any = None):
        self.model_path = model_path
        self.adapter = adapter
        self.config = config or InferenceConfig()
        self.parser = ToolCallParser()
        self.engine = engine

    def _resolve_engine(self):
        if self.engine is not None:
            return self.engine
        if not self.model_path:
            return None
        from asic_ai.inference.engine import TransformersEngine
        self.engine = TransformersEngine(self.model_path)
        return self.engine

    def _resolve_env(self, task: Dict[str, Any]):
        from asic_ai.reward.reward import RewardFunction
        from asic_ai.training.rl_env import CircuitDesignEnv

        reward_fn = RewardFunction.from_eval_task(task)
        env = CircuitDesignEnv(self.adapter, reward_fn,
                               max_steps=self.config.max_steps)
        env.reset(task)
        return env

    def run_task(self, task: dict) -> InferenceResult:
        """Run the model on a single eval task."""
        engine = self._resolve_engine()
        if engine is None:
            # Never a run of dummy text: a missing model is an error.
            return InferenceResult(
                task_id=task.get("id", "unknown"), passed=False, final_score=0.0,
                steps=0, wall_time=0.0, trajectory=[], reward_breakdown={},
                prompt_tokens=0, completion_tokens=0,
                error="no inference engine: pass engine=... or a model_path")
        if self.adapter is None:
            return InferenceResult(
                task_id=task.get("id", "unknown"), passed=False, final_score=0.0,
                steps=0, wall_time=0.0, trajectory=[], reward_breakdown={},
                prompt_tokens=0, completion_tokens=0,
                error="no simulator adapter")

        return run_agent_loop(
            task, engine, self._resolve_env(task),
            max_steps=self.config.max_steps,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            user_prompt=task.get("spec") if isinstance(task.get("spec"), str) else None,
        )

    def run_eval(self, task_dir: str) -> EvalReport:
        """Run every task under `task_dir` and aggregate."""
        from pathlib import Path

        import yaml

        results: List[InferenceResult] = []
        for path in sorted(Path(task_dir).rglob("*.yaml")):
            with open(path, encoding="utf-8") as fh:
                task = yaml.safe_load(fh)
            results.append(self.run_task(task))

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        return EvalReport(
            total_tasks=total,
            passed_tasks=passed,
            success_rate=(passed / total) if total else 0.0,
            average_score=(sum(r.final_score for r in results) / total) if total else 0.0,
            task_results=results,
        )
