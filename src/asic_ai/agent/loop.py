"""Agent loop façade.

The design document's Layer 1: the model's working rhythm --

    spec -> topology -> netlist -> simulate -> read -> diagnose -> fix -> repeat

The body of `run()` used to be a list of comments inside a `while` that only
incremented a counter, then reported `status = "max_steps_reached"`. A loop that
never ran anything claimed to have exhausted its step budget.

The loop itself now lives in `asic_ai.inference.runner.run_agent_loop`, which is
the one implementation the eval runner and InferenceRunner also use. This class
stays as the façade its existing callers expect, and delegates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    max_steps: int = 20
    max_retries_same_error: int = 3
    checkpoint_interval: int = 5
    use_optimizer_for_analog: bool = True
    escalation_threshold: int = 10
    temperature: float = 0.7


class EvalTask(BaseModel):
    spec: Dict[str, Any]
    task_type: str = "analog"


class Trajectory(BaseModel):
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "running"
    final_result: Optional[Dict[str, Any]] = None


class AgentLoop:
    """Main agent loop: plan -> act -> observe -> decide.

    The model is NOT a one-shot netlist generator. It is an agent:
    spec -> topology -> netlist -> simulate -> read result -> diagnose -> fix ->
    re-simulate, until the spec is met.

    `model` must be a ModelEngine (anything with .generate(messages, **kwargs)).
    Given something else, this reports status="error" rather than running a
    loop that produces nothing and calling it a completed episode.
    """

    def __init__(self, model: Any, simulator: Any, optimizer: Any,
                 config: AgentConfig):
        self.model = model
        self.simulator = simulator
        self.optimizer = optimizer
        self.config = config

    async def run(self, task: EvalTask) -> Trajectory:
        """Run the full agent loop for a design task."""
        if not hasattr(self.model, "generate"):
            return Trajectory(
                steps=[], status="error",
                final_result={"error": (
                    f"{type(self.model).__name__} is not a ModelEngine: it has "
                    "no generate(). Pass an engine from asic_ai.inference "
                    "(LlamaServerEngine, TransformersEngine, APIEngine).")})
        if self.simulator is None:
            return Trajectory(steps=[], status="error",
                              final_result={"error": "no simulator adapter"})

        from asic_ai.inference.runner import run_agent_loop
        from asic_ai.reward.reward import RewardFunction
        from asic_ai.training.rl_env import CircuitDesignEnv

        task_dict: Dict[str, Any] = {
            "id": getattr(task, "task_type", "task"),
            "specs": task.spec,
        }

        try:
            reward_fn = RewardFunction.from_eval_task(task_dict)
        except Exception as exc:
            return Trajectory(steps=[], status="error",
                              final_result={"error": f"reward function: {exc}"})

        env = CircuitDesignEnv(self.simulator, reward_fn,
                               max_steps=self.config.max_steps)
        env.reset(task_dict)

        result = run_agent_loop(task_dict, self.model, env,
                                max_steps=self.config.max_steps,
                                temperature=self.config.temperature)

        if result.error:
            status = "error"
        elif result.passed:
            status = "success"
        elif result.steps >= self.config.max_steps:
            status = "max_steps_reached"
        else:
            status = "stopped"

        return Trajectory(
            steps=result.trajectory,
            status=status,
            final_result={
                "score": result.final_score,
                "passed": result.passed,
                "measured": result.reward_breakdown,
                "error": result.error,
            },
        )

    def _should_use_optimizer(self, action_type: str) -> bool:
        """Analog sizing -> optimizer. Digital/topology -> LLM only."""
        return self.config.use_optimizer_for_analog and action_type.lower() == "analog_sizing"

    def _check_stuck(self, trajectory: Trajectory) -> bool:
        """Detect if the agent is repeating the same mistakes.

        Repetition, not step count: the previous version returned True once the
        trajectory was merely long, so a productive episode was called stuck.
        """
        calls = [tuple(sorted(s.get("tool_calls", []))) for s in trajectory.steps]
        recent = calls[-self.config.max_retries_same_error:]
        return (len(recent) >= self.config.max_retries_same_error
                and len(set(recent)) == 1 and recent[0] != ())

    def _escalate(self, reason: str) -> None:
        """Ask the user when the agent is stuck or uncertain."""
        import logging
        logging.getLogger(__name__).warning("ESCALATION: %s", reason)
