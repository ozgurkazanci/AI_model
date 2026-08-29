"""RL Environment for ASIC circuit design.

Wraps the simulator adapter into a gym-like environment for GRPO training.
The agent receives a task specification, produces design iterations, and
receives reward from the simulator.

This is the bridge between the LLM agent and the RL training loop.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from asic_ai.adapters import spec_extract

logger = logging.getLogger(__name__)


@dataclass
class DesignState:
    """Current state of a circuit design episode."""
    task_id: str
    task_specs: dict[str, Any]
    netlist: str = ""
    step: int = 0
    sim_results: dict[str, Any] = field(default_factory=dict)
    analyses: dict[str, Any] = field(default_factory=dict)
    """Typed adapter results per analysis ('dc', 'ac', 'tran', 'noise', 'stb').

    sim_results is the flat model_dump() merge, keyed by SCHEMA FIELD names, and
    is kept for the observation text. The reward must never be scored off it:
    RewardFunction looks up SPEC names, so a schema-keyed dict scores every spec
    at -1.0. spec_extract.extract_specs() converts these typed results into
    spec-name-keyed scalars in the units the task declares.
    """
    history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    success: bool = False
    total_reward: float = 0.0


@dataclass
class StepResult:
    """Result of one design step."""
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class CircuitDesignEnv:
    """Gym-like environment for circuit design RL.

    Episode flow:
        1. reset(task) -> initial observation (spec description)
        2. step(action) -> observation, reward, done, info
           - action is a tool call (sim.ac, netlist.patch, etc.)
           - observation is the tool result
           - reward comes from spec.check via RewardFunction
        3. Episode ends when: spec met, max steps, or convergence failure

    Usage:
        env = CircuitDesignEnv(adapter, reward_fn, max_steps=20)
        obs = env.reset(task)
        while not done:
            action = model.generate(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        adapter: Any,
        reward_fn: Any,
        max_steps: int = 20,
        early_stop_threshold: float = 0.95,
        step_penalty: float = 0.005,
    ):
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.early_stop_threshold = early_stop_threshold
        self.step_penalty = step_penalty
        self.state: DesignState | None = None
        self._episode_start: float = 0.0

    def reset(self, task: dict[str, Any]) -> str:
        """Start a new design episode.

        Args:
            task: Eval task dict with 'id', 'specs', 'description', 'pdk', etc.

        Returns:
            Initial observation string (task specification).
        """
        self.state = DesignState(
            task_id=task.get("id", "unknown"),
            task_specs=task.get("specs", {}),
        )
        self._episode_start = time.time()

        # Format initial observation
        specs_str = json.dumps(task.get("specs", {}), indent=2)
        observation = (
            f"Design Task: {task.get('description', task.get('id', 'unknown'))}\n"
            f"PDK: {task.get('pdk', 'sky130')}\n"
            f"Supply: {task.get('supply', 1.8)}V\n"
            f"Load: {task.get('load', 'unspecified')}\n"
            f"Specifications:\n{specs_str}\n\n"
            f"Design a circuit that meets ALL specifications. "
            f"Use available tools to simulate and verify."
        )

        self.state.history.append({
            "step": 0,
            "role": "system",
            "content": observation,
        })

        return observation

    def step(self, action: dict[str, Any]) -> StepResult:
        """Execute one design step.

        Args:
            action: Tool call dict with 'name' and 'arguments'.
                    e.g., {"name": "sim.ac", "arguments": {"netlist": "..."}}

        Returns:
            StepResult with observation, reward, done, info.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        self.state.step += 1
        tool_name = action.get("name", "")
        tool_args = action.get("arguments", {})

        # Execute the tool
        observation, tool_success = self._execute_tool(tool_name, tool_args)

        # Track netlist updates
        if tool_name == "netlist.patch" and tool_success:
            # Update current netlist from patch result
            if "netlist" in tool_args:
                self.state.netlist = tool_args["netlist"]

        # Compute reward
        reward = self._compute_reward(tool_name, observation, tool_success)
        self.state.total_reward += reward

        # Check termination
        done = False
        if self.state.step >= self.max_steps:
            done = True
            logger.debug("Episode ended: max steps reached")
        elif self.state.success:
            done = True
            logger.debug("Episode ended: specs met")
        elif not tool_success and tool_name.startswith("sim."):
            # Simulation convergence failure
            reward -= 0.1  # Extra penalty for convergence failure
            # Don't end episode — let model try to fix

        self.state.done = done

        # Record history
        self.state.history.append({
            "step": self.state.step,
            "action": action,
            "observation": observation[:500],  # Truncate for memory
            "reward": reward,
            "done": done,
        })

        info = {
            "step": self.state.step,
            "total_reward": self.state.total_reward,
            "success": self.state.success,
            "tool_name": tool_name,
            "duration_sec": time.time() - self._episode_start,
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool call and return (observation, success)."""
        try:
            if tool_name == "sim.dc":
                result = self._run_sim("dc", args)
            elif tool_name == "sim.ac":
                result = self._run_sim("ac", args)
            elif tool_name == "sim.tran":
                result = self._run_sim("tran", args)
            elif tool_name == "sim.noise":
                result = self._run_sim("noise", args)
            elif tool_name == "sim.stb":
                result = self._run_sim("stb", args)
            elif tool_name == "sim.corners":
                result = self._run_corners(args)
            elif tool_name == "sim.mc":
                result = self._run_sim("mc", args)
            elif tool_name == "spec.check":
                result = self._run_spec_check(args)
            elif tool_name == "netlist.patch":
                result = {"status": "ok", "message": "Netlist updated"}
            elif tool_name == "lint.check":
                result = {"status": "ok", "errors": [], "warnings": []}
            elif tool_name.startswith("pdk."):
                result = self._run_pdk_query(tool_name, args)
            elif tool_name == "opt.suggest":
                result = {"status": "ok", "suggestion": "Increase W of M1 by 2x"}
            elif tool_name == "meas.eval":
                result = {"value": 0.0, "unit": "V"}
            else:
                return f"Unknown tool: {tool_name}", False

            self.state.sim_results.update(result if isinstance(result, dict) else {})
            return json.dumps(result, default=str), True

        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)}), False

    def _run_sim(self, sim_type: str, args: dict) -> dict:
        """Run simulation via adapter."""
        netlist = args.get("netlist", self.state.netlist)
        if not netlist:
            return {"error": "No netlist provided"}

        # Use mock adapter's methods if available
        if hasattr(self.adapter, sim_type):
            method = getattr(self.adapter, sim_type)
            # Mock adapter returns Pydantic models, convert to dict
            result = method(netlist, args)
            if hasattr(result, "model_dump"):
                # Keep the typed result: spec_extract needs the structure, not
                # the flattened dump, to derive spec-name-keyed scalars.
                self.state.analyses[sim_type] = result
                return result.model_dump()
            return result if isinstance(result, dict) else {"result": str(result)}

        return {"status": "simulated", "type": sim_type}

    def _run_corners(self, args: dict) -> dict:
        """Run corner simulation."""
        if hasattr(self.adapter, "corners"):
            netlist = args.get("netlist", self.state.netlist)
            corners = args.get("corners", ["tt", "ss", "ff"])
            result = self.adapter.corners(netlist, corners)
            if isinstance(result, list):
                return {"corners": [r.model_dump() if hasattr(r, "model_dump") else r for r in result]}
            return result if isinstance(result, dict) else {"result": str(result)}
        return {"corners": []}

    def _run_spec_check(self, args: dict) -> dict:
        """Measure the task's specs from the stored analyses, then score them.

        The measured values are keyed by SPEC name and expressed in the unit the
        task declared, which is what RewardFunction expects. Specs that cannot
        be derived from the analyses that were actually run are reported in
        `unmeasurable` rather than dropped: RewardFunction reads a missing spec
        as -1.0, which is indistinguishable from a design that was measured and
        failed, and that silently pins the reward to the floor.
        """
        specs = args.get("specs", self.state.task_specs)

        explicit = args.get("results")
        if explicit:
            # Caller supplied measurements directly; trust them as spec-keyed.
            measured = dict(explicit)
            unmeasurable: dict[str, str] = {}
        else:
            extraction = spec_extract.extract_specs(
                specs,
                dc=self.state.analyses.get("dc"),
                ac=self.state.analyses.get("ac"),
                tran=self.state.analyses.get("tran"),
                noise=self.state.analyses.get("noise"),
                stb=self.state.analyses.get("stb"),
            )
            measured = extraction.values
            unmeasurable = extraction.unmeasurable

        if unmeasurable:
            logger.info(
                "spec.check: %d/%d specs measurable; not derivable: %s",
                len(measured), len(specs), sorted(unmeasurable),
            )

        score = self._score(specs, measured)
        if score is None:
            return {
                "score": 0.0,
                "passed": False,
                "error": "no usable reward function",
                "measured": measured,
                "unmeasurable": unmeasurable,
            }

        self.state.success = score >= self.early_stop_threshold
        return {
            "score": score,
            "passed": self.state.success,
            "specs_checked": len(specs),
            "specs_measured": len(measured),
            "measured": measured,
            "unmeasurable": unmeasurable,
        }

    def _score(self, specs: dict[str, Any],
               measured: dict[str, float]) -> float | None:
        """Score measured values, accepting either reward_fn convention.

        Two incompatible shapes are in use in this repo:
          - the closure built by rl_grpo.create_reward_fn(), called as
            reward_fn(task_specs, {"measurements": ...}) -> float
          - a bare RewardFunction instance, which has NO __call__ and must be
            driven through .compute(results=...) -> RewardResult
        Calling the second like the first raises TypeError, which a bare
        `except Exception: pass` used to swallow, silently scoring 0.0.
        """
        if self.reward_fn is None:
            return None

        compute = getattr(self.reward_fn, "compute", None)
        if callable(compute):
            try:
                return float(compute(results=measured).total_reward)
            except Exception:
                logger.exception("RewardFunction.compute failed")
                return None

        if callable(self.reward_fn):
            try:
                return float(self.reward_fn(specs, {"measurements": measured}))
            except Exception:
                logger.exception("reward_fn callable failed")
                return None

        logger.error("reward_fn is neither callable nor a RewardFunction: %r",
                     type(self.reward_fn).__name__)
        return None

    def _run_pdk_query(self, tool_name: str, args: dict) -> dict:
        """Mock PDK query responses."""
        if tool_name == "pdk.list_devices":
            return {
                "devices": [
                    {"name": "nfet_01v8", "type": "nmos", "vth": 0.45, "w_range": [0.42e-6, 100e-6]},
                    {"name": "pfet_01v8", "type": "pmos", "vth": -0.45, "w_range": [0.42e-6, 100e-6]},
                ]
            }
        elif tool_name == "pdk.device_query":
            w = args.get("W", 1e-6)
            l = args.get("L", 180e-9)
            vgs = args.get("VGS", 0.6)
            gm = 2e-3 * (w / l) * max(0, vgs - 0.45)
            return {"gm": gm, "gds": gm / 50, "id": gm * 0.15, "vth": 0.45, "ft": gm / (2 * 3.14159 * 20e-15)}
        elif tool_name == "pdk.get_corners":
            return {"corners": ["tt", "ss", "ff", "sf", "fs"], "temp_range": [-40, 125], "vdd_range": [1.62, 1.98]}

        return {}

    def _compute_reward(self, tool_name: str, observation: str, success: bool) -> float:
        """Compute step reward."""
        reward = 0.0

        # Step penalty (encourages efficiency)
        reward -= self.step_penalty

        # Simulation reward (running sim is good, shows progress)
        if tool_name.startswith("sim.") and success:
            reward += 0.02

        # Spec check reward (checking specs is very valuable)
        if tool_name == "spec.check" and success:
            try:
                result = json.loads(observation)
                score = result.get("score", 0)
                reward += score * 0.5  # Scale spec score to reward
                if result.get("passed"):
                    reward += 1.0  # Big bonus for passing all specs
            except (json.JSONDecodeError, TypeError):
                pass

        # Corner analysis reward (shows thoroughness)
        if tool_name == "sim.corners" and success:
            reward += 0.05

        return reward

    def get_episode_summary(self) -> dict[str, Any]:
        """Get summary of completed episode."""
        if self.state is None:
            return {}

        return {
            "task_id": self.state.task_id,
            "steps": self.state.step,
            "total_reward": self.state.total_reward,
            "success": self.state.success,
            "duration_sec": time.time() - self._episode_start,
            "history_length": len(self.state.history),
        }
