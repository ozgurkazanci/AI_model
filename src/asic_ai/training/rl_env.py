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

from asic_ai import serialization
from asic_ai.adapters import measure, spec_extract

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
    analysis_netlists: dict[str, str] = field(default_factory=dict)
    """The deck each stored analysis was actually run on.

    `netlist` is "the deck the episode is currently working on"; this is "the
    deck that produced dc/ac/tran/noise/stb". They are not the same thing the
    moment an agent runs one analysis on the design and the next on a separate
    testbench, and the reward must be measured against the deck the numbers
    came from. Handing spec_extract the CURRENT netlist let a stability
    testbench decide the supply polarities of an operating point taken from a
    different circuit.
    """
    noise_freq: float | None = None
    """Frequency, in Hz, at which the task declares its noise DENSITY spec."""
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
        specs = task.get("specs", {})
        self.state = DesignState(
            task_id=task.get("id", "unknown"),
            task_specs=specs,
            # A noise density is a value at a frequency and the task is the
            # only thing that knows which. Read it from a task-level
            # `noise_freq`, else off the spec itself (`at_freq`). Without it
            # the density is refused on any 1/f spectrum, and a spec that can
            # never be measured is a silent -1.0 on the task.
            noise_freq=(task.get("noise_freq")
                        or spec_extract.spec_noise_freq(specs)),
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
            # Don't end episode -- let model try to fix

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
            # serialization.dumps, never json.dumps: a raw dump writes
            # -Infinity and NaN for the non-finite floats an AC result can
            # legitimately carry, and this text goes straight to the model and
            # on into trajectories and SFT files, where it is not loadable.
            return serialization.dumps(result), True

        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return serialization.dumps({"error": str(e)}), False

    def _run_sim(self, sim_type: str, args: dict) -> dict:
        """Run simulation via adapter.

        A netlist supplied INLINE with the call becomes the episode's current
        netlist. The tool schema allows an inline netlist and the SFT data
        demonstrates it, but state.netlist used to be written only by
        netlist.patch, so an agent that called sim.dc directly ran the
        simulation on its deck and then handed spec.check a netlist of None.
        Without the deck, supply_current cannot tell a 0 V ammeter from a rail:
        on a deck with a Vsense in series with the load that turned a true
        198 uA into 378 uA (91 pct high) and a +0.3364 score into -0.5965,
        entirely according to whether netlist.patch had been called first.

        An inline netlist is adopted ONLY AFTER the tool has succeeded, the
        same way netlist.patch is guarded in step(). Adopting it first meant a
        FAILED call replaced the episode's deck with the deck that failed:
        sim.dc on the design (a Vdd plus a 0 V Vsense ammeter), then a sim.ac
        on a stability testbench with no .ac card, raises NgspiceError and
        tool_success is False -- and state.netlist was already the testbench.
        spec.check then resolved the supply polarities of the design's
        operating point out of a deck that has no Vsense, so the ammeter was
        summed as a second rail and idd read 0.36 mA against a true 0.18 mA:
        exactly 2x, the D7/N5 failure returning by a third route.
        """
        netlist = args.get("netlist", self.state.netlist)
        if not netlist:
            return {"error": "No netlist provided"}
        inline = args.get("netlist")

        # Use mock adapter's methods if available
        if hasattr(self.adapter, sim_type):
            method = getattr(self.adapter, sim_type)
            # Mock adapter returns Pydantic models, convert to dict
            result = method(netlist, args)
            failed = isinstance(result, dict) and result.get("error")
            if not failed:
                if inline:
                    self.state.netlist = inline
                self.state.analysis_netlists[sim_type] = netlist
            if hasattr(result, "model_dump"):
                # Keep the typed result: spec_extract needs the structure, not
                # the flattened dump, to derive spec-name-keyed scalars.
                self.state.analyses[sim_type] = result
                return result.model_dump()
            return result if isinstance(result, dict) else {"result": str(result)}

        # No adapter method: nothing was simulated, so nothing is adopted. A
        # deck the simulator never saw must not become the deck the reward is
        # measured against.
        return {"status": "simulated", "type": sim_type}

    def _run_corners(self, args: dict) -> dict:
        """Run corner simulation.

        The inline netlist is adopted only after the corner run has succeeded,
        for the reason spelled out in _run_sim: a deck that FAILED must never
        become the deck the reward is measured against.
        """
        if hasattr(self.adapter, "corners"):
            netlist = args.get("netlist", self.state.netlist)
            corners = args.get("corners", ["tt", "ss", "ff"])
            result = self.adapter.corners(netlist, corners)
            if args.get("netlist") and not (
                    isinstance(result, dict) and result.get("error")):
                self.state.netlist = args["netlist"]
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
        # The TASK's specs are the contract being scored. They used to be
        # overridable by the call's own `specs` argument, and the 824g eval's
        # two "passes" were exactly that override: edge_detector_001 re-sent
        # the task specs MINUS the 250 MHz clock spec its own numbers failed,
        # and decoder_3et al. sent booleans it then asserted. The argument now
        # only fills a vacuum (a chat session with no task specs); with a task
        # loaded, the task decides what is checked.
        specs = self.state.task_specs or args.get("specs", {})

        # Caller-supplied `results` are treated as a CLAIM, never as the
        # measurement. Scoring always derives from the analyses that actually
        # ran: in the 824g eval, 73 of 97 verifiable claimed values appeared
        # in no prior observation -- the model invented numbers, sent them
        # here, and the old trust-the-caller path scored them. An env that
        # scores unexecuted physics is the fabrication pattern this repo
        # documents, wearing a new hat.
        claimed_raw = args.get("results") or {}
        claimed = {k: v for k, v in dict(claimed_raw).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        extraction = spec_extract.extract_specs(
            specs,
            dc=self.state.analyses.get("dc"),
            ac=self.state.analyses.get("ac"),
            tran=self.state.analyses.get("tran"),
            noise=self.state.analyses.get("noise"),
            stb=self.state.analyses.get("stb"),
            # The deck is the only thing that can tell a 0 V sense source
            # from a supply rail, or spot a current-source-biased block
            # whose supply current is not in the operating point at all --
            # but it has to be the deck the numbers CAME FROM, not
            # whichever deck the episode has reached by now.
            netlist=self._reward_netlist(),
            noise_freq=args.get("noise_freq", self.state.noise_freq),
        )
        measured = extraction.values
        unmeasurable = extraction.unmeasurable

        # Feedback on the claim, so an honest model can calibrate and a
        # fabricating one is told exactly which numbers had no basis.
        claim_mismatch: dict[str, str] = {}
        for k, v in claimed.items():
            if k in measured:
                m = measured[k]
                denom = max(abs(m), abs(v), 1e-30)
                if abs(m - v) / denom > 0.05:
                    claim_mismatch[k] = (f"claimed {v:g}, measured {m:g}")
            else:
                claim_mismatch[k] = (f"claimed {v:g}, but no analysis run "
                                     "here produced this quantity")

        if unmeasurable:
            logger.info(
                "spec.check: %d/%d specs measurable; not derivable: %s",
                len(measured), len(specs), sorted(unmeasurable),
            )

        # Score ONLY the specs that were actually measured. Handing the full
        # spec dict to RewardFunction scores every unmeasurable spec at
        # SCORE_CLIP_MIN = -1.0, which makes the reward depend on how the model
        # happened to set up the sweep rather than on the circuit. Measured on
        # one identical, spec-meeting design: sweeping from 0.01 Hz scored
        # +0.63, from 1 Hz scored -0.00, purely because dc_gain became
        # unmeasurable. That is a gradient toward "sweep from 0.01 Hz", not
        # toward a better amplifier.
        scored_specs = {k: v for k, v in specs.items() if k in measured}

        if specs and not scored_specs:
            # Nothing was measurable. This must not read as a pass, and it must
            # not read as a design failure either -- it is a missing analysis.
            return {
                "score": 0.0,
                "passed": False,
                "error": "no spec could be measured from the analyses run so far",
                "specs_checked": len(specs),
                "specs_measured": 0,
                "coverage": 0.0,
                "measured": measured,
                "unmeasurable": unmeasurable,
                **({"claim_mismatch": claim_mismatch} if claim_mismatch else {}),
            }

        score = self._score(scored_specs, measured)
        if score is None:
            return {
                "score": 0.0,
                "passed": False,
                "error": "no usable reward function",
                "measured": measured,
                "unmeasurable": unmeasurable,
            }

        # The gradient comes from what was measured; SUCCESS additionally
        # requires that everything was measured. Otherwise a design could be
        # declared finished with half its specs never checked.
        coverage = len(scored_specs) / len(specs) if specs else 0.0
        self.state.success = (score >= self.early_stop_threshold
                              and not unmeasurable)
        return {
            "score": score,
            "passed": self.state.success,
            "specs_checked": len(specs),
            "specs_measured": len(scored_specs),
            "coverage": coverage,
            "measured": measured,
            "unmeasurable": unmeasurable,
            **({"claim_mismatch": claim_mismatch} if claim_mismatch else {}),
        }

    def _reward_netlist(self) -> str | None:
        """The deck spec.check must resolve its measurements against.

        Prefer the deck the DC analysis was actually run on: idd is by far the
        largest consumer of the netlist, and it is read out of that operating
        point. Fall back to the noise deck (the .noise card is what decides
        whether an input-referred density is volts or amperes), then to the
        episode's current deck.

        Whichever it is, it is checked the way ngspice_shared._netlist_for
        checks its own: every '<name>#branch' in the operating point must be an
        element card of that deck. That guard exists precisely so a stale deck
        can never be applied to an earlier result, and the reward path used to
        walk straight past it by handing state.netlist to spec_extract.
        """
        netlist = (self.state.analysis_netlists.get("dc")
                   or self.state.analysis_netlists.get("noise")
                   or self.state.netlist or None)
        if not netlist:
            return None
        dc = self.state.analyses.get("dc")
        op = getattr(dc, "op_points", None) or {}
        wanted = {k.split("#")[0].lower().split(".")[-1]
                  for k in op if "#branch" in str(k).lower()}
        if wanted and not wanted <= measure.parse_deck_sources(netlist).elements:
            logger.warning(
                "spec.check: the deck on hand does not contain the source(s) "
                "%s that this operating point reports a branch current for, so "
                "it did not produce this result and is NOT used to resolve the "
                "supply current", sorted(wanted),
            )
            return None
        return netlist

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
