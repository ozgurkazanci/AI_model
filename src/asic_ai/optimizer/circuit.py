"""Turn a parameterised netlist into something an optimizer can search.

This is the glue the design document's Layer 3 needs: the LLM chooses the
topology and writes a netlist with the device sizes left as placeholders, and
the optimizer searches those placeholders against real simulation.

    netlist template  ->  substitute sizes  ->  simulate  ->  measure specs
          ^                                                        |
          +--------------------  score  <-------------------------+

Every step of that loop was fake until recently: the adapter returned zeros, no
converter existed between simulation results and spec names, and the optimizer
never evaluated anything. The loop is only meaningful now that all three are
real, which is why this module did not exist before.

Sizes are written in SPICE engineering notation (4.2e-05 -> "42u") because that
is what the netlists in this project use and what a designer reading the output
expects.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from asic_ai.optimizer.base import OptimizationObjective, OptParam, OptResult

log = logging.getLogger(__name__)

__all__ = [
    "format_spice",
    "substitute",
    "template_placeholders",
    "build_eval_fn",
    "optimize_sizing",
]

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_PREFIXES = (
    (1e12, "t"), (1e9, "g"), (1e6, "x"), (1e3, "k"),
    (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"),
    (1e-12, "p"), (1e-15, "f"),
)


def format_spice(value: float) -> str:
    """4.2e-05 -> '42u'. Matches the convention used by data/perturbation.py."""
    if value == 0:
        return "0"
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"cannot write {value!r} into a netlist")
    a = abs(value)
    for scale, suffix in _PREFIXES:
        if a >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value * 1e18:g}a"


def template_placeholders(template: str) -> List[str]:
    """Names the template expects, in order of first appearance."""
    seen: List[str] = []
    for name in _PLACEHOLDER_RE.findall(template):
        if name not in seen:
            seen.append(name)
    return seen


def substitute(template: str, values: Mapping[str, float]) -> str:
    """Fill every {placeholder}. Raises if any is missing.

    Deliberately strict: a netlist with an unsubstituted "{W1}" is a parse error
    inside the simulator, reported as a convergence failure, which the optimizer
    would then read as "this design is bad" and search away from.
    """
    missing = [n for n in template_placeholders(template) if n not in values]
    if missing:
        raise KeyError(f"netlist template has unfilled placeholders: {missing}")

    def repl(m: "re.Match[str]") -> str:
        return format_spice(float(values[m.group(1)]))

    return _PLACEHOLDER_RE.sub(repl, template)


def build_eval_fn(template: str,
                  adapter: Any,
                  specs: Mapping[str, Any],
                  *,
                  analyses: Sequence[str] = ("dc", "ac"),
                  output_signal: Optional[str] = "out",
                  reward_fn: Any = None,
                  supply_sources: Optional[Iterable[str]] = None,
                  on_step: Optional[Callable[[dict], None]] = None,
                  ) -> Callable[[Dict[str, float]], float]:
    """Build a score-to-MAXIMIZE function over the template's placeholders.

    Args:
        template: netlist with {name} placeholders for the sizes to search.
        adapter: a SimulatorAdapter. Its methods accept netlist TEXT.
        specs: the eval task's spec block, spec_name -> {min/max/target/unit}.
        analyses: which analyses to run per candidate. Each is a simulation, so
            this is the main cost knob.
        reward_fn: a RewardFunction. Built from `specs` when omitted.
        on_step: optional callback receiving a record per evaluation.

    A candidate whose simulation fails, or whose specs cannot be measured,
    scores -1.0 rather than raising: non-convergence is a real property of a
    bad design and the search should move away from it, not stop.
    """
    from asic_ai.adapters import spec_extract
    from asic_ai.reward.reward import RewardFunction, SpecTarget

    if reward_fn is None:
        targets = [
            SpecTarget(name=name,
                       min_val=(d or {}).get("min"),
                       max_val=(d or {}).get("max"),
                       target_val=(d or {}).get("target"),
                       weight=(d or {}).get("weight", 1.0),
                       unit=(d or {}).get("unit", ""))
            for name, d in specs.items()
            if any(k in (d or {}) for k in ("min", "max", "target"))
        ]
        if not targets:
            raise ValueError("no spec in `specs` has a min, max or target")
        reward_fn = RewardFunction(specs=targets)

    def evaluate(values: Dict[str, float]) -> float:
        record: Dict[str, Any] = {"params": dict(values)}
        try:
            netlist = substitute(template, values)
        except (KeyError, ValueError) as exc:
            record["error"] = str(exc)
            if on_step:
                on_step(record)
            raise  # a template error is a bug, not a bad design

        results: Dict[str, Any] = {}
        for kind in analyses:
            method = getattr(adapter, kind, None)
            if method is None:
                continue
            try:
                results[kind] = method(netlist, None)
            except Exception as exc:
                # Non-convergence, an unparseable deck, a device out of range.
                record["error"] = f"{kind}: {exc}"
                if on_step:
                    on_step(record)
                return -1.0

        extraction = spec_extract.extract_specs(
            specs,
            dc=results.get("dc"), ac=results.get("ac"), tran=results.get("tran"),
            noise=results.get("noise"), stb=results.get("stb"),
            output_signal=output_signal,
            supply_sources=supply_sources,
            # The deck is the only thing that can tell a 0 V sense source from a
            # supply rail, or spot a current-source-biased block. Without it the
            # supply current is a guess, and it is being optimised against.
            netlist=netlist,
        )
        record["measured"] = dict(extraction.values)
        record["unmeasurable"] = dict(extraction.unmeasurable)

        if not extraction.values:
            record["score"] = -1.0
            if on_step:
                on_step(record)
            return -1.0

        # Score only what was measured, for the same reason rl_env does: an
        # unmeasurable spec reads as -1.0 and would drown the real gradient.
        measurable = {k: v for k, v in specs.items() if k in extraction.values}
        rf = reward_fn
        if len(measurable) != len(specs):
            rf = RewardFunction(specs=[s for s in reward_fn.specs.values()
                                       if s.name in extraction.values])
        score = float(rf.compute(results=extraction.values).total_reward)
        record["score"] = score
        if on_step:
            on_step(record)
        return score

    return evaluate


def optimize_sizing(template: str,
                    variables: Sequence[Mapping[str, Any]],
                    specs: Mapping[str, Any],
                    adapter: Any,
                    *,
                    max_iterations: int = 60,
                    analyses: Sequence[str] = ("dc", "ac"),
                    output_signal: Optional[str] = "out",
                    optimizer: Any = None,
                    seed: int = 0) -> OptResult:
    """Size a parameterised netlist against its specs. What opt.suggest needs.

    `variables` is the tool-contract shape: a list of {name, min, max} dicts.
    Widths and lengths are searched on a log scale by default, because they span
    decades and a linear search spends nearly all its samples at the top end.
    """
    from asic_ai.optimizer.scipy_opt import ScipyOptimizer

    placeholders = set(template_placeholders(template))
    params: List[OptParam] = []
    for v in variables:
        name = v.get("name")
        if not name:
            raise ValueError(f"variable without a name: {v}")
        if name not in placeholders:
            raise ValueError(
                f"variable {name!r} does not appear in the netlist template; "
                f"template expects {sorted(placeholders)}")
        lo, hi = v.get("min"), v.get("max")
        if lo is None or hi is None:
            raise ValueError(f"variable {name!r} needs both min and max")
        params.append(OptParam(
            name=name, min_val=float(lo), max_val=float(hi),
            initial=v.get("initial"),
            log_scale=bool(v.get("log_scale", float(hi) / max(float(lo), 1e-300) >= 100.0)),
            fixed=bool(v.get("fixed", False)),
        ))

    unfilled = placeholders - {p.name for p in params}
    if unfilled:
        raise ValueError(f"template placeholders with no variable: {sorted(unfilled)}")

    objective = OptimizationObjective(parameters=params, objectives=[])
    eval_fn = build_eval_fn(template, adapter, specs, analyses=analyses,
                            output_signal=output_signal)
    opt = optimizer or ScipyOptimizer(seed=seed)
    return opt.optimize(objective, eval_fn, max_iterations=max_iterations)
