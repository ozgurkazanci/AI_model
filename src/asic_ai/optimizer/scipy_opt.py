"""A numerical optimizer that actually searches.

The design document calls this layer the most-skipped and most-consequential
part of an analog design system:

    "The model picks the topology, the optimizer sizes the devices. Projects
     that skip this hit a wall on analog. Asking an LLM to produce transistor
     widths is the wrong architecture."

What was here before never called eval_fn at all. It returned each parameter's
lower bound, a score of 0.0, and converged=True -- a search that reports success
without searching. Measured: 0 evaluations for a 25-iteration request.

This implementation uses scipy, which is already a dependency. BoTorch/Ax and
cma are not installed; the wrappers for those now refuse rather than pretending
(see bayesian.py, cmaes.py).

CONVENTION: eval_fn returns a score to MAXIMIZE. RewardFunction.compute returns
higher-is-better, and this layer exists to be driven by it, so the whole stack
points the same way. scipy minimizes internally; the negation happens here and
nowhere else.

EVALUATIONS ARE SPICE RUNS. `max_iterations` is a hard cap on eval_fn calls, not
a suggestion: at roughly 0.1-3 s per simulation an uncapped global search is an
overnight job. The budget is enforced by a counter that stops the search and
returns the best point found so far.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from asic_ai.optimizer.base import (
    NumericalOptimizer, OptimizationObjective, OptParam, OptResult,
)

log = logging.getLogger(__name__)

__all__ = ["ScipyOptimizer", "BudgetExhausted"]


class BudgetExhausted(Exception):
    """Raised internally when the evaluation budget runs out."""


class _Budget:
    """Counts eval_fn calls, remembers the best point, and stops at the cap."""

    def __init__(self, eval_fn: Callable[[Dict[str, float]], float], cap: int):
        self._eval = eval_fn
        self.cap = cap
        self.n = 0
        self.best_score = -math.inf
        self.best_params: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def __call__(self, params: Dict[str, float]) -> float:
        if self.n >= self.cap:
            raise BudgetExhausted()
        self.n += 1
        try:
            score = float(self._eval(params))
        except Exception as exc:
            # A simulation that fails to converge is information, not a crash.
            # Score it worse than any real design so the search moves away.
            log.debug("eval failed at %s: %s", params, exc)
            score = -math.inf
            self.history.append({"iteration": self.n, "params": dict(params),
                                 "score": None, "error": str(exc)})
            return -1e6
        if math.isnan(score):
            score = -math.inf
        self.history.append({"iteration": self.n, "params": dict(params),
                             "score": score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = dict(params)
        return score


def _free_params(objective: OptimizationObjective) -> List[OptParam]:
    return [p for p in objective.parameters if not p.fixed]


def _fixed_values(objective: OptimizationObjective) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in objective.parameters:
        if p.fixed:
            out[p.name] = p.initial if p.initial is not None else p.min_val
    return out


def _encode(p: OptParam, value: float) -> float:
    """Parameter value -> search coordinate."""
    if p.log_scale:
        return math.log10(max(value, 1e-300))
    return value


def _decode(p: OptParam, x: float) -> float:
    """Search coordinate -> parameter value.

    Device widths and lengths span decades, so a linear search wastes almost all
    its samples at the large end. log_scale=True makes the search uniform in
    orders of magnitude, which is how these are actually chosen.
    """
    if p.log_scale:
        return float(10.0 ** x)
    return float(x)


def _bounds(params: List[OptParam]) -> List[Tuple[float, float]]:
    return [(_encode(p, p.min_val), _encode(p, p.max_val)) for p in params]


class ScipyOptimizer(NumericalOptimizer):
    """Global search (differential evolution) with a local polish (Nelder-Mead).

    Differential evolution because the objective is non-convex, discontinuous
    where a simulation fails to converge, and has no usable gradient. Nelder-Mead
    afterwards because DE gets close but wastes budget on the last decimal.
    """

    def __init__(self, seed: int = 0, polish: bool = True,
                 popsize: int = 8, tol: float = 1e-6):
        self.seed = seed
        self.polish = polish
        self.popsize = popsize
        self.tol = tol
        # Successive suggest_next() calls must differ, or a caller asking for a
        # batch of candidates gets the same point N times. Seeding only from
        # len(history) made every repeated call identical.
        self._suggest_calls = 0

    # -- helpers ----------------------------------------------------------

    def _wrap(self, budget: _Budget, params: List[OptParam],
              fixed: Dict[str, float]) -> Callable[[Any], float]:
        """scipy minimizes; the project's convention is maximize."""
        def f(x) -> float:
            values = dict(fixed)
            for p, xi in zip(params, list(x)):
                values[p.name] = _decode(p, float(xi))
            return -budget(values)
        return f

    def _initial(self, params: List[OptParam]) -> List[float]:
        out = []
        for p in params:
            if p.initial is not None:
                out.append(_encode(p, p.initial))
            else:
                lo, hi = _encode(p, p.min_val), _encode(p, p.max_val)
                out.append(0.5 * (lo + hi))
        return out

    # -- API --------------------------------------------------------------

    def optimize(self, objective: OptimizationObjective,
                 eval_fn: Callable[[Dict[str, float]], float],
                 max_iterations: int = 100) -> OptResult:
        free = _free_params(objective)
        fixed = _fixed_values(objective)

        if not free:
            # Nothing to search. Report the one point honestly rather than
            # claiming a converged optimisation.
            budget = _Budget(eval_fn, max(1, max_iterations))
            try:
                budget(dict(fixed))
            except BudgetExhausted:
                pass
            return OptResult(best_params=budget.best_params or dict(fixed),
                             best_score=budget.best_score if budget.n else 0.0,
                             iterations=budget.n, history=budget.history,
                             converged=False)

        budget = _Budget(eval_fn, max_iterations)
        f = self._wrap(budget, free, fixed)
        bounds = _bounds(free)
        converged = False

        try:
            from scipy.optimize import differential_evolution, minimize
        except ImportError as exc:  # pragma: no cover - scipy is a dependency
            raise ImportError("ScipyOptimizer requires scipy") from exc

        # Reserve a slice of the budget for the local polish.
        de_budget = max(1, int(max_iterations * (0.75 if self.polish else 1.0)))
        budget.cap = de_budget

        try:
            maxiter = max(1, de_budget // max(1, self.popsize * len(free)))
            res = differential_evolution(
                f, bounds, seed=self.seed, maxiter=maxiter,
                popsize=self.popsize, tol=self.tol, polish=False,
                init="sobol" if len(free) > 1 else "latinhypercube",
            )
            converged = bool(getattr(res, "success", False))
        except BudgetExhausted:
            log.debug("global search stopped at the evaluation budget")
        except Exception as exc:
            log.warning("differential_evolution failed: %s", exc)

        if self.polish and budget.best_params:
            budget.cap = max_iterations
            start = [_encode(p, budget.best_params[p.name]) for p in free]
            try:
                res = minimize(f, start, method="Nelder-Mead",
                               bounds=bounds,
                               options={"maxfev": max_iterations - budget.n,
                                        "xatol": 1e-8, "fatol": 1e-10})
                converged = converged or bool(getattr(res, "success", False))
            except BudgetExhausted:
                pass
            except Exception as exc:
                log.debug("polish failed: %s", exc)

        if budget.n == 0:
            raise RuntimeError("optimizer made no evaluations; check max_iterations")

        return OptResult(
            best_params=budget.best_params,
            best_score=budget.best_score,
            iterations=budget.n,
            history=budget.history,
            converged=converged,
        )

    def suggest_next(self, objective: OptimizationObjective,
                     history: List[Dict[str, Any]]) -> Dict[str, float]:
        """One point to evaluate next, for a caller running its own loop.

        Space-filling while there is little history, then a shrinking local
        perturbation around the best point seen. Deliberately simple: a caller
        that wants a real search should use optimize().
        """
        free = _free_params(objective)
        values = _fixed_values(objective)
        self._suggest_calls += 1
        rng = random.Random((self.seed, len(history), self._suggest_calls).__hash__())

        scored = [h for h in history
                  if isinstance(h, dict) and h.get("score") is not None
                  and isinstance(h.get("params"), dict)]

        if not scored:
            for p in free:
                if p.initial is not None:
                    values[p.name] = p.initial
                else:
                    lo, hi = _encode(p, p.min_val), _encode(p, p.max_val)
                    values[p.name] = _decode(p, rng.uniform(lo, hi))
            return values

        best = max(scored, key=lambda h: h["score"])
        # Shrink the neighbourhood as evidence accumulates. The multiplier is
        # small on purpose: this is a LOCAL perturbation around the best point,
        # and a sigma comparable to the whole range would just be another
        # uniform sample wearing a normal distribution's clothes.
        span = 1.0 / (1.0 + 0.5 * len(scored))
        for p in free:
            lo, hi = _encode(p, p.min_val), _encode(p, p.max_val)
            centre = _encode(p, float(best["params"].get(p.name, p.min_val)))
            sigma = (hi - lo) * span * 0.15
            x = min(hi, max(lo, rng.gauss(centre, sigma)))
            values[p.name] = _decode(p, x)
        return values
