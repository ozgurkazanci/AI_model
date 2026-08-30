"""Bayesian optimization. Requires BoTorch/Ax, which are not installed here.

This class previously "fell back to scipy" in a warning message and then
returned each parameter's lower bound with a score of 0.0 and converged=True,
without calling eval_fn once. A search that reports success without searching
is worse than one that fails, because the caller acts on the result.

It now delegates to ScipyOptimizer when BoTorch is absent -- a real search --
and says so, rather than claiming to be Bayesian.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from asic_ai.optimizer.base import NumericalOptimizer, OptimizationObjective, OptResult
from asic_ai.optimizer.scipy_opt import ScipyOptimizer

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only where BoTorch is installed
    import ax  # noqa: F401
    import botorch  # noqa: F401
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


class BayesianOptimizer(NumericalOptimizer):
    """Bayesian optimization when BoTorch is available, a real search otherwise."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._fallback = ScipyOptimizer(seed=seed)
        if not HAS_BOTORCH:
            log.info("BoTorch/Ax not installed; using differential evolution "
                     "plus Nelder-Mead instead. This is a real search, but it "
                     "is not Bayesian optimization.")

    @property
    def is_bayesian(self) -> bool:
        """False when running on the fallback, so callers need not guess."""
        return HAS_BOTORCH

    def optimize(self, objective: OptimizationObjective,
                 eval_fn: Callable[[Dict[str, float]], float],
                 max_iterations: int = 100) -> OptResult:
        if HAS_BOTORCH:  # pragma: no cover - no BoTorch on this machine
            return self._optimize_ax(objective, eval_fn, max_iterations)
        return self._fallback.optimize(objective, eval_fn, max_iterations)

    def suggest_next(self, objective: OptimizationObjective,
                     history: List[Dict[str, Any]]) -> Dict[str, float]:
        return self._fallback.suggest_next(objective, history)

    def _optimize_ax(self, objective: OptimizationObjective,
                     eval_fn: Callable[[Dict[str, float]], float],
                     max_iterations: int) -> OptResult:  # pragma: no cover
        raise NotImplementedError(
            "BoTorch is installed but the Ax loop is not implemented here. "
            "Use ScipyOptimizer, or implement this rather than returning a "
            "result that was never evaluated.")
