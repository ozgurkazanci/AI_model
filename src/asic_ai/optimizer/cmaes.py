"""CMA-ES. Requires the `cma` package, which is not installed here.

This class previously returned each parameter's lower bound with a score of 0.0,
iterations=0 and converged=False, without calling eval_fn. The zero score in
particular is indistinguishable from a real evaluation of a mediocre design.

It now raises when `cma` is missing rather than returning a fabricated result.
Callers that just want a working optimizer should use get_optimizer("scipy").
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from asic_ai.optimizer.base import NumericalOptimizer, OptimizationObjective, OptResult
from asic_ai.optimizer.scipy_opt import ScipyOptimizer

log = logging.getLogger(__name__)

try:
    import cma  # noqa: F401
    HAS_CMA = True
except ImportError:
    HAS_CMA = False


class CmaesOptimizer(NumericalOptimizer):
    """CMA-ES. Raises unless the `cma` package is installed."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def optimize(self, objective: OptimizationObjective,
                 eval_fn: Callable[[Dict[str, float]], float],
                 max_iterations: int = 100) -> OptResult:
        if not HAS_CMA:
            raise ImportError(
                "CmaesOptimizer needs the 'cma' package (pip install cma). "
                "Use get_optimizer('scipy') for a search that works with the "
                "dependencies already present.")
        raise NotImplementedError(  # pragma: no cover - no cma on this machine
            "cma is installed but the CMA-ES loop is not implemented here.")

    def suggest_next(self, objective: OptimizationObjective,
                     history: List[Dict[str, Any]]) -> Dict[str, float]:
        # Space-filling suggestions do not need CMA-ES and are useful on their own.
        return ScipyOptimizer(seed=self.seed).suggest_next(objective, history)
