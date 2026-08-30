"""Numerical optimizers for analog design."""

from .base import NumericalOptimizer, OptimizationObjective, OptParam, OptObjective, OptConstraint, OptResult
from .bayesian import BayesianOptimizer
from .cmaes import CmaesOptimizer
from .scipy_opt import ScipyOptimizer
from .circuit import (
    build_eval_fn, optimize_sizing, substitute, template_placeholders,
    format_spice,
)

def get_optimizer(method: str = 'scipy') -> NumericalOptimizer:
    """Factory function for instantiating optimizers.

    Defaults to 'scipy' because it is the only one whose dependencies are
    present, and therefore the only one that actually searches.
    """
    if method == 'scipy':
        return ScipyOptimizer()
    elif method == 'bayesian':
        return BayesianOptimizer()
    elif method == 'cmaes':
        return CmaesOptimizer()
    else:
        raise ValueError(f"Unknown optimizer method: {method}")

__all__ = [
    "NumericalOptimizer",
    "OptimizationObjective",
    "OptParam", 
    "OptObjective",
    "OptConstraint",
    "OptResult",
    "BayesianOptimizer",
    "ScipyOptimizer",
    "build_eval_fn",
    "optimize_sizing",
    "substitute",
    "template_placeholders",
    "format_spice",
    "CmaesOptimizer",
    "get_optimizer"
]
