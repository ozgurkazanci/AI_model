"""Numerical optimizers for analog design."""

from .base import NumericalOptimizer, OptimizationObjective, OptParam, OptObjective, OptConstraint, OptResult
from .bayesian import BayesianOptimizer
from .cmaes import CmaesOptimizer

def get_optimizer(method: str = 'bayesian') -> NumericalOptimizer:
    """Factory function for instantiating optimizers."""
    if method == 'bayesian':
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
    "CmaesOptimizer",
    "get_optimizer"
]
