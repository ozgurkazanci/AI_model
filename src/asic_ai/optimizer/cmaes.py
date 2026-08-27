import warnings
from typing import Any, Callable, Dict, List

from asic_ai.optimizer.base import NumericalOptimizer, OptimizationObjective, OptResult

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class CmaesOptimizer(NumericalOptimizer):
    """CMA-ES optimizer."""
    
    def __init__(self):
        if not HAS_CMA:
            warnings.warn("cma package not found. Optimization will fail if executed.")
            
    def optimize(self, objective: OptimizationObjective, 
                 eval_fn: Callable[[Dict[str, float]], float],
                 max_iterations: int = 100) -> OptResult:
        if not HAS_CMA:
            raise ImportError("cma package required for CmaesOptimizer")
            
        best_params = {p.name: p.initial or p.min_val for p in objective.parameters}
        return OptResult(
            best_params=best_params,
            best_score=0.0,
            iterations=0,
            history=[],
            converged=False
        )
            
    def suggest_next(self, objective: OptimizationObjective,
                     history: List[Dict[str, Any]]) -> Dict[str, float]:
        if not history:
            return {p.name: p.initial or (p.min_val + p.max_val) / 2 for p in objective.parameters}
        return history[-1]
