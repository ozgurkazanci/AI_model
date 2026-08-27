from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel

class OptParam(BaseModel):
    name: str
    min_val: float
    max_val: float
    initial: Optional[float] = None
    log_scale: bool = False
    fixed: bool = False

class OptObjective(BaseModel):
    name: str
    target: float
    direction: str = "min" # or "max"
    weight: float = 1.0

class OptConstraint(BaseModel):
    name: str
    expression: str
    limit: float

class OptimizationObjective(BaseModel):
    parameters: List[OptParam]
    objectives: List[OptObjective]
    constraints: List[OptConstraint] = []

class OptResult(BaseModel):
    best_params: Dict[str, float]
    best_score: float
    iterations: int
    history: List[Dict[str, Any]]
    converged: bool

class NumericalOptimizer(ABC):
    @abstractmethod
    def optimize(self, objective: OptimizationObjective, 
                 eval_fn: Callable[[Dict[str, float]], float],
                 max_iterations: int = 100) -> OptResult:
        """Run full optimization loop."""
        pass
    
    @abstractmethod  
    def suggest_next(self, objective: OptimizationObjective,
                     history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Suggest next point to evaluate (for async/batched optimization)."""
        pass
