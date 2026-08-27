import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

class RetryStrategy(Enum):
    SAME_APPROACH = "SAME_APPROACH"
    MODIFY_PARAMS = "MODIFY_PARAMS"
    CHANGE_TOPOLOGY = "CHANGE_TOPOLOGY"
    ESCALATE = "ESCALATE"

class StrategyManager:
    """Manages agent retry strategies based on historical errors."""
    
    def __init__(self):
        self.error_history: List[str] = []
        
    def analyze_failure(self, current_results: Dict[str, Any], spec: Dict[str, Any], history: List[Any]) -> RetryStrategy:
        """Determines the next strategy based on past failures."""
        if len(history) >= 5:
            return RetryStrategy.CHANGE_TOPOLOGY
        if len(history) >= 3:
            return RetryStrategy.MODIFY_PARAMS
        return RetryStrategy.SAME_APPROACH

class TopologySelector:
    """Selects topology candidates based on task specification."""
    
    def suggest_topology(self, spec: Dict[str, Any]) -> List[str]:
        """Suggests a ranked list of topologies."""
        task_type = spec.get("type", "unknown")
        if task_type == "analog":
            return ["two_stage_ota", "folded_cascode", "telescopic"]
        elif task_type == "digital":
            return ["pipelined", "combinational", "fsm"]
        return ["default_topology"]
