import asyncio
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    max_steps: int = 20
    max_retries_same_error: int = 3
    checkpoint_interval: int = 5
    use_optimizer_for_analog: bool = True
    escalation_threshold: int = 10
    temperature: float = 0.7

class EvalTask(BaseModel):
    spec: Dict[str, Any]
    task_type: str = "analog"

class Trajectory(BaseModel):
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "running"
    final_result: Optional[Dict[str, Any]] = None

class AgentLoop:
    """Main agent loop: plan -> act -> observe -> decide.
    
    The model is NOT a one-shot netlist generator. It's an agent that:
    spec -> topology -> netlist -> simulate -> read result -> diagnose -> fix -> re-simulate -> until spec met
    """
    
    def __init__(self, model: Any, simulator: Any, optimizer: Any, config: AgentConfig):
        self.model = model
        self.simulator = simulator
        self.optimizer = optimizer
        self.config = config
    
    async def run(self, task: EvalTask) -> Trajectory:
        """Run the full agent loop for a design task."""
        trajectory = Trajectory()
        steps_taken = 0
        
        while steps_taken < self.config.max_steps:
            # 1. Parse spec
            # 2. Select topology (LLM decision)
            # 3. Generate initial netlist (LLM)
            # 4. Lint check (catch errors before simulation)
            # 5. Simulate
            # 6. Check spec
            # 7. If not met: diagnose (LLM), fix (LLM + optimizer), goto 5
            # 8. If met: run corners + MC
            # 9. If corners fail: adjust, goto 5
            # 10. Return trajectory
            
            if self._check_stuck(trajectory):
                self._escalate("Agent is repeating the same mistakes.")
                trajectory.status = "escalated"
                break
                
            steps_taken += 1
            
            if steps_taken % self.config.checkpoint_interval == 0:
                self._checkpoint(trajectory)
                
        if trajectory.status == "running":
            trajectory.status = "max_steps_reached"
            
        return trajectory
    
    def _should_use_optimizer(self, action_type: str) -> bool:
        """Analog sizing -> optimizer. Digital/topology -> LLM only."""
        return self.config.use_optimizer_for_analog and action_type.lower() == "analog_sizing"
    
    def _check_stuck(self, trajectory: Trajectory) -> bool:
        """Detect if agent is repeating same mistakes."""
        return len(trajectory.steps) >= self.config.escalation_threshold
    
    def _escalate(self, reason: str) -> None:
        """Ask user when agent is stuck or uncertain."""
        print(f"ESCALATION: {reason}")
        
    def _checkpoint(self, trajectory: Trajectory) -> None:
        """Save a checkpoint of the current trajectory."""
        pass
