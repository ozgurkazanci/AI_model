from typing import Any, Dict, List
from asic_ai.agent.loop import Trajectory, EvalTask

class DesignMemory:
    """Manages storage and retrieval of design trajectories."""
    
    def __init__(self):
        self.past_trajectories: List[Trajectory] = []
        
    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Save completed designs."""
        self.past_trajectories.append(trajectory)
        
    def query_similar(self, spec: Dict[str, Any], k: int = 5) -> List[Trajectory]:
        """Find similar past designs."""
        # Simple placeholder for similarity search
        return self.past_trajectories[-k:]
        
    def get_context_for_task(self, task: EvalTask) -> str:
        """Build context string including PDK info, similar designs, and topology hints."""
        return f"Context for task {task.task_type} with spec {task.spec}"
        
    def prune(self, max_tokens: int) -> None:
        """Keep context within budget."""
        pass

class ContextBuilder:
    """Assembles the prompt for the LLM."""
    
    def __init__(self, memory: DesignMemory):
        self.memory = memory
        
    def build_prompt(self, system_prompt: str, pdk_context: str, task: EvalTask, current_state: Dict[str, Any], history: List[Any]) -> str:
        """Assembles the full context, ensuring token limits are respected."""
        # Priority: spec > current state > history > reference designs
        context = self.memory.get_context_for_task(task)
        prompt = f"{system_prompt}\n\nPDK: {pdk_context}\n\nTask: {task.spec}\n\nState: {current_state}\n\nHistory: {history}\n\nContext: {context}"
        return prompt
