import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel, Field

from asic_ai import serialization

T = TypeVar("T", bound="Trajectory")

class ToolCall(BaseModel):
    name: str = Field(..., description="Name of the tool, e.g., 'sim.ac', 'netlist.patch'")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    call_id: str = Field(..., description="Unique identifier for the tool call")

class TrajectoryStep(BaseModel):
    step_index: int
    role: str = Field(..., description="'system' | 'user' | 'assistant' | 'tool'")
    content: Optional[str] = Field(None, description="Text content for thinking, topology selection, diagnosis")
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

class Trajectory(BaseModel):
    id: str
    task_id: str
    steps: List[TrajectoryStep]
    success: bool
    final_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float

    def to_chat_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI chat format for SFT."""
        chat = []
        for step in self.steps:
            msg = {"role": step.role}
            if step.content is not None:
                msg["content"] = step.content
            if step.tool_call is not None:
                msg["tool_calls"] = [
                    {
                        "id": step.tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": step.tool_call.name,
                            "arguments": serialization.dumps(step.tool_call.arguments)
                        }
                    }
                ]
            if step.tool_result is not None:
                msg["content"] = serialization.dumps(step.tool_result)
                msg["tool_call_id"] = step.tool_call.call_id if step.tool_call else ""
            chat.append(msg)
        return chat

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls: Type[T], line: str) -> T:
        """Deserialize from JSONL."""
        return cls.model_validate_json(line)

    def validate(self) -> List[str]:
        """Check format consistency."""
        errors = []
        for step in self.steps:
            if step.role not in ('system', 'user', 'assistant', 'tool'):
                errors.append(f"Step {step.step_index}: Invalid role '{step.role}'")
            if step.role == 'tool' and step.tool_result is None:
                errors.append(f"Step {step.step_index}: Tool role must have tool_result")
            if step.tool_call is not None and step.role != 'assistant':
                errors.append(f"Step {step.step_index}: Only assistant can emit tool_calls")
        return errors

    @classmethod
    def filter_successful(cls, trajectories: List[T]) -> List[T]:
        """Only keep successful trajectories (rejection sampling)."""
        return [t for t in trajectories if t.success]

    def get_tool_calls(self) -> List[ToolCall]:
        """Extract all tool calls from the trajectory."""
        return [step.tool_call for step in self.steps if step.tool_call is not None]

class TrajectoryDataset:
    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        self.trajectories = trajectories or []

    def load(self, path: str) -> None:
        """Load from a directory of JSONL files."""
        p = Path(path)
        if p.is_dir():
            files = list(p.glob("*.jsonl"))
            for f in files:
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip():
                            self.trajectories.append(Trajectory.from_jsonl(line))

    def save(self, path: str) -> None:
        """Save to a JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for t in self.trajectories:
                f.write(t.to_jsonl() + "\n")

    def statistics(self) -> Dict[str, Any]:
        """Count, success rate, avg steps, avg score."""
        count = len(self.trajectories)
        if count == 0:
            return {"count": 0}
        
        successes = sum(1 for t in self.trajectories if t.success)
        total_steps = sum(len(t.steps) for t in self.trajectories)
        total_score = sum(t.final_score for t in self.trajectories)
        
        return {
            "count": count,
            "success_rate": successes / count,
            "avg_steps": total_steps / count,
            "avg_score": total_score / count
        }

    def filter_successful(self) -> 'TrajectoryDataset':
        """Filter to only keep successful trajectories (rejection sampling)."""
        return TrajectoryDataset(
            trajectories=[t for t in self.trajectories if t.success]
        )

    def to_sft_dataset(self) -> List[List[Dict[str, Any]]]:
        """Convert all to SFT format."""
        return [t.to_chat_format() for t in self.trajectories]

    def split(self, train_ratio: float = 0.9, seed: int = 42) -> Tuple['TrajectoryDataset', 'TrajectoryDataset']:
        """Split dataset into train and test sets."""
        random.seed(seed)
        shuffled = self.trajectories.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]
        
        return TrajectoryDataset(train_data), TrajectoryDataset(test_data)
