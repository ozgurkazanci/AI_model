from typing import Any, Dict, List
from pydantic import BaseModel
from .trajectory import Trajectory, TrajectoryDataset

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]

class DatasetValidationReport(BaseModel):
    is_valid: bool
    total_trajectories: int
    invalid_trajectories: int
    errors: List[str]
    warnings: List[str]

def validate_tool_call_format(tool_call: Dict[str, Any]) -> bool:
    """Strict JSON schema validation against the frozen contract."""
    # Placeholder for actual JSON schema validation logic
    if not isinstance(tool_call, dict):
        return False
    if "name" not in tool_call or not isinstance(tool_call["name"], str):
        return False
    if "arguments" not in tool_call or not isinstance(tool_call["arguments"], dict):
        return False
    return True

def validate_trajectory(trajectory: Trajectory) -> ValidationResult:
    """Validates the schema of a trajectory."""
    errors = []
    
    # Check general schema consistency
    format_errors = trajectory.validate()
    errors.extend(format_errors)
    
    # Specific tool call validation
    assistant_seen = False
    for step in trajectory.steps:
        if step.role == 'assistant':
            assistant_seen = True
            
        if step.tool_call:
            tc_dict = step.tool_call.model_dump()
            if not validate_tool_call_format(tc_dict):
                errors.append(f"Step {step.step_index}: Invalid tool call format")
                
        # Role sequence check
        if step.role == 'tool':
            if not assistant_seen:
                errors.append(f"Step {step.step_index}: Tool role appeared before any assistant role")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

def validate_dataset(dataset: TrajectoryDataset) -> DatasetValidationReport:
    """Validates an entire dataset of trajectories."""
    errors = []
    warnings = []
    invalid_count = 0
    total = len(dataset.trajectories)
    
    for i, t in enumerate(dataset.trajectories):
        res = validate_trajectory(t)
        if not res.is_valid:
            invalid_count += 1
            for err in res.errors:
                errors.append(f"Trajectory {t.id}: {err}")
    
    if invalid_count > 0:
        warnings.append(f"{invalid_count} out of {total} trajectories are invalid.")
        
    return DatasetValidationReport(
        is_valid=invalid_count == 0,
        total_trajectories=total,
        invalid_trajectories=invalid_count,
        errors=errors,
        warnings=warnings
    )
