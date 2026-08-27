import re
from typing import Any, Dict
from pydantic import BaseModel, Field
from .schema import SignalData

class MeasResult(BaseModel):
    """Structured measurement result."""
    value: float = Field(..., description="The measured scalar value.")
    unit: str = Field(..., description="The unit of the measurement (e.g., 'V', 's', 'dB').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context about the measurement.")


def eval(signal: SignalData, expr: str) -> MeasResult:
    """
    Evaluate a measurement expression on structured signal data.
    Supported expressions: 'max', 'min', 'rise_time', 'fall_time', 'settling_time', 
    'overshoot', 'cross', 'avg', 'rms', 'pp'
    
    Args:
        signal: Structured signal data containing x and y values.
        expr: Measurement expression string.
        
    Returns:
        MeasResult: The computed measurement.
    """
    expr = expr.strip().lower()
    
    y_vals = signal.y_values
    x_vals = signal.x_values
    
    if not y_vals or not x_vals or len(y_vals) != len(x_vals):
        raise ValueError("Invalid signal data provided for measurement.")

    val = 0.0
    unit = ""
    metadata = {}

    if expr == "max":
        val = max(y_vals)
        metadata = {"index": y_vals.index(val), "x_at_max": x_vals[y_vals.index(val)]}
    elif expr == "min":
        val = min(y_vals)
        metadata = {"index": y_vals.index(val), "x_at_min": x_vals[y_vals.index(val)]}
    elif expr == "pp":
        val = max(y_vals) - min(y_vals)
    elif expr == "avg":
        val = sum(y_vals) / len(y_vals)
    elif expr == "rms":
        val = (sum(y ** 2 for y in y_vals) / len(y_vals)) ** 0.5
    else:
        # Placeholder for more complex measurements like rise_time, settling_time.
        # In a real implementation, these would use numpy/scipy for robust calculations.
        raise NotImplementedError(f"Measurement expression '{expr}' is not fully implemented yet.")

    return MeasResult(value=val, unit=unit, metadata=metadata)
