import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, confloat, conint


class ActionType(str, Enum):
    """Enumeration of possible agent actions."""
    SIMULATE = "SIMULATE"
    MEASURE = "MEASURE"
    QUERY_PDK = "QUERY_PDK"
    PATCH_NETLIST = "PATCH_NETLIST"
    LINT = "LINT"
    SET_SPEC = "SET_SPEC"
    OPTIMIZE = "OPTIMIZE"


class SimParams(BaseModel):
    """Parameters for a simulation run."""
    analysis_type: str = Field(..., description="Type of analysis, e.g., 'dc', 'ac', 'tran'.")
    start: Optional[float] = Field(None, description="Start value for sweep or time.")
    stop: Optional[float] = Field(None, description="Stop value for sweep or time.")
    step: Optional[float] = Field(None, description="Step size.")
    points: Optional[int] = Field(None, description="Number of points for AC sweep.")
    sweep_var: Optional[str] = Field(None, description="Variable to sweep (e.g., 'v1').")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional simulator options.")


class PVTCorner(BaseModel):
    """Process, Voltage, and Temperature corner definition."""
    process: str = Field(..., description="Process corner, e.g., 'tt', 'ss', 'ff'.")
    voltage: float = Field(..., description="Supply voltage scaling or absolute value.")
    temperature: float = Field(..., description="Temperature in Celsius.")


# ==========================================
# Simulation Results
# ==========================================

class SignalData(BaseModel):
    """Structured data for a signal (voltage, current, etc.)."""
    name: str = Field(..., description="Name of the signal, e.g., 'v(out)'.")
    x_values: List[float] = Field(..., description="Independent variable values (time, freq, etc.).")
    y_values: List[float] = Field(..., description="Dependent variable values.")


class DCResult(BaseModel):
    """DC operating point or DC sweep results."""
    op_points: Dict[str, float] = Field(default_factory=dict, description="Operating points (node -> voltage/current).")
    sweeps: Dict[str, SignalData] = Field(default_factory=dict, description="DC sweep data if applicable.")


class ACResult(BaseModel):
    """AC analysis results."""
    frequencies: List[float] = Field(..., description="Frequency points.")
    signals: Dict[str, SignalData] = Field(default_factory=dict, description="Magnitude/phase or complex data.")


class TranResult(BaseModel):
    """Transient analysis results."""
    time: List[float] = Field(..., description="Time points.")
    signals: Dict[str, SignalData] = Field(default_factory=dict, description="Transient signals.")


class NoiseResult(BaseModel):
    """Noise analysis results."""
    frequencies: List[float] = Field(..., description="Frequency points.")
    input_noise: SignalData = Field(..., description="Input-referred noise.")
    output_noise: SignalData = Field(..., description="Output noise.")


class StabilityResult(BaseModel):
    """Stability analysis results."""
    phase_margin: float = Field(..., description="Phase margin in degrees.")
    gain_margin: float = Field(..., description="Gain margin in dB.")
    loop_gain: SignalData = Field(..., description="Loop gain over frequency.")


class CornerResult(BaseModel):
    """Results from a single PVT corner simulation."""
    corner: PVTCorner
    dc: Optional[DCResult] = None
    ac: Optional[ACResult] = None
    tran: Optional[TranResult] = None
    stb: Optional[StabilityResult] = None


class MonteCarloResult(BaseModel):
    """Monte Carlo analysis results."""
    seed: int = Field(..., description="Random seed used.")
    runs: int = Field(..., description="Number of runs.")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted metrics per run.")


# ==========================================
# Netlist & Linting
# ==========================================

class NetlistPatchOperation(BaseModel):
    """A single netlist patching operation."""
    op: str = Field(..., description="Operation type: 'add_instance', 'remove_instance', 'modify_param', 'add_net', 'remove_net', 'rename_net'.")
    target: str = Field(..., description="Target identifier (instance name, net name, etc.).")
    value: Optional[str] = Field(None, description="Value or definition to apply.")


class NetlistPatch(BaseModel):
    """A set of diff-based patch operations for a netlist."""
    operations: List[NetlistPatchOperation] = Field(default_factory=list, description="List of operations to apply.")


class LintError(BaseModel):
    """A single linting error."""
    node: Optional[str] = Field(None, description="Node or element name involved.")
    line: Optional[int] = Field(None, description="Line number if available.")
    message: str = Field(..., description="Error or warning message.")
    severity: str = Field("error", description="'error' or 'warning'.")


class LintResult(BaseModel):
    """Overall linting results."""
    errors: List[LintError] = Field(default_factory=list)
    passed: bool = Field(..., description="True if no severe errors were found.")


# ==========================================
# Specifications & PDK
# ==========================================

class SpecDefinition(BaseModel):
    """Definition of a single specification."""
    target: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    weight: float = Field(1.0, description="Importance weight of this spec.")


class SpecCheckDetail(BaseModel):
    """Details of a single specification check."""
    target_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    actual: float
    met: bool
    score: float


class SpecCheckResult(BaseModel):
    """Result of checking all specifications."""
    score: float = Field(..., description="Overall weighted score [-1, 1].")
    breakdown: Dict[str, SpecCheckDetail] = Field(..., description="Detailed breakdown per spec.")


class DeviceQueryResult(BaseModel):
    """Structured data returned from a PDK device query."""
    model: str
    W: float
    L: float
    VGS: float
    VDS: float
    VSB: float
    gm: float
    gds: float
    id: float
    ft: float
    cgs: float
    cgd: float
    cdb: float
    vth: float
    region: str = Field(..., description="Operating region: 'cutoff', 'linear', 'saturation'.")


# ==========================================
# Agent & Environment
# ==========================================

class AgentAction(BaseModel):
    """An action taken by the RL agent."""
    action_type: ActionType
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the specific tool call.")


class AgentObservation(BaseModel):
    """The observation state returned to the agent."""
    netlist_state: str = Field(..., description="Current state of the netlist.")
    last_results: Dict[str, Any] = Field(default_factory=dict, description="Summary of the last simulation or action results.")
    spec_status: SpecCheckResult
    step_count: int = Field(..., description="Current step count in the episode.")


def get_json_schema() -> str:
    """Export the JSON schema for all relevant models."""
    models = {
        "ActionType": ActionType,
        "SimParams": SimParams,
        "PVTCorner": PVTCorner,
        "DCResult": DCResult,
        "ACResult": ACResult,
        "TranResult": TranResult,
        "NoiseResult": NoiseResult,
        "StabilityResult": StabilityResult,
        "CornerResult": CornerResult,
        "MonteCarloResult": MonteCarloResult,
        "NetlistPatch": NetlistPatch,
        "LintResult": LintResult,
        "SpecCheckResult": SpecCheckResult,
        "DeviceQueryResult": DeviceQueryResult,
        "AgentAction": AgentAction,
        "AgentObservation": AgentObservation,
    }
    schemas = {name: model.model_json_schema() for name, model in models.items() if hasattr(model, 'model_json_schema')}
    return json.dumps(schemas, indent=2)
