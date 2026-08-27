import json
from typing import List, Dict, Any, Tuple

SYSTEM_PROMPT = """You are an expert ASIC circuit designer.
You use precision simulation tools to design, analyze, and optimize analog and mixed-signal circuits.
Always think step by step. Propose hypotheses, verify them with simulations, and adjust the design based on structured simulation results.
Constraints:
- Do not hallucinate specifications. Check them strictly.
- Always simulate before claiming success.
- Never memorize PDK parameters; always query the PDK tools to retrieve correct device parameters, corners, and limits.
"""

TOOL_DEFINITIONS = [
  {"type": "function", "function": {"name": "sim.dc", "description": "Run a DC simulation.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.ac", "description": "Run an AC simulation.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.tran", "description": "Run a Transient simulation.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.noise", "description": "Run a Noise simulation.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.stb", "description": "Run a Stability (STB) simulation.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.corners", "description": "Run simulations across process corners.", "parameters": {}}},
  {"type": "function", "function": {"name": "sim.mc", "description": "Run Monte Carlo simulations.", "parameters": {}}},
  {"type": "function", "function": {"name": "meas.eval", "description": "Evaluate measurement expressions on simulation results.", "parameters": {}}},
  {"type": "function", "function": {"name": "spec.check", "description": "Check results against target specifications.", "parameters": {}}},
  {"type": "function", "function": {"name": "pdk.device_query", "description": "Query specific device parameters from the PDK.", "parameters": {}}},
  {"type": "function", "function": {"name": "pdk.list_devices", "description": "List available devices in the PDK.", "parameters": {}}},
  {"type": "function", "function": {"name": "pdk.get_corners", "description": "Get available process corners.", "parameters": {}}},
  {"type": "function", "function": {"name": "netlist.patch", "description": "Apply a patch to modify the current netlist.", "parameters": {}}},
  {"type": "function", "function": {"name": "lint.check", "description": "Check the netlist for lint errors.", "parameters": {}}},
  {"type": "function", "function": {"name": "opt.suggest", "description": "Suggest optimization directions based on gradients or sensitivities.", "parameters": {}}}
]

def format_trajectory_for_sft(trajectory: Any) -> List[Dict[str, Any]]:
    """Convert a trajectory to the exact SFT training format.
    Format: chatml with tool_call tags
    """
    formatted_messages = []
    # Simplified version for now
    formatted_messages.append({
        "role": "system", 
        "content": SYSTEM_PROMPT + "\n\nAvailable Tools:\n" + json.dumps(TOOL_DEFINITIONS, indent=2)
    })
    
    # Process trajectory messages
    for msg in trajectory.messages:
        # Assuming msg is a dict with role and content
        if msg.get("role") != "system":
            formatted_messages.append(msg)
            
    return formatted_messages

def validate_sft_format(messages: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate that a formatted trajectory exactly matches the expected format.
    Returns (is_valid, list of errors)."""
    errors = []
    
    if not messages:
        return False, ["Messages list is empty."]
        
    if messages[0].get("role") != "system":
        errors.append("First message must be a system prompt.")
        
    for idx, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"Message {idx} missing 'role'.")
        if msg.get("role") not in ["system", "user", "assistant", "tool"]:
            errors.append(f"Message {idx} has invalid role: {msg.get('role')}.")
            
    return len(errors) == 0, errors
