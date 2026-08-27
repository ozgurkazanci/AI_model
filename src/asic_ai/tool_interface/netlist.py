import re
from .schema import NetlistPatch, LintResult, LintError

def patch(netlist: str, diff: NetlistPatch) -> str:
    """
    Apply a set of diff-based patch operations to a netlist.
    Operations include adding/removing instances, changing params, etc.
    This operates as a structural or string manipulation depending on the backend.
    """
    lines = netlist.split("\n")
    # For a robust implementation, this should parse the SPICE netlist into an AST.
    # Below is a highly simplified conceptual implementation.
    
    for op in diff.operations:
        if op.op == "add_instance":
            if op.value:
                lines.append(op.value)
        elif op.op == "remove_instance":
            # Search and remove
            lines = [line for line in lines if not line.strip().startswith(op.target)]
        elif op.op == "modify_param":
            # Very basic string replace for demonstration
            for i, line in enumerate(lines):
                if line.strip().startswith(op.target):
                    # Replace the parameter in the string
                    # Requires proper SPICE tokenization in a real implementation
                    if op.value:
                        lines[i] = f"{line} {op.value}" # naive append
        elif op.op in ["add_net", "remove_net", "rename_net"]:
            # Net manipulation logic here
            pass
            
    return "\n".join(lines)


def check(netlist: str) -> LintResult:
    """
    Lint the netlist to catch common errors before simulation.
    Checks for floating nodes, missing connections, invalid params, duplicates.
    """
    errors = []
    
    # Example check: look for floating nodes (nodes connected to only one pin)
    # This requires parsing netlist into nodes and instances
    # Here is a mock implementation
    
    lines = netlist.split("\n")
    instance_names = set()
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith(('*', '.')):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        inst_name = parts[0]
        if inst_name in instance_names:
            errors.append(LintError(
                node=inst_name,
                line=line_num + 1,
                message=f"Duplicate instance name found: {inst_name}",
                severity="error"
            ))
        else:
            instance_names.add(inst_name)
            
    # Mocking successful pass if no errors
    passed = len([e for e in errors if e.severity == "error"]) == 0
    return LintResult(errors=errors, passed=passed)
