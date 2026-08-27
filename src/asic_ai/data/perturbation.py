"""
Synthetic perturbation pipeline for ASIC AI project.
"""

import re
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

# SPICE unit prefixes mapping to multipliers
SPICE_PREFIXES = {
    't': 1e12,
    'g': 1e9,
    'meg': 1e6,
    'x': 1e6,
    'k': 1e3,
    'm': 1e-3,
    'u': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
    'f': 1e-15,
    'a': 1e-18,
}

def parse_spice_value(s: str) -> float:
    """Parse a SPICE string value like '10u' into a float 10e-6."""
    s = s.strip().lower()
    if not s:
        return 0.0
    
    # Extract numeric part and unit part
    match = re.match(r'^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z]*)$', s)
    if not match:
        try:
            return float(s)
        except ValueError:
            return 0.0
            
    val_str, unit_str = match.groups()
    val = float(val_str)
    
    if unit_str in SPICE_PREFIXES:
        val *= SPICE_PREFIXES[unit_str]
    elif unit_str == 'meg':
        val *= SPICE_PREFIXES['meg']
    
    return val

def format_spice_value(f: float) -> str:
    """Format a float 10e-6 into a SPICE string '10u' using best SI prefix."""
    if f == 0:
        return "0"
        
    abs_f = abs(f)
    if abs_f >= 1e12: return f"{f/1e12:g}t"
    elif abs_f >= 1e9: return f"{f/1e9:g}g"
    elif abs_f >= 1e6: return f"{f/1e6:g}x"
    elif abs_f >= 1e3: return f"{f/1e3:g}k"
    elif abs_f >= 1: return f"{f:g}"
    elif abs_f >= 1e-3: return f"{f*1e3:g}m"
    elif abs_f >= 1e-6: return f"{f*1e6:g}u"
    elif abs_f >= 1e-9: return f"{f*1e9:g}n"
    elif abs_f >= 1e-12: return f"{f*1e12:g}p"
    elif abs_f >= 1e-15: return f"{f*1e15:g}f"
    else: return f"{f*1e18:g}a"

def parse_instance_line(line: str) -> dict:
    """Parse 'XM1 net1 INM net3 VSS nfet_01v8 W=10u L=180n m=4' into structured data."""
    parts = line.split()
    if not parts:
        return {}
    
    name = parts[0]
    
    # Try to find parameters W= L= m= etc.
    params_start_idx = len(parts)
    for i, part in enumerate(parts):
        if '=' in part:
            params_start_idx = i
            break
            
    nodes_and_model = parts[1:params_start_idx]
    
    model = nodes_and_model[-1] if nodes_and_model else ""
    nodes = nodes_and_model[:-1] if len(nodes_and_model) > 1 else []
    
    params = {}
    for part in parts[params_start_idx:]:
        if '=' in part:
            k, v = part.split('=', 1)
            params[k] = v
            
    return {
        'name': name,
        'nodes': nodes,
        'model': model,
        'params': params,
        'original_line': line
    }

def modify_instance_param(line: str, param: str, new_value: str) -> str:
    """Change a parameter in an instance line."""
    if f"{param}=" not in line and f" {param} =" not in line:
        return line + f" {param}={new_value}"
        
    # Replace the existing parameter
    pattern = rf'\b{param}\s*=\s*\S+'
    replacement = f"{param}={new_value}"
    return re.sub(pattern, replacement, line)

def find_instances(netlist: str, pattern: str) -> List[Tuple[int, str]]:
    """Find matching instance lines with line numbers (0-indexed)."""
    results = []
    lines = netlist.split('\n')
    regex = re.compile(pattern)
    for i, line in enumerate(lines):
        # Ignore comments
        if line.strip().startswith('*'):
            continue
        if regex.search(line):
            results.append((i, line))
    return results

class PerturbedCircuit(BaseModel):
    original_netlist: str
    perturbed_netlist: str
    perturbations_applied: List[str]

class Perturbation(ABC):
    @abstractmethod
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        """Apply the perturbation, returning (new_netlist, description)"""
        pass

class BiasShift(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        # Find current sources Ixxx
        instances = find_instances(netlist, r'^I')
        if not instances:
            return netlist, "BiasShift: No current sources found to shift."
            
        i, line = random.choice(instances)
        parts = line.split()
        if len(parts) >= 4:
            val_str = parts[3]
            try:
                val = parse_spice_value(val_str)
                factor = random.uniform(0.3, 3.0)
                new_val = val * factor
                parts[3] = format_spice_value(new_val)
                lines[i] = ' '.join(parts)
                return '\n'.join(lines), f"BiasShift: Shifted current source {parts[0]} by factor {factor:.2f}"
            except Exception:
                pass
        return netlist, "BiasShift: Failed to parse or modify current source."

class RemoveComponent(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[CR]')
        if not instances:
            return netlist, "RemoveComponent: No capacitors or resistors found."
            
        i, line = random.choice(instances)
        lines[i] = f"* REMOVED BY PERTURBATION: {line}"
        return '\n'.join(lines), f"RemoveComponent: Removed component {line.split()[0]}"

class ScaleWL(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[XM]\w+')
        if not instances:
            return netlist, "ScaleWL: No transistors found."
            
        i, line = random.choice(instances)
        param = random.choice(['W', 'L'])
        parsed = parse_instance_line(line)
        if param in parsed['params']:
            try:
                val = parse_spice_value(parsed['params'][param])
                factor = random.uniform(0.5, 2.0)
                new_val = val * factor
                lines[i] = modify_instance_param(line, param, format_spice_value(new_val))
                return '\n'.join(lines), f"ScaleWL: Scaled {param} of {parsed['name']} by factor {factor:.2f}"
            except Exception:
                pass
        return netlist, f"ScaleWL: Could not scale {param} for {parsed.get('name', 'unknown')}."

class MisconnectNode(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[XM]\w+')
        if not instances:
            return netlist, "MisconnectNode: No transistors found."
            
        i, line = random.choice(instances)
        parsed = parse_instance_line(line)
        nodes = parsed['nodes']
        if len(nodes) >= 2:
            # swap two nodes
            idx1, idx2 = random.sample(range(len(nodes)), 2)
            nodes[idx1], nodes[idx2] = nodes[idx2], nodes[idx1]
            # rebuild line
            parts = line.split()
            parts[1:1+len(nodes)] = nodes
            lines[i] = ' '.join(parts)
            return '\n'.join(lines), f"MisconnectNode: Swapped nodes {idx1} and {idx2} of {parsed['name']}"
            
        return netlist, "MisconnectNode: Transistor has not enough nodes."

class ChangeLoad(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^C')
        if not instances:
            return netlist, "ChangeLoad: No capacitors found."
            
        i, line = random.choice(instances)
        parts = line.split()
        if len(parts) >= 4:
            val_str = parts[3]
            try:
                val = parse_spice_value(val_str)
                factor = random.uniform(0.1, 10.0)
                new_val = val * factor
                parts[3] = format_spice_value(new_val)
                lines[i] = ' '.join(parts)
                return '\n'.join(lines), f"ChangeLoad: Changed load cap {parts[0]} by factor {factor:.2f}"
            except Exception:
                pass
                
        return netlist, "ChangeLoad: Failed to modify load capacitor."

class MirrorRatioBroken(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[XM]\w+')
        if not instances:
            return netlist, "MirrorRatioBroken: No transistors found."
            
        i, line = random.choice(instances)
        parsed = parse_instance_line(line)
        if 'm' in parsed['params']:
            try:
                m_val = float(parsed['params']['m'])
                m_val = max(1.0, m_val + random.choice([-1.0, 1.0]))
                lines[i] = modify_instance_param(line, 'm', str(int(m_val)))
                return '\n'.join(lines), f"MirrorRatioBroken: Changed multiplier m of {parsed['name']} to {m_val}"
            except Exception:
                pass
                
        return netlist, "MirrorRatioBroken: Could not break mirror ratio."

class SwapDevices(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[XM]\w+')
        if len(instances) < 2:
            return netlist, "SwapDevices: Not enough transistors to swap."
            
        i1, line1 = random.choice(instances)
        i2, line2 = random.choice(instances)
        if i1 == i2:
            return netlist, "SwapDevices: Same device chosen twice."
            
        parsed1 = parse_instance_line(line1)
        parsed2 = parse_instance_line(line2)
        
        # simple swap of models
        m1 = parsed1['model']
        m2 = parsed2['model']
        if m1 and m2:
            lines[i1] = line1.replace(m1, m2)
            lines[i2] = line2.replace(m2, m1)
            return '\n'.join(lines), f"SwapDevices: Swapped models between {parsed1['name']} and {parsed2['name']}"
            
        return netlist, "SwapDevices: Failed to swap devices."

class ParameterDrift(Perturbation):
    def apply(self, netlist: str, seed: int) -> Tuple[str, str]:
        random.seed(seed)
        lines = netlist.split('\n')
        instances = find_instances(netlist, r'^[XM]\w+')
        
        mod_count = 0
        for i, line in instances:
            parsed = parse_instance_line(line)
            mod_line = line
            for param in ['W', 'L', 'm']:
                if param in parsed['params']:
                    try:
                        val = parse_spice_value(parsed['params'][param])
                        drift = random.uniform(0.7, 1.3)
                        new_val = val * drift
                        if param == 'm':
                            new_val = max(1.0, round(new_val))
                        mod_line = modify_instance_param(mod_line, param, format_spice_value(new_val) if param != 'm' else str(int(new_val)))
                        mod_count += 1
                    except Exception:
                        pass
            lines[i] = mod_line
            
        if mod_count > 0:
            return '\n'.join(lines), f"ParameterDrift: Applied drift to {mod_count} parameters."
        return netlist, "ParameterDrift: No parameters drifted."

class PerturbationPipeline:
    def __init__(self, perturbations: List[Perturbation]):
        self.perturbations = perturbations
        
    def generate(self, netlist: str, seed: int, num_perturbations: int = 1) -> PerturbedCircuit:
        random.seed(seed)
        current_netlist = netlist
        applied_descriptions = []
        
        for i in range(num_perturbations):
            p = random.choice(self.perturbations)
            # Use a deterministic but varying seed for each step
            step_seed = seed + i
            current_netlist, desc = p.apply(current_netlist, step_seed)
            applied_descriptions.append(desc)
            
        return PerturbedCircuit(
            original_netlist=netlist,
            perturbed_netlist=current_netlist,
            perturbations_applied=applied_descriptions
        )
