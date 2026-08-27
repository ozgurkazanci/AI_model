import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field

class PerturbedCircuit(BaseModel):
    original_netlist: str
    perturbed_netlist: str
    perturbation_types: List[str]
    description: str

class Perturbation(ABC):
    @abstractmethod
    def apply(self, netlist: str) -> Tuple[str, str]:
        """Applies the perturbation to the netlist.
        Returns:
            Tuple containing the perturbed netlist and a description of the change.
        """
        pass

class BiasShift(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        # Implementation placeholder
        factor = random.uniform(0.3, 3.0)
        return netlist + f"\n* BiasShift applied: x{factor:.2f}", f"Multiplied bias current by {factor:.2f}"

class RemoveComponent(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Component Removed", "Deleted a compensation cap or bias resistor"

class ScaleWL(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        scale = random.uniform(0.1, 10.0)
        return netlist + f"\n* ScaleWL applied: x{scale:.2f}", f"Scaled W/L ratio by {scale:.2f}"

class MisconnectNode(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Node Misconnected", "Connected a node to the wrong net"

class ChangeLoad(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Load Changed", "Modified load capacitance without updating compensation"

class MirrorRatioBroken(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Mirror Ratio Broken", "Changed mirror ratio (e.g., 1:4 became 1:2)"

class SwapDevices(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Devices Swapped", "Swapped NMOS/PMOS or two device positions"

class ParameterDrift(Perturbation):
    def apply(self, netlist: str) -> Tuple[str, str]:
        return netlist + "\n* Parameter Drift", "Small random drift applied to multiple parameters"

class PerturbationPipeline:
    def __init__(self):
        self.registry: Dict[Type[Perturbation], float] = {}

    def register(self, perturbation_type: Type[Perturbation], weight: float) -> None:
        self.registry[perturbation_type] = weight

    def generate(self, netlist: str, n: int, seed: int) -> List[PerturbedCircuit]:
        random.seed(seed)
        results = []
        if not self.registry:
            return results

        types, weights = zip(*self.registry.items())
        
        for _ in range(n):
            num_perturbations = random.randint(1, min(3, len(types)))
            chosen_types = random.choices(types, weights=weights, k=num_perturbations)
            
            current_netlist = netlist
            descriptions = []
            applied_types = []
            
            for p_type in chosen_types:
                perturbation_instance = p_type()
                current_netlist, desc = perturbation_instance.apply(current_netlist)
                descriptions.append(desc)
                applied_types.append(p_type.__name__)
            
            results.append(PerturbedCircuit(
                original_netlist=netlist,
                perturbed_netlist=current_netlist,
                perturbation_types=applied_types,
                description=" | ".join(descriptions)
            ))
            
        return results
