from typing import Protocol, List, Type
from .schema import (
    SimParams, PVTCorner, DCResult, ACResult, TranResult, 
    NoiseResult, StabilityResult, CornerResult, MonteCarloResult
)

class SimulatorInterface(Protocol):
    """
    Abstract protocol for simulator backends. 
    Adapters (e.g., ngspice, spectre, xyce) must implement this interface.
    """

    def dc(self, netlist: str, params: SimParams) -> DCResult:
        """Run DC analysis (operating point or sweep)."""
        ...

    def ac(self, netlist: str, params: SimParams) -> ACResult:
        """Run AC analysis."""
        ...

    def tran(self, netlist: str, params: SimParams) -> TranResult:
        """Run Transient analysis."""
        ...

    def noise(self, netlist: str, params: SimParams) -> NoiseResult:
        """Run Noise analysis."""
        ...

    def stb(self, netlist: str, params: SimParams) -> StabilityResult:
        """Run Stability analysis."""
        ...

    def corners(self, netlist: str, pvt_list: List[PVTCorner]) -> List[CornerResult]:
        """Run simulations across multiple PVT corners."""
        ...

    def mc(self, netlist: str, n: int, seed: int) -> MonteCarloResult:
        """Run Monte Carlo analysis."""
        ...


class SimulatorRegistry:
    """Registry to manage and instantiate simulator backends."""
    
    _backends: dict[str, Type[SimulatorInterface]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[SimulatorInterface]) -> None:
        """Register a new simulator backend."""
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str, **kwargs) -> SimulatorInterface:
        """Instantiate and return a simulator backend by name."""
        if name not in cls._backends:
            raise ValueError(f"Simulator backend '{name}' not found. Available: {list(cls._backends.keys())}")
        return cls._backends[name](**kwargs)
