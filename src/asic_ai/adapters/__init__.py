from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.adapters.ngspice import NgspiceAdapter
from asic_ai.adapters.verilator import VerilatorAdapter
from asic_ai.adapters.opensta import OpenSTAAdapter
from asic_ai.adapters.nabla import NablaAdapter

def get_adapter(backend: str = 'ngspice', **kwargs) -> SimulatorAdapter:
    """Factory function for creating simulator adapters."""
    config = AdapterConfig(**kwargs)
    
    if backend == 'ngspice':
        return NgspiceAdapter(config)
    elif backend == 'ngspice_shared':
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        return NgspiceSharedAdapter(config)
    elif backend == 'mock':
        from asic_ai.adapters.mock import MockSimulatorAdapter
        return MockSimulatorAdapter(config)
    elif backend == 'verilator':
        return VerilatorAdapter(config)
    elif backend == 'opensta':
        return OpenSTAAdapter(config)
    elif backend == 'nabla':
        return NablaAdapter(config)
    elif backend == 'spectre' or backend == 'spectre_wsl':
        from asic_ai.adapters.spectre_wsl import SpectreWSLAdapter
        return SpectreWSLAdapter(config)
    else:
        raise ValueError(f"Unknown simulator backend: {backend}")

__all__ = [
    'SimulatorAdapter',
    'AdapterConfig',
    'NgspiceAdapter',
    'VerilatorAdapter',
    'OpenSTAAdapter',
    'NablaAdapter',
    'get_adapter'
]
