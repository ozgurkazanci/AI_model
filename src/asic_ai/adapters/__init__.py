from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.adapters.ngspice import NgspiceAdapter
from asic_ai.adapters.verilator import VerilatorAdapter
from asic_ai.adapters.opensta import OpenSTAAdapter
from asic_ai.adapters.nabla import NablaAdapter

def get_adapter(backend: str = 'ngspice_shared', **kwargs) -> SimulatorAdapter:
    """Factory function for creating simulator adapters.

    Defaults to 'ngspice_shared'. The previous default was 'ngspice', the
    subprocess adapter, whose seven result constructors ALL used field names
    that do not exist on the frozen schema, so every method raised a
    ValidationError -- a caller who did not name a backend got the one
    adapter where nothing worked. It also needs an `ngspice` executable,
    which KiCad does not ship: only ngspice.dll, which 'ngspice_shared' drives.

    `pdk` and `corner` are accepted for the ngspice_shared backend and select a
    foundry model deck resolved from configuration (see adapters/pdk_deck.py).
    They are ignored by the other backends.
    """
    pdk = kwargs.pop('pdk', None)
    corner = kwargs.pop('corner', 'tt')
    config = AdapterConfig(**kwargs)

    if backend == 'ngspice':
        return NgspiceAdapter(config)
    elif backend == 'ngspice_shared':
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        return NgspiceSharedAdapter(config, pdk=pdk, corner=corner)
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
