"""ngspice smoke tests — verifies ngspice works on this system.

Uses the shared library (DLL) from KiCad if available,
falls back to binary on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Check for ngspice DLL (KiCad) or binary
try:
    from asic_ai.adapters.ngspice_shared import find_ngspice_dll
    HAS_DLL = find_ngspice_dll() is not None
except ImportError:
    HAS_DLL = False

HAS_BINARY = shutil.which("ngspice") is not None
HAS_NGSPICE = HAS_DLL or HAS_BINARY

pytestmark = pytest.mark.skipif(
    not HAS_NGSPICE,
    reason="ngspice not available (no DLL or binary)",
)

MINIMAL_MODELS = """\
.model sky130_fd_pr__nfet_01v8 nmos level=1 vto=0.45 kp=200u
.model sky130_fd_pr__pfet_01v8 pmos level=1 vto=-0.45 kp=100u
"""


class TestNgspiceSmoke:
    """Smoke tests using ngspice shared library."""

    def test_ngspice_dll_loads(self):
        """DLL loads and initializes."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        from asic_ai.adapters.base import AdapterConfig
        with tempfile.TemporaryDirectory() as td:
            adapter = NgspiceSharedAdapter(AdapterConfig(binary_path="", work_dir=td))
            assert adapter._lib is not None

    def test_simple_dc_sweep(self):
        """Run a simple DC sweep via shared library."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        from asic_ai.adapters.base import AdapterConfig
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = NgspiceSharedAdapter(AdapterConfig(binary_path="", work_dir=td))
            cir = Path(td) / "inv.cir"
            cir.write_text(f"""\
* CMOS Inverter
{MINIMAL_MODELS}
VDD vdd 0 DC 1.8
Vin in 0 DC 0
M1 out in 0 0 sky130_fd_pr__nfet_01v8 W=1u L=0.15u
M2 out in vdd vdd sky130_fd_pr__pfet_01v8 W=2u L=0.15u
.dc Vin 0 1.8 0.01
.end
""")
            result = adapter.dc(str(cir), SimParams(analysis_type="dc"))
            points = sum(len(s.x_values) for s in result.sweeps.values())
            assert points > 10, f"Expected >10 data points, got {points}"

    def test_ac_analysis(self):
        """Run an AC analysis."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        from asic_ai.adapters.base import AdapterConfig
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = NgspiceSharedAdapter(AdapterConfig(binary_path="", work_dir=td))
            cir = Path(td) / "rc.cir"
            cir.write_text("""\
* RC Filter
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac dec 10 1 1G
.end
""")
            result = adapter.ac(str(cir), SimParams(analysis_type="ac"))
            assert len(result.frequencies) > 0

    def test_transient(self):
        """Run a transient simulation."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        from asic_ai.adapters.base import AdapterConfig
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = NgspiceSharedAdapter(AdapterConfig(binary_path="", work_dir=td))
            cir = Path(td) / "tran.cir"
            cir.write_text("""\
* Pulse Response
V1 in 0 PULSE(0 1.8 0 1n 1n 5u 10u)
R1 in out 1k
C1 out 0 1n
.tran 0.1u 20u
.end
""")
            result = adapter.tran(str(cir), SimParams(analysis_type="tran"))
            assert len(result.time) > 0


class TestAdapterImport:
    """Unit tests that don't require ngspice."""

    def test_ngspice_adapter_exists(self):
        from asic_ai.adapters.ngspice import NgspiceAdapter
        assert NgspiceAdapter is not None

    def test_ngspice_shared_adapter_exists(self):
        from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
        assert NgspiceSharedAdapter is not None

    def test_nabla_adapter_exists(self):
        from asic_ai.adapters.nabla import NablaAdapter
        assert NablaAdapter is not None
