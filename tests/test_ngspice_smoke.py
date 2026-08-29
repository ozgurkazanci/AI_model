"""ngspice smoke tests -- verifies ngspice works on this system.

Uses the shared library (DLL) from KiCad if available, falls back to the binary
on PATH for the presence check.

These assert REAL NUMBERS, not point counts. The previous versions asserted
things like "points > 10", which a fabricated [0.0]*181 satisfies trivially;
that is exactly how the zeros bug survived 67 phases.
"""
from __future__ import annotations

import math
import shutil
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

RC_F3DB = 1.0 / (2.0 * math.pi * 1000.0 * 1e-9)  # 159154.9431 Hz


def _adapter(work_dir: str):
    from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
    from asic_ai.adapters.base import AdapterConfig
    return NgspiceSharedAdapter(AdapterConfig(binary_path="", work_dir=work_dir))


class TestNgspiceSmoke:
    """Smoke tests using ngspice shared library."""

    def test_ngspice_dll_loads(self):
        """DLL loads and initializes."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(td)
            assert adapter._lib is not None

    def test_simple_dc_sweep(self):
        """CMOS inverter VTC: rails at both ends, trip point near VDD/2."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(td)
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
            out = result.sweeps["out"]
            assert len(out.x_values) == 181
            # Real sweep axis: 0 V to 1.8 V in 10 mV steps, not range(n).
            assert out.x_values[0] == pytest.approx(0.0, abs=1e-12)
            assert out.x_values[-1] == pytest.approx(1.8, abs=1e-9)
            assert out.x_values[1] - out.x_values[0] == pytest.approx(0.01, abs=1e-9)
            # Real VTC: high rail at Vin=0, low rail at Vin=VDD.
            assert out.y_values[0] == pytest.approx(1.8, abs=0.01)
            assert out.y_values[-1] == pytest.approx(0.0, abs=0.01)

            # Switching threshold, where Vout crosses Vin.
            trip = None
            for i in range(len(out.x_values) - 1):
                d0 = out.y_values[i] - out.x_values[i]
                d1 = out.y_values[i + 1] - out.x_values[i + 1]
                if d0 * d1 < 0:
                    trip = out.x_values[i] + (-d0) * (
                        out.x_values[i + 1] - out.x_values[i]) / (d1 - d0)
                    break
            assert trip is not None
            assert trip == pytest.approx(0.9, abs=0.15)

    def test_ac_analysis(self):
        """RC 1k/1n low-pass: -3 dB at 159.1549 kHz, -45 deg there."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters import measure
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(td)
            cir = Path(td) / "rc.cir"
            cir.write_text("""\
* RC Filter
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac dec 100 1 1G
.end
""")
            result = adapter.ac(str(cir), SimParams(analysis_type="ac"))
            assert result.frequencies[0] == pytest.approx(1.0, rel=1e-9)
            assert result.frequencies[-1] == pytest.approx(1e9, rel=1e-6)

            m = adapter.measure_ac(result, "out")
            assert m["dc_gain_db"] == pytest.approx(0.0, abs=0.01)
            assert m["bandwidth_3db"] == pytest.approx(RC_F3DB, rel=0.01)
            assert m["rolloff_db_per_dec"] == pytest.approx(-20.0, abs=0.2)

            phase = result.signals["vp(out)"].y_values
            at_pole = measure.value_at_freq(result.frequencies, phase,
                                            m["bandwidth_3db"])
            assert at_pole == pytest.approx(-45.0, abs=1.0)

    def test_transient(self):
        """RC pulse response: tau = 1 us, 63.2 pct in one tau."""
        if not HAS_DLL:
            pytest.skip("No ngspice DLL")
        from asic_ai.adapters import measure
        from asic_ai.tool_interface.schema import SimParams

        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(td)
            cir = Path(td) / "tran.cir"
            cir.write_text("""\
* Pulse Response
V1 in 0 PULSE(0 1.8 0 1n 1n 50u 100u)
R1 in out 1k
C1 out 0 1n
.tran 10n 10u
.end
""")
            result = adapter.tran(str(cir), SimParams(analysis_type="tran"))
            assert result.time[0] == pytest.approx(0.0, abs=1e-12)
            assert result.time[-1] == pytest.approx(10e-6, rel=1e-6)

            y = result.signals["out"].y_values
            assert y[-1] == pytest.approx(1.8, abs=0.01)
            t63 = measure.crossing_time(result.time, y, 1.8 * (1 - math.exp(-1)))
            assert t63 == pytest.approx(1e-6, rel=0.03)

            m = adapter.measure_tran(result, "out")
            assert m["rise_time"] == pytest.approx(math.log(9.0) * 1e-6, rel=0.01)


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
