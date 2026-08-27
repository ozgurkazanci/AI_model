"""ngspice smoke test — verifies the adapter works with a real simulation.

Skip automatically if ngspice is not installed on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

NGSPICE_AVAILABLE = shutil.which("ngspice") is not None

pytestmark = pytest.mark.skipif(
    not NGSPICE_AVAILABLE,
    reason="ngspice not installed or not on PATH",
)

MINIMAL_MODELS = """\
.model sky130_fd_pr__nfet_01v8 nmos level=1 vto=0.45 kp=200u
.model sky130_fd_pr__pfet_01v8 pmos level=1 vto=-0.45 kp=100u
"""


class TestNgspiceSmoke:
    """Smoke tests that require ngspice binary."""

    def test_ngspice_version(self):
        result = subprocess.run(
            ["ngspice", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout + result.stderr
        assert "ngspice" in output.lower()

    def test_simple_dc_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models.spice"
            output_file = Path(tmpdir) / "output.txt"
            netlist_path = Path(tmpdir) / "test.spice"

            model_path.write_text(MINIMAL_MODELS)

            netlist = f"""\
* CMOS Inverter DC Sweep
.include {str(model_path).replace(chr(92), '/')}
VDD vdd 0 dc 1.8
VIN in 0 dc 0.9
XM1 out in vdd vdd sky130_fd_pr__pfet_01v8 W=1u L=150n
XM2 out in 0 0 sky130_fd_pr__nfet_01v8 W=0.5u L=150n
.dc VIN 0 1.8 0.01
.control
run
wrdata {str(output_file).replace(chr(92), '/')} v(out)
.endc
.end
"""
            netlist_path.write_text(netlist)

            result = subprocess.run(
                ["ngspice", "-b", str(netlist_path)],
                capture_output=True, text=True, timeout=30, cwd=tmpdir,
            )

            assert output_file.exists(), f"No output. ngspice: {result.stderr}"
            lines = [l for l in output_file.read_text().split("\n") if l.strip() and not l.startswith("#")]
            assert len(lines) > 10


class TestAdapterImport:
    """Unit tests that don't require ngspice binary."""

    def test_ngspice_adapter_exists(self):
        from asic_ai.adapters.ngspice import NgspiceAdapter
        assert NgspiceAdapter is not None

    def test_nabla_adapter_exists(self):
        from asic_ai.adapters.nabla import NablaAdapter
        assert NablaAdapter is not None
