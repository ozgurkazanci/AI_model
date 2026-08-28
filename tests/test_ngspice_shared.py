"""Integration tests for ngspice shared library adapter.

Requires KiCad with ngspice.dll. Tests are skipped if not available.
"""
from __future__ import annotations

import pytest

try:
    from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter, find_ngspice_dll
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.tool_interface.schema import SimParams, PVTCorner
    HAS_NGSPICE = find_ngspice_dll() is not None
except ImportError:
    HAS_NGSPICE = False

skipif_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE, reason="ngspice DLL not found")


@pytest.fixture
def adapter(tmp_path):
    config = AdapterConfig(binary_path="", work_dir=str(tmp_path))
    return NgspiceSharedAdapter(config)


@pytest.fixture
def resistor_divider(tmp_path):
    cir = tmp_path / "divider.cir"
    cir.write_text("""* Resistor Divider
V1 vdd 0 DC 1.8
R1 vdd out 10k
R2 out 0 10k
.dc V1 0 3.3 0.1
.end
""")
    return str(cir)


@pytest.fixture
def rc_filter(tmp_path):
    cir = tmp_path / "rc.cir"
    cir.write_text("""* RC Low-Pass Filter
V1 in 0 AC 1 DC 0
R1 in out 1k
C1 out 0 1n
.ac dec 10 1 1G
.end
""")
    return str(cir)


@skipif_no_ngspice
class TestNgspiceSharedAdapter:

    def test_adapter_loads(self, adapter):
        assert adapter._lib is not None

    def test_dc_simulation(self, adapter, resistor_divider):
        params = SimParams(analysis_type="dc")
        result = adapter.dc(resistor_divider, params)
        assert result is not None

    def test_ac_simulation(self, adapter, rc_filter):
        params = SimParams(analysis_type="ac")
        result = adapter.ac(rc_filter, params)
        assert result is not None

    def test_tran_simulation(self, adapter, tmp_path):
        cir = tmp_path / "tran.cir"
        cir.write_text("""* Transient Test
V1 in 0 PULSE(0 1.8 0 1n 1n 5u 10u)
R1 in out 1k
C1 out 0 1n
.tran 0.1u 20u
.end
""")
        params = SimParams(analysis_type="tran")
        result = adapter.tran(str(cir), params)
        assert result is not None

    def test_nmos_iv_curves(self, adapter, tmp_path):
        cir = tmp_path / "nmos.cir"
        cir.write_text("""* NMOS I-V
.model nch nmos level=1 vto=0.5 kp=100u
M1 drain gate 0 0 nch W=10u L=1u
Vgs gate 0 DC 0.9
Vds drain 0 DC 0
.dc Vds 0 1.8 0.05 Vgs 0.5 1.0 0.1
.end
""")
        params = SimParams(analysis_type="dc")
        result = adapter.dc(str(cir), params)
        assert result is not None
