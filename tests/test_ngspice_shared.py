"""Integration tests for the ngspice shared library adapter.

These are KNOWN-ANSWER tests: every assertion below compares against a value
that is derivable on paper from the netlist, not against "is not None". The
adapter used to fabricate SignalData(x=range(n), y=[0.0]*n) for every analysis,
and the old assertions passed happily on that. TestNoFabricatedData is the
regression guard that would have caught it.

Every circuit here is passive R/L/C or a LEVEL=1 toy model, so the whole class
runs on any machine with ngspice and needs no PDK. The TSMC tests at the bottom
are gated on the foundry deck being installed and skip cleanly without it.
"""
from __future__ import annotations

import math

import pytest

try:
    from asic_ai.adapters.ngspice_shared import (
        NgspiceSharedAdapter, NgspiceError, find_ngspice_dll, netlist_text,
        set_supply_voltage,
    )
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters import measure, pdk_deck
    from asic_ai.tool_interface.schema import SimParams, PVTCorner
    HAS_NGSPICE = find_ngspice_dll() is not None
    IMPORT_OK = True
except ImportError:  # pragma: no cover
    HAS_NGSPICE = False
    IMPORT_OK = False

skipif_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE, reason="ngspice DLL not found")

HAS_PDK = bool(IMPORT_OK) and pdk_deck.pdk_available("tsmc65")
skipif_no_pdk = pytest.mark.skipif(
    not (HAS_NGSPICE and HAS_PDK),
    reason="TSMC CRN65GPLUS model deck not installed on this machine",
)

# Analytic reference values, all derived on paper.
BOLTZMANN = 1.380649e-23
RC_F3DB = 1.0 / (2.0 * math.pi * 1000.0 * 1e-9)      # 159154.9431 Hz
RC_TAU = 1000.0 * 1e-9                               # 1 us
RC_RISE_10_90 = math.log(9.0) * RC_TAU               # 2.197225 us
FRAC_63 = 1.0 - math.exp(-1.0)                       # 0.6321205588


@pytest.fixture
def adapter(tmp_path):
    config = AdapterConfig(binary_path="", work_dir=str(tmp_path))
    return NgspiceSharedAdapter(config)


@pytest.fixture
def pdk_adapter(tmp_path):
    config = AdapterConfig(binary_path="", work_dir=str(tmp_path))
    return NgspiceSharedAdapter(config, pdk="tsmc65")


DIVIDER_SWEEP = """* Resistor Divider
V1 vdd 0 DC 1.8
R1 vdd out 10k
R2 out 0 10k
.dc V1 0 3.3 0.1
.end
"""

DIVIDER_OP = """* Resistor Divider operating point
V1 vdd 0 DC 1.8
R1 vdd out 10k
R2 out 0 10k
.op
.end
"""

RC_AC = """* RC Low-Pass Filter
V1 in 0 AC 1 DC 0
R1 in out 1k
C1 out 0 1n
.ac dec 500 1 100Meg
.end
"""

RC_TRAN = """* RC pulse response, tau = 1 us
V1 in 0 PULSE(0 1 0 1n 1n 50u 100u)
R1 in out 1k
C1 out 0 1n
.tran 10n 10u
.end
"""

RC_NOISE = """* Thermal noise of a 1k resistor into 1n
V1 in 0 AC 1 DC 0
R1 in out 1k
C1 out 0 1n
.noise v(out) V1 dec 500 1 1G
.end
"""

# Two buffered poles at 1 kHz and 100 kHz behind a gain of 1000 (60 dB). The
# VCVS buffers keep the stages from loading each other so the response is
# exactly A0 / ((1 + jf/1k)(1 + jf/100k)).
TWO_POLE_LOOP = """* Buffered two-pole loop gain
Vin in 0 AC 1 DC 0
E1 a 0 in 0 1000
R1 a b 1k
C1 b 0 159.1549431n
E2 c 0 b 0 1
R2 c out 1k
C2 out 0 1.591549431n
.ac dec 200 1 100Meg
.end
"""

TEMPCO_DIVIDER = """* Divider with a temperature coefficient on the top leg
V1 vdd 0 DC 1.8
R1 vdd out 10k tc1=0.01
R2 out 0 10k
.op
.end
"""


def _two_pole_phase_margin() -> float:
    """Analytic phase margin of TWO_POLE_LOOP, computed from first principles."""
    def loop(f: float) -> complex:
        return 1000.0 / ((1 + 1j * f / 1000.0) * (1 + 1j * f / 100000.0))
    lo, hi = 1.0, 1e8
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if abs(loop(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    fu = math.sqrt(lo * hi)
    t = loop(fu)
    return 180.0 + math.degrees(math.atan2(t.imag, t.real))


# ---------------------------------------------------------------------------
# Pure measurement helpers. No simulator needed, so these always run.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not IMPORT_OK, reason="asic_ai not importable")
class TestMeasureHelpers:

    def test_db3_is_exact_not_three(self):
        assert measure.DB3 == pytest.approx(3.010299956639812, abs=1e-12)
        assert measure.DB3 != 3.0

    def test_unwrap_is_noop_when_not_wrapped(self):
        p = [0.0, -10.0, -30.0, -90.0]
        assert measure.unwrap_deg(p) == pytest.approx(p)

    def test_unwrap_recovers_minus_270(self):
        wrapped = [0.0, -90.0, -179.0, 179.0, 90.0]
        out = measure.unwrap_deg(wrapped)
        assert out[-1] == pytest.approx(-270.0, abs=1e-9)

    def test_normalize_dc_phase_of_inverting_loop(self):
        raw = [180.0, 170.0, 100.0, 10.0]
        out = measure.normalize_dc_phase(measure.unwrap_deg(raw))
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[-1] == pytest.approx(-170.0, abs=1e-9)

    def test_transfer_function_divides_by_input(self):
        # An AC 2 stimulus must not read 6 dB high.
        gain_db, _ = measure.transfer_function([complex(2.0, 0.0)],
                                               [complex(2.0, 0.0)])
        assert gain_db[0] == pytest.approx(0.0, abs=1e-12)

    def test_single_pole_metrics_are_analytic(self):
        freqs = [10.0 ** (i / 100.0) for i in range(0, 801)]
        gain_db, phase = [], []
        for f in freqs:
            h = 1000.0 / (1 + 1j * f / 1000.0)
            gain_db.append(20 * math.log10(abs(h)))
            phase.append(math.degrees(math.atan2(h.imag, h.real)))
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["dc_gain_db"] == pytest.approx(60.0, abs=0.01)
        assert m["bandwidth_3db"] == pytest.approx(1000.0, rel=1e-3)
        assert m["ugb"] == pytest.approx(1e6, rel=1e-3)
        # A single pole never reaches -180 deg, so the gain margin is infinite
        # and must be reported as None, never as 0.0.
        assert m["gain_margin"] is None
        assert m["f_180"] is None

    def test_supply_current_sign_convention(self):
        idd = measure.supply_current({"v1#branch": -9e-05, "out": 0.9})
        assert idd == pytest.approx(9e-05, rel=1e-12)

    def test_supply_current_sums_magnitudes(self):
        idd = measure.supply_current({"v1#branch": -1e-3, "v2#branch": 1e-3})
        assert idd == pytest.approx(2e-3, rel=1e-12)

    def test_integrate_noise_flat_spectrum(self):
        freqs = [float(i) for i in range(1, 10001)]
        spec = [2.0e-9] * len(freqs)
        total = measure.integrate_noise(freqs, spec)
        assert total == pytest.approx(2.0e-9 * math.sqrt(9999.0), rel=1e-9)

    def test_metrics_return_none_not_zero(self):
        freqs = [1.0, 10.0, 100.0]
        gain = [-20.0, -30.0, -40.0]  # never above 0 dB
        m = measure.ac_metrics(freqs, gain, [0.0, -45.0, -90.0])
        assert m["ugb"] is None
        assert m["phase_margin"] is None

    def test_netlist_text_accepts_text_and_path(self, tmp_path):
        text = "* t\nV1 a 0 1\n.end\n"
        assert netlist_text(text) == text
        f = tmp_path / "x.cir"
        f.write_text(text, encoding="utf-8")
        assert netlist_text(str(f)) == text

    def test_set_supply_voltage_rewrites_dc(self):
        out, n = set_supply_voltage("V1 vdd 0 DC 1.8\nR1 vdd 0 1k\n", 1.62)
        assert n == 1
        assert "V1 vdd 0 DC 1.62" in out

    def test_set_supply_voltage_leaves_signal_sources_alone(self):
        src = "Vin in 0 DC 0 AC 1\nV1 vdd 0 DC 1.8\n"
        out, n = set_supply_voltage(src, 1.0)
        assert n == 1
        assert "Vin in 0 DC 0 AC 1" in out


# ---------------------------------------------------------------------------
# Known-answer physics against the real DLL
# ---------------------------------------------------------------------------

@skipif_no_ngspice
class TestNgspiceSharedAdapter:

    def test_adapter_loads(self, adapter):
        assert adapter._lib is not None

    def test_dc_sweep_divider_is_exactly_half(self, adapter):
        """R1 == R2, so v(out) must be v(vdd)/2 at EVERY swept point."""
        result = adapter.dc(DIVIDER_SWEEP, SimParams(analysis_type="dc"))
        sig = result.sweeps["out"]
        assert len(sig.x_values) == 34
        assert len(sig.y_values) == len(sig.x_values)
        # The x axis is the real swept source voltage, 0 V to 3.3 V.
        assert sig.x_values[0] == pytest.approx(0.0, abs=1e-12)
        assert sig.x_values[-1] == pytest.approx(3.3, abs=1e-9)
        for x, y in zip(sig.x_values, sig.y_values):
            assert y == pytest.approx(x / 2.0, abs=1e-12)
        assert sig.y_values[-1] == pytest.approx(1.65, abs=1e-9)

    def test_operating_point_values(self, adapter):
        """1.8 V across 10k + 10k: v(out) = 0.9 V exactly, idd = 90 uA."""
        result = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        assert result.op_points["vdd"] == pytest.approx(1.8, abs=1e-12)
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)
        # ngspice uses the passive sign convention on the source, so a supply
        # that delivers current reports a NEGATIVE branch current.
        assert result.op_points["v1#branch"] == pytest.approx(-90e-6, rel=1e-9)
        assert result.op_points["v1#branch"] < 0.0
        assert adapter.measure_idd(result) == pytest.approx(1.8 / 20e3, rel=1e-9)
        assert adapter.measure_idd(result) == pytest.approx(90e-6, rel=1e-9)

    def test_ac_rc_lowpass_matches_theory(self, adapter):
        """RC 1k/1n: 0 dB passband, -3 dB at 159.1549 kHz, -20 dB/decade."""
        result = adapter.ac(RC_AC, SimParams(analysis_type="ac"))
        assert result.frequencies[0] == pytest.approx(1.0, rel=1e-9)
        assert result.frequencies[-1] == pytest.approx(1e8, rel=1e-6)

        m = adapter.measure_ac(result, "out")
        assert m["dc_gain_db"] == pytest.approx(0.0, abs=0.01)
        assert m["bandwidth_3db"] == pytest.approx(RC_F3DB, rel=0.01)
        # Rolloff far above the pole is a single pole, -20 dB per decade.
        assert m["rolloff_db_per_dec"] == pytest.approx(-20.0, abs=0.2)

        phase = result.signals["vp(out)"]
        phase_at_f3db = measure.value_at_freq(result.frequencies, phase.y_values,
                                              m["bandwidth_3db"])
        assert phase_at_f3db == pytest.approx(-45.0, abs=1.0)

    def test_ac_signal_naming_scheme(self, adapter):
        result = adapter.ac(RC_AC, SimParams(analysis_type="ac"))
        assert "vdb(out)" in result.signals
        assert "vp(out)" in result.signals
        assert result.signals["vdb(out)"].name == "vdb(out)"
        # x_values of every signal is the frequency axis itself.
        assert result.signals["vdb(out)"].x_values == pytest.approx(result.frequencies)

    def test_tran_rc_step_matches_theory(self, adapter):
        """tau = 1 us: 63.2 pct in one tau, 10-90 rise time = ln(9)*tau."""
        result = adapter.tran(RC_TRAN, SimParams(analysis_type="tran"))
        assert result.time[0] == pytest.approx(0.0, abs=1e-12)
        assert result.time[-1] == pytest.approx(10e-6, rel=1e-6)

        t = result.time
        y = result.signals["out"].y_values
        assert y[-1] == pytest.approx(1.0, abs=0.001)

        t63 = measure.crossing_time(t, y, FRAC_63)
        assert t63 == pytest.approx(RC_TAU, rel=0.03)

        m = adapter.measure_tran(result, "out")
        assert m["rise_time"] == pytest.approx(RC_RISE_10_90, rel=0.01)
        assert m["overshoot_pct"] == pytest.approx(0.0, abs=0.1)

    def test_noise_matches_johnson_nyquist(self, adapter):
        """Input-referred noise of a 1k resistor must be sqrt(4kTR)."""
        result = adapter.noise(RC_NOISE, SimParams(analysis_type="noise"))
        assert len(result.frequencies) > 100
        expected = math.sqrt(4 * BOLTZMANN * 300.15 * 1000.0)
        assert result.input_noise.y_values[0] == pytest.approx(expected, rel=1e-3)
        assert result.input_noise.name == "inoise_spectrum"
        assert result.output_noise.name == "onoise_spectrum"
        # Band-limited output noise, sqrt(4kTR*f0*(atan(f_hi/f0)-atan(f_lo/f0))).
        f0 = RC_F3DB
        analytic = math.sqrt(
            4 * BOLTZMANN * 300.15 * 1000.0 * f0
            * (math.atan(1e9 / f0) - math.atan(1.0 / f0))
        )
        total = measure.integrate_noise(result.frequencies,
                                        result.output_noise.y_values)
        assert total == pytest.approx(analytic, rel=0.005)

    def test_stb_two_pole_loop(self, adapter):
        """60 dB DC loop gain, poles at 1 kHz and 100 kHz."""
        result = adapter.stb(TWO_POLE_LOOP, SimParams(analysis_type="ac"))
        assert result.loop_gain.y_values[0] == pytest.approx(60.0, abs=0.01)
        assert result.phase_margin == pytest.approx(_two_pole_phase_margin(), abs=0.5)
        assert 17.0 < result.phase_margin < 20.0
        # Two poles can never reach -180 deg, so the gain margin is infinite.
        assert math.isinf(result.gain_margin)
        assert result.loop_gain.name == "loop_gain_db"

    def test_stb_raises_when_loop_never_reaches_unity(self, adapter):
        """And the message must name the PEAK gain, not the gain at f_start.

        The old message said "loop gain never crosses 0 dB" whatever the data
        was, because the ugb guard looked at gain_db[0]. On an AC-coupled loop
        with 40 dB of mid-band gain it therefore contradicted itself.
        """
        with pytest.raises(NgspiceError, match="never exceeds 0 dB"):
            adapter.stb(RC_AC, SimParams(analysis_type="ac"))
        with pytest.raises(NgspiceError, match="peak loop gain"):
            adapter.stb(RC_AC, SimParams(analysis_type="ac"))

    def test_nmos_iv_curves_are_monotonic(self, adapter):
        """Nested .dc sweep: drain current must grow with Vgs, not stay zero."""
        cir = """* NMOS I-V
.model nch nmos level=1 vto=0.5 kp=100u
M1 drain gate 0 0 nch W=10u L=1u
Vgs gate 0 DC 0.9
Vds drain 0 DC 0
.dc Vds 0 1.8 0.05 Vgs 0.5 1.0 0.1
.end
"""
        result = adapter.dc(cir, SimParams(analysis_type="dc"))
        # The gate staircase is the outer sweep variable and must be present.
        gate = result.sweeps["gate"]
        assert min(gate.y_values) == pytest.approx(0.5, abs=1e-9)
        assert max(gate.y_values) == pytest.approx(1.0, abs=1e-9)
        # Drain current flows out of Vds, so its branch current is negative.
        ids = result.sweeps["vds#branch"]
        assert min(ids.y_values) < -1e-6
        assert abs(min(ids.y_values)) > abs(max(ids.y_values))


# ---------------------------------------------------------------------------
# The regression guard for the fabricated-zeros bug
# ---------------------------------------------------------------------------

@skipif_no_ngspice
class TestNoFabricatedData:
    """Fails if the adapter ever goes back to zeros and range(n) indices.

    The pre-Phase-68 adapter regex-grepped "No. of Data Rows : N" out of the
    console and returned SignalData(x=range(N), y=[0.0]*N) for every analysis.
    Every assertion in this class is false for that data and true for real data.
    """

    def test_dc_y_values_are_not_all_zeros(self, adapter):
        result = adapter.dc(DIVIDER_SWEEP, SimParams(analysis_type="dc"))
        sig = result.sweeps["out"]
        assert any(y != 0.0 for y in sig.y_values), "y_values are all zeros"

    def test_dc_x_values_are_not_an_index_range(self, adapter):
        result = adapter.dc(DIVIDER_SWEEP, SimParams(analysis_type="dc"))
        sig = result.sweeps["out"]
        index_range = [float(i) for i in range(len(sig.x_values))]
        assert sig.x_values != index_range, "x_values is just range(n)"
        # Consecutive x steps are 0.1 V, not 1.0.
        assert sig.x_values[1] - sig.x_values[0] == pytest.approx(0.1, abs=1e-9)

    def test_dc_response_actually_depends_on_the_circuit(self, adapter):
        """Two different dividers must produce two different answers."""
        one_to_one = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        three_to_one = adapter.dc(
            DIVIDER_OP.replace("R1 vdd out 10k", "R1 vdd out 30k"),
            SimParams(analysis_type="dc"),
        )
        assert one_to_one.op_points["out"] == pytest.approx(0.9, abs=1e-12)
        assert three_to_one.op_points["out"] == pytest.approx(0.45, abs=1e-12)

    def test_op_points_are_populated(self, adapter):
        result = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        assert result.op_points, "op_points was empty (the old adapter always was)"

    def test_ac_frequencies_are_hertz_not_indices(self, adapter):
        result = adapter.ac(RC_AC, SimParams(analysis_type="ac"))
        n = len(result.frequencies)
        assert result.frequencies != [float(i) for i in range(n)]
        assert result.frequencies[0] == pytest.approx(1.0, rel=1e-9)
        assert result.frequencies[-1] > 1e7
        mag = result.signals["vdb(out)"].y_values
        assert any(v != 0.0 for v in mag)
        # A real low-pass rolls off; a zero vector would be flat.
        assert mag[0] - mag[-1] > 50.0

    def test_tran_time_is_seconds_not_indices(self, adapter):
        result = adapter.tran(RC_TRAN, SimParams(analysis_type="tran"))
        n = len(result.time)
        assert result.time != [float(i) for i in range(n)]
        assert result.time[-1] == pytest.approx(10e-6, rel=1e-6)
        y = result.signals["out"].y_values
        assert any(v != 0.0 for v in y)
        assert max(y) > 0.9


# ---------------------------------------------------------------------------
# Netlist input handling and failure detection
# ---------------------------------------------------------------------------

@skipif_no_ngspice
class TestNetlistInputModes:

    def test_accepts_a_file_path(self, adapter, tmp_path):
        cir = tmp_path / "divider.cir"
        cir.write_text(DIVIDER_OP, encoding="utf-8")
        result = adapter.dc(str(cir), SimParams(analysis_type="dc"))
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)

    def test_accepts_netlist_text(self, adapter):
        """training/rl_env.py passes netlist TEXT, per the frozen protocol."""
        result = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)

    def test_accepts_a_plain_dict_of_params(self, adapter):
        """rl_env.py passes a raw args dict where SimParams is declared."""
        result = adapter.dc(DIVIDER_OP, {"netlist": DIVIDER_OP})
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)

    def test_first_line_is_never_swallowed(self, adapter):
        """A netlist starting with a device line must keep that device.

        SPICE consumes line 1 as the title. Without the adapter's unconditional
        title, this deck loses V1 and silently returns 0.0 V everywhere.
        """
        headless = "V1 vdd 0 DC 1.8\nR1 vdd out 10k\nR2 out 0 10k\n.op\n.end\n"
        result = adapter.dc(headless, SimParams(analysis_type="dc"))
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)

    def test_missing_dot_end_is_supplied(self, adapter):
        no_end = "* no end\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\nR2 out 0 10k\n.op\n"
        result = adapter.dc(no_end, SimParams(analysis_type="dc"))
        assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)

    def test_analysis_card_synthesised_from_params(self, adapter):
        no_card = "* rc\nV1 in 0 AC 1 DC 0\nR1 in out 1k\nC1 out 0 1n\n.end\n"
        result = adapter.ac(
            no_card,
            SimParams(analysis_type="ac", start=1.0, stop=1e8, points=100),
        )
        m = adapter.measure_ac(result, "out")
        assert m["bandwidth_3db"] == pytest.approx(RC_F3DB, rel=0.01)

    def test_refuses_to_invent_a_sweep(self, adapter):
        no_card = "* rc\nV1 in 0 AC 1 DC 0\nR1 in out 1k\nC1 out 0 1n\n.end\n"
        with pytest.raises(NgspiceError, match="Refusing to invent"):
            adapter.ac(no_card, SimParams(analysis_type="ac"))


@skipif_no_ngspice
class TestFailureDetection:

    def test_syntax_error_raises(self, adapter):
        broken = "* broken\nV1 vdd 0 DC 1.8\nR1 vdd out ten_k\n.op\n.end\n"
        with pytest.raises(NgspiceError):
            adapter.dc(broken, SimParams(analysis_type="dc"))

    def test_unknown_model_raises(self, adapter):
        broken = "* broken\nV1 d 0 DC 1\nM1 d d 0 0 nosuchmodel W=1u L=1u\n.op\n.end\n"
        with pytest.raises(NgspiceError):
            adapter.dc(broken, SimParams(analysis_type="dc"))

    def test_no_analysis_declared_raises(self, adapter):
        empty = "* nothing\nV1 a 0 DC 1\nR1 a 0 1k\n.end\n"
        with pytest.raises(NgspiceError):
            adapter.tran(empty, SimParams(analysis_type="tran"))

    def test_failed_run_does_not_return_the_previous_circuit(self, adapter):
        """The stale-plot trap: a broken deck must not inherit good numbers."""
        good = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        assert good.op_points["out"] == pytest.approx(0.9, abs=1e-12)

        broken = "* broken\nV1 alpha 0 DC 1.8\nR1 alpha beta ten_k\n.op\n.end\n"
        with pytest.raises(NgspiceError):
            adapter.dc(broken, SimParams(analysis_type="dc"))

        # And the adapter still works afterwards.
        again = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
        assert again.op_points["out"] == pytest.approx(0.9, abs=1e-12)


# ---------------------------------------------------------------------------
# Corners and Monte Carlo
# ---------------------------------------------------------------------------

@skipif_no_ngspice
class TestCornersAndMonteCarlo:

    def test_corners_vary_supply(self, adapter):
        corners = [PVTCorner(process="tt", voltage=1.8, temperature=27.0),
                   PVTCorner(process="ss", voltage=1.62, temperature=27.0)]
        results = adapter.corners(DIVIDER_OP, corners)
        assert len(results) == 2
        assert results[0].dc.op_points["out"] == pytest.approx(0.90, abs=1e-9)
        assert results[1].dc.op_points["out"] == pytest.approx(0.81, abs=1e-9)

    def test_corners_vary_temperature(self, adapter):
        """R1 has tc1=0.01, so heating it must move the divider ratio."""
        corners = [PVTCorner(process="tt", voltage=1.8, temperature=27.0),
                   PVTCorner(process="tt", voltage=1.8, temperature=125.0)]
        results = adapter.corners(TEMPCO_DIVIDER, corners)
        cold = results[0].dc.op_points["out"]
        hot = results[1].dc.op_points["out"]
        assert cold == pytest.approx(0.9, abs=1e-9)
        assert hot < cold - 0.2

    def test_corners_accept_plain_strings(self, adapter):
        """rl_env.py passes ['tt', 'ss', 'ff'], not PVTCorner objects."""
        results = adapter.corners(TEMPCO_DIVIDER, ["tt", "ss", "ff"])
        assert [r.corner.process for r in results] == ["tt", "ss", "ff"]
        outs = [r.dc.op_points["out"] for r in results]
        # Named corners still carry a temperature, so the results differ.
        assert len(set(round(v, 9) for v in outs)) == 3

    def test_mc_refuses_without_statistical_variation(self, adapter):
        with pytest.raises(NgspiceError, match="no source of statistical variation"):
            adapter.mc(DIVIDER_OP, 3, 42)

    def test_mc_with_netlist_defined_distribution(self, adapter):
        cir = """* agauss divider
.param rv=agauss(10k,1k,1)
V1 vdd 0 DC 1.8
R1 vdd out 'rv'
R2 out 0 10k
.op
.end
"""
        result = adapter.mc(cir, 6, 5)
        assert result.runs == 6
        assert result.seed == 5
        assert len(result.results) == 6
        outs = [row["op.out"] for row in result.results]
        assert len(set(outs)) > 1, "Monte Carlo produced identical runs"
        assert all(0.5 < v < 1.3 for v in outs)
        # Same seed, same draws.
        repeat = adapter.mc(cir, 6, 5)
        assert [row["op.out"] for row in repeat.results] == pytest.approx(outs)

    def test_mc_accepts_the_rl_env_args_dict(self, adapter):
        cir = """* agauss divider
.param rv=agauss(10k,1k,1)
V1 vdd 0 DC 1.8
R1 vdd out 'rv'
R2 out 0 10k
.op
.end
"""
        result = adapter.mc(cir, {"n": 3, "seed": 11})
        assert result.runs == 3
        assert result.seed == 11


# ---------------------------------------------------------------------------
# TSMC CRN65GPLUS. Gated: skips cleanly when the deck is not installed.
# NDA: these assert SIMULATED TERMINAL BEHAVIOUR ONLY. No model parameter
# value from the deck appears here, and none may be added.
# ---------------------------------------------------------------------------

@skipif_no_pdk
class TestTsmc65Pdk:

    INVERTER = """* CMOS inverter VTC, 65nm core devices
VDD vdd 0 DC 1.0
VIN in 0 DC 0
XM1 out in 0 0 nch_mac l=60n w=1u
XM2 out in vdd vdd pch_mac l=60n w=2u
.dc VIN 0 1.0 0.005
.end
"""

    NMOS_OP = """* nch_mac operating point
VDD d 0 DC 1.0
VG g 0 DC 1.0
XM1 d g 0 0 nch_mac l=60n w=1u
.op
.end
"""

    @staticmethod
    def _trip_point(result) -> float:
        """Where the inverter VTC crosses vout == vin."""
        sig = result.sweeps["out"]
        x, y = sig.x_values, sig.y_values
        for i in range(len(x) - 1):
            d0, d1 = y[i] - x[i], y[i + 1] - x[i + 1]
            if d0 == 0.0:
                return x[i]
            if d0 * d1 < 0:
                return x[i] + (-d0) * (x[i + 1] - x[i]) / (d1 - d0)
        raise AssertionError("inverter VTC never crosses vout == vin")

    def test_deck_resolves_to_an_ngspice_safe_path(self):
        deck = pdk_deck.ensure_local_deck("tsmc65")
        assert deck is not None and deck.is_file()
        assert " " not in str(deck)
        assert not str(deck).startswith("\\\\")

    def test_lib_lines_name_the_corner_section(self):
        lines = pdk_deck.lib_lines("tsmc65", "ss")
        assert any(line.startswith(".lib") and line.endswith(" SS") for line in lines)
        assert any(line.startswith(".param") for line in lines)

    def test_inverter_trip_point_is_plausible(self, pdk_adapter):
        result = pdk_adapter.dc(self.INVERTER, SimParams(analysis_type="dc"))
        trip = self._trip_point(result)
        # A 2:1 sized inverter on a 1.0 V supply should switch near mid rail.
        assert 0.35 < trip < 0.65, f"implausible trip point {trip}"

    def test_skew_corners_move_the_trip_point(self, pdk_adapter):
        """SF and FS must give measurably different switching thresholds."""
        sf = self._trip_point(pdk_adapter.dc(
            self.INVERTER, SimParams(analysis_type="dc", options={"corner": "sf"})))
        fs = self._trip_point(pdk_adapter.dc(
            self.INVERTER, SimParams(analysis_type="dc", options={"corner": "fs"})))
        assert abs(sf - fs) > 0.02, f"corners are not switching: sf={sf} fs={fs}"
        # Slow-N/fast-P pulls the threshold up, fast-N/slow-P pulls it down.
        assert sf > fs

    def test_slow_and_fast_corners_move_the_drive_current(self, pdk_adapter):
        def drive(corner: str) -> float:
            r = pdk_adapter.dc(
                self.NMOS_OP,
                SimParams(analysis_type="dc", options={"corner": corner}),
            )
            return abs(r.op_points["vdd#branch"])
        slow, typ, fast = drive("ss"), drive("tt"), drive("ff")
        assert slow < typ < fast, f"corner ordering wrong: {slow} {typ} {fast}"
        assert fast / slow > 1.2, "SS and FF are not measurably different"

    def test_nominal_runs_are_bit_reproducible(self, pdk_adapter):
        """sigma is pinned per instance, so mismatch cannot vary the nominal."""
        a = pdk_adapter.dc(self.NMOS_OP, SimParams(analysis_type="dc"))
        b = pdk_adapter.dc(self.NMOS_OP, SimParams(analysis_type="dc"))
        assert a.op_points["vdd#branch"] == b.op_points["vdd#branch"]

    def test_statistical_monte_carlo(self, pdk_adapter):
        result = pdk_adapter.mc(self.NMOS_OP, 5, 3)
        currents = [abs(row["op.vdd#branch"]) for row in result.results]
        assert len(currents) == 5
        assert len(set(currents)) == 5, "MC produced identical runs"
        mean = sum(currents) / len(currents)
        spread = (max(currents) - min(currents)) / mean
        assert 0.005 < spread < 0.60, f"implausible MC spread {spread}"
        repeat = pdk_adapter.mc(self.NMOS_OP, 5, 3)
        assert [abs(r["op.vdd#branch"]) for r in repeat.results] == currents

    def test_io_device_at_core_length_fails_loudly(self, pdk_adapter):
        """Thick oxide devices bin out below their minimum gate length."""
        cir = """* IO device with an illegal gate length
VDD vdd 0 DC 2.5
VIN in 0 DC 0
XM1 out in 0 0 nch_25_mac l=60n w=1u
RL vdd out 10k
.op
.end
"""
        with pytest.raises(NgspiceError) as exc:
            pdk_adapter.dc(cir, SimParams(analysis_type="dc"))
        # The message must not leak deck text while a PDK is loaded.
        assert "withheld" in str(exc.value)


@pytest.mark.skipif(not IMPORT_OK, reason="asic_ai not importable")
class TestPdkConfigDegradesCleanly:
    """The repo must stay functional and green with no PDK installed."""

    def test_unknown_pdk_is_not_available(self):
        assert pdk_deck.pdk_available("no_such_pdk") is False
        assert pdk_deck.get_pdk_config("no_such_pdk") is None
        assert pdk_deck.lib_lines("no_such_pdk", "tt") == []

    def test_missing_deck_yields_no_lib_lines(self, monkeypatch):
        monkeypatch.setenv("ASIC_AI_TSMC65_DECK", "/definitely/not/here/deck.l")
        pdk_deck.reload_config()
        try:
            assert pdk_deck.pdk_available("tsmc65") is False
            assert pdk_deck.lib_lines("tsmc65", "tt") == []
            assert pdk_deck.describe("tsmc65")["available"] is False
        finally:
            monkeypatch.delenv("ASIC_AI_TSMC65_DECK", raising=False)
            pdk_deck.reload_config()

    @skipif_no_ngspice
    def test_simulation_still_works_without_the_deck(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASIC_AI_TSMC65_DECK", "/definitely/not/here/deck.l")
        pdk_deck.reload_config()
        try:
            adapter = NgspiceSharedAdapter(
                AdapterConfig(binary_path="", work_dir=str(tmp_path)), pdk="tsmc65")
            assert adapter.pdk_ready() is False
            result = adapter.dc(DIVIDER_OP, SimParams(analysis_type="dc"))
            assert result.op_points["out"] == pytest.approx(0.9, abs=1e-12)
        finally:
            monkeypatch.delenv("ASIC_AI_TSMC65_DECK", raising=False)
            pdk_deck.reload_config()

    def test_corner_sections_are_the_core_ones(self):
        """The 1.0V core lives in the bare sections, not the *_18 family."""
        assert pdk_deck.corner_section("tsmc65", "tt") == "TT"
        assert pdk_deck.corner_section("tsmc65", "ff") == "FF"
        assert pdk_deck.ngbehavior("tsmc65") == "hsa"

    def test_devices_are_subcircuit_wrappers(self):
        devices = pdk_deck.device_names("tsmc65")
        assert devices["nmos"].endswith("_mac")
        assert devices["pmos"].endswith("_mac")

    def test_instance_params_pin_sigma(self):
        netlist = "XM1 d g 0 0 nch_mac l=60n w=1u\nR1 a b 1k\n"
        out = pdk_deck.apply_instance_params(netlist, "tsmc65")
        assert "sigma=0" in out.splitlines()[0]
        assert out.splitlines()[1] == "R1 a b 1k"

    def test_instance_params_do_not_double_apply(self):
        netlist = "XM1 d g 0 0 nch_mac l=60n w=1u sigma=1\n"
        out = pdk_deck.apply_instance_params(netlist, "tsmc65")
        assert out.count("sigma=") == 1


@skipif_no_ngspice
class TestContractParamNames:
    """The frozen contract's parameter names must reach the adapter.

    TOOL_DEFINITIONS documents sim.ac(start_freq, stop_freq,
    points_per_decade) and sim.tran(stop_time, step_time); until 2026-09-01
    _analysis_card read only start/stop/step/points, so a call written EXACTLY
    as the contract documents was refused with "params do not supply
    start/stop frequency". The 945ex eval hit that refusal 214 times.
    Reverting the alias fix makes each of these raise NgspiceError again.
    """

    RC = (
        "* RC low-pass, no analysis card\n"
        ".model nch nmos level=1 vto=0.5 kp=200u\n"
        "V1 in 0 DC 0 AC 1\n"
        "R1 in out 1k\n"
        "C1 out 0 1n\n"
        ".end\n"
    )
    RC_PULSE = (
        "* RC step response, no analysis card\n"
        "V1 in 0 PULSE(0 1 1u 1n 1n 1m 2m)\n"
        "R1 in out 1k\n"
        "C1 out 0 1n\n"
        ".end\n"
    )

    def _adapter(self, tmp_path):
        return NgspiceSharedAdapter(
            AdapterConfig(binary_path="", work_dir=str(tmp_path)))

    def test_ac_accepts_contract_names(self, tmp_path):
        r = self._adapter(tmp_path).ac(
            self.RC, {"start_freq": 10.0, "stop_freq": 1e8,
                      "points_per_decade": 5})
        assert len(r.frequencies) > 10
        assert any(len(s.y_values) == len(r.frequencies)
                   for s in r.signals.values())
        # the sweep actually honoured the contract bounds
        assert abs(r.frequencies[0] - 10.0) < 1.0
        assert r.frequencies[-1] <= 1e8 * 1.01

    def test_tran_accepts_contract_names(self, tmp_path):
        r = self._adapter(tmp_path).tran(
            self.RC_PULSE, {"stop_time": 20e-6, "step_time": 0.2e-6})
        assert len(r.time) > 10
        # RC tau = 1us: by 20us the step has fully settled to 1V
        vout = next(s for n, s in r.signals.items() if "out" in n.lower())
        assert abs(vout.y_values[-1] - 1.0) < 0.05

    def test_generic_spellings_still_work(self, tmp_path):
        r = self._adapter(tmp_path).ac(
            self.RC, {"start": 10.0, "stop": 1e8, "points": 5})
        assert len(r.frequencies) > 10


@skipif_no_ngspice
class TestWedgedEngineRecovery:
    """One bad deck must not poison the remaining 76 eval tasks.

    The 824g eval's first malformed deck left ngspice in 'cannot recover and
    awaits to be reset or detached'; every later run then failed with
    "there aren't any circuits loaded" -- 32 cascade failures before the run
    was killed. Recovery is a full DLL reload, triggered by that console
    marker. Reverting the reload call leaves test_run_path_reloads failing.
    """

    GOOD = "* ok\nV1 in 0 DC 1\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"

    def test_reload_produces_a_working_engine(self, tmp_path):
        from asic_ai.adapters.ngspice_shared import _NgspiceLibrary
        a = NgspiceSharedAdapter(
            AdapterConfig(binary_path="", work_dir=str(tmp_path)))
        before = _NgspiceLibrary._instance
        assert before is not None
        a._ng = _NgspiceLibrary.reload()
        a._lib = a._ng.dll
        assert _NgspiceLibrary._instance is not before, "must be a NEW engine"
        r = a.dc(self.GOOD, {})
        assert len(r.op_points) >= 2, "reloaded engine must actually simulate"

    def test_run_path_reloads_on_the_wedge_marker(self, tmp_path, monkeypatch):
        from asic_ai.adapters import ngspice_shared as ns
        a = NgspiceSharedAdapter(
            AdapterConfig(binary_path="", work_dir=str(tmp_path)))
        wedged_console = ["Error: ngspice.dll cannot recover and awaits to be "
                          "reset or detached"]
        monkeypatch.setattr(a._ng, "console_lines", lambda: wedged_console)
        reloaded = []
        monkeypatch.setattr(ns._NgspiceLibrary, "reload",
                            classmethod(lambda cls: reloaded.append(1) or a._ng))
        with pytest.raises(ns.NgspiceError):
            a.dc(self.GOOD, {})
        assert reloaded, "the wedge marker must trigger a DLL reload"
