"""Regression guards for 16 confirmed defects in the DERIVED METRIC layer.

Every test here reproduces the ORIGINAL probe that found the defect and would
fail against the code as it stood before the fix. The raw ctypes vector
extraction was never in doubt and is not retested here; what was broken is
everything built on top of it.

Why these matter more than an ordinary bug: the simulator is the reward source
for GRPO. A metric that is confidently wrong is worse than one that is missing,
because a missing metric is reported as unmeasurable and dropped, while a wrong
one trains the policy on a smooth, false gradient. Each test below therefore
asserts the correct NUMBER, not merely "is not None".

Reference values are analytic. Where the write-up that reported a defect quoted
an approximate expectation, the exact value is derived here and the difference
is explained in the test.
"""
from __future__ import annotations

import logging
import math

import pytest

try:
    from asic_ai.adapters import measure, pdk_deck
    from asic_ai.adapters.ngspice_shared import (
        NgspiceSharedAdapter, find_ngspice_dll, _dc_sweep_axis,
    )
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters import spec_extract as sx
    from asic_ai.tool_interface.schema import SimParams
    HAS_NGSPICE = find_ngspice_dll() is not None
except ImportError:  # pragma: no cover
    HAS_NGSPICE = False

skipif_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE,
                                       reason="ngspice DLL not found")

TWOPI = 2.0 * math.pi

# The AC-coupling corner of a 1 uF cap into 1 kOhm, and the amplifier poles.
FZ = 1.0 / (TWOPI * 1e-6 * 1e3)          # 159.1549431 Hz
FP = 159154.9430918954                    # 159.15 kHz
RC_FP = 1591.5494309189535                # 100 nF into 1 kOhm


@pytest.fixture
def adapter(tmp_path):
    return NgspiceSharedAdapter(AdapterConfig(binary_path="",
                                              work_dir=str(tmp_path)))


def _log_sweep(fstart: float, fstop: float, per_dec: int) -> list[float]:
    n = int(round(math.log10(fstop / fstart) * per_dec))
    return [fstart * 10.0 ** (i / per_dec) for i in range(n + 1)]


def _response(freqs, hfun):
    gain_db, phase_deg = [], []
    for f in freqs:
        h = hfun(f)
        gain_db.append(20.0 * math.log10(abs(h)))
        phase_deg.append(math.degrees(math.atan2(h.imag, h.real)))
    return gain_db, phase_deg


def _two_zero_three_pole(f, K=10000.0):
    """Non-inverting, two C-R coupling zeros at FZ, three poles at FP."""
    u = 1j * f / FZ
    v = 1j * f / FP
    return K * (u / (1 + u)) ** 2 / (1 + v) ** 3


def _ac_coupled_amp(f, K=100.0):
    """One coupling zero at FZ, three poles at FP. K = 100 (40 dB mid-band)."""
    u = 1j * f / FZ
    v = 1j * f / FP
    return K * (u / (1 + u)) / (1 + v) ** 3


def _single_pole_lowpass(f, fp=RC_FP):
    return 1.0 / (1 + 1j * f / fp)


# ===========================================================================
# D1  Phase margin must not depend on where the sweep starts
# ===========================================================================

class TestD1PhaseMarginIsSweepStartInvariant:
    """normalize_dc_phase inferred "this loop is inverting" from phase[0].

    A real phase lead at the bottom of the sweep (two AC-coupling capacitors,
    a textbook multistage amplifier) was read as an inversion and 180 deg was
    subtracted from the whole curve. Before the fix, only f_start varied and
    the answer moved by exactly 180 deg:

        f_start 10 Hz   raw phase[0] = +172.799  ->  PM -262.0135  GM -79.978
        f_start 100 Hz  raw phase[0] = +115.608  ->  PM -262.0135  GM -79.978
        f_start 159 Hz  raw phase[0] =  +89.884  ->  PM  -82.0135  GM -61.921
        f_start 1000 Hz raw phase[0] =  +17.006  ->  PM  -82.0135  GM -61.921

    The gain data is identical in all four rows (ugb is bit-identical), so only
    the phase bookkeeping changed. -82.0135 deg is the correct answer.
    """

    FSTARTS = (10.0, 100.0, 159.0, 1000.0)
    PM_EXPECTED = -82.0135
    GM_EXPECTED = -61.9208

    def _metrics(self, fstart):
        freqs = _log_sweep(fstart, 1e9, 400)
        gain_db, phase = _response(freqs, _two_zero_three_pole)
        return measure.ac_metrics(freqs, gain_db, phase), phase[0]

    def test_every_sweep_start_gives_the_same_phase_margin(self):
        margins, raw_first = [], []
        for fstart in self.FSTARTS:
            m, p0 = self._metrics(fstart)
            margins.append(m["phase_margin"])
            raw_first.append(p0)
        # The raw first sample really does span 156 deg, so this is the case
        # that used to split the answer in two.
        assert max(raw_first) - min(raw_first) > 150.0
        assert max(margins) - min(margins) < 1e-3, margins
        for pm in margins:
            assert pm == pytest.approx(self.PM_EXPECTED, abs=1e-3)

    def test_every_sweep_start_gives_the_same_gain_margin(self):
        gms = [self._metrics(f)[0]["gain_margin"] for f in self.FSTARTS]
        assert max(gms) - min(gms) < 1e-3, gms
        for gm in gms:
            assert gm == pytest.approx(self.GM_EXPECTED, abs=1e-3)

    def test_no_inversion_is_inferred_for_a_non_inverting_amplifier(self):
        for fstart in self.FSTARTS:
            m, _ = self._metrics(fstart)
            assert m["phase_inversion_k"] == 0.0

    def test_inversion_is_still_removed_when_it_is_real(self):
        """A genuinely inverting loop must still be normalised."""
        freqs = _log_sweep(1.0, 1e7, 100)
        gain_db, phase = _response(
            freqs, lambda f: -1000.0 / (1 + 1j * f / 1000.0))
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["phase_inversion_k"] == pytest.approx(1.0)
        # Without the shift this reads 180 deg larger and looks stable.
        assert m["phase_margin"] == pytest.approx(90.0, abs=0.5)

    def test_phase_inversion_shift_refuses_an_ambiguous_reference(self):
        """A reference phase near -90 deg is not evidence of an inversion."""
        assert measure.phase_inversion_shift([-90.0]) == 0
        assert measure.phase_inversion_shift([-100.0]) == 0
        assert measure.phase_inversion_shift([-175.0]) == -1
        assert measure.phase_inversion_shift([178.0]) == 1


# ===========================================================================
# D2  .dc on anything but a voltage source returned a fabricated x axis
# ===========================================================================

class TestD2SweepAxis:
    """`x_name = max(vecs, key=len)` cannot discriminate in a DC sweep.

    Every vector has the SAME length there, so max() returned whichever key
    ngSpice_AllVecs happened to list first. Only "v-sweep" was recognised;
    ngspice also emits "i-sweep", "temp-sweep" and "res-sweep".
    """

    def test_axis_is_taken_from_the_designated_vector_not_the_longest(self):
        # All four vectors are the same length, which is exactly why a length
        # heuristic is meaningless here.
        vecs = {"v1#branch": [0.0] * 5, "out": [0.0] * 5, "vdd": [0.0] * 5,
                "temp-sweep": [-40.0, 0.0, 40.0, 80.0, 120.0]}
        assert _dc_sweep_axis(vecs) == "temp-sweep"
        assert max(vecs, key=lambda k: len(vecs[k])) == "v1#branch"

    @pytest.mark.parametrize("name", ["v-sweep", "i-sweep", "temp-sweep",
                                      "res-sweep"])
    def test_all_four_sweep_kinds_are_recognised(self, name):
        assert _dc_sweep_axis({"out": [0.0], name: [0.0]}) == name

    def test_no_designated_axis_means_no_fabricated_axis(self):
        assert _dc_sweep_axis({"out": [0.0], "vdd": [0.0]}) is None


@skipif_no_ngspice
class TestD2SweepAxisAgainstNgspice:

    TEMPCO = """* tempco divider, R1 tc1=0.002
V1 vdd 0 DC 1
R1 vdd out 1k tc1=0.002
R2 out 0 1k
.dc temp -40 120 40
.end
"""

    ISWEEP = """* current sweep into 1k
I1 0 out DC 0
R1 out 0 1k
.dc i1 0 1m 0.25m
.end
"""

    def test_temperature_sweep_x_axis_is_temperature_in_degc(self, adapter):
        """Before: x_values was v1#branch, in AMPERES, and nearly constant.

        Any consumer computing dV/dT divided by zero. The y values were right
        all along: R0*(1 + tc1*dT) gives 0.5359056806 at -40 C, both ways.
        """
        result = adapter.dc(self.TEMPCO, SimParams(analysis_type="dc"))
        sig = result.sweeps["out"]
        assert sig.x_values == pytest.approx([-40.0, 0.0, 40.0, 80.0, 120.0])
        expected = [1.0 / (1.0 + (1.0 + 0.002 * (t - 27.0)))
                    for t in (-40.0, 0.0, 40.0, 80.0, 120.0)]
        assert sig.y_values == pytest.approx(expected, rel=1e-9)
        assert sig.y_values[0] == pytest.approx(0.5359056806, rel=1e-9)
        # And a real temperature coefficient can now be computed.
        dvdt = (sig.y_values[-1] - sig.y_values[0]) / (sig.x_values[-1]
                                                       - sig.x_values[0])
        assert dvdt == pytest.approx(-4.9e-4, rel=0.05)

    def test_temperature_sweep_no_longer_demotes_the_axis_to_a_signal(self, adapter):
        result = adapter.dc(self.TEMPCO, SimParams(analysis_type="dc"))
        assert "temp-sweep" not in result.sweeps
        assert "v1#branch" in result.sweeps      # a real signal, restored

    def test_current_sweep_is_not_transposed(self, adapter):
        """Before: the only signal was 'i-sweep' plotted against v(out), and
        sweeps['out'] raised KeyError."""
        result = adapter.dc(self.ISWEEP, SimParams(analysis_type="dc"))
        assert "i-sweep" not in result.sweeps
        sig = result.sweeps["out"]
        assert sig.x_values == pytest.approx([0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3])
        assert sig.y_values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


# ===========================================================================
# D3  ugb and phase margin were suppressed for any AC-coupled amplifier
# D5  bandwidth_3db was referenced to the first sweep point
# ===========================================================================

class TestD3D5AcCoupledAmplifier:
    """1 uF input coupling cap (corner 159 Hz), K = 100, three poles at 159 kHz.

    An ordinary AC-coupled stage, swept 0.1 Hz to 1 GHz:

        gain[0] = -24.0364 dB   max gain = 39.9850 dB   gain[-1] = -187.8908 dB

    Before the fix the guard was keyed on gain_db[0] rather than max(gain_db):

        ugb          = None       phase_margin = None
        gain_margin  = -21.929    f_180        = 275787.25   <- STILL COMPUTED

    so the returned dict reported a gain margin while refusing a phase margin
    for the same response, which crosses 0 dB unambiguously.
    """

    FREQS = _log_sweep(0.1, 1e9, 400)

    @pytest.fixture
    def metrics(self):
        gain_db, phase = _response(self.FREQS, _ac_coupled_amp)
        return measure.ac_metrics(self.FREQS, gain_db, phase), gain_db

    def test_the_probe_is_the_one_that_was_reported(self, metrics):
        _, gain_db = metrics
        assert gain_db[0] == pytest.approx(-24.0364, abs=1e-3)
        assert max(gain_db) == pytest.approx(39.9850, abs=1e-3)
        assert gain_db[-1] == pytest.approx(-187.8908, abs=1e-3)

    def test_ugb_is_found_although_the_sweep_starts_below_0_db(self, metrics):
        m, _ = metrics
        # Exact value from a bisection on the analytic response: 721383.6548.
        assert m["ugb"] == pytest.approx(721383.65, rel=1e-5)

    def test_phase_margin_is_reported(self, metrics):
        m, _ = metrics
        # Analytic phase at the exact ugb is +127.3373 deg principal value,
        # i.e. -232.6627 unwrapped, so PM = -52.6627 deg.
        assert m["phase_margin"] == pytest.approx(-52.6627, abs=0.01)

    def test_gain_margin_and_phase_margin_are_now_consistent(self, metrics):
        m, _ = metrics
        assert m["gain_margin"] == pytest.approx(-21.9295, abs=1e-3)
        assert m["f_180"] == pytest.approx(275787.36, rel=1e-6)
        # The old dict reported one and refused the other for the same curve.
        assert (m["gain_margin"] is None) == (m["phase_margin"] is None)

    def test_bandwidth_is_referenced_to_the_passband_not_to_f_start(self, metrics):
        m, _ = metrics
        # Before: 2079683.12 Hz, because the -3 dB level was taken from the
        # -24 dB gain at the bottom of the sweep. That is 25.6x too high.
        #
        # The exact upper -3 dB edge, referenced to the TRUE peak gain of
        # 39.9849643 dB, is 81367.61 Hz (bisection on the analytic response).
        # The original write-up quoted 81141 Hz from (1+u^2)^1.5 = sqrt(2),
        # which references an idealised 40 dB instead of the actual peak; that
        # 0.015 dB difference moves the edge by 0.27 pct on a -37 dB/dec slope.
        assert m["bandwidth_3db"] == pytest.approx(81367.61, rel=1e-4)
        assert m["f_3db_hi"] == pytest.approx(81367.61, rel=1e-4)
        assert m["bandwidth_3db"] < 100e3

    def test_both_band_edges_are_reported_for_a_band_pass_response(self, metrics):
        m, _ = metrics
        assert m["f_3db_lo"] == pytest.approx(158.6063, rel=1e-4)
        assert "BAND-PASS" in m["notes"]["bandwidth_3db"]
        assert "UPPER" in m["notes"]["bandwidth_3db"]

    def test_low_frequency_gain_is_kept_but_not_called_a_dc_gain(self, metrics):
        m, _ = metrics
        assert m["low_freq_gain_db"] == pytest.approx(-24.0364, abs=1e-3)
        assert m["dc_gain_db"] is None
        assert m["dc_gain_valid"] == 0.0
        assert "does not reach DC" in m["notes"]["dc_gain_db"]
        assert m["passband_gain_db"] == pytest.approx(39.9850, abs=1e-3)

    def test_a_unity_gain_cascade_is_not_defeated_by_negative_zero(self):
        """-0.0 <= 0.0 is True, which used to suppress ugb on its own."""
        freqs = [1.0, 10.0, 100.0]
        gain = [-0.0, -0.0, -0.0]
        m = measure.ac_metrics(freqs, gain, [0.0, 0.0, 0.0])
        # Exactly 0 dB everywhere really has no unity-gain CROSSING, but the
        # refusal must name the peak gain, not the first sample.
        assert m["ugb"] is None
        assert "peak gain" in m["notes"]["ugb"]

    def test_gain_below_unity_everywhere_still_refuses(self):
        m = measure.ac_metrics([1.0, 10.0, 100.0], [-20.0, -30.0, -40.0],
                               [0.0, -45.0, -90.0])
        assert m["ugb"] is None
        assert m["phase_margin"] is None


@skipif_no_ngspice
class TestD3StbMessage:

    AC_COUPLED_LOOP = """* AC-coupled loop, 40 dB mid-band, single pole
Vin in 0 DC 0 AC 1
Cc in mid 1u
Rc mid 0 1k
Eamp amp 0 mid 0 100
Rp amp out 1k
Cp out 0 1n
.ac dec 200 0.1 1e9
.end
"""

    def test_stb_no_longer_contradicts_itself(self, adapter):
        """The old refusal said "never crosses 0 dB (max gain 39.98 dB)".

        40 dB of mid-band gain and one pole at 159154.94 Hz put the unity-gain
        frequency at fp*sqrt(100^2 - 1) = 15914698.5 Hz, where the pole has
        contributed -atan(99.995) = -89.4271 deg, so the phase margin is
        90.5735 deg. Before the fix this deck raised instead.
        """
        result = adapter.stb(self.AC_COUPLED_LOOP,
                             SimParams(analysis_type="ac",
                                       options={"loop_out": "out",
                                                "loop_in": "in"}))
        assert result.phase_margin == pytest.approx(90.5735, abs=0.01)
        assert math.isinf(result.gain_margin)


# ===========================================================================
# D4  dc_gain_valid was a function of sweep density, not of reaching DC
# ===========================================================================

class TestD4DcGainValid:
    """RC low-pass, pole 1591.549 Hz, swept `.ac lin 20001 100000 200000`.

    The whole sweep is ABOVE the pole. Before the fix:

        dc_gain_db    = -35.9647   dc_gain_valid = 1.0   (true DC gain 0 dB)
        bandwidth_3db =  141430.31 Hz                    (true 1591.5494 Hz)

    Adjacent samples on ANY fine sweep agree to less than 0.01 dB, so the only
    guard the API offered could never fail. It read "valid" while dc_gain_db
    was 36 dB wrong. That is the one metric consumers were told to trust.
    """

    @pytest.fixture
    def metrics(self):
        freqs = [1e5 + i * (1e5 / 20000.0) for i in range(20001)]
        gain_db, phase = _response(freqs, _single_pole_lowpass)
        return measure.ac_metrics(freqs, gain_db, phase), freqs, gain_db

    def test_adjacent_samples_still_agree_to_better_than_a_hundredth_of_a_db(
            self, metrics):
        """The old test would still pass here. That is the point."""
        _, _, gain_db = metrics
        assert abs(gain_db[0] - gain_db[1]) < 0.01

    def test_a_sweep_entirely_above_the_pole_is_not_a_valid_dc_gain(self, metrics):
        m, _, _ = metrics
        assert m["dc_gain_valid"] == 0.0
        assert m["dc_gain_db"] is None
        assert m["low_freq_gain_db"] == pytest.approx(-35.9647, abs=1e-3)
        assert m["low_slope_db_per_dec"] == pytest.approx(-20.0, abs=0.1)

    def test_bandwidth_is_refused_rather_than_reported_89x_high(self, metrics):
        m, _, _ = metrics
        assert m["bandwidth_3db"] is None
        note = m["notes"]["bandwidth_3db"]
        assert "refused" in note and "passband lies below f_start" in note

    def test_a_sweep_that_does_reach_dc_is_still_accepted(self):
        freqs = _log_sweep(1.0, 1e6, 100)
        gain_db, phase = _response(freqs, _single_pole_lowpass)
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["dc_gain_valid"] == 1.0
        assert m["dc_gain_db"] == pytest.approx(0.0, abs=1e-4)
        assert m["bandwidth_3db"] == pytest.approx(RC_FP, rel=1e-3)

    def test_the_guard_is_a_slope_not_a_sample_spacing(self):
        """Same circuit, same band, 100x the density: same verdict."""
        verdicts = []
        for n in (201, 20001):
            freqs = [1e5 + i * (1e5 / (n - 1)) for i in range(n)]
            gain_db, _ = _response(freqs, _single_pole_lowpass)
            verdicts.append(measure.ac_metrics(freqs, gain_db)["dc_gain_valid"])
        assert verdicts == [0.0, 0.0]


# ===========================================================================
# D6  SS and FF had their temperatures swapped
# ===========================================================================

class TestD6CornerConvention:
    """SS = slow process / LOW VDD / 125 C, FF = fast / HIGH VDD / -40 C.

    The voltages were already right (ss 0.9, ff 1.1 on tsmc65) but the
    temperatures were inverted, so within each corner process+voltage and
    temperature pushed in OPPOSITE directions and the corner spread was
    partially cancelled. The error was consistent across every site, which
    makes it a convention error rather than a typo.
    """

    def test_pdk_knowledge_tables(self):
        from asic_ai.data.pdk_knowledge import get_pdk_params
        for pdk in ("sky130", "gf180mcu", "tsmc65"):
            corners = get_pdk_params(pdk)["corners"]
            assert corners["ss"]["temperature"] == 125, pdk
            assert corners["ff"]["temperature"] == -40, pdk
            assert corners["tt"]["temperature"] == 27, pdk
            # And every axis of the corner pushes the same way.
            assert corners["ss"]["voltage"] < corners["tt"]["voltage"], pdk
            assert corners["ff"]["voltage"] > corners["tt"]["voltage"], pdk

    @pytest.mark.parametrize("pdk", ["tsmc65", "sky130"])
    def test_pdk_deck_corner_pvt(self, pdk):
        assert pdk_deck.corner_pvt(pdk, "ss")["temperature"] == 125.0
        assert pdk_deck.corner_pvt(pdk, "ff")["temperature"] == -40.0

    def test_generic_corner_temperatures_in_the_adapter(self):
        from asic_ai.adapters.ngspice_shared import _GENERIC_CORNER_TEMPS
        assert _GENERIC_CORNER_TEMPS["ss"] == 125.0
        assert _GENERIC_CORNER_TEMPS["ff"] == -40.0
        assert _GENERIC_CORNER_TEMPS["tt"] == 27.0

    def test_eda_tools_yaml(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "configs" / "eda_tools.yaml").read_text(
            encoding="utf-8"))
        found = list(_iter_corner_pvt(cfg))
        assert found, "no corner_pvt block found in configs/eda_tools.yaml"
        for block in found:
            assert block["ss"]["temperature"] == 125.0
            assert block["ff"]["temperature"] == -40.0


def _iter_corner_pvt(node):
    if isinstance(node, dict):
        for key, val in node.items():
            if key == "corner_pvt" and isinstance(val, dict):
                yield val
            else:
                yield from _iter_corner_pvt(val)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_corner_pvt(item)


@skipif_no_ngspice
class TestD6CornerConventionIsApplied:

    TEMPCO = """* tempco divider
V1 vdd 0 DC 1.8
R1 vdd out 10k tc1=0.01
R2 out 0 10k
.op
.end
"""

    def test_named_ss_corner_runs_hot(self, adapter):
        """corners(NL, ['ss']) used to apply T = -40 C."""
        results = adapter.corners(self.TEMPCO, ["ss"])
        assert results[0].corner.temperature == 125.0

    def test_named_ff_corner_runs_cold(self, adapter):
        results = adapter.corners(self.TEMPCO, ["ff"])
        assert results[0].corner.temperature == -40.0

    def test_ss_and_ff_still_bracket_tt(self, adapter):
        outs = {r.corner.process: r.dc.op_points["out"]
                for r in adapter.corners(self.TEMPCO, ["tt", "ss", "ff"])}
        assert outs["ss"] < outs["tt"] < outs["ff"]


# ===========================================================================
# D7  measure_idd counted any branch whose source name begins with "v"
# ===========================================================================

class TestD7SupplyCurrent:
    """A 1 mA current-source-biased stage with a 0 V ammeter beside it.

        I1 0 vdd DC 1m      the actual 1 mA supply
        R1 vdd 0 1k
        Vsense vdd n2 DC 0  a 0 V ammeter, standard practice
        R2 n2 0 1meg

    Before: measure_idd -> 9.99e-07 against a true supply current of 1.0e-03.
    Off by 1000x, silently.
    """

    OP = {"vsense#branch": 9.99000999000999e-07,
          "n2": 0.999000999000999,
          "vdd": 0.999000999000999}

    DECK = """* current-source biased stage with a 0 V sense source
I1 0 vdd DC 1m
R1 vdd 0 1k
Vsense vdd n2 DC 0
R2 n2 0 1meg
.op
.end
"""

    def test_a_zero_volt_sense_source_is_not_a_supply(self):
        rep = measure.supply_current_report(self.OP, netlist=self.DECK)
        assert rep.value is None
        assert "vsense" in rep.excluded
        assert "sense source" in rep.excluded["vsense"]

    def test_a_biasing_current_source_is_called_out(self):
        rep = measure.supply_current_report(self.OP, netlist=self.DECK)
        joined = " ".join(rep.warnings)
        assert "i1" in joined
        assert "no branch vector" in joined
        assert rep.ambiguous

    def test_without_a_netlist_the_answer_is_flagged_not_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            value = measure.supply_current(self.OP)
        assert value == pytest.approx(9.99000999e-07)
        assert "no netlist was supplied" in caplog.text
        assert "sense source" in caplog.text

    def test_an_ordinary_rail_is_unaffected_and_unflagged(self):
        rep = measure.supply_current_report(
            {"v1#branch": -9e-05, "out": 0.9, "vdd": 1.8},
            netlist="* d\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\nR2 out 0 10k\n.end\n")
        assert rep.value == pytest.approx(9e-05, rel=1e-12)
        assert rep.warnings == []
        assert rep.sources == ["v1#branch".split("#")[0]]

    def test_an_explicit_source_list_always_wins(self):
        assert measure.supply_current(self.OP, sources=["vsense"]) == \
            pytest.approx(9.99000999e-07)

    def test_named_source_that_is_absent_is_reported_not_invented(self):
        rep = measure.supply_current_report({"v1#branch": -1e-3},
                                            sources=["vdd"])
        assert rep.value is None
        assert "vdd" in " ".join(rep.warnings)

    def test_ac_only_stimulus_is_excluded_but_a_parameterised_rail_is_not(self):
        deck = ("* x\nVin in 0 AC 1\nV1 vdd 0 DC {vsup}\n"
                "R1 vdd 0 1k\n.end\n")
        rep = measure.supply_current_report(
            {"vin#branch": 0.0, "v1#branch": -1.8e-3}, netlist=deck)
        assert rep.sources == ["v1"]
        assert rep.value == pytest.approx(1.8e-3)
        assert "vin" in rep.excluded

    def test_multiple_rails_are_summed_but_flagged(self):
        rep = measure.supply_current_report({"v1#branch": -1e-3,
                                             "v2#branch": 1e-3})
        assert rep.value == pytest.approx(2e-3, rel=1e-12)
        assert any("summed" in w for w in rep.warnings)


@skipif_no_ngspice
class TestD7SupplyCurrentAgainstNgspice:

    def test_measure_idd_refuses_the_sense_source(self, adapter):
        result = adapter.dc(TestD7SupplyCurrent.DECK,
                            SimParams(analysis_type="dc"))
        assert result.op_points["vsense#branch"] == pytest.approx(
            9.99000999e-07, rel=1e-6)
        assert adapter.measure_idd(result) is None

    def test_measure_idd_still_works_on_an_ordinary_divider(self, adapter):
        deck = ("* divider\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\n"
                "R2 out 0 10k\n.op\n.end\n")
        result = adapter.dc(deck, SimParams(analysis_type="dc"))
        assert adapter.measure_idd(result) == pytest.approx(90e-6, rel=1e-9)

    def test_a_stale_netlist_is_never_applied_to_an_older_result(self, adapter):
        deck = ("* divider\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\n"
                "R2 out 0 10k\n.op\n.end\n")
        result = adapter.dc(deck, SimParams(analysis_type="dc"))
        adapter.dc(TestD7SupplyCurrent.DECK, SimParams(analysis_type="dc"))
        # v1 is not an element of the second deck, so it is not applied.
        assert adapter.measure_idd(result) == pytest.approx(90e-6, rel=1e-9)


# ===========================================================================
# L1 .. L7
# ===========================================================================

class TestLowSeverity:

    def test_l1_crossing_freq_does_not_skip_a_flat_reach(self):
        """[10, 0, 0, -10] reaches 0 dB at f = 10, not at f = 100."""
        got = measure.crossing_freq([1.0, 10.0, 100.0, 1000.0],
                                    [10.0, 0.0, 0.0, -10.0], 0.0, direction=-1)
        assert got == pytest.approx(10.0)

    def test_l1_a_gain_curve_that_flattens_at_0_db_reports_the_first_reach(self):
        freqs = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        gain = [20.0, 0.0, 0.0, 0.0, -20.0]
        assert measure.ac_metrics(freqs, gain)["ugb"] == pytest.approx(10.0)

    def test_l2_settling_time_is_none_when_the_waveform_never_settles(self):
        """A square wave used to return t[-1], reading as "settles at the end"."""
        assert measure.settling_time(list(range(10)), [0, 1] * 5) is None

    def test_l2_a_single_in_band_sample_is_not_evidence_of_settling(self):
        t = [float(i) for i in range(10)]
        y = [0.0] * 9 + [1.0]             # exactly one sample inside the band
        assert measure.settling_time(t, y) is None
        # Two consecutive in-band samples IS evidence, and is still reported.
        assert measure.settling_time(t, [0.0] * 8 + [1.0, 1.0]) ==             pytest.approx(8.0)

    def test_l2_a_genuinely_settled_waveform_still_reports_a_time(self):
        t = [float(i) for i in range(20)]
        y = [0.0] * 5 + [1.0] * 15
        assert measure.settling_time(t, y) == pytest.approx(5.0)

    def test_l3_step_at_t0_matches_the_delayed_step(self):
        """Identical series RLC, zeta = 0.5, analytic overshoot 16.30335 pct.

        Before: the delayed step gave 16.303358 pct (exact) and the step at
        t = 0 gave 19.91 pct with y_initial 0.181 instead of 0.0 and a rise
        time 20 pct short, with nothing to distinguish the two cases.
        """
        zeta, wn = 0.5, TWOPI * 1e5
        wd = wn * math.sqrt(1 - zeta ** 2)

        def step(x):
            if x < 0:
                return 0.0
            return 1.0 - math.exp(-zeta * wn * x) * (
                math.cos(wd * x)
                + zeta / math.sqrt(1 - zeta ** 2) * math.sin(wd * x))

        n = 4001
        t = [i * 100e-6 / (n - 1) for i in range(n)]
        delayed = measure.tran_metrics(t, [step(x - 20e-6) for x in t])
        at_zero = measure.tran_metrics(t, [step(x) for x in t])

        analytic = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))
        assert analytic == pytest.approx(16.30335, abs=1e-4)
        assert delayed["overshoot_pct"] == pytest.approx(analytic, rel=1e-6)
        assert at_zero["overshoot_pct"] == pytest.approx(analytic, rel=2e-3)
        assert at_zero["y_initial"] == pytest.approx(0.0, abs=1e-3)
        assert at_zero["rise_time"] == pytest.approx(delayed["rise_time"],
                                                     rel=2e-3)

    def test_l3_a_noisy_quiescent_window_keeps_its_averaging(self):
        """The quiescence test is on DRIFT, so zero-mean noise does not trip it."""
        import random
        rng = random.Random(7)
        n = 4000
        y = [(0.0 if i < n // 2 else 1.0) + rng.gauss(0, 0.01)
             for i in range(n)]
        y0, y1 = measure.settled_levels(y)
        assert y0 == pytest.approx(0.0, abs=3e-3)
        assert y1 == pytest.approx(1.0, abs=3e-3)

    def test_l4_db20_of_an_exact_zero_is_minus_infinity(self):
        assert measure.db20(0.0) == -math.inf
        assert measure.db20(1.0) == 0.0
        assert measure.db20(0.0, floor=1e-300) == pytest.approx(-6000.0)

    def test_l4_a_zero_divisor_does_not_fabricate_a_phase(self):
        gain, phase = measure.transfer_function([complex(1.0, 1.0)],
                                                [complex(0.0, 0.0)])
        assert math.isnan(gain[0])
        assert math.isnan(phase[0])

    def test_l4_a_nan_sample_does_not_poison_the_unwrap(self):
        out = measure.unwrap_deg([0.0, math.nan, -90.0, -179.0, 179.0])
        assert math.isnan(out[1])
        assert out[-1] == pytest.approx(-181.0, abs=1e-9)

    def test_l4_a_nan_sample_does_not_poison_ac_metrics(self):
        freqs = [1.0, 10.0, 100.0, 1000.0]
        gain = [20.0, math.nan, 10.0, -10.0]
        m = measure.ac_metrics(freqs, gain, [0.0, math.nan, -45.0, -90.0])
        assert m["peak_gain_db"] == pytest.approx(20.0)
        assert m["ugb"] == pytest.approx(316.2277, rel=1e-3)

    def test_l5_value_at_freq_refuses_outside_the_swept_band(self):
        freqs, values = [1.0, 10.0, 1e3], [0.0, -10.0, -40.0]
        assert measure.value_at_freq(freqs, values, 1e9) is None
        assert measure.value_at_freq(freqs, values, 0.1) is None
        # Opt back in deliberately, and the old answer is still available.
        assert measure.value_at_freq(freqs, values, 1e9, clamp=True) == -40.0
        # Endpoints and interior points are unchanged.
        assert measure.value_at_freq(freqs, values, 1.0) == 0.0
        assert measure.value_at_freq(freqs, values, 1e3) == -40.0
        assert measure.value_at_freq(freqs, values, 100.0) == pytest.approx(-25.0)

    def test_l7_the_docstring_states_the_measured_bias(self):
        doc = " ".join((measure.__doc__ or "").split())
        assert "It is NOT exact for a single-pole rolloff" in doc
        assert "biased LOW" in doc
        assert "dec 5 -> -7.6e-4" in doc
        assert "does NOT average out" in doc

    def test_l7_the_bias_is_biased_low_and_shrinks_with_density(self):
        errs = []
        for per_dec in (5, 10, 100, 1000):
            freqs = _log_sweep(1.0, 1e6, per_dec)
            gain_db, _ = _response(freqs, _single_pole_lowpass)
            b = measure.ac_metrics(freqs, gain_db)["bandwidth_3db"]
            errs.append((b - RC_FP) / RC_FP)
        assert errs[0] == pytest.approx(-7.6e-4, rel=0.05)
        assert errs[1] == pytest.approx(-4.2e-4, rel=0.05)
        assert errs[2] == pytest.approx(-3.9e-5, rel=0.05)
        assert abs(errs[3]) < 1e-8
        # Systematic, one-sided: it does not average out across RL candidates.
        assert all(e < 0 for e in errs[:3])


@skipif_no_ngspice
class TestL6NodeEqualToTheSweepAxis:
    """Sweeping v1 on a divider, node 'vdd' vanished from sweeps entirely.

    The drop test was `y_values == x_values`, so any node that happens to equal
    the swept variable was deleted. Which node that is depends on topology, so
    it was silent data loss with a topology-dependent victim.
    """

    DECK = """* three-resistor divider swept on v1
V1 vdd 0 DC 1.8
R1 vdd a 10k
R2 a b 10k
R3 b 0 10k
.dc v1 0 1.8 0.45
.end
"""

    def test_the_supply_node_survives(self, adapter):
        result = adapter.dc(self.DECK, SimParams(analysis_type="dc"))
        assert "vdd" in result.sweeps
        sig = result.sweeps["vdd"]
        assert sig.x_values == pytest.approx(sig.y_values)
        assert sig.y_values == pytest.approx([0.0, 0.45, 0.9, 1.35, 1.8])

    def test_the_other_nodes_are_still_there_and_still_right(self, adapter):
        result = adapter.dc(self.DECK, SimParams(analysis_type="dc"))
        assert set(result.sweeps) == {"vdd", "a", "b", "v1#branch"}
        for x, y in zip(result.sweeps["a"].x_values,
                        result.sweeps["a"].y_values):
            assert y == pytest.approx(2.0 * x / 3.0, abs=1e-12)


# ===========================================================================
# Consumers downstream of every metric
# ===========================================================================

class TestConsumersStillWork:

    def test_spec_extract_reports_a_refusal_reason_not_a_wrong_number(self):
        from asic_ai.tool_interface.schema import ACResult, SignalData
        freqs = [1e5 + i * (1e5 / 200.0) for i in range(201)]
        gain_db, phase = _response(freqs, _single_pole_lowpass)
        ac = ACResult(
            frequencies=freqs,
            signals={
                "vdb(out)": SignalData(name="vdb(out)", x_values=freqs,
                                       y_values=gain_db),
                "vp(out)": SignalData(name="vp(out)", x_values=freqs,
                                      y_values=phase),
            },
        )
        ext = sx.extract_specs({"dc_gain": {"min": 40, "unit": "dB"}},
                               ac=ac, output_signal="out")
        assert "dc_gain" not in ext.values
        assert "does not reach DC" in ext.unmeasurable["dc_gain"]

    def test_spec_extract_reports_the_mid_band_gain_of_an_ac_coupled_stage(self):
        from asic_ai.tool_interface.schema import ACResult, SignalData
        freqs = _log_sweep(0.1, 1e9, 100)
        gain_db, phase = _response(freqs, _ac_coupled_amp)
        ac = ACResult(
            frequencies=freqs,
            signals={
                "vdb(out)": SignalData(name="vdb(out)", x_values=freqs,
                                       y_values=gain_db),
                "vp(out)": SignalData(name="vp(out)", x_values=freqs,
                                      y_values=phase),
            },
        )
        ext = sx.extract_specs(
            {"gain": {"min": 35, "unit": "dB"},
             "ugb": {"min": 1e5, "unit": "Hz"},
             "pm": {"min": -60, "unit": "deg"}},
            ac=ac, output_signal="out")
        # Before: gain was -24.04 dB (the bottom of the sweep) and both ugb
        # and pm were missing entirely.
        assert ext.values["gain"] == pytest.approx(39.985, abs=0.01)
        assert ext.values["ugb"] == pytest.approx(721383.65, rel=1e-3)
        assert ext.values["pm"] == pytest.approx(-52.66, abs=0.1)

    def test_spec_extract_passes_the_netlist_through_to_idd(self):
        from asic_ai.tool_interface.schema import DCResult
        dc = DCResult(op_points=TestD7SupplyCurrent.OP, sweeps={})
        ext = sx.extract_specs({"idd": {"max": 2000, "unit": "uA"}}, dc=dc,
                               netlist=TestD7SupplyCurrent.DECK)
        assert "idd" not in ext.values
        assert "no branch vector" in ext.unmeasurable["idd"]

    def test_spec_extract_dimension_table_covers_every_alias(self):
        for metric in set(sx.ALIASES.values()):
            assert metric in sx.DIMENSIONS, metric
