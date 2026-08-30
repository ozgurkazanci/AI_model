"""Regression guards for the fourth pass: F1 through F9.

tests/test_metric_defects.py guards the 16 original defects and
tests/test_metric_regressions.py the ones the second and third passes
introduced. This file guards the fourth set, and three of its nine were
introduced by the pass that fixed the third set:

    F1  flat_band certified a resonant peak as the passband whenever the peak
        sat in the last span_dec decades       (introduced by R2)
    F2  the "did a level ever form" guard was on the step branch only, and the
        same circuit escaped it by changing its drive form   (R7 incomplete)
    F3  abs() removed the SIGN dependence of slew_rate but not the EDGE
        dependence, turning a false-fail into a false-pass  (introduced by R10)
    F4  `if twins and not netlist` deleted the only "exactly 2x" warning on a
        deck whose DC values cannot be read                 (introduced by R4)
    F5  ac_metrics raised ValueError on a NaN phase, reachable from a real deck
    F6  idd = 0.0 with an empty warning list -- a perfect score on a budget
    F7  a pulse with ringing: the region ended at the first ring trough
    F8  prop_delay refused with the wrong reason
    F9  `100.0e3` is a STRING under PyYAML, not a float

Every test below reproduces the probe that found the defect and asserts the
correct NUMBER, and every one of them FAILS against the code as it stood before
the corresponding fix -- that was checked by reverting each fix in a scratch
copy of src/ and running the test against it.

Two shapes repeat across the nine and both are guarded here by construction:

  A SELF-REFERENTIAL THRESHOLD -- a tolerance derived from the very quantity it
  is meant to validate. F1's flat-band tolerance scaled with the window span,
  so it vanished as the window shrank; F2's drift tolerance scaled with the
  excursion of the record, so it vanished as the record captured less of the
  step. Every threshold touched here is anchored to something the failure mode
  cannot move: a caller-supplied span floor, the fastest movement in the
  record, the stimulus, or an exact zero.

  A GUARD SUPPRESSED BECAUSE A TWIN WAS ASSUMED TO COVER IT. F4 suppressed the
  twins warning whenever a netlist was passed, on the strength of a polarity
  test that classifies by `> 0.0` / `< 0.0` -- and a NaN is neither.
"""
from __future__ import annotations

import math
import pathlib
import re

import pytest
import yaml

from asic_ai.adapters import measure
from asic_ai.adapters import spec_extract as sx

try:
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.ngspice_shared import (
        NgspiceSharedAdapter, find_ngspice_dll,
    )
    from asic_ai.tool_interface.schema import SimParams
    HAS_NGSPICE = find_ngspice_dll() is not None
except ImportError:  # pragma: no cover
    HAS_NGSPICE = False

skipif_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE,
                                       reason="ngspice DLL not found")

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter(tmp_path):
    return NgspiceSharedAdapter(AdapterConfig(binary_path="",
                                              work_dir=str(tmp_path)))


# ===========================================================================
# F5  ac_metrics raised ValueError on a NaN phase, reachable through stb()
# ===========================================================================

# f[0] == 0 makes low_frequency_slope return None, so `expected` is None and
# the 2*pi branch correction runs; on an AC-coupled response H(0) = 0, so
# transfer_function emits (-inf, NaN) there and int(round(NaN/360)) raises.
AC_COUPLED_FROM_DC = """* AC-coupled broken-loop testbench swept from 0 Hz
Vin in 0 DC 0 AC 1
Cc in mid 1u
R1 mid 0 15.9k
E1 amp 0 mid 0 100
R2 amp out 1k
C2 out 0 100n
.ac lin 201 0 1e6
.end
"""


def _ac_coupled_from_dc_response():
    """The same shape in pure Python: a zero at 10 Hz, swept from 0 Hz."""
    freqs = [0.0] + [1e6 * i / 200 for i in range(1, 201)]
    num = []
    for f in freqs:
        s = complex(0.0, 2.0 * math.pi * f)
        wz = 2.0 * math.pi * 10.0
        num.append(100.0 * (s / wz) / (1.0 + s / wz))
    gain_db, phase_deg = measure.transfer_function(num, None)
    return freqs, gain_db, phase_deg


class TestF5ANaNPhaseIsNotAnArithmeticError:
    """`k = 2 * int(round(ph[0] / 360.0))` on a sample with no phase."""

    def test_the_premise_the_first_sample_really_has_no_transfer_function(self):
        _, g, p = _ac_coupled_from_dc_response()
        assert g[0] == -float("inf")
        assert math.isnan(p[0])

    def test_ac_metrics_does_not_raise(self):
        f, g, p = _ac_coupled_from_dc_response()
        m = measure.ac_metrics(f, g, p)   # was ValueError: cannot convert NaN
        assert m["passband_gain_db"] == pytest.approx(40.0, abs=0.05)

    def test_the_zero_hz_sample_refuses_the_dc_gain_with_its_reason(self):
        """The twin of the `at_dc` guard: a 0 Hz sample with no magnitude.

        `if f[0] <= 0.0: reaches_dc = True` published -inf as the DC gain and
        made it the passband reference. The 0 Hz sample DOES answer the
        question -- it says the gain at DC is zero -- so this is a refusal with
        evidence, not an "I cannot tell".
        """
        f, g, p = _ac_coupled_from_dc_response()
        m = measure.ac_metrics(f, g, p)
        assert m["dc_gain_db"] is None
        assert m["dc_gain_valid"] == 0.0
        assert "|H(0)| = 0" in m["notes"]["dc_gain_db"]

    def test_a_dc_coupled_sweep_from_zero_hz_still_reaches_dc(self):
        """R1's fix must survive: a 0 Hz sample WITH a magnitude is the DC point."""
        freqs = [0.0] + [1e6 * i / 200 for i in range(1, 201)]
        num = [100.0 / (1.0 + complex(0.0, f / 1e4)) for f in freqs]
        g, p = measure.transfer_function(num, None)
        m = measure.ac_metrics(freqs, g, p)
        assert m["dc_gain_valid"] == 1.0
        assert m["dc_gain_db"] == pytest.approx(40.0, abs=1e-6)

    def test_extract_specs_does_not_raise_either(self):
        f, g, p = _ac_coupled_from_dc_response()
        ac = {"frequencies": f,
              "signals": {"vdb(out)": {"name": "vdb(out)", "x_values": f,
                                       "y_values": g},
                          "vp(out)": {"name": "vp(out)", "x_values": f,
                                      "y_values": p}}}
        ext = sx.extract_specs({"gain": {"min": 30.0, "unit": "dB"}}, ac=ac,
                               output_signal="out")
        assert ext.values["gain"] == pytest.approx(40.0, abs=0.05)


@skipif_no_ngspice
class TestF5BStbSurvivesASweepThatStartsAtZeroHz:
    """ngspice_shared._build_ac drops the f = 0 sample; stb() does not.

    stb() feeds ac_metrics the RAW vectors, so `AD.ac(deck)` was fine and
    `AD.stb(deck)` died on the same deck. In the RL loop _execute_tool swallows
    it, so sim.stb reported tool_success=False forever on that testbench and
    the design scored at the floor for a sweep-form reason.
    """

    def test_ac_is_fine_and_keeps_the_zero_hz_sample_in_frequencies(self, adapter):
        result = adapter.ac(AC_COUPLED_FROM_DC, SimParams(analysis_type="ac"))
        assert result.frequencies[0] == 0.0
        assert len(result.frequencies) == 201

    def test_stb_returns_a_real_phase_margin(self, adapter):
        stb = adapter.stb(AC_COUPLED_FROM_DC, SimParams(analysis_type="ac"))
        # One dominant pole inside the loop: 90 deg, and no -180 crossing.
        assert stb.phase_margin == pytest.approx(90.0, abs=1.5)
        assert stb.gain_margin == float("inf")


# ===========================================================================
# F1  flat_band certified a resonant peak as the passband
# ===========================================================================

# 35.1 dB of mid-band gain and a Q = 2.2 pole pair at 1 MHz, AC-coupled at
# 10 Hz so the sweep does not reach DC and flat_band decides the passband.
RESONANT_DECK = """* AC-coupled stage, 35.1 dB midband, Q = 2.2 pole pair at 1 MHz
Vin in 0 DC 0 AC 1
Cc in mid 1u
Rin mid 0 15.9k
E1 amp 0 mid 0 56.8
Rs amp n1 72.34
L1 n1 out 25.330u
C1 out 0 1n
.ac dec {N} 1k {FSTOP}
.end
"""


def _resonant(adapter, per_dec, fstop="1meg"):
    """(ACResult, freqs, gain_db, phase_deg).

    The phase matters: without it the AC-coupling zero below the sweep cannot
    be seen, the sweep is judged to reach DC, and flat_band -- the function
    under test -- is never called at all.
    """
    deck = RESONANT_DECK.replace("{N}", str(per_dec)).replace("{FSTOP}", fstop)
    result = adapter.ac(deck, SimParams(analysis_type="ac"))
    sig = result.signals["vdb(out)"]
    return (result, sig.x_values, sig.y_values,
            result.signals["vp(out)"].y_values)


@skipif_no_ngspice
class TestF1TheFlatBandWindowMustBeBigEnoughToSupportItsTest:
    """The window at sample i ran until span_dec decades OR THE ARRAY ENDED.

    In the last span_dec decades the window is truncated, and once it holds
    only TWO samples the peak-to-peak test IS the two-point test -- and a
    two-sample window straddling a resonance has near-equal endpoints, the
    exact failure the docstring says peak-to-peak exists to prevent.

    Measured on this deck at `dec 20 1k 1meg`: the window at sample 59
    (891251 Hz) spanned 0.05 decades and held two samples, its spread was
    5.8e-03 dB against a 1.25e-02 dB tolerance, and 41.94 dB of resonant peak
    was published as passband_gain_db. A 35 dB amplifier passed `min: 40 dB`,
    and the peakier the design the higher the number.
    """

    def test_the_premise_the_deck_really_does_peak(self, adapter):
        _, f, g, ph = _resonant(adapter, 100)
        m = measure.ac_metrics(f, g, ph)
        assert m["peak_gain_db"] == pytest.approx(42.16, abs=0.1)
        assert m["f_peak"] == pytest.approx(950e3, rel=0.02)

    @pytest.mark.parametrize("per_dec", [10, 20, 50, 100, 200, 500])
    def test_the_passband_is_the_midband_at_every_sweep_density(self, adapter,
                                                                per_dec):
        """The answer used to be a function of SWEEP DENSITY ALONE.

            pts/dec    10     20     50    100    200    500
            gain    35.12  41.94  35.13  35.13  35.13  35.13
        """
        result, f, g, ph = _resonant(adapter, per_dec)
        m = measure.ac_metrics(f, g, ph)
        assert m["passband_gain_db"] == pytest.approx(35.12, abs=0.05)
        ext = sx.extract_specs({"gain": {"min": 40.0, "unit": "dB"}},
                               ac=result, output_signal="out")
        assert ext.values["gain"] < 40.0            # was 41.94 at dec 20
        assert "peaking, not gain" in m["notes"]["passband_gain_db"]

    @pytest.mark.parametrize("per_dec", [100, 400, 2000])
    def test_stopping_the_sweep_on_the_peak_does_not_restore_it(self, adapter,
                                                               per_dec):
        """`.ac dec 2000 1k 946939` put the peak in the LAST window: 42.08 dB."""
        _, f, g, ph = _resonant(adapter, per_dec, fstop="946939")
        m = measure.ac_metrics(f, g, ph)
        assert m["passband_gain_db"] == pytest.approx(35.13, abs=0.05)

    def test_a_two_sample_window_is_refused_not_evaluated(self):
        """The mechanism, in isolation: a window with no interior.

        Three samples straddling a peak symmetrically: the two ENDPOINTS are
        equal, so a two-sample window reports a spread of exactly zero and
        certifies the peak. Sample 1 must not win.
        """
        freqs = [1.0, 10.0 ** 0.05, 10.0 ** 0.1]
        values = [10.0, 12.0, 10.0]
        # Windows: 0 spans 0.1 dec with 3 samples -> spread 2 dB, refused.
        #          1 spans 0.05 dec with 2 samples -> INCOMPLETE, refused.
        assert measure.flat_band(freqs, values) == (None, None)

    def test_a_genuinely_flat_band_is_still_found(self):
        freqs = [10.0 ** (i / 20.0) for i in range(61)]
        values = [20.0] * 61
        i, v = measure.flat_band(freqs, values)
        assert i == 0 and v == pytest.approx(20.0)

    def test_the_last_windows_of_the_sweep_are_the_refused_ones(self):
        """A window that runs off the end of the array is not a flat window."""
        freqs = [10.0 ** (i / 20.0) for i in range(61)]
        values = [20.0] * 61
        i, _ = measure.flat_band(freqs, values[:])
        assert i is not None
        # The last complete window starts span_dec decades below the end.
        last_ok = max(j for j in range(61)
                      if math.log10(freqs[60] / freqs[j]) >= 0.1 - 1e-12)
        assert measure.flat_band(freqs[last_ok + 1:],
                                 values[last_ok + 1:]) == (None, None)


# ===========================================================================
# F2  the "did a level ever form" guard was on the step branch only
# ===========================================================================

PULSE_RC = """* RC stage, tau = 5 us, driven by a pulse of width {W}
V1 in 0 PULSE(0 1.8 0 1n 1n {W} {PER})
R1 in out 5k
C1 out 0 1n
.tran 10n {TSTOP}
.end
"""

SLOW_STAGE = """* R = 1 Meg, C = 1 uF: tau = 1 s, run for 1 ms
{SRC}
R1 in out 1meg
C1 out 0 1u
.tran 1u 1m
.end
"""


@skipif_no_ngspice
class TestF2ATheTopOfAPulseIsAFinalLevelToo:
    """waveform_levels guarded the step tail and not the pulse top.

    The pulse is the drive form this module's own docstring says every
    inverter, buffer and ring oscillator in the eval set uses. Real ngspice,
    tau = 5 us, true 10-90 rise 10.986 us, only the drive pulse width changing:

        width  2u: kind=pulse top=0.5936 V rise=1.592 us  -> PASSES max 5us
        width 10u: kind=pulse top=1.5539 V rise=7.051 us
        width 50u: kind=pulse top=1.7999 V rise=10.984 us  correct

    The top never formed, nothing said so, and the stage passed by 3x.
    """

    def _run(self, adapter, w, per, tstop):
        deck = (PULSE_RC.replace("{W}", w).replace("{PER}", per)
                .replace("{TSTOP}", tstop))
        result = adapter.tran(deck, SimParams(analysis_type="tran"))
        sig = result.signals["out"]
        return result, sig.x_values, sig.y_values

    @pytest.mark.parametrize("w,per,tstop,top", [("2u", "8u", "16u", 0.5936),
                                                 ("10u", "40u", "80u", 1.5539)])
    def test_a_top_that_never_formed_is_refused(self, adapter, w, per, tstop,
                                                top):
        result, t, y = self._run(adapter, w, per, tstop)
        lv = measure.waveform_levels(y, t=t)
        # The premise: the plateau really is short of the 1.8 V asymptote.
        assert lv.y1 == pytest.approx(top, rel=5e-3)
        assert lv.kind == "unsettled"           # was "pulse"
        assert "TOP OF THE PULSE NEVER FORMED" in lv.note
        m = measure.tran_metrics(t, y)
        assert m["rise_time"] is None           # was 1.59 us / 7.05 us
        ext = sx.extract_specs({"rise_time": {"max": 5.0, "unit": "us"}},
                               tran=result, output_signal="out")
        assert "rise_time" not in ext.values    # was a PASS at 3x margin
        assert "rise_time" in ext.unmeasurable

    def test_a_pulse_wide_enough_is_still_measured(self, adapter):
        result, t, y = self._run(adapter, "50u", "200u", "400u")
        m = measure.tran_metrics(t, y)
        assert m["waveform_kind"] == "pulse"
        assert m["rise_time"] == pytest.approx(10.9839e-6, rel=1e-3)

    def test_the_threshold_is_anchored_to_the_edge_not_to_the_excursion(self,
                                                                       adapter):
        """settle_rate_ratio is rate_end / rate_max, and rate_max is the edge.

        The failure mode cannot slow the edge down -- it is the same edge
        either way -- so the anchor cannot shrink with the error. The ratio
        tracks exp(-W/tau), i.e. the fraction of the step still to come.
        """
        measured = []
        for w, per, tstop in (("2u", "8u", "16u"), ("10u", "40u", "80u"),
                              ("30u", "120u", "240u"), ("50u", "200u", "400u")):
            _, t, y = self._run(adapter, w, per, tstop)
            lv = measure.waveform_levels(y, t=t)
            k = max(2, int((lv.i_end - lv.i_edge + 1) * 0.02))
            ratio, _, _ = measure.settle_rate_ratio(y, t, lv.i_end, k)
            assert ratio is not None
            measured.append(ratio)
        # It falls monotonically with the pulse width and tracks exp(-W/tau)
        # once the window is a small part of the plateau -- that is what makes
        # it "the fraction of the step still to come" rather than a number
        # that depends on how much of the step this record happened to catch.
        assert measured == sorted(measured, reverse=True)
        assert measured[1] == pytest.approx(math.exp(-2.0), rel=0.15)
        assert measured[2] == pytest.approx(math.exp(-6.0), rel=0.15)
        assert measured[3] == pytest.approx(math.exp(-10.0), rel=0.25)
        # The two that must land on opposite sides of the threshold.
        assert measured[1] > measure.SETTLE_RATE_FRAC * 5.0
        assert measured[3] < measure.SETTLE_RATE_FRAC / 1.5


@skipif_no_ngspice
class TestF2BTheSameCircuitCannotEscapeByChangingItsDriveForm:
    """R7's own circuit, under two stimuli.

        PWL step drive : kind=unsettled  rise None  settling None  (correct)
        PULSE drive    : kind=step  rise 399.79 us  settling 490.05 us
                         notes {} -> BOTH PASS a `max: 1 ms` budget on a
                         circuit 1000x too slow, unmeasurable empty

    The output of the second is genuinely flat to 0.05 pct over the last half
    of the record -- better formed than this repo's own canonical 5-tau pulse
    -- so no settling test on `out` alone can separate them. The DRIVE can, and
    the drive is an immune anchor: it is fixed before the circuit responds.
    """

    SPECS = {"rise_time": {"max": 1.0, "unit": "ms"},
             "settling_time": {"max": 1.0, "unit": "ms"}}

    def _run(self, adapter, src):
        deck = SLOW_STAGE.replace("{SRC}", src)
        return adapter.tran(deck, SimParams(analysis_type="tran"))

    def test_the_pwl_step_drive_is_refused(self, adapter):
        result = self._run(adapter, "V1 in 0 PWL(0 0 1n 1.8)")
        ext = sx.extract_specs(self.SPECS, tran=result, output_signal="out")
        assert ext.values == {}
        assert sorted(ext.unmeasurable) == ["rise_time", "settling_time"]

    def test_the_pulse_drive_is_refused_the_same_way(self, adapter):
        result = self._run(adapter, "V1 in 0 PULSE(0 1.8 0 1n 1n 500u 1)")
        # The premise: on the output alone this record looks perfectly settled.
        sig = result.signals["out"]
        lv = measure.waveform_levels(sig.y_values, t=sig.x_values)
        assert lv.kind == "step"
        assert lv.y1 == pytest.approx(0.0009, rel=0.05)
        ext = sx.extract_specs(self.SPECS, tran=result, output_signal="out")
        assert ext.values == {}                 # was rise 0.39979, settling 0.49
        assert sorted(ext.unmeasurable) == ["rise_time", "settling_time"]
        assert "DRIVE WAS REMOVED" in ext.unmeasurable["rise_time"]

    def test_the_canonical_five_tau_deck_is_not_caught_by_it(self, adapter):
        """The repo's own transient deck must keep its IEEE 181 answer."""
        deck = ("* scripts/agent_ngspice.py\n"
                "V1 in 0 PULSE(0 1.8 0 1n 1n 5u 10u)\n"
                "R1 in out 1k\nC1 out 0 1n\n.tran 0.1u 20u\n.end\n")
        result = adapter.tran(deck, SimParams(analysis_type="tran"))
        ext = sx.extract_specs({"rise_time": {"max": 5.0, "unit": "us"}},
                               tran=result, output_signal="out")
        assert ext.values["rise_time"] == pytest.approx(2.197, rel=0.05)


# ===========================================================================
# F3  abs() removed the sign dependence of slew_rate but not the edge one
# ===========================================================================

# tr = 40 us, tf = 4 us: a 10x asymmetric stage. Only the .tran start moves.
ASYMMETRIC_EDGES = """* tr = 40 us, tf = 4 us
V1 in 0 PULSE(0 1.8 0 40u 4u 60u 200u)
R1 in out 100
C1 out 0 1p
{TRAN}
.end
"""


@skipif_no_ngspice
class TestF3TheSlewRateSpecDoesNotDependOnWhichEdgeOpensTheRECORD:
    """measure.slew_rate only ever measures the FIRST edge.

        .tran 100n 160u 0      opens on the SLOW edge: 0.045 V/us -> FAIL
        .tran 100n 260u 95u    opens on the FAST edge: 0.450 V/us -> PASS

    a 10x swing in the score for the same circuit. At HEAD both failed; R10's
    abs() made one of them PASS -- the optimistic direction, which is worse
    than the deterministic false-fail it replaced.
    """

    SPECS = {"slew_rate": {"min": 0.1, "unit": "V/us"}}

    def _run(self, adapter, tran):
        deck = ASYMMETRIC_EDGES.replace("{TRAN}", tran)
        return adapter.tran(deck, SimParams(analysis_type="tran"))

    def test_the_premise_the_first_edge_really_does_change(self, adapter):
        slow = self._run(adapter, ".tran 100n 160u 0")
        fast = self._run(adapter, ".tran 100n 260u 95u")
        for result, sign in ((slow, 1.0), (fast, -1.0)):
            sig = result.signals["out"]
            m = measure.tran_metrics(sig.x_values, sig.y_values)
            assert math.copysign(1.0, m["slew_rate"]) == sign
        s = measure.tran_metrics(slow.signals["out"].x_values,
                                 slow.signals["out"].y_values)["slew_rate"]
        f = measure.tran_metrics(fast.signals["out"].x_values,
                                 fast.signals["out"].y_values)["slew_rate"]
        assert abs(f) == pytest.approx(10.0 * abs(s), rel=0.02)

    def test_the_spec_scores_the_same_circuit_the_same(self, adapter):
        slow = sx.extract_specs(self.SPECS, tran=self._run(
            adapter, ".tran 100n 160u 0"), output_signal="out")
        fast = sx.extract_specs(self.SPECS, tran=self._run(
            adapter, ".tran 100n 260u 95u"), output_signal="out")
        assert slow.values["slew_rate"] == pytest.approx(0.045, rel=1e-3)
        assert fast.values["slew_rate"] == pytest.approx(0.045, rel=1e-3)
        assert (fast.values["slew_rate"]
                == pytest.approx(slow.values["slew_rate"], rel=1e-6))

    def test_the_worst_edge_is_what_the_spec_takes(self, adapter):
        result = self._run(adapter, ".tran 100n 260u 95u")
        sig = result.signals["out"]
        m = measure.tran_metrics(sig.x_values, sig.y_values)
        assert m["slew_edges"] == 2
        assert m["slew_rate_worst"] == pytest.approx(45000.0, rel=1e-3)
        assert abs(m["slew_rate"]) == pytest.approx(450000.0, rel=1e-3)

    def test_the_note_always_names_the_edge(self, adapter):
        """tran_metrics used to emit a slew_rate note only when it was absent."""
        result = self._run(adapter, ".tran 100n 160u 0")
        sig = result.signals["out"]
        m = measure.tran_metrics(sig.x_values, sig.y_values)
        note = m["notes"]["slew_rate"]
        assert "FIRST edge" in note and "rising" in note
        assert "slew_rate_worst" in note

    def test_a_single_edge_record_is_unchanged(self):
        """R10's own probe: one falling exponential, one number."""
        tau = 20e-9
        n = 2001
        t = [i * 1e-6 / (n - 1) for i in range(n)]
        y = [1.8 * math.exp(-x / tau) if x > 0 else 1.8 for x in t]
        tran = {"time": t, "signals": {"out": {"name": "out", "x_values": t,
                                              "y_values": y}}}
        ext = sx.extract_specs({"slew_rate": {"min": 10.0, "unit": "V/us"}},
                               tran=tran, output_signal="out")
        assert ext.values["slew_rate"] == pytest.approx(32.769, rel=1e-3)


# ===========================================================================
# F4  `if twins and not netlist` on a deck whose DC values cannot be read
# ===========================================================================

DUAL_RAIL_LITERAL = """* dual rail, literal DC values
Vdd vdd 0 DC 1.8
Vss vss 0 DC -1.8
R1 vdd vss 3.6k
.op
.end
"""

DUAL_RAIL_PARAM = """* dual rail, {param} DC values
.param vsup=1.8
Vdd vdd 0 DC {vsup}
Vss vss 0 DC {-vsup}
R1 vdd vss 3.6k
.op
.end
"""


@skipif_no_ngspice
class TestF4TheTwinsWarningSurvivesADeckThatCannotBeRead:
    """The suppression assumed a twin that cannot fire on a NaN.

    The polarity test classifies by source_dc_value(...) > 0.0 / < 0.0 and the
    ammeter test by == 0.0. A `.param`-valued rail parses to NaN, which is none
    of the three, so the deck resolves NOTHING -- and the twins net that used
    to catch the 2x was suppressed on the strength of it. +-1.8 V across
    3.6 k, one 1.000 mA current:

        literal DC values : idd 0.001, vss excluded, no warning     correct
        {vsup} parameters : idd 0.002, only "2 branches were summed"     2x
        no netlist at all : idd 0.002, plus the full "exactly 2x" warning

    Passing the netlist made the diagnosis strictly WORSE than not passing it.
    """

    def _op(self, adapter, deck):
        return adapter.dc(deck, SimParams(analysis_type="op")).op_points

    def test_the_premise_the_param_rail_really_does_parse_to_nan(self):
        deck = measure.parse_deck_sources(DUAL_RAIL_PARAM)
        assert math.isnan(measure.source_dc_value("vdd", deck))
        assert math.isnan(measure.source_dc_value("vss", deck))
        lit = measure.parse_deck_sources(DUAL_RAIL_LITERAL)
        assert measure.source_dc_value("vss", lit) == -1.8

    def test_the_literal_deck_still_resolves_it_silently(self, adapter):
        op = self._op(adapter, DUAL_RAIL_LITERAL)
        rep = measure.supply_current_report(op, None, DUAL_RAIL_LITERAL)
        assert rep.value == pytest.approx(1e-3, rel=1e-6)
        assert rep.warnings == []

    def test_the_param_deck_is_warned_about(self, adapter):
        op = self._op(adapter, DUAL_RAIL_PARAM)
        rep = measure.supply_current_report(op, None, DUAL_RAIL_PARAM)
        assert rep.value == pytest.approx(2e-3, rel=1e-6)   # still 2x ...
        twin = [w for w in rep.warnings if "exactly 2x" in w]
        assert twin, rep.warnings          # ... but no longer silently so
        assert "could not be" in twin[0]

    def test_passing_the_netlist_is_never_worse_than_not_passing_it(self, adapter):
        op = self._op(adapter, DUAL_RAIL_PARAM)
        with_deck = measure.supply_current_report(op, None, DUAL_RAIL_PARAM)
        without = measure.supply_current_report(op, None, None)
        assert (len([w for w in with_deck.warnings if "exactly 2x" in w])
                == len([w for w in without.warnings if "exactly 2x" in w]))


# ===========================================================================
# F6  idd = 0.0 with an empty warning list
# ===========================================================================

OPEN_SUPPLY = """* the 1.8 V rail reaches the circuit only through a capacitor
Vdd vdd 0 DC 1.8
Cc vdd out 1u
R1 out 0 10k
.op
.end
"""


@skipif_no_ngspice
class TestF6AnExactlyZeroSupplyBranchIsNotAMeasurement:
    """For a current budget 0.0 is a PERFECT value.

        op_points {vdd#branch: 0.0, out: 0.0, vdd: 1.8}
        idd 0.0, sources [vdd], excluded {}, warnings []
        spec idd <= 0.2 mA -> {idd: 0.0}, unmeasurable []

    Disconnect the supply and the current budget is met perfectly. Identical on
    the explicit-`sources` path, which the docstring calls "the only input that
    actually knows which sources are supplies".
    """

    @pytest.fixture
    def op(self, adapter):
        return adapter.dc(OPEN_SUPPLY, SimParams(analysis_type="op")).op_points

    def test_the_premise_the_branch_really_is_exactly_zero(self, op):
        assert op["vdd#branch"] == 0.0

    @pytest.mark.parametrize("kwargs", [{"netlist": OPEN_SUPPLY},
                                        {"sources": ["vdd"]},
                                        {}])
    def test_every_path_refuses_it_with_a_reason(self, op, kwargs):
        rep = measure.supply_current_report(op, **kwargs)
        assert rep.value is None                    # was 0.0
        assert any("EXACTLY 0 A" in w for w in rep.warnings)
        assert any("PERFECT" in w for w in rep.warnings)

    def test_the_spec_is_unmeasurable_on_both_paths(self, adapter):
        result = adapter.dc(OPEN_SUPPLY, SimParams(analysis_type="op"))
        specs = {"idd": {"max": 0.2, "unit": "mA"}}
        by_deck = sx.extract_specs(specs, dc=result, netlist=OPEN_SUPPLY,
                                   output_signal="out")
        by_name = sx.extract_specs(specs, dc=result, supply_sources=["vdd"],
                                   output_signal="out")
        for ext in (by_deck, by_name):
            assert "idd" not in ext.values          # was 0.0, a clean PASS
            assert "idd" in ext.unmeasurable

    def test_a_real_supply_current_is_untouched(self, adapter):
        deck = ("* an ordinary rail\nVdd vdd 0 DC 1.8\nR1 vdd 0 1.8k\n"
                ".op\n.end\n")
        result = adapter.dc(deck, SimParams(analysis_type="op"))
        rep = measure.supply_current_report(result.op_points, None, deck)
        assert rep.value == pytest.approx(1e-3, rel=1e-6)
        assert rep.warnings == []


# ===========================================================================
# F7  a pulse with ringing: the region ended at the first ring trough
# ===========================================================================

def _ringing_pulse():
    """zeta = 0.16, a 19 us plateau at 1.0 V, then the return edge.

    i_ret is the first sample below 50 pct of dmax, and dmax is the OVERSHOOT
    PEAK. On any pulse whose overshoot exceeds ~50 pct the first ring trough
    falls below that level, so i_ret lands there instead of on the return
    edge; the rate-bounded walk-back then correctly walks back up the trough
    to the ring peak and stops.
    """
    zeta = 0.16
    wd = math.pi / 1.51e-6
    wn = wd / math.sqrt(1.0 - zeta ** 2)

    def step(x):
        if x <= 0.0:
            return 0.0
        return 1.0 - math.exp(-zeta * wn * x) * (
            math.cos(wd * x) + zeta / math.sqrt(1 - zeta ** 2) * math.sin(wd * x))

    n = 6000
    t = [i * 30e-6 / (n - 1) for i in range(n)]
    y = [step(x) - step(x - 19e-6) for x in t]
    return t, y


class TestF7AnOvershootPeakIsNotTheTopOfAPulse:
    """truth: base 0.0, top 1.0, peak 1.60097 -> overshoot 60.097 pct.

    observed: kind=pulse y1=1.5998 i_end=302/5999 (t_end 1.510 us of 30 us),
              overshoot_pct 0.0756, settling_time 1.45e-06, y_final 1.5998
    """

    def test_the_premise_the_ring_trough_really_is_below_half_the_peak(self):
        t, y = _ringing_pulse()
        peak = max(y)
        assert peak == pytest.approx(1.60097, rel=1e-3)
        first_peak = y.index(peak)
        trough = min(y[first_peak:first_peak + 800])
        assert trough < 0.5 * peak          # 0.639 against 0.8005

    def test_the_region_runs_to_the_real_return_edge(self):
        t, y = _ringing_pulse()
        lv = measure.waveform_levels(y, t=t)
        assert lv.kind == "pulse"
        assert t[lv.i_end] == pytest.approx(19e-6, rel=5e-3)   # was 1.51e-6
        assert lv.i_end > 3000                                 # was 302

    def test_the_top_is_the_plateau_and_the_overshoot_is_reported(self):
        t, y = _ringing_pulse()
        m = measure.tran_metrics(t, y)
        assert m["y_final"] == pytest.approx(1.0, rel=5e-3)     # was 1.5998
        assert m["overshoot_pct"] == pytest.approx(60.1, rel=0.02)  # was 0.0756

    def test_the_spec_sees_the_overshoot(self):
        t, y = _ringing_pulse()
        tran = {"time": t, "signals": {"out": {"name": "out", "x_values": t,
                                              "y_values": y}}}
        ext = sx.extract_specs({"overshoot": {"max": 10.0, "unit": "%"}},
                               tran=tran, output_signal="out")
        assert ext.values["overshoot"] > 50.0       # was 0.0756, a clean PASS

    def test_a_pulse_without_ringing_is_unchanged(self):
        tau = 1e-6
        t = [i * 20e-6 / 2000 for i in range(2001)]
        top = 1.8 * (1.0 - math.exp(-5.0))
        y = [1.8 * (1 - math.exp(-x / tau)) if x < 5e-6
             else top * math.exp(-(x - 5e-6) / tau) for x in t]
        lv = measure.waveform_levels(y, t=t)
        assert lv.kind == "pulse"
        assert lv.y1 == pytest.approx(top, rel=5e-3)
        assert t[lv.i_end] == pytest.approx(5e-6, abs=2e-7)


# ===========================================================================
# F8  prop_delay refused with the wrong reason
# ===========================================================================

class TestF8ThePropDelayRefusalSaysWhatIsActuallyWrong:
    """A result holding ONLY a stimulus was told it "carries no input signal".

    _pick_output falls back to the first non-branch vector, so on a result with
    just `in` it returns `in`, _pick_input returns `in` too, and the two are
    the same vector. The refusal was correct; its REASON was the exact opposite
    of what is wrong, and it sent the reader off to name an `input_signal` that
    was already there.
    """

    T = [i * 200e-9 / 400 for i in range(401)]

    def _tran(self, **sigs):
        return {"time": self.T,
                "signals": {k: {"name": k, "x_values": self.T, "y_values": v}
                            for k, v in sigs.items()}}

    def _step_at(self, t0):
        return [0.0 if x < t0 else 1.8 for x in self.T]

    def test_only_a_stimulus_is_present(self):
        tran = self._tran(**{"in": self._step_at(20e-9)})
        ext = sx.extract_specs({"delay": {"max": 100.0, "unit": "ns"}},
                               tran=tran)
        why = ext.unmeasurable["delay"]
        assert "SAME vector" in why
        assert "no input signal" not in why      # it has one; it lacks an OUTPUT
        assert "output_signal" in why

    def test_naming_the_same_vector_twice_says_so_too(self):
        tran = self._tran(**{"in": self._step_at(20e-9),
                             "out": self._step_at(60e-9)})
        ext = sx.extract_specs({"delay": {"max": 100.0, "unit": "ns"}},
                               tran=tran, output_signal="in",
                               input_signal="in")
        assert "SAME vector" in ext.unmeasurable["delay"]

    def test_a_result_with_no_stimulus_keeps_its_own_reason(self):
        """R11's message must survive for the case it was written for."""
        tran = self._tran(out=self._step_at(60e-9), vout=self._step_at(100e-9))
        ext = sx.extract_specs({"delay": {"max": 100.0, "unit": "ns"}},
                               tran=tran, output_signal="vout")
        assert "no input signal" in ext.unmeasurable["delay"]

    def test_a_real_pair_is_still_measured(self):
        tran = self._tran(**{"in": self._step_at(20e-9),
                             "vout": self._step_at(100e-9)})
        ext = sx.extract_specs({"delay": {"max": 100.0, "unit": "ns"}},
                               tran=tran, output_signal="vout")
        assert ext.values["delay"] == pytest.approx(80.0, abs=1.0)


# ===========================================================================
# F9  `100.0e3` is a STRING under PyYAML, not a float
# ===========================================================================

# PyYAML implements YAML 1.1, whose float resolver requires a DOT and a SIGNED
# exponent. '100.0e3', '1e9' and '2e-9' therefore load as str, and any eval
# task written that way makes its spec silently unmeasurable.
_NUMERIC_TEXT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

# Keys whose value is a NUMBER: a spec bound, a target, or the frequency a
# bound is stated at. `unit`, `tolerance` and the task's own prose are text.
_NUMERIC_KEYS = frozenset({"min", "max", "target", "typ", "nominal", "weight",
                           "at_freq", "freq", "frequency"})


def _task_files():
    return sorted((REPO / "eval" / "tasks").rglob("*.yaml"))


def test_there_are_eval_tasks_to_check():
    assert len(_task_files()) >= 70


@pytest.mark.parametrize("path", _task_files(), ids=lambda p: p.name)
def test_every_spec_bound_parses_to_a_number(path):
    """PyYAML 1.1 needs a dot AND a sign in the exponent for a float literal.

    Found in eval/tasks: flash_adc_3bit_001 `min: 1.0e9`, pll_charge_pump_001
    `target: 1.0e9`, spectre_emx_inductor_001 `target: 2e-9` and two
    `freq: 5e9`, spectre_pss_vco_001 `target: 1e9`, tia_001
    `at_freq: 100.0e3` -- seven bounds, every one of them a string.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = []

    def walk(node, trail):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, trail + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, trail + [str(i)])
        elif isinstance(node, str) and trail and trail[-1] in _NUMERIC_KEYS:
            bad.append(("/".join(trail), node))

    walk(doc, [])
    assert not bad, (
        f"{path.name}: these are numeric fields that PyYAML loaded as STRINGS "
        f"-- write 1.0e+9, not 1e9 or 1.0e9: {bad}"
    )


@pytest.mark.parametrize("path", _task_files(), ids=lambda p: p.name)
def test_no_value_anywhere_in_a_task_looks_like_a_number_but_is_not(path):
    """The wider net: any scalar that READS as a number must BE one.

    A bound is not the only place this bites -- a `unit` is text and a
    `tolerance: 10%` is text, but anything that a reader would take for a
    number and PyYAML took for a string is a trap wherever it sits.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = []

    def walk(node, trail):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, trail + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, trail + [str(i)])
        elif isinstance(node, str) and _NUMERIC_TEXT.match(node.strip()):
            bad.append(("/".join(trail), node))

    walk(doc, [])
    assert not bad, f"{path.name}: numeric-looking strings: {bad}"


def test_the_yaml_forms_that_do_and_do_not_work():
    """The rule itself, so the next author does not have to rediscover it."""
    loaded = yaml.safe_load(
        "a: 1.0e+9\nb: 1.0e9\nc: 1e9\nd: 2.0e-9\ne: 2e-9\nf: 1000000000.0\n")
    assert isinstance(loaded["a"], float)     # dot AND signed exponent
    assert isinstance(loaded["b"], str)       # no sign
    assert isinstance(loaded["c"], str)       # no dot, no sign
    assert isinstance(loaded["d"], float)     # dot AND signed exponent
    assert isinstance(loaded["e"], str)       # no dot
    assert isinstance(loaded["f"], float)     # plain decimal is always safe
