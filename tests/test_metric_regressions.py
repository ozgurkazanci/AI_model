"""Regression guards for the defects the FIRST round of metric fixes introduced.

tests/test_metric_defects.py guards 16 original defects. Fixing them introduced
a second set, and this file guards those. Every test below reproduces the probe
that found the regression and asserts the correct NUMBER, and every one of them
FAILS against the code as it stood before the corresponding fix -- that was
checked by reverting each fix in a scratch copy of src/ and running the test
against it.

The shape of every regression here is the same and it is the dangerous one: a
metric that used to refuse (None, with a reason) started returning a confident
number that flatters the design. An unstable loop reported a +111.8 deg phase
margin; a notched loop reported +89.6 deg where it actually closes at
-14.1 deg; a 1.8 mA design was scored at 3.6 mA; an AC-coupled amplifier whose
gain at DC is zero reported 40.0 dB of DC gain. The simulator is the reward
source for GRPO, so each of those is a smooth false gradient, which is strictly
worse than a missing metric.
"""
from __future__ import annotations

import json
import math

import pytest

try:
    from asic_ai import serialization
    from asic_ai.adapters import measure
    from asic_ai.adapters import spec_extract as sx
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

TWOPI = 2.0 * math.pi


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


# ===========================================================================
# N1  The inversion must not be inferred from a single sample against 0 deg
# ===========================================================================

# One pole at 10 kHz followed by a Q = 20 resonance at 12 kHz. Non-inverting:
# H(0) = +1 exactly, and the phase at the bottom of the sweep is 0.
FP1, F0, QRES = 10e3, 12e3, 20.0
L_RES = 1e-3
C_RES = 1.0 / (L_RES * (TWOPI * F0) ** 2)
R_RES = math.sqrt(L_RES / C_RES) / QRES
C_POLE = 1.0 / (TWOPI * FP1 * 1e3)

RESONANT_LOOP = f"""* resonant loop: pole at 10 kHz, RLC resonance at 12 kHz, Q = 20
* Unity VCVSs only, so there is NO inversion anywhere and H(DC) = +1.
Vin in 0 DC 0 AC 1
E1 a 0 in 0 1
R1 a b 1k
C1 b 0 {C_POLE:.10g}
E2 c 0 b 0 1
L1 c d {L_RES:.10g}
R3 d e {R_RES:.10g}
C3 e 0 {C_RES:.10g}
E3 out 0 e 0 1
.ac dec 400 1 1e7
.end
"""


def _resonant(f):
    s = 1j * TWOPI * f
    w0 = TWOPI * F0
    return 1.0 / ((1 + s / (TWOPI * FP1))
                  * (1 + s / (QRES * w0) + (s / w0) ** 2))


class TestN1InversionIsNotFabricatedOnAResonance:
    """The peak-gain sample is "mid-band" only for a band-pass response.

    On a resonant response the peak sits exactly where the REAL phase lag is
    near -180 deg, and the D1 fix read that lag as a sign inversion, subtracted
    180 deg from the whole curve, and reported

        phase_margin  +111.751575   against a true -68.248453  (UNSTABLE)
        gain_margin   inf           against a true -41.233660 dB

    Removing 180 deg also destroys the -180 deg crossing, so f_180 became None
    and stb() turned that into float("inf") -- the most dangerous reading there
    is for a loop that is actually unstable. The note even claimed the
    inference came from "the mid-band phase which does not move when f_start
    moves".

    The inversion is now judged at the BOTTOM of the sweep against the
    minimum-phase Bode estimate for the magnitude slope there, so it answers
    the only question that matters -- is the sign of the response at DC
    negative -- instead of asking whether some sample happens to sit near
    +/-180 deg.
    """

    FREQS = _log_sweep(1.0, 1e7, 400)

    @pytest.fixture
    def metrics(self):
        gain_db, phase = _response(self.FREQS, _resonant)
        return measure.ac_metrics(self.FREQS, gain_db, phase), gain_db, phase

    def test_the_peak_really_does_sit_near_minus_180_deg(self, metrics):
        """The premise of the defect: this is the shape that fooled it."""
        _, gain_db, phase = metrics
        peak_i = max(range(len(gain_db)), key=lambda i: gain_db[i])
        ph = measure.unwrap_deg(phase)
        assert abs(ph[peak_i] + 180.0) < 60.0
        # The OLD rule is still wrong about this data, which is why the answer
        # had to stop depending on it.
        assert measure.phase_inversion_shift(ph, ref_index=peak_i) == -1

    def test_no_inversion_is_inferred(self, metrics):
        m, _, _ = metrics
        assert m["phase_inversion_k"] == 0.0

    def test_the_loop_is_reported_as_unstable(self, metrics):
        m, _, _ = metrics
        # Exact PM from a bisection on the analytic response.
        assert m["phase_margin"] == pytest.approx(-49.7902, abs=0.01)
        assert m["phase_margin"] < 0.0
        # Before: this same curve reported PM + 180 and looked comfortable.
        assert m["phase_margin"] + 180.0 == pytest.approx(130.2098, abs=0.01)

    def test_the_gain_margin_is_finite_and_negative(self, metrics):
        m, _, _ = metrics
        assert m["gain_margin"] is not None
        assert math.isfinite(m["gain_margin"])
        assert m["gain_margin"] == pytest.approx(-19.6212, abs=0.01)
        assert m["f_180"] == pytest.approx(12248.8, rel=1e-3)

    def test_a_genuine_inversion_is_still_removed(self):
        """The fix must not cost the ability to see a real inversion."""
        freqs = _log_sweep(1.0, 1e7, 100)
        gain_db, phase = _response(
            freqs, lambda f: -1000.0 / (1 + 1j * f / 1000.0))
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["phase_inversion_k"] == pytest.approx(1.0)
        assert m["phase_margin"] == pytest.approx(90.0, abs=0.5)


@skipif_no_ngspice
class TestN1AgainstNgspice:
    """The same loop as a real deck: passive RLC plus unity VCVSs."""

    def test_the_deck_really_has_no_inversion(self, adapter):
        ac = adapter.ac(RESONANT_LOOP, SimParams(analysis_type="ac"))
        assert ac.signals["vp(out)"].y_values[0] == pytest.approx(0.0, abs=0.01)

    def test_stb_reports_the_instability(self, adapter):
        result = adapter.stb(RESONANT_LOOP,
                             SimParams(analysis_type="ac",
                                       options={"loop_out": "out",
                                                "loop_in": "in"}))
        assert result.phase_margin == pytest.approx(-49.7902, abs=0.05)
        # Before: +111.75 here, and an INFINITE gain margin for an unstable
        # loop, because removing the fabricated 180 deg destroyed the -180 deg
        # crossing.
        assert result.phase_margin < 0.0
        assert not math.isinf(result.gain_margin)
        assert result.gain_margin == pytest.approx(-19.6212, abs=0.05)


class TestN1SweepStartInvarianceOfTheGainMargin:
    """D1's own invariant failed on the commonest shape of all.

    A plain non-inverting 3-pole amplifier has peak_i = 0 at every sweep start,
    so the peak-referenced rule was the FIRST-sample rule it replaced. With the
    poles placed so that -180 deg falls on 100 kHz:

        f_start 1 .. 1e3  gain_margin +35.5641 dB
        f_start 1e5       gain_margin None, "infinite; unconditionally stable"

    The 360 deg branch of atan2 is the whole story: at f_start = 1e5 the true
    phase is -180 deg and the principal value is +180 deg, so the curve never
    crossed -180 and the loop was declared unconditionally stable.
    """

    FP = 1e5 / math.tan(math.radians(60.0))     # -180 deg lands on 100 kHz
    K = 8.0 * 10.0 ** (-35.5641 / 20.0)         # gain margin +35.5641 dB there

    def _gm(self, fstart):
        freqs = _log_sweep(fstart, 1e9, 400)
        gain_db, phase = _response(
            freqs, lambda f: self.K / (1 + 1j * f / self.FP) ** 3)
        return measure.ac_metrics(freqs, gain_db, phase)

    @pytest.mark.parametrize("fstart", [1.0, 10.0, 100.0, 1000.0, 1e5])
    def test_the_gain_margin_does_not_move_with_the_sweep_start(self, fstart):
        m = self._gm(fstart)
        assert m["gain_margin"] == pytest.approx(35.5641, abs=1e-3)

    def test_the_branch_correction_is_reported_as_a_branch_not_an_inversion(self):
        """At f_start = 2e5 the true phase is -222 deg and atan2 reports +138.

        The 360 deg the unwrap cannot know about is removed, and the note says
        plainly that an EVEN number of 180 deg turns is a branch artefact and
        not a sign inversion.
        """
        m = self._gm(2e5)
        assert m["phase_inversion_k"] == 2.0
        assert "branch" in m["notes"]["phase"]

    def test_a_sweep_that_starts_past_minus_180_refuses_instead_of_lying(self):
        """At f_start = 2e5 the crossing is BELOW the sweep. Refuse, and say so.

        The number cannot be recovered from data that does not contain it: the
        worst case inside the sweep is +53.6 dB against a true +35.6 dB, which
        is optimistic. What must never happen again is the old answer -- None
        with "the gain margin is infinite; unconditionally stable".
        """
        m = self._gm(2e5)
        assert m["gain_margin"] is None
        note = m["notes"]["gain_margin"]
        assert "below the sweep" in note.lower()
        assert "NOT an infinite margin" in note
        assert "unconditionally stable" not in note


class TestN1ALoopSittingAtMinus180:
    """Phase exactly -180 deg at every frequency: PM 0, and NOT stable.

    The old rule read the -180 deg at the reference sample as an inversion,
    shifted the whole curve to 0 deg, and reported a phase margin of 180.0 deg
    for a loop that is marginally unstable, with a gain margin of None under
    the note "infinite; unconditionally stable".
    """

    FREQS = _log_sweep(1.0, 1e4, 100)
    F_UNITY = 100.0

    @pytest.fixture
    def metrics(self):
        gain = [20.0 * math.log10(self.F_UNITY / f) for f in self.FREQS]
        phase = [-180.0] * len(self.FREQS)
        return measure.ac_metrics(self.FREQS, gain, phase)

    def test_no_inversion_is_inferred(self, metrics):
        assert metrics["phase_inversion_k"] == 0.0

    def test_the_phase_margin_is_zero_not_one_hundred_and_eighty(self, metrics):
        assert metrics["ugb"] == pytest.approx(self.F_UNITY, rel=1e-9)
        assert metrics["phase_margin"] == pytest.approx(0.0, abs=1e-9)

    def test_the_gain_margin_is_not_infinite(self, metrics):
        """A phase that SITS at -180 deg is the opposite of unconditionally stable.

        There is no unique phase crossover frequency when the phase is -180 deg
        everywhere, so the margin is reported at the worst case -- the highest
        gain anywhere in that region -- which is never optimistic. What it must
        not be is None-with-"infinite".
        """
        assert metrics["gain_margin"] is not None
        assert math.isfinite(metrics["gain_margin"])
        assert metrics["gain_margin"] <= 0.0
        assert "NOT infinite" in metrics["notes"]["gain_margin"]


# ===========================================================================
# N4  ugb is the LOOP CLOSURE, not the first 0 dB crossing
# ===========================================================================

def _notched(f):
    """5 poles and one Q = 25 zero pair: the loop gain crosses 0 dB 3 times.

    40 dB at DC, a dominant pole at 100 Hz, a notch at 1 kHz that takes the
    gain 8 dB below unity, a recovery, and the real closure at 93 kHz. The
    peak gain is the FIRST sweep sample, so start_index=peak_i is no
    protection at all.
    """
    s = 1j * TWOPI * f
    wz = TWOPI * 1000.0
    num = 1 + s / (25.0 * wz) + (s / wz) ** 2
    den = ((1 + s / (TWOPI * 100.0)) * (1 + s / (TWOPI * 1e4)) ** 2
           * (1 + s / (TWOPI * 3e4)) ** 2)
    return 100.0 * num / den


class TestN4UnityGainIsTheLoopClosure:
    """crossing_freq(..., start_index=peak_i) returns the FIRST falling crossing.

    On a notched loop gain that is the edge of the notch, not the closure. The
    loop climbs back above 0 dB after it and closes for real two decades
    higher, where the phase margin is NEGATIVE. The error is always toward
    "safe", and the gain margin computed by the same call disagreed with it in
    sign with nothing to flag the contradiction.
    """

    FREQS = _log_sweep(1.0, 1e8, 500)

    @pytest.fixture
    def metrics(self):
        gain_db, phase = _response(self.FREQS, _notched)
        return measure.ac_metrics(self.FREQS, gain_db, phase), gain_db, phase

    def test_the_loop_really_does_cross_zero_db_more_than_once(self, metrics):
        _, gain_db, _ = metrics
        downs = measure.all_crossings(self.FREQS, gain_db, 0.0, direction=-1)
        ups = measure.all_crossings(self.FREQS, gain_db, 0.0, direction=1)
        assert len(downs) == 2 and len(ups) == 1
        assert ups[0] > downs[0]

    def test_ugb_is_the_last_crossing_not_the_first(self, metrics):
        m, gain_db, _ = metrics
        downs = measure.all_crossings(self.FREQS, gain_db, 0.0, direction=-1)
        assert m["ugb"] == pytest.approx(downs[-1], rel=1e-9)
        assert m["ugb"] == pytest.approx(93058.25, rel=1e-4)
        assert downs[0] == pytest.approx(954.39, rel=1e-4)
        assert m["ugb"] > downs[0] * 90.0

    def test_the_phase_margin_says_the_loop_is_unstable(self, metrics):
        m, gain_db, phase = metrics
        downs = measure.all_crossings(self.FREQS, gain_db, 0.0, direction=-1)
        ph = measure.unwrap_deg(phase)
        at_first = 180.0 + measure.value_at_freq(self.FREQS, ph, downs[0])
        # The old answer was comfortably positive at the FIRST crossing...
        assert at_first == pytest.approx(104.638, abs=0.01)
        # ... and the loop is in fact unstable where it actually closes.
        assert m["phase_margin"] == pytest.approx(-41.960, abs=0.01)
        assert m["phase_margin"] < 0.0
        # The error is 146.6 deg, and always toward "safe".
        assert at_first - m["phase_margin"] > 140.0

    def test_the_multi_crossing_response_is_flagged(self, metrics):
        m, _, _ = metrics
        note = m["notes"]["ugb"]
        assert "LOOP CLOSURE" in note
        assert "optimistic" in note

    def test_the_two_margins_no_longer_contradict_each_other(self, metrics):
        """A +104 deg phase margin beside a -15.6 dB gain margin, unflagged.

        Both now say the same thing about the same loop.
        """
        m, _, _ = metrics
        assert m["gain_margin"] == pytest.approx(-15.617, abs=0.01)
        assert m["gain_margin"] < 0.0 and m["phase_margin"] < 0.0

    def test_a_gain_curve_that_flattens_at_0_db_still_reports_the_first_reach(self):
        """L1's guard. A flat run AT the level is one crossing, not two."""
        freqs = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        gain = [20.0, 0.0, 0.0, 0.0, -20.0]
        assert measure.ac_metrics(freqs, gain)["ugb"] == pytest.approx(10.0)


# ===========================================================================
# N12  a flat magnitude has no meaningful 0 dB crossing
# ===========================================================================

def test_n12_a_flat_magnitude_refuses_ugb_instead_of_reporting_noise():
    freqs = _log_sweep(1.0, 1e6, 20)
    gain = [1e-9 * math.sin(i) for i in range(len(freqs))]
    phase = [-30.0] * len(freqs)
    m = measure.ac_metrics(freqs, gain, phase)
    assert m["ugb"] is None
    assert m["phase_margin"] is None
    assert "floating-point noise" in m["notes"]["ugb"]


# ===========================================================================
# N7  a flat span proves you are in A passband, not that it reaches DC
# ===========================================================================

def _ac_coupled_40db(f):
    """Coupling corner 10 Hz, 40 dB mid-band, load pole at 1 MHz."""
    u = 1j * f / 10.0
    return 100.0 * (u / (1 + u)) / (1 + 1j * f / 1e6)


class TestN7DcGainMustNotBeCertifiedByFlatnessAlone:
    """dc_gain_valid = 1.0 for an amplifier whose gain at DC is exactly zero.

        f_start 1 Hz     low_slope +19.7492  valid 0.0  dc_gain None   correct
        f_start 1 kHz    low_slope  +0.0016  valid 1.0  dc_gain +39.9996
        f_start 10 kHz   low_slope  -0.0025  valid 1.0  dc_gain +39.9996

    A series capacitor makes the true gain at DC -inf dB. spec_extract emitted
    39.9996 with no note at all, because the dc_gain_db note is written only on
    a refusal. This is D3/D5 inverted: D4 refused when it should have reported,
    and this reports when it must refuse.

    The magnitude cannot see the coupling corner from 100x above it (2e-4 dB),
    but the PHASE can: the corner leaves atan(fz/f) of lead that decays as 1/f,
    and separating that 1/f term from the -f lag of the poles above the sweep
    identifies it exactly.
    """

    @pytest.mark.parametrize("fstart", [1e3, 1e4])
    def test_an_ac_coupled_stage_is_refused_however_flat_it_looks(self, fstart):
        freqs = _log_sweep(fstart, 1e8, 400)
        gain_db, phase = _response(freqs, _ac_coupled_40db)
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert abs(m["low_slope_db_per_dec"]) < measure.DC_SLOPE_TOL_DB_PER_DEC
        assert m["dc_gain_valid"] == 0.0
        assert m["dc_gain_db"] is None
        note = m["notes"]["dc_gain_db"]
        assert "does not reach DC" in note
        assert "1/f LEAD" in note
        # The note names the coupling corner it found, to within a factor 2.
        assert "AC-coupling capacitor" in note

    def test_the_mid_band_gain_is_still_reported_under_its_own_name(self):
        freqs = _log_sweep(1e3, 1e8, 400)
        gain_db, phase = _response(freqs, _ac_coupled_40db)
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["low_freq_gain_db"] == pytest.approx(39.9996, abs=1e-3)
        assert m["passband_gain_db"] == pytest.approx(40.0, abs=0.01)

    def test_spec_extract_reports_it_as_unmeasurable_with_the_reason(self):
        freqs = _log_sweep(1e3, 1e8, 400)
        gain_db, phase = _response(freqs, _ac_coupled_40db)
        ac = {"frequencies": freqs,
              "signals": {"vdb(out)": {"name": "vdb(out)", "x_values": freqs,
                                       "y_values": gain_db},
                          "vp(out)": {"name": "vp(out)", "x_values": freqs,
                                      "y_values": phase}}}
        ext = sx.extract_specs({"dc_gain": {"min": 30, "unit": "dB"}},
                               ac=ac, output_signal="out")
        assert "dc_gain" not in ext.values
        assert "does not reach DC" in ext.unmeasurable["dc_gain"]

    def test_a_genuinely_dc_coupled_response_is_still_certified(self):
        """The guard must not cost the case it was built for."""
        fp = 1591.5494309189535
        freqs = _log_sweep(1.0, 1e6, 100)
        gain_db, phase = _response(freqs, lambda f: 1.0 / (1 + 1j * f / fp))
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["dc_gain_valid"] == 1.0
        assert m["dc_gain_db"] == pytest.approx(0.0, abs=1e-4)
        assert m["bandwidth_3db"] == pytest.approx(fp, rel=1e-3)

    def test_a_zero_ABOVE_the_sweep_does_not_trip_the_guard(self):
        """A left-half-plane zero above f_start also leads -- but grows with f.

        Only the 1/f component means a zero BELOW the sweep, and only a zero
        below the sweep blocks DC.
        """
        freqs = _log_sweep(1.0, 1e6, 200)
        gain_db, phase = _response(
            freqs, lambda f: 100.0 * (1 + 1j * f / 5e3)
            / (1 + 1j * f / 100.0) / (1 + 1j * f / 1e6))
        m = measure.ac_metrics(freqs, gain_db, phase)
        assert m["dc_gain_valid"] == 1.0
        assert m["dc_gain_db"] == pytest.approx(40.0, abs=0.01)


# ===========================================================================
# N15  a band-stop response has no single -3 dB bandwidth
# ===========================================================================

def test_n15_a_band_stop_does_not_report_its_stop_band_edge_as_a_bandwidth():
    freqs = _log_sweep(1.0, 1e6, 200)
    gain_db, phase = _response(
        freqs,
        lambda f: (1 + (1j * f / 1e3) ** 2 + 1e-6j * f / 1e3)
        / (1 + 1j * f / 700.0 + (1j * f / 1e3) ** 2))
    m = measure.ac_metrics(freqs, gain_db, phase)
    assert m["bandwidth_3db"] is None
    assert m["f_3db_hi"] is not None          # the first edge is still reported
    note = m["notes"]["bandwidth_3db"]
    assert "NOTCH" in note
    assert "stop band" in note


# ===========================================================================
# N2 / N10 / N11  every pulse and periodic transient
# ===========================================================================

PULSE_DECK = """* the repo's own transient deck, scripts/agent_ngspice.py
V1 in 0 PULSE(0 1.8 0 1n 1n 5u 10u)
R1 in out 1k
C1 out 0 1n
.tran 0.1u 20u
.end
"""

# tau = R*C = 1 us; the pulse is 5 tau wide, so the top of the first pulse is
# 1.8*(1 - exp(-5)) = 1.7878731 V, not the 1.8 V asymptote.
PULSE_TAU = 1e-6
PULSE_TOP = 1.8 * (1.0 - math.exp(-5.0))


@skipif_no_ngspice
class TestN2PulseTransientsAreMeasuredOnTheFirstEdge:
    """settled_levels() took the mean of the LAST 2 pct as "the final level".

    Any waveform that returns to where it started therefore had a step of
    almost nothing, and the whole 10/90 ladder was referenced to a
    microvolt-scale span. On this deck, run for real:

        rise_time      6.39632e-09 s   against 2.19722e-06 s   (343x fast)
        overshoot_pct  12374.85 pct    against 0 pct
        slew_rate      1.793 V/us      against 0.655 V/us
        y_final        0.01433 V       against 1.8 V

    and spec_extract emitted all three as measured values. This is every
    inverter, buffer, ring oscillator and clocked block in the eval set.
    """

    @pytest.fixture
    def metrics(self, adapter):
        result = adapter.tran(PULSE_DECK, SimParams(analysis_type="tran"))
        return adapter.measure_tran(result, "out"), result

    def test_the_waveform_is_recognised_as_a_pulse(self, metrics):
        m, result = metrics
        y = result.signals["out"].y_values
        # The premise: it really does return to where it started.
        assert abs(y[-1] - y[0]) < 0.02 * (max(y) - min(y))
        assert m["waveform_kind"] == "pulse"
        assert "RETURNS" in m["notes"]["levels"]

    def test_the_top_level_is_the_plateau_not_the_end_of_the_record(self, metrics):
        m, _ = metrics
        assert m["y_final"] == pytest.approx(PULSE_TOP, rel=5e-3)
        assert m["y_final"] > 1.7           # was 0.01433
        assert m["y_initial"] == pytest.approx(0.0, abs=1e-3)

    def test_rise_time_is_the_rise_time(self, metrics):
        m, _ = metrics
        # 10-90 of the OBSERVED excursion (IEEE 181 base and top). The
        # asymptotic ln(9)*tau = 2.19722 us is 2.7 pct larger because the pulse
        # is only 5 tau wide, so the top is 1.78787 V and not 1.8 V.
        ideal = PULSE_TAU * math.log(9.0)
        assert m["rise_time"] == pytest.approx(ideal, rel=0.05)
        assert m["rise_time"] > 2.0e-6      # was 6.4e-9

    def test_overshoot_is_zero_on_a_monotone_edge(self, metrics):
        m, _ = metrics
        assert m["overshoot_pct"] == 0.0    # was 12374.85

    def test_slew_rate_matches_the_chord(self, metrics):
        m, _ = metrics
        chord = 0.8 * (m["y_final"] - m["y_initial"]) / m["rise_time"]
        assert m["slew_rate"] == pytest.approx(chord, rel=1e-9)
        assert m["slew_rate"] / 1e6 == pytest.approx(0.655, rel=0.05)

    def test_spec_extract_carries_the_same_numbers(self, adapter):
        result = adapter.tran(PULSE_DECK, SimParams(analysis_type="tran"))
        ext = sx.extract_specs(
            {"rise_time": {"max": 5.0, "unit": "us"},
             "slew_rate": {"min": 0.1, "unit": "V/us"},
             "overshoot": {"max": 10.0, "unit": "%"}},
            tran=result, output_signal="out")
        assert ext.values["rise_time"] == pytest.approx(2.19722, rel=0.05)
        assert ext.values["slew_rate"] == pytest.approx(0.655, rel=0.05)
        assert ext.values["overshoot"] == 0.0


def test_n2_a_plain_step_is_still_measured_as_a_step():
    """The pulse path must not capture a waveform that really does step."""
    tau = 1e-6
    t = [i * 10e-6 / 2000 for i in range(2001)]
    y = [1.0 - math.exp(-x / tau) for x in t]
    m = measure.tran_metrics(t, y)
    assert m["waveform_kind"] == "step"
    assert m["y_final"] == pytest.approx(1.0, abs=1e-3)
    assert m["rise_time"] == pytest.approx(tau * math.log(9.0), rel=1e-3)


def test_n11_a_monotone_waveform_has_exactly_zero_overshoot():
    """max(y) against the mean of the tail invents an overshoot that is not there."""
    tau = 1e-6
    t = [i * 5e-6 / 500 for i in range(501)]
    y = [1.0 - math.exp(-x / tau) for x in t]
    assert measure.overshoot_pct(y) == 0.0


def test_n10_settling_time_is_measured_against_the_first_plateau():
    """A pulse settles inside its own pulse, or not at all.

    Referring the band to the mean of the tail of the record is
    self-referential: the tail is inside its own band by construction.
    """
    tau = 1e-6
    t = [i * 20e-6 / 2000 for i in range(2001)]
    y = [1.8 * (1 - math.exp(-x / tau)) if x < 5e-6
         else 1.8 * (1 - math.exp(-5.0)) * math.exp(-(x - 5e-6) / tau)
         for x in t]
    m = measure.tran_metrics(t, y)
    assert m["waveform_kind"] == "pulse"
    st = m["settling_time"]
    assert st is not None
    # It settles to within 2 pct of the plateau at t = tau*ln(1/0.0198) = 3.92 us
    # and the plateau ends at 5 us; a settling time outside that window would be
    # measured against the wrong level.
    assert 3.0e-6 < st < 5.0e-6


def test_l3_overshoot_of_a_ringing_step_is_unchanged():
    """The original L3 guard, restated: the pulse path must not touch it."""
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
    analytic = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))
    m = measure.tran_metrics(t, [step(x) for x in t])
    assert m["waveform_kind"] == "step"
    assert m["overshoot_pct"] == pytest.approx(analytic, rel=2e-3)


# ===========================================================================
# N5 / N6 / N8 / N9  supply current
# ===========================================================================

DUAL_RAIL = """* dual rail, 3.6k across the rails: ONE current of 1.000 mA
Vdd vdd 0 DC 1.8
Vss vss 0 DC -1.8
R1 vdd vss 3.6k
.op
.end
"""

SUBCKT_NAME_COLLISION = """* top rail V1, and a 0 V ammeter also called V1 inside blk
V1 vdd 0 DC 1.8
X1 vdd 0 blk
.subckt blk p n
V1 p mid DC 0
R1 mid n 1k
.ends
.end
"""

SUBCKT_AMMETER = """* a 0 V ammeter inside a subckt, with no name collision at all
V1 vdd 0 DC 1.8
X1 vdd 0 blk
.subckt blk p n
Vsense p mid DC 0
R1 mid n 1k
.ends
.end
"""

PULSED_RAIL = """* a rail declared with PULSE, and a real 0 V sense source
Vdd vdd 0 PULSE(1.8 0 1m 1n 1n 1m 2m)
Vsense vdd out DC 0
R1 out 0 1k
.op
.end
"""


class TestN5SupplyCurrentIsNotDoubleCounted:
    """Summing the magnitude of every rail counts a dual supply twice.

        Vdd 1.8 / Vss -1.8 / 3.6k across the rails: hand 1.000 mA, was 2.000 mA

    A design drawing 1.8 mA against a 2 mA budget was scored at 3.6 mA and
    failed. The netlist carries the polarities and was not consulted; the
    "less than a thousandth of the largest" heuristic cannot fire here, because
    both branches carry the SAME current.
    """

    OP = {"vdd#branch": -1.0e-3, "vss#branch": 1.0e-3, "vdd": 1.8, "vss": -1.8}

    def test_the_negative_rail_is_the_return_path_not_a_second_supply(self):
        rep = measure.supply_current_report(self.OP, netlist=DUAL_RAIL)
        assert rep.value == pytest.approx(1.0e-3, rel=1e-9)
        assert rep.sources == ["vdd"]
        assert "return path" in rep.excluded["vss"]
        assert rep.warnings == []

    def test_two_positive_rails_are_still_summed(self):
        """A 1.8 V analog rail and a 3.3 V IO rail really are two supplies."""
        deck = ("* two rails\nV1 vdd 0 DC 1.8\nV2 vio 0 DC 3.3\n"
                "R1 vdd 0 1k\nR2 vio 0 1k\n.end\n")
        rep = measure.supply_current_report(
            {"v1#branch": -1.8e-3, "v2#branch": -3.3e-3}, netlist=deck)
        assert rep.value == pytest.approx(5.1e-3, rel=1e-9)

    def test_without_a_netlist_the_double_count_is_at_least_flagged(self):
        rep = measure.supply_current_report({"v1#branch": -1e-3,
                                             "v2#branch": 1e-3})
        # The historical answer is kept -- polarity is genuinely unknowable
        # from an operating point -- but it is no longer silent about it.
        assert rep.value == pytest.approx(2e-3, rel=1e-12)
        joined = " ".join(rep.warnings)
        assert "exactly 2x" in joined
        assert rep.ambiguous


class TestN6SubcircuitScope:
    """_parse_source_cards let .subckt/.ends through and parsed the body flat.

    With a top rail V1 and a 0 V ammeter also called V1 inside blk, the
    subcircuit entry OVERWROTE the top-level one, so the REAL rail was excluded
    as "declared DC 0 ... 0 V sense source" and the ammeter was kept:

        branches  v1#branch = -1.801800e-03   v.x1.v1#branch = 1.799998e-06
        reported  1.799998e-06 A              hand 1.801800e-03 A   (1000x low)

    with an empty warnings list, so `ambiguous` was False. That is D7's own
    failure mode restored, and made worse: before this fix the same deck read
    1.80360e-3, only 0.1 pct high.
    """

    def test_a_subckt_source_does_not_overwrite_a_top_level_rail(self):
        op = {"v1#branch": -1.8018e-3, "v.x1.v1#branch": 1.799998e-06}
        rep = measure.supply_current_report(op, netlist=SUBCKT_NAME_COLLISION)
        assert rep.value == pytest.approx(1.8018e-3, rel=1e-9)
        assert rep.sources == ["v1"]
        assert "v.x1.v1" in rep.excluded

    def test_a_subckt_ammeter_is_matched_by_its_hierarchical_name(self):
        """v.x1.vsense was never found in the source table, so it was summed."""
        op = {"v1#branch": -1.8e-3, "v.x1.vsense#branch": 1.8e-3}
        rep = measure.supply_current_report(op, netlist=SUBCKT_AMMETER)
        assert rep.value == pytest.approx(1.8e-3, rel=1e-9)
        assert "v.x1.vsense" in rep.excluded

    def test_the_parser_keeps_the_two_scopes_apart(self):
        deck = measure.parse_deck_sources(SUBCKT_NAME_COLLISION)
        assert deck.top["v1"] == pytest.approx(1.8)
        assert deck.subckt_v["blk"]["v1"] == 0.0
        assert deck.instances["x1"] == "blk"
        assert measure.source_dc_value("v1", deck) == pytest.approx(1.8)
        assert measure.source_dc_value("v.x1.v1", deck) == 0.0


class TestN9ATransientSourceIsNotASenseSource:
    """"Vdd vdd 0 PULSE(1.8 0 ...)" has an operating-point value of 1.8, not 0.

    Reading it as DC 0 excluded the real rail as a 0 V sense source.
    """

    @pytest.mark.parametrize("card,expected", [
        ("Vdd vdd 0 PULSE(1.8 0 1m 1n 1n 1m 2m)", 1.8),
        ("Vdd vdd 0 SIN(1.8 0.1 1k)", 1.8),
        ("Vdd vdd 0 EXP(1.8 0 1m 1u 2m 1u)", 1.8),
        ("Vdd vdd 0 PWL(0 1.8 1m 1.8)", 1.8),
        ("Vin in 0 AC 1", 0.0),
        ("Vsense a b DC 0", 0.0),
        ("V1 a b", 0.0),
    ])
    def test_the_operating_point_value_of_a_v_card(self, card, expected):
        deck = measure.parse_deck_sources(f"* t\n{card}\n.end\n")
        name = card.split()[0].lower()
        assert deck.top[name] == pytest.approx(expected)

    def test_a_pulsed_rail_is_kept_and_the_sense_source_excluded(self):
        rep = measure.supply_current_report(
            {"vdd#branch": -1.8e-3, "vsense#branch": 1.8e-3},
            netlist=PULSED_RAIL)
        assert rep.value == pytest.approx(1.8e-3, rel=1e-9)
        assert rep.sources == ["vdd"]
        assert "vsense" in rep.excluded


@skipif_no_ngspice
class TestSupplyCurrentAgainstNgspice:
    """N5, N6 and N8 end to end: the adapter must find its own deck."""

    def test_n5_dual_rail(self, adapter):
        result = adapter.dc(DUAL_RAIL, SimParams(analysis_type="dc"))
        assert adapter.measure_idd(result) == pytest.approx(1.0e-3, rel=1e-9)

    def test_n6_subckt_name_collision(self, adapter):
        result = adapter.dc(SUBCKT_NAME_COLLISION, SimParams(analysis_type="dc"))
        assert "v.x1.v1#branch" in result.op_points
        assert adapter.measure_idd(result) == pytest.approx(1.8e-3, rel=1e-6)

    def test_n8_a_deck_holding_a_subckt_source_still_matches_its_own_result(
            self, adapter):
        """_netlist_for compared 'v.x1.vsense' against top-level cards only.

        No deck has an element card called that, so the deck failed to match
        its OWN result, no netlist was applied, and the ammeter was summed as a
        second supply: exactly 2x.
        """
        result = adapter.dc(SUBCKT_AMMETER, SimParams(analysis_type="dc"))
        assert adapter._netlist_for(result) is not None
        assert adapter.measure_idd(result) == pytest.approx(1.8e-3, rel=1e-6)

    def test_a_stale_netlist_is_still_never_applied(self, adapter):
        """N8's fix must not weaken the guard it relaxes."""
        deck = ("* divider\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\n"
                "R2 out 0 10k\n.op\n.end\n")
        result = adapter.dc(deck, SimParams(analysis_type="dc"))
        adapter.dc(SUBCKT_AMMETER, SimParams(analysis_type="dc"))
        assert adapter.measure_idd(result) == pytest.approx(90e-6, rel=1e-9)


# ===========================================================================
# C2  nothing non-finite may enter a pydantic result or a JSON dump
# ===========================================================================

DC_RAIL_AC_DECK = """* an AC 1 input and a DC-only supply rail
Vin in 0 DC 0 AC 1
Vdd vdd 0 DC 1.8
R1 in out 1k
R2 out vdd 1k
C1 out 0 1n
.ac dec 10 1 1e6
.end
"""


def _strict(value):  # pragma: no cover - only called on a failure
    raise ValueError(f"not valid JSON: {value}")


@skipif_no_ngspice
class TestC2NonFiniteValuesCannotBeSerialised:
    """_build_ac ran transfer_function over EVERY AC vector.

    L4 changed the zero cases from -6000.0/0.0 to -inf/NaN, which is right for
    the metric layer. It is not right for a pydantic result: on this ordinary
    deck three of seven signals were non-finite at all 61 samples, and
    rl_env.py did json.dumps(result, default=str) on it and handed the model
    183 -Infinity and 183 NaN tokens. Python tolerates that; json.loads with a
    strict parse_constant, jq, JavaScript and the HuggingFace datasets loader
    do not, and the same payload reaches data/trajectory.py and the FROZEN
    data/format.py, so a trajectory-to-SFT conversion could write -Infinity
    into a .jsonl.
    """

    def test_no_ac_signal_carries_a_non_finite_sample(self, adapter):
        result = adapter.ac(DC_RAIL_AC_DECK, SimParams(analysis_type="ac"))
        for name, sig in result.signals.items():
            assert all(math.isfinite(v) for v in sig.y_values), name
            assert all(math.isfinite(v) for v in sig.x_values), name
            assert len(sig.x_values) == len(sig.y_values), name

    def test_the_result_survives_a_strict_json_round_trip(self, adapter):
        result = adapter.ac(DC_RAIL_AC_DECK, SimParams(analysis_type="ac"))
        text = json.dumps(result.model_dump(), allow_nan=False)
        json.loads(text, parse_constant=_strict)

    def test_a_signal_with_no_ac_content_is_dropped_not_reported_as_minus_inf(
            self, adapter):
        result = adapter.ac(DC_RAIL_AC_DECK, SimParams(analysis_type="ac"))
        # The DC-only rail has no transfer function at any frequency.
        assert "vdb(vdd)" not in result.signals
        # The signal that does have one is still there.
        assert "vdb(out)" in result.signals


def test_c2_the_serialisation_boundary_replaces_non_finite_floats():
    payload = {"a": float("-inf"), "b": [1.0, float("nan"), float("inf")],
               "c": {"d": 2.0}}
    assert serialization.count_non_finite(payload) == 3
    text = serialization.dumps(payload)
    back = json.loads(text, parse_constant=_strict)
    assert back == {"a": None, "b": [1.0, None, None], "c": {"d": 2.0}}


def test_c2_rl_env_never_emits_a_non_finite_token():
    from asic_ai.training import rl_env

    class _Adapter:
        def ac(self, netlist, args):
            return {"frequencies": [1.0, 2.0],
                    "vdb(vdd)": [float("-inf"), float("-inf")],
                    "vp(vdd)": [float("nan"), float("nan")]}

    env = rl_env.CircuitDesignEnv(_Adapter(), reward_fn=None)
    env.reset({"id": "t", "specs": {}})
    obs, ok = env._execute_tool("sim.ac", {"netlist": "* x\n.end\n"})
    assert ok
    assert "-Infinity" not in obs and "NaN" not in obs
    json.loads(obs, parse_constant=_strict)


# ===========================================================================
# C4  an inline netlist must reach spec.check
# ===========================================================================

SENSE_DECK = """* Idd metered by a 0 V sense source: the true supply current is 198 uA
VDD vdd 0 DC 1.8
R1 vdd 0 100k
Vsense vdd n1 DC 0
R2 n1 0 10k
.op
.end
"""


@skipif_no_ngspice
class TestC4InlineNetlistReachesSpecCheck:
    """state.netlist was written ONLY by netlist.patch.

    _run_sim read the netlist from args and never wrote it back, so an agent
    that called sim.dc with an inline netlist -- which the schema allows and
    the SFT data demonstrates -- handed spec.check a netlist of None. Without
    the deck, supply_current cannot tell the 0 V ammeter from the rail:

        netlist.patch first = True  -> idd 198.0 uA
        netlist.patch first = False -> idd 378.0 uA   (91 pct high)
    """

    SPECS = {"idd": {"max": 250.0, "unit": "uA"}}

    def _env(self, adapter):
        from asic_ai.training import rl_env
        env = rl_env.CircuitDesignEnv(adapter, reward_fn=None)
        env.reset({"id": "sense", "specs": self.SPECS})
        return env

    def test_idd_is_the_same_with_and_without_netlist_patch(self, adapter):
        from asic_ai.training import rl_env

        env = self._env(adapter)
        env.step({"name": "netlist.patch", "arguments": {"netlist": SENSE_DECK}})
        env.step({"name": "sim.dc", "arguments": {"netlist": SENSE_DECK}})
        with_patch = env._run_spec_check({"specs": self.SPECS})

        env2 = self._env(adapter)
        env2.step({"name": "sim.dc", "arguments": {"netlist": SENSE_DECK}})
        without = env2._run_spec_check({"specs": self.SPECS})

        assert with_patch["measured"]["idd"] == pytest.approx(198.0, rel=1e-3)
        assert without["measured"]["idd"] == pytest.approx(198.0, rel=1e-3)
        assert without["measured"]["idd"] == with_patch["measured"]["idd"]


# ===========================================================================
# C5  _summarize_run still guessed the sweep axis
# ===========================================================================

MC_TEMP_DECK = """* a temperature sweep with a statistical resistor
V1 vdd 0 DC 1.8
R1 vdd out {agauss(10k, 100, 1)}
R2 out 0 10k
.dc temp -40 125 5
.end
"""


@skipif_no_ngspice
class TestC5MonteCarloSweepAxis:
    """"v-sweep if present else time" is the heuristic _dc_sweep_axis replaced.

    Reachable through mc() -> MonteCarloResult.results, where a temperature or
    current sweep axis was summarised as though it were a measured circuit
    quantity ('temp-sweep.min = -40.0').
    """

    def test_the_temperature_axis_is_not_reported_as_a_circuit_quantity(
            self, adapter):
        result = adapter.mc(MC_TEMP_DECK, n=2, seed=0)
        assert result.results
        for run in result.results:
            assert "temp-sweep.min" not in run
            assert "temp-sweep.max" not in run
        # The real measured node is still summarised.
        assert any(k.startswith("out.") for k in result.results[0])


# ===========================================================================
# C1  the training data must teach ONE corner convention
# ===========================================================================

class TestC1TrainingDataCornerConvention:
    """data/sft/train_final.jsonl taught both conventions six records apart.

        :873  "SS corner at -40C is the worst case ...", a table TT 27C /
              SS -40C / FF 125C, and "Root cause at SS -40C: loop gain drops
              23 dB" -- the OLD convention with a causal justification attached.
        :879  TT (27C) / SS (125C) -- the NEW one.

    The generator is scripts/generate_debug_sft.py; the data is regenerated
    from it, never hand-edited.
    """

    OLD = ("ss_n40", "ff_125", "SS -40C", "FF 125C", "SS (-40", "FF (125")

    def _sft_files(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        return sorted((root / "data" / "sft").glob("*.jsonl"))

    def test_no_training_file_teaches_the_old_convention(self):
        offenders = []
        for path in self._sft_files():
            text = path.read_text(encoding="utf-8")
            for token in self.OLD:
                if token in text:
                    offenders.append(f"{path.name}: {token!r}")
        assert not offenders, offenders

    def test_no_generator_writes_the_old_convention(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for path in sorted((root / "scripts").glob("generate_*.py")):
            text = path.read_text(encoding="utf-8")
            for token in self.OLD:
                if token in text:
                    offenders.append(f"{path.name}: {token!r}")
        assert not offenders, offenders

    def test_the_corner_example_is_present_and_uses_the_new_convention(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        text = (root / "data" / "sft" / "debug_v1.jsonl").read_text(
            encoding="utf-8")
        assert "SS 125C" in text
        assert "ss_125" in text and "ff_n40" in text


# ===========================================================================
# N13  a noise density is a value AT A FREQUENCY
# ===========================================================================

class TestN13NoiseDensityIsNotWhateverFStartHappensToBe:
    """input_noise_density was spectrum[0]: D3/D5, unfixed, for noise.

    The same circuit reports its 1/f corner from a sweep that starts at 1 Hz
    and its thermal floor from one that starts at 10 kHz, and nothing said
    which. A density is a value at a frequency, so either the caller names the
    frequency or the spectrum has to be demonstrably flat where it is read.
    """

    FC = 1e3          # 1/f corner
    FLOOR = 2e-9      # V/sqrt(Hz)
    DECK = "* t\nVin in 0 DC 0 AC 1\nR1 in out 1k\n.noise v(out) Vin dec 50 1 1e6\n.end\n"

    def _noise(self, fstart, fstop=1e6, per_dec=50):
        n = int(round(math.log10(fstop / fstart) * per_dec))
        freqs = [fstart * 10.0 ** (i / per_dec) for i in range(n + 1)]
        spec = [self.FLOOR * math.sqrt(1.0 + self.FC / f) for f in freqs]
        return {"frequencies": freqs,
                "input_noise": {"name": "inoise_spectrum", "x_values": freqs,
                                "y_values": spec},
                "output_noise": {"name": "onoise_spectrum", "x_values": freqs,
                                 "y_values": spec}}

    def test_a_sloping_spectrum_is_refused_not_reported(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                               noise=self._noise(1.0), netlist=self.DECK)
        assert "noise" not in ext.values
        note = ext.unmeasurable["noise"]
        assert "1/f corner" in note
        assert "noise_freq" in note

    def test_a_flat_spectrum_is_still_reported(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                               noise=self._noise(1e5), netlist=self.DECK)
        assert ext.values["noise"] == pytest.approx(2.0, rel=0.02)

    def test_a_named_frequency_is_honoured(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                               noise=self._noise(1.0), netlist=self.DECK,
                               noise_freq=1e4)
        expected = self.FLOOR * math.sqrt(1.0 + self.FC / 1e4) / 1e-9
        assert ext.values["noise"] == pytest.approx(expected, rel=0.01)

    def test_the_answer_no_longer_depends_on_where_the_sweep_started(self):
        at_1hz = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                                  noise=self._noise(1.0), netlist=self.DECK,
                                  noise_freq=1e4).values["noise"]
        at_1khz = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                                   noise=self._noise(1e3), netlist=self.DECK,
                                   noise_freq=1e4).values["noise"]
        assert at_1hz == pytest.approx(at_1khz, rel=1e-9)

    def test_the_integrated_rms_is_unaffected(self):
        ext = sx.extract_specs({"noise_rms": {"max": 100, "unit": "uV"}},
                               noise=self._noise(1.0), netlist=self.DECK)
        assert ext.values["noise_rms"] > 0.0


# ===========================================================================
# C6  a mapping defect: an input-referred noise density has two unit families
# ===========================================================================

class TestC6NoiseDensityUnitFamily:
    """One _SCALES table held V/sqrt(Hz) AND pA/sqrt(Hz) for the same metric.

    ngspice refers inoise_spectrum to the source named on the .noise card, so
    the measurement is a VOLTAGE density for 'noise v(out) Vin' and a CURRENT
    density for 'noise v(out) Iin'. With both families in one table, the TIA
    task's `unit: pA/sqrt(Hz)` divided a V/sqrt(Hz) measurement by 1e-12: a
    silent error of twelve orders of magnitude, in the direction that makes
    every design look like it passes.
    """

    FLAT = {"frequencies": [1e3, 1e4, 1e5],
            "input_noise": {"name": "inoise_spectrum",
                            "x_values": [1e3, 1e4, 1e5],
                            "y_values": [2e-9, 2e-9, 2e-9]}}
    V_DECK = "* v\nVin in 0 AC 1\nR1 in out 1k\n.noise v(out) Vin dec 10 1 1e6\n.end\n"
    I_DECK = "* i\nIin 0 in AC 1\nR1 in out 1k\n.noise v(out) Iin dec 10 1 1e6\n.end\n"

    def test_the_noise_card_decides_the_family(self):
        assert sx.noise_input_kind(self.V_DECK) == "v"
        assert sx.noise_input_kind(self.I_DECK) == "i"
        assert sx.noise_input_kind(None) is None

    def test_a_current_unit_is_refused_for_a_voltage_driven_noise_run(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "pA/sqrt(Hz)"}},
                               noise=self.FLAT, netlist=self.V_DECK)
        assert "noise" not in ext.values
        assert "not a noise_density_v unit" in ext.unmeasurable["noise"]

    def test_a_current_unit_is_accepted_for_a_current_driven_noise_run(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "pA/sqrt(Hz)"}},
                               noise=self.FLAT, netlist=self.I_DECK)
        assert ext.values["noise"] == pytest.approx(2e-9 / 1e-12)

    def test_without_the_netlist_the_family_is_unknown_and_it_says_so(self):
        ext = sx.extract_specs({"noise": {"max": 10, "unit": "nV/sqrt(Hz)"}},
                               noise=self.FLAT)
        assert "noise" not in ext.values
        assert "1e12" in ext.unmeasurable["noise"]


def test_c6_the_stb_docstring_no_longer_claims_a_dc_normalisation():
    """It said "its DC value normalised to ~0 deg", which is the removed rule."""
    from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter
    doc = " ".join((NgspiceSharedAdapter.stb.__doc__ or "").split())
    assert "normalised to ~0 deg" not in doc
    assert "LOOP CLOSURE" in doc
    assert "BOTTOM of the sweep" in doc


def test_c6_crossing_freq_no_longer_claims_ac_metrics_uses_it_for_ugb():
    doc = " ".join((measure.crossing_freq.__doc__ or "").split())
    assert "unity-gain crossing ABOVE the peak-gain frequency" not in doc
    assert "closure_freq" in doc
