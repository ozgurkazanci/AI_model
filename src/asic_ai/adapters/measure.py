"""Scalar measurement helpers for simulation waveforms.

Pure Python. No ctypes, no simulator dependency, so every function here can be
unit tested against an analytic waveform. The ngspice adapter uses these to turn
raw vectors into the scalar metrics the reward function consumes
(dc_gain_db, ugb, phase_margin, gain_margin, bandwidth_3db, idd, rise_time, ...).

Conventions used throughout this module:

  - Frequency in Hz, time in seconds, gain in dB (20*log10), phase in degrees.
  - Frequency-domain interpolation is linear in (log10(f), value) space, which
    is what SPICE post-processors use. It is NOT exact for a single-pole
    rolloff: the -3 dB frequency it recovers is biased LOW, and the bias is a
    function of sweep density only. Measured against an RC with
    fp = 1591.5494309 Hz: dec 5 -> -7.6e-4 relative, dec 10 -> -4.2e-4,
    dec 100 -> -3.9e-5, dec 1000 -> +3.7e-9. The bias is systematic and does
    NOT average out across RL candidates, so score candidates against each
    other only at equal sweep density, and use >= 100 points/decade when the
    bandwidth itself is being optimised.
  - The -3 dB point uses the exact constant 10*log10(2) = 3.010299956639812,
    never the rounded 3.0. The rounded constant biases f_3db by about 0.24 pct
    against a -20 dB/decade slope, which is large enough to reorder two
    near-equal RL candidates.
  - Phase is ALWAYS unwrapped before a phase margin is taken. A wrapped atan2
    turns a genuinely unstable loop (true phase -200 deg, reported +160 deg)
    into a phase margin of 340 deg, which an RL policy will happily exploit.
  - A metric that is not defined for the data (no 0 dB crossing, no -180 deg
    crossing, no rising edge) returns None. It never returns 0.0, because 0.0
    is a meaningful and usually terrible value for these metrics.
  - "Undefined" is signalled two ways, deliberately kept apart. A SCALAR METRIC
    that the data does not define is None. A per-sample GAIN whose magnitude is
    exactly zero is -inf (a real, ordered value that arithmetic can carry), and
    a per-sample PHASE that does not exist at all -- a zero divisor in
    transfer_function, or a zero-magnitude response -- is NaN. Every scan in
    this module skips NaN samples rather than letting them poison a result.

Nothing here assumes the sweep starts at DC
-------------------------------------------
ac_metrics() used to take gain_db[0] as "the DC gain" and reference every other
metric to it. That is false for any AC-coupled or band-pass response, and it is
false for a sweep that simply starts above the dominant pole. All four
consequences (a phase margin 180 deg out, a suppressed ugb, a "valid" DC gain
that is 36 dB wrong, a bandwidth 25x high) came from that one assumption, so it
is gone:

  - dc_gain_db is reported ONLY when the sweep demonstrably reaches DC, judged
    by the low-frequency SLOPE (see low_frequency_slope) AND by the absence of
    a low-frequency phase LEAD (see low_frequency_phase_lead), not by comparing
    two adjacent samples. Otherwise it is None and notes["dc_gain_db"] says
    why; the gain at the bottom of the sweep is always available as
    low_freq_gain_db, under a name that does not claim to be a DC gain.
  - The -3 dB edges are referenced to the passband gain, which is the DC gain
    when the sweep reaches DC and the peak gain otherwise.
  - The unity-gain frequency is the LOOP CLOSURE: the last frequency at which
    the gain is above 0 dB. Not the first crossing, which on a notched loop
    gain reports a phase margin that is 100 deg too optimistic.
  - The inversion of an inverting loop is inferred from the phase at the BOTTOM
    of the sweep measured against the minimum-phase Bode estimate for the
    magnitude slope there (90 deg per 20 dB/decade). See
    phase_inversion_shift. Judging it from any single sample against a fixed
    0 deg -- the first sample, or the peak-gain sample -- fabricates a 180 deg
    inversion on every response whose phase happens to pass near +/-180 there,
    which is exactly where the peak of a resonance sits.

Every ac_metrics() key that is None for a reason carries that reason in the
returned dict under "notes". Consumers already handle None; they cannot handle
a confident lie.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

log = logging.getLogger(__name__)

# 10*log10(2). The half-power point, exactly.
DB3 = 10.0 * math.log10(2.0)

# "Does this sweep reach DC" guard. The low-frequency slope is measured over
# DC_SLOPE_SPAN_DEC decades from the bottom of the sweep. For a single pole at
# fp the gain error at f0 is |dc_error| = 4.343*(f0/fp)^2 dB while the local
# slope is 20*(f0/fp)^2 dB/dec, so |dc_error| ~= 0.217*|slope|. A tolerance of
# 0.25 dB/dec therefore certifies the reported DC gain to about 0.054 dB.
DC_SLOPE_SPAN_DEC = 0.1
DC_SLOPE_TOL_DB_PER_DEC = 0.25

# Below this the reference phase is too far from a multiple of 180 deg for an
# inversion to be inferred at all, and no shift is applied. See
# phase_inversion_shift.
PHASE_INVERSION_TOL_DEG = 60.0

# Minimum-phase Bode relation, degrees of phase per dB/decade of magnitude
# slope: 90 deg per 20 dB/dec. This is what makes an inversion inferable at
# ALL: it says what the phase SHOULD be at the bottom of the sweep given the
# magnitude there, so a departure of a whole 180 deg is evidence of a sign
# inversion rather than of accumulated pole lag.
BODE_DEG_PER_DB_PER_DEC = 4.5

# A minimum-phase response that reaches DC cannot LEAD its own magnitude slope
# at the bottom of the sweep. A residual lead that DECAYS as 1/f is the
# signature of a zero below f_start -- an AC-coupling capacitor -- and a zero
# below f_start means the gain at DC is zero, whatever the flat magnitude at
# f_start says. A coupling corner 100x below f_start moves the magnitude by
# only 2e-4 dB (undetectable) but still leaves atan(1/100) = 0.57 deg of lead,
# so the phase is the only instrument fine enough for this.
DC_PHASE_LEAD_TOL_DEG = 0.01
DC_PHASE_LEAD_RATIO = 2.0

# The RETURN EDGE of a pulse is identified by its RATE, measured against the one
# segment that is certainly on it: the segment that crosses the 50 pct level.
# A top that is still settling descends too, so a walk-back that only asks
# "is this sample lower than the last" runs straight through a lead network's
# droop, an AC-coupled stage's droop or any pulse with top tilt, and lands on
# the overshoot peak. Any segment slower than this fraction of the crossing
# segment is a plateau, not an edge: real return edges run at 1x to 100x the
# crossing rate, real droops at 1e-3 of it or less.
RETURN_EDGE_RATE_FRAC = 0.1

# How closely a negative rail's current has to match the positive rails' before
# it is called their RETURN PATH rather than a supply of its own. Polarity is
# already strong evidence, so this only has to rule out a grossly different
# current: a real dual-rail block leaks some of its current to ground and its
# two rails are near, not equal. The no-netlist `twins` heuristic uses a much
# tighter 0.1 pct because there it is the ONLY evidence there is.
#
# Erring loose costs the R4 direction (an independent rail excluded, idd under
# budget, optimistic); erring tight costs the N5 direction (a return path
# counted twice, idd 2x, pessimistic). The optimistic error is the dangerous
# one for a reward, so this stays well below the smallest genuine imbalance
# either probe shows: R4's case A differs by 50 pct and case C by 75 pct.
RAIL_RETURN_MATCH_FRAC = 0.05


# ----------------------------------------------------------------------------
# Generic numeric helpers
# ----------------------------------------------------------------------------

def _is_nan(v: float) -> bool:
    return v != v


def db20(magnitude: float, floor: Optional[float] = None) -> float:
    """20*log10(|magnitude|).

    An exact zero returns -inf, NOT a large finite number. A finite floor of
    -6000 dB is worse than useless downstream: it survives a mean(), it
    survives a reward, and it reads as "very small gain" instead of "no signal
    at all". Pass an explicit `floor` magnitude only when a caller genuinely
    wants clamping.
    """
    m = abs(float(magnitude))
    if _is_nan(m):
        return math.nan
    if floor is not None and m < floor:
        m = float(floor)
    if m == 0.0:
        return -math.inf
    if math.isinf(m):
        return math.inf
    return 20.0 * math.log10(m)


def unwrap_deg(phase: Sequence[float]) -> list[float]:
    """Remove 2*pi wraps from a phase sequence in degrees.

    Exact no-op when the sequence never wraps, so it is always safe to apply.
    Assumes adjacent samples differ by less than 180 deg, which requires a
    reasonably dense sweep (>= 50 points/decade near a sharp pole pair).

    NaN samples (an undefined phase, see transfer_function) are passed through
    as NaN and do not take part in the unwrap accumulator, so one undefined
    sample cannot shift every sample after it.
    """
    if not phase:
        return []
    out: list[float] = []
    prev_raw: Optional[float] = None
    acc: float = 0.0
    for p in phase:
        v = float(p)
        if not math.isfinite(v):
            out.append(math.nan)
            continue
        if prev_raw is None:
            acc = v
            out.append(acc)
            prev_raw = v
            continue
        d = v - prev_raw
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        acc += d
        out.append(acc)
        prev_raw = v
    return out


def bode_phase_estimate(slope_db_per_dec: Optional[float]) -> Optional[float]:
    """Phase a MINIMUM-PHASE response has where its magnitude has this slope.

    The Bode gain-phase relation, in its one-line asymptotic form: 90 deg of
    lag per 20 dB/decade of rolloff, so 4.5 deg per dB/dec. Exact on a single
    asymptote and at a real pole (-10 dB/dec <-> -45 deg); worst case about
    9 deg per pole half a decade off the corner, and about 25 deg for three
    coincident poles. That is well inside the 60 deg window
    phase_inversion_shift allows, which is the only thing it is used for.

    Returns None when there is no slope to work from.
    """
    if slope_db_per_dec is None:
        return None
    return BODE_DEG_PER_DB_PER_DEC * float(slope_db_per_dec)


def phase_inversion_shift(phase_unwrapped: Sequence[float], ref_index: int = 0,
                          tol_deg: float = PHASE_INVERSION_TOL_DEG,
                          expected_deg: float = 0.0) -> int:
    """How many whole 180 deg turns to remove at ONE reference sample.

    Returns k, so that `phase - 180*k` puts the reference sample near
    `expected_deg`. `expected_deg` is what a NON-inverting minimum-phase
    response would show there; ac_metrics gets it from the magnitude slope via
    bode_phase_estimate(), which is what makes the answer independent of where
    the sweep starts.

    Judging an inversion against a FIXED 0 deg -- the old behaviour, whether
    the reference was sample 0 or the peak-gain sample -- fabricates an
    inversion on any response whose phase passes near +/-180 deg at the
    reference. On a resonant response the peak-gain sample sits at exactly such
    a phase, and the resulting 180 deg error turns an unstable loop
    (PM -68.2 deg) into a comfortable one (+111.8 deg).

    When the residual after removing k turns is further than `tol_deg` from
    `expected_deg` the 180 deg question is UNANSWERABLE from this data, and no
    inversion is claimed. The nearest whole 360 deg turn is still removed
    (k is then even): a 2*pi offset is an artefact of atan2's principal branch,
    never information, and leaving it in is what made the gain margin of a
    plain 3-pole amplifier vanish whenever the sweep started above its poles.

    An ODD k is a real inversion. An EVEN k is a branch correction only.
    """
    n = len(phase_unwrapped)
    if n == 0:
        return 0
    i = min(max(0, ref_index), n - 1)
    p = float(phase_unwrapped[i])
    if not math.isfinite(p) or not math.isfinite(float(expected_deg)):
        return 0
    d = p - float(expected_deg)
    k = int(round(d / 180.0))
    if k == 0:
        return 0
    if abs(d - 180.0 * k) > tol_deg:
        return 2 * int(round(d / 360.0))
    return k


def normalize_dc_phase(phase_unwrapped: Sequence[float]) -> list[float]:
    """Shift an unwrapped phase so the LOWEST-FREQUENCY point sits near 0 deg.

    DEPRECATED for metric work, and no longer used by ac_metrics. It infers the
    inversion of a loop from the phase of the first sweep sample alone, so a
    real phase lead at the bottom of the sweep (two AC-coupling capacitors, for
    instance) is misread as an inversion and the whole curve is moved 180 deg.
    The resulting phase margin depends on where the sweep starts, which no
    metric may do. Use phase_inversion_shift() against a mid-band sample.

    Kept because it is a meaningful transform in its own right for a response
    that genuinely starts at DC.
    """
    if not phase_unwrapped:
        return []
    first = float(phase_unwrapped[0])
    if not math.isfinite(first):
        return [float(p) for p in phase_unwrapped]
    k = round(first / 180.0)
    if k == 0:
        return [float(p) for p in phase_unwrapped]
    return [float(p) - 180.0 * k for p in phase_unwrapped]


def transfer_function(out: Sequence[complex],
                      inp: Optional[Sequence[complex]] = None
                      ) -> tuple[list[float], list[float]]:
    """Complex out/in -> (gain in dB, phase in degrees).

    Divides by the input vector. Do not skip that: an `AC 2` stimulus makes
    |out| alone read 6.02 dB high. When `inp` is None the output is used as-is,
    which is only correct for a unity-amplitude source.

    A sample whose divisor is zero has NO transfer function. It is returned as
    (NaN, NaN), never as (0 dB, 0 deg): a fabricated 0 deg at one frequency is
    indistinguishable from a real measurement and silently moves a phase
    margin. A sample whose ratio is exactly zero is returned as (-inf, NaN):
    the magnitude is genuinely zero, the phase genuinely does not exist.
    """
    gain_db: list[float] = []
    phase_deg: list[float] = []
    undefined = 0
    for i, o in enumerate(out):
        h = complex(o)
        if inp is not None:
            d = complex(inp[i])
            if d == 0:
                undefined += 1
                gain_db.append(math.nan)
                phase_deg.append(math.nan)
                continue
            h = h / d
        if h == 0:
            gain_db.append(-math.inf)
            phase_deg.append(math.nan)
            continue
        gain_db.append(db20(abs(h)))
        phase_deg.append(math.degrees(math.atan2(h.imag, h.real)))
    if undefined:
        log.warning(
            "transfer_function: %d of %d samples have a zero divisor and are "
            "returned as NaN; the stimulus vector is zero there",
            undefined, len(gain_db),
        )
    return gain_db, phase_deg


def _interp(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
    """Linear inverse interpolation: the x where the segment crosses y."""
    if y1 == y0:
        return x0
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0)


def _seg_crosses(a: float, b: float, level: float) -> tuple[bool, bool]:
    """(falling, rising) for one segment, counting a segment that REACHES level.

    A segment that ends exactly on the level counts as a crossing. The old test
    (`a >= level > b`) missed it, and the companion `if a == b: continue` then
    skipped the flat run that followed, so a gain curve that flattens at
    exactly 0 dB reported its unity-gain frequency a full decade late. A run of
    samples that all sit exactly on the level is not itself a crossing (a == b)
    -- the segment that arrived at the level already was.
    """
    if _is_nan(a) or _is_nan(b):
        return False, False
    return (a >= level >= b and a > b), (a <= level <= b and a < b)


def crossing_freq(freqs: Sequence[float], values: Sequence[float], level: float,
                  direction: int = 0, start_index: int = 0) -> Optional[float]:
    """First frequency where `values` crosses `level`, interpolated in log10(f).

    direction: -1 falling only, +1 rising only, 0 either. Returns None when the
    level is never crossed. `start_index` begins the scan at a later sample.

    THE FIRST crossing is not the unity-gain frequency of a loop and not the
    band edge of a response that has several. ac_metrics uses closure_freq()
    for the loop closure and all_crossings() when it has to reason about the
    whole set; this function is for the cases where "the first one" really is
    the answer, such as the upper -3 dB edge of a single-passband response.
    """
    n = min(len(freqs), len(values))
    for i in range(max(0, start_index), n - 1):
        a, b = float(values[i]), float(values[i + 1])
        falling, rising = _seg_crosses(a, b, level)
        if direction < 0 and not falling:
            continue
        if direction > 0 and not rising:
            continue
        if direction == 0 and not (falling or rising):
            continue
        f0, f1 = float(freqs[i]), float(freqs[i + 1])
        if f0 <= 0.0 or f1 <= 0.0:
            return _interp(f0, a, f1, b, level)
        lf = _interp(math.log10(f0), a, math.log10(f1), b, level)
        return 10.0 ** lf
    return None


def last_crossing_freq_below(freqs: Sequence[float], values: Sequence[float],
                             level: float, stop_index: int,
                             direction: int = 0) -> Optional[float]:
    """Crossing of `level` nearest below sample `stop_index`, scanning backwards.

    Used for the LOW-side -3 dB edge of a band-pass response, which is the last
    time the gain climbs through (passband - 3 dB) before the peak.
    """
    n = min(len(freqs), len(values))
    hi = min(stop_index, n - 1)
    for i in range(hi - 1, -1, -1):
        a, b = float(values[i]), float(values[i + 1])
        falling, rising = _seg_crosses(a, b, level)
        if direction < 0 and not falling:
            continue
        if direction > 0 and not rising:
            continue
        if direction == 0 and not (falling or rising):
            continue
        f0, f1 = float(freqs[i]), float(freqs[i + 1])
        if f0 <= 0.0 or f1 <= 0.0:
            return _interp(f0, a, f1, b, level)
        lf = _interp(math.log10(f0), a, math.log10(f1), b, level)
        return 10.0 ** lf
    return None


def all_crossings(freqs: Sequence[float], values: Sequence[float], level: float,
                  direction: int = 0, start_index: int = 0) -> list[float]:
    """EVERY crossing of `level`, interpolated in log10(f), in sweep order.

    crossing_freq() returns the first one. A response can have several -- a
    notched loop gain crosses 0 dB three times -- and which one a metric wants
    is a decision that has to be made explicitly, not by taking whichever came
    first.
    """
    out: list[float] = []
    n = min(len(freqs), len(values))
    for i in range(max(0, start_index), n - 1):
        a, b = float(values[i]), float(values[i + 1])
        falling, rising = _seg_crosses(a, b, level)
        if direction < 0 and not falling:
            continue
        if direction > 0 and not rising:
            continue
        if direction == 0 and not (falling or rising):
            continue
        f0, f1 = float(freqs[i]), float(freqs[i + 1])
        if f0 <= 0.0 or f1 <= 0.0:
            out.append(_interp(f0, a, f1, b, level))
            continue
        out.append(10.0 ** _interp(math.log10(f0), a, math.log10(f1), b, level))
    return out


def closure_freq(freqs: Sequence[float], values: Sequence[float],
                 level: float) -> Optional[float]:
    """The frequency above which `values` stays at or below `level`, forever.

    This is the LOOP CLOSURE, and it is what a unity-gain frequency has to be:
    the last frequency at which the gain is still above unity. Taking the FIRST
    falling crossing instead is optimistic in exactly the dangerous direction.
    On a loop gain with a Q = 25 notch the first crossing is at 618.03 Hz
    (phase margin +89.58 deg, "comfortable"), the loop climbs back above 0 dB,
    and the real closure is at 148554.84 Hz where the phase margin is
    -14.07 deg and the loop is UNSTABLE.

    A run of samples sitting exactly ON the level is not a re-crossing: the
    answer is where the response first ARRIVED at the level, so a gain curve
    that flattens at exactly 0 dB and only later falls away still reports the
    frequency at which it reached 0 dB.

    Returns None when the response is above `level` at the top of the sweep
    (it has not closed inside the sweep) or never above it at all.
    """
    n = min(len(freqs), len(values))
    last_above: Optional[int] = None
    for i in range(n - 1, -1, -1):
        v = float(values[i])
        if _is_nan(v):
            continue
        if v > level:
            last_above = i
            break
    if last_above is None or last_above >= n - 1:
        return None
    nxt: Optional[int] = None
    for j in range(last_above + 1, n):
        if not _is_nan(float(values[j])):
            nxt = j
            break
    if nxt is None:
        return None
    a, b = float(values[last_above]), float(values[nxt])
    f0, f1 = float(freqs[last_above]), float(freqs[nxt])
    if f0 <= 0.0 or f1 <= 0.0:
        return _interp(f0, a, f1, b, level)
    return 10.0 ** _interp(math.log10(f0), a, math.log10(f1), b, level)


def value_at_freq(freqs: Sequence[float], values: Sequence[float],
                  f_target: float, clamp: bool = False) -> Optional[float]:
    """Interpolate `values` at `f_target`, linear in log10(f).

    Returns None when `f_target` lies outside the swept band. It used to return
    the nearest endpoint with no flag, so asking for the gain at 1 GHz on a
    1 Hz..1 kHz sweep answered with the 1 kHz sample and nothing said so. Pass
    clamp=True to opt back into that behaviour deliberately.
    """
    n = min(len(freqs), len(values))
    if n == 0 or f_target <= 0.0:
        return None
    f_lo, f_hi = float(freqs[0]), float(freqs[n - 1])
    if f_target < f_lo:
        return float(values[0]) if clamp else None
    if f_target > f_hi:
        return float(values[n - 1]) if clamp else None
    if f_target == f_lo:
        return float(values[0])
    if f_target == f_hi:
        return float(values[n - 1])
    lt = math.log10(f_target)
    for i in range(n - 1):
        f0, f1 = float(freqs[i]), float(freqs[i + 1])
        if f0 <= 0.0 or f1 <= 0.0:
            continue
        l0, l1 = math.log10(f0), math.log10(f1)
        if l0 <= lt <= l1:
            if l1 == l0:
                return float(values[i])
            t = (lt - l0) / (l1 - l0)
            return float(values[i]) + t * (float(values[i + 1]) - float(values[i]))
    return None


# ----------------------------------------------------------------------------
# AC / frequency domain metrics
# ----------------------------------------------------------------------------

def low_frequency_slope(freqs: Sequence[float], values: Sequence[float],
                        span_dec: float = DC_SLOPE_SPAN_DEC
                        ) -> tuple[Optional[float], Optional[float]]:
    """(slope in units/decade at the bottom of the sweep, span used in decades).

    The slope is taken from the first sample to the first sample at least
    `span_dec` decades above it, so it is normalised by log-frequency and
    cannot be faked by making the sweep finer. Comparing two ADJACENT samples
    -- the old dc_gain_valid test -- measures nothing at all: on a 20001-point
    linear sweep two neighbours are 2e-4 decades apart and agree to 0.004 dB
    even while the response falls at a full -20 dB/decade.

    Returns (None, None) when there is no usable low-frequency span (fewer than
    two finite samples, or a non-positive start frequency).
    """
    return local_slope(freqs, values, 0, span_dec)


def local_slope(freqs: Sequence[float], values: Sequence[float],
                start_index: int = 0, span_dec: float = DC_SLOPE_SPAN_DEC
                ) -> tuple[Optional[float], Optional[float]]:
    """(slope in units/decade upward from `start_index`, span used in decades).

    The same measurement low_frequency_slope makes, at an arbitrary sample.
    Used to compare the phase against the Bode estimate at more than one place
    in the sweep, which is how a zero BELOW the sweep is told apart from a zero
    above it: the first leaves a lead that decays as 1/f, the second a lead
    that grows with f.
    """
    n = min(len(freqs), len(values))
    i0 = max(0, int(start_index))
    if n - i0 < 2:
        return None, None
    f0 = float(freqs[i0])
    g0 = float(values[i0])
    if f0 <= 0.0 or not math.isfinite(g0):
        return None, None
    j: Optional[int] = None
    for i in range(i0 + 1, n):
        fi = float(freqs[i])
        if fi <= f0 or not math.isfinite(float(values[i])):
            continue
        j = i
        if math.log10(fi / f0) >= span_dec:
            break
    if j is None:
        return None, None
    span = math.log10(float(freqs[j]) / f0)
    if span <= 1e-12:
        return None, None
    return (float(values[j]) - g0) / span, span


def flat_band(freqs: Sequence[float], values: Sequence[float],
              span_dec: float = DC_SLOPE_SPAN_DEC,
              tol_db_per_dec: float = DC_SLOPE_TOL_DB_PER_DEC
              ) -> tuple[Optional[int], Optional[float]]:
    """(index, value) of the FLAT window with the HIGHEST value, or (None, None).

    A "window" at sample i is the run of samples from i up to the first one at
    least `span_dec` decades above it (or the end of the sweep). It is FLAT when
    the PEAK-TO-PEAK spread of the whole run stays inside `tol_db_per_dec` times
    the span the run actually covers -- the same certification the DC guard
    makes, normalised the same way, so it cannot be faked by making the sweep
    finer or coarser.

    THE PEAK-TO-PEAK TEST IS THE POINT. A two-point slope over the same window
    is exactly zero on the flank of a resonance, because a window straddling
    the peak has equal ENDPOINTS: an under-damped amplifier with 35 dB of
    mid-band gain and a Q = 2.2 pole pair has a window at 0.82*f0 whose slope
    is 0.000 dB/decade and whose gain is 41.1 dB, one dB under the resonant
    peak. A slope test therefore certifies the resonance as the passband, which
    is the whole defect. The spread over that window is a full dB and the
    peak-to-peak test rejects it; the first window it accepts is the real
    mid-band, at 35.04 dB.

    Used by ac_metrics to find the passband of a response whose sweep does not
    reach DC. The highest-valued flat window is the passband because a passband
    is, by definition, the flat region a response is designed to operate in,
    and any flatter-but-lower region is a stop band or a rolloff shelf.
    """
    n = min(len(freqs), len(values))
    best_i: Optional[int] = None
    best_v: Optional[float] = None
    for i in range(n):
        f0 = float(freqs[i])
        v0 = float(values[i])
        if f0 <= 0.0 or not math.isfinite(v0):
            continue
        if best_v is not None and v0 <= best_v:
            continue                    # cannot win; skip the window scan
        lo = hi = v0
        j: Optional[int] = None
        for k in range(i + 1, n):
            fk = float(freqs[k])
            vk = float(values[k])
            if fk <= f0 or not math.isfinite(vk):
                continue
            lo = min(lo, vk)
            hi = max(hi, vk)
            j = k
            if math.log10(fk / f0) >= span_dec:
                break
        if j is None:
            continue
        span = math.log10(float(freqs[j]) / f0)
        if span <= 1e-12:
            continue
        if (hi - lo) > tol_db_per_dec * span:
            continue
        best_v, best_i = v0, i
    return best_i, best_v


def low_frequency_phase_lead(freqs: Sequence[float], gain_db: Sequence[float],
                             phase_unwrapped: Optional[Sequence[float]],
                             ratio: float = DC_PHASE_LEAD_RATIO
                             ) -> tuple[Optional[float], Optional[float],
                                        Optional[float]]:
    """(1/f lead component at f_start, residual at f_start, residual at ratio*f_start).

    The RESIDUAL is the measured phase minus the minimum-phase Bode estimate
    for the magnitude slope at the same frequency. Two different things put a
    residual there, and they must not be confused:

      - a pole ABOVE the sweep contributes a lag that grows in proportion to f
        (-57.3 * f/fp degrees),
      - a zero BELOW the sweep contributes a lead that decays as 1/f
        (+57.3 * fz/f degrees).

    Only the second one means the response does not reach DC, and on a real
    amplifier the first one is the larger of the two: an AC-coupled stage with
    a 10 Hz coupling corner and a 1 MHz load pole, swept from 10 kHz, shows
    +0.057 deg of coupling lead buried under -0.573 deg of load-pole lag. A
    single residual sample cannot see it. Two samples, one decade-fraction
    apart, separate the 1/f term from the f term exactly:

        r(f) = A/f - B*f     ->     A/f0 = r*(r0*r - r1)/(r*r - 1)

    for f1 = ratio*f0, which for ratio = 2 is (4*r0 - 2*r1)/3. The ratio is
    kept small on purpose: the Bode residual of a pole is -57.3x + 90x^2 in
    x = f/fp, and a wide span lets the quadratic term corrupt the separation.

    Returns (None, ...) when there is no phase, no usable slope, or no second
    sample at `ratio` times f_start.
    """
    if phase_unwrapped is None:
        return None, None, None
    n = min(len(freqs), len(gain_db), len(phase_unwrapped))
    if n < 2 or float(freqs[0]) <= 0.0:
        return None, None, None

    def _resid(i: Optional[int]) -> Optional[float]:
        if i is None:
            return None
        p = float(phase_unwrapped[i])
        if not math.isfinite(p):
            return None
        slope, _ = local_slope(freqs, gain_db, i)
        est = bode_phase_estimate(slope)
        if est is None:
            return None
        return p - est

    r0 = _resid(0)
    f_up = float(freqs[0]) * float(ratio)
    j: Optional[int] = None
    for i in range(1, n):
        if float(freqs[i]) >= f_up:
            j = i
            break
    r1 = _resid(j)
    if r0 is None:
        return None, r0, r1
    if r1 is None:
        # No second point: the 1/f term cannot be separated from the f term.
        # Report the raw residual, which is conservative -- it can only
        # over-report a lead when the poles above the sweep are negligible.
        return r0, r0, None
    rr = float(freqs[j]) / float(freqs[0])
    denom = rr * rr - 1.0
    if abs(denom) < 1e-12:
        return r0, r0, r1
    lead = rr * (r0 * rr - r1) / denom
    return lead, r0, r1


def ac_metrics(freqs: Sequence[float], gain_db: Sequence[float],
               phase_deg: Optional[Sequence[float]] = None,
               normalize_phase: bool = True) -> dict[str, Any]:
    """Standard small-signal metrics from one frequency response.

    Returns, all Optional[float] except "notes":

      low_freq_gain_db     gain at the lowest swept frequency. Always present.
                           Says nothing about DC; it is named for what it is.
      dc_gain_db           the DC gain, ONLY when the sweep demonstrably
                           reaches DC. None otherwise, with a reason in notes.
      dc_gain_valid        1.0 / 0.0 / None (cannot tell). This is a real test
                           of the low-frequency slope, not of sweep density.
      low_slope_db_per_dec measured slope at the bottom of the sweep.
      peak_gain_db, f_peak the maximum gain and where it occurs.
      passband_gain_db     THE gain of the response, and the level the -3 dB
                           edges are referenced to: the DC gain when the sweep
                           reaches DC, else the gain of the FLAT region
                           (flat_band). It is the peak only when the peak sits
                           in a flat region. None, with a reason in notes, when
                           there is no flat region inside the sweep at all --
                           reporting a resonant peak as the gain rewards
                           peaking, and reporting the end of a rolloff as the
                           gain understates the design by the whole of it.
      bandwidth_3db        the UPPER -3 dB edge, in Hz, referenced to
                           passband_gain_db. None when the passband itself is
                           not inside the sweep.
      f_3db_lo, f_3db_hi   both -3 dB edges. f_3db_lo is non-None only for a
                           band-pass response; then the -3 dB SPAN is
                           f_3db_hi - f_3db_lo and notes says so.
      ugb                  the LOOP CLOSURE: the last frequency at which the
                           gain is still above 0 dB. Looked for whenever the
                           peak gain exceeds 0 dB. Refused when the magnitude
                           is flat to within a millionth of a dB, because then
                           the crossing is floating-point noise.
      phase_margin         180 + phase(ugb), after removing any whole inversion
                           inferred at the BOTTOM of the sweep against the
                           Bode estimate for the magnitude slope there.
      gain_margin, f_180   -gain at the WORST -180 deg crossing (the one with
                           the highest gain). None only when the phase never
                           reaches -180 deg at all, which is a genuinely
                           infinite margin; a phase that SITS at -180 without
                           crossing reports its worst-case gain, never None.
      phase_inversion_k    whole 180 deg turns removed. 0 for a non-inverting
                           response; an ODD value is a real sign inversion, an
                           EVEN one is only atan2's 2*pi branch.
      rolloff_db_per_dec   gain change over the top decade of the sweep.
      notes                metric name -> why it is None or what it refers to.
    """
    out: dict[str, Any] = {
        "dc_gain_db": None, "dc_gain_valid": None, "low_freq_gain_db": None,
        "low_slope_db_per_dec": None, "peak_gain_db": None, "f_peak": None,
        "passband_gain_db": None, "bandwidth_3db": None, "f_3db_lo": None,
        "f_3db_hi": None, "ugb": None, "phase_margin": None,
        "gain_margin": None, "f_180": None, "phase_inversion_k": None,
        "rolloff_db_per_dec": None,
    }
    notes: dict[str, str] = {}
    out["notes"] = notes

    n = min(len(freqs), len(gain_db))
    if n < 2:
        notes["*"] = "fewer than two frequency points; nothing is defined"
        return out

    f = [float(v) for v in freqs[:n]]
    g = [float(v) for v in gain_db[:n]]

    if math.isfinite(g[0]):
        out["low_freq_gain_db"] = g[0]

    peak_i: Optional[int] = None
    for i, v in enumerate(g):
        if not math.isfinite(v):
            continue
        if peak_i is None or v > g[peak_i]:
            peak_i = i
    if peak_i is None:
        notes["*"] = "no finite gain sample in the sweep"
        return out
    peak = g[peak_i]
    out["peak_gain_db"] = peak
    out["f_peak"] = f[peak_i]

    # -- phase bookkeeping, BEFORE the DC decision ---------------------------
    # The DC decision needs the phase (see low_frequency_phase_lead), and the
    # phase needs the magnitude slope, so the slope is measured first and the
    # unwrap/inversion is settled here rather than at the end of the function.
    slope, span = low_frequency_slope(f, g)
    out["low_slope_db_per_dec"] = slope

    ph: Optional[list[float]] = None
    k = 0
    if phase_deg is not None:
        ph = unwrap_deg(list(phase_deg)[:n])

        # A sample AT 0 Hz is the DC point, and the inversion question is
        # exactly "what is the sign of H at DC", so it answers itself. A
        # minimum-phase network has no rolloff-induced lag at DC, so a
        # non-inverting response shows 0 deg there by construction and no Bode
        # estimate is needed. Measured on real decks: an inverting loop swept
        # with `.ac lin n 0 fstop` reads exactly 180.000 at f = 0 and a
        # non-inverting one exactly 0.000.
        #
        # Treating f = 0 as "no measurable slope, therefore no information"
        # inverted the logic and disabled the inference on the one sweep form
        # that settles it outright.
        #
        # The magnitude is the guard: an AC-coupled network has H(0) = 0, and a
        # sample with no transfer function is already dropped upstream, so a
        # surviving 0 Hz sample with a finite magnitude is a real DC reading.
        at_dc = (f[0] == 0.0 and n > 0 and math.isfinite(float(g[0])))
        expected = 0.0 if at_dc else bode_phase_estimate(slope)

        if normalize_phase and expected is not None:
            k = phase_inversion_shift(ph, ref_index=0, expected_deg=expected)
            if k:
                ph = [p - 180.0 * k for p in ph]
        elif normalize_phase:
            # The 180 deg question is unanswerable here, but a 2*pi offset is
            # an artefact of atan2's principal branch and never information.
            # Skipping the call entirely left it in, which is what made a plain
            # 3-pole amplifier's gain margin vanish.
            k = 2 * int(round(ph[0] / 360.0)) if ph else 0
            if k:
                ph = [p - 180.0 * k for p in ph]
        out["phase_inversion_k"] = float(k)
        out["phase_reference"] = "dc" if at_dc else (
            "bode" if expected is not None else "unresolved")
        if k:
            kind = ("an inversion of" if k % 2 else
                    "a 2*pi branch artefact of")
            if at_dc:
                where = ("the sample at 0 Hz, which IS the DC point, where a "
                         "non-inverting minimum-phase response shows 0 deg")
            elif expected is not None:
                where = (f"f_start = {f[0]:g} Hz, where a minimum-phase "
                         f"response with the measured {slope:.4g} dB/decade "
                         f"slope would show {expected:.4g} deg")
            else:
                where = (f"f_start = {f[0]:g} Hz; the 180 deg question was "
                         f"unanswerable from this data, so only the 2*pi "
                         f"branch was corrected")
            notes["phase"] = (
                f"{kind} {180.0 * k:+g} deg was removed. It was inferred at "
                f"{where}. An ODD multiple of 180 deg is a real sign "
                f"inversion; an EVEN one is only atan2's principal branch, "
                f"which carries no information."
            )
        elif normalize_phase and expected is None:
            notes["phase"] = (
                "no inversion was inferred: the magnitude slope at the bottom "
                "of the sweep is not measurable, so there is nothing to "
                "compare the phase against"
            )

    # -- does this sweep actually reach DC? ---------------------------------
    lead, resid0, resid1 = low_frequency_phase_lead(f, g, ph)
    if f[0] <= 0.0:
        reaches_dc: Optional[bool] = True
    elif slope is None:
        reaches_dc = None
    elif abs(slope) > DC_SLOPE_TOL_DB_PER_DEC:
        reaches_dc = False
    elif lead is not None and lead > DC_PHASE_LEAD_TOL_DEG:
        # Flat magnitude, but the phase carries a 1/f lead that the flatness
        # cannot explain. Only a zero BELOW the sweep does that.
        reaches_dc = False
        f_zero = f[0] * math.tan(math.radians(min(lead, 89.0)))
        notes["dc_gain_db"] = (
            f"the sweep does not reach DC. The magnitude IS flat at "
            f"f_start = {f[0]:g} Hz ({slope:.4g} dB/decade), but the phase "
            f"carries a 1/f LEAD of {lead:.4g} deg there (residual "
            f"{resid0:.4g} deg at {f[0]:g} Hz against "
            f"{resid1:.4g} deg at {f[0] * DC_PHASE_LEAD_RATIO:g} Hz). Only a "
            f"zero BELOW the sweep leads like that, and here it sits at about "
            f"{f_zero:.4g} Hz -- an AC-coupling capacitor, whose gain at DC is "
            f"zero however flat the magnitude looks at f_start. The "
            f"{g[0]:.4g} dB measured there is reported as low_freq_gain_db. "
            f"Sweep from below {f_zero:.4g} Hz to measure what DC does."
        )
    else:
        reaches_dc = True

    if reaches_dc:
        out["dc_gain_db"] = g[0]
        out["dc_gain_valid"] = 1.0
        if f[0] > 0.0 and lead is None:
            notes["dc_gain_db"] = (
                f"the magnitude is flat at f_start = {f[0]:g} Hz "
                f"({slope:.4g} dB/decade) and NO PHASE was supplied, so the "
                "one instrument fine enough to see an AC-coupling zero below "
                "the sweep was not available. Flatness alone cannot tell a "
                "response that reaches DC from one whose coupling corner is "
                "far below f_start. Pass the phase to get that check."
            )
    elif reaches_dc is None:
        out["dc_gain_valid"] = None
        notes["dc_gain_db"] = (
            "cannot tell whether the sweep reaches DC: no usable "
            "low-frequency span (need two finite samples above 0 Hz)"
        )
    else:
        out["dc_gain_valid"] = 0.0
        notes.setdefault("dc_gain_db", (
            f"the sweep does not reach DC. At f_start = {f[0]:g} Hz the "
            f"response still slopes {slope:.4g} dB/decade (tolerance "
            f"{DC_SLOPE_TOL_DB_PER_DEC:g} dB/dec over {span:.3g} decades), so "
            f"the {g[0]:.4g} dB measured there is a point on a rolloff, not a "
            "DC gain. It is reported as low_freq_gain_db. Extend the sweep "
            "downward to measure the DC gain."
        ))

    # -- passband reference for the -3 dB edges ------------------------------
    # passband_gain_db IS the number the reward reads as "the gain of this
    # amplifier" (spec_extract maps gain / gain_db / gain_min / gain_max /
    # conversion_gain / linearity onto it). It used to be the PEAK gain
    # whenever the sweep did not reach DC, and that is a gradient toward
    # under-damped designs: an amplifier with 35 dB of mid-band gain and a
    # Q = 2.2 pole pair reported 42.07 dB, so it passed a `min: 40 dB` spec,
    # and the less damped the design the higher the number. Worse, it was
    # published even in the case the module had just refused -- a low-pass
    # swept entirely above its pole reported 20 dB of "gain" for a 60 dB
    # amplifier while notes said the passband was not in the sweep at all.
    #
    # So the passband is now the flat region of the response (flat_band), the
    # peak is only ever the passband when the peak sits IN a flat region, and
    # when there is no flat region inside the sweep the gain is refused.
    passband_outside = False
    dc_is_the_passband = out["dc_gain_db"] is not None
    flat_i, flat_g = (None, None) if dc_is_the_passband else flat_band(f, g)
    if dc_is_the_passband:
        ref: Optional[float] = float(out["dc_gain_db"])
        ref_kind = "DC"
        out["passband_gain_db"] = ref
    elif flat_g is not None:
        # There IS a flat region inside the sweep, and that is the passband --
        # wherever the peak happens to sit. A high-pass swept well above its
        # corner has its maximum at the LAST sample and a perfectly good
        # passband; only the absence of a flat region says the passband is
        # outside the band.
        ref = float(flat_g)
        ref_kind = "flat passband"
        out["passband_gain_db"] = ref
        if peak - ref > DC_SLOPE_TOL_DB_PER_DEC:
            notes["passband_gain_db"] = (
                f"the passband is the flat region at {f[flat_i]:g} Hz "
                f"({ref:.4g} dB), NOT the peak of {peak:.4g} dB at "
                f"{f[peak_i]:g} Hz. The response peaks {peak - ref:.4g} dB "
                f"above its own passband, which is peaking, not gain."
            )
    elif peak_i == 0 or peak_i == n - 1:
        # The response only falls away from an END of the sweep, and the sweep
        # does not reach DC, so the flat top it falls away FROM is outside the
        # swept band. Nothing here can be referenced to it. The mirror case
        # (peak at the TOP, a high-pass swept below its corner) had no refusal
        # branch at all and reported 19.96 dB for a 40 dB stage.
        ref, ref_kind = peak, "peak"
        passband_outside = True
        where = (f"below f_start = {f[0]:g} Hz" if peak_i == 0
                 else f"above f_stop = {f[n - 1]:g} Hz")
        direction = "lower" if peak_i == 0 else "higher"
        reason = (
            f"refused: the peak gain {peak:.4g} dB is the "
            f"{'FIRST' if peak_i == 0 else 'LAST'} sweep sample and the sweep "
            f"does not reach DC, so the response only falls away from the end "
            f"of the band and its passband lies {where}. The "
            f"{peak:.4g} dB measured at that end is a point on a rolloff, not "
            f"a passband gain, and reporting it as one understates a low-pass "
            f"swept above its pole by the whole of its gain. Extend the sweep "
            f"to {direction} frequencies."
        )
        notes["passband_gain_db"] = reason
        notes["bandwidth_3db"] = reason
    else:
        # Nowhere in the sweep is the magnitude flat, and the peak is INSIDE
        # the band: this response has no passband to measure and its peak is a
        # resonance, a notch flank or a point on a slope -- not a gain. The
        # -3 dB machinery still runs against the peak (referencing an edge to a
        # HIGHER level can only shorten the reported bandwidth, which is the
        # safe direction) but the gain itself is refused.
        ref, ref_kind = peak, "peak"
        notes["passband_gain_db"] = (
            f"refused: no window of the sweep is flat to within "
            f"{DC_SLOPE_TOL_DB_PER_DEC:g} dB/decade, so this response has "
            f"no passband inside it. The peak gain {peak:.4g} dB at "
            f"{f[peak_i]:g} Hz is a resonance, a notch flank or a point on "
            f"a slope; reporting it as THE gain rewards peaking, because "
            f"the less damped the design the higher the peak. peak_gain_db "
            f"still carries it under a name that says what it is."
        )

    if not passband_outside:
        out["f_3db_hi"] = crossing_freq(f, g, ref - DB3, direction=-1,
                                        start_index=peak_i)
        if peak_i > 0:
            out["f_3db_lo"] = last_crossing_freq_below(
                f, g, ref - DB3, peak_i, direction=1)
        out["bandwidth_3db"] = out["f_3db_hi"]
        # A response that climbs back INTO the passband above the first upper
        # edge is not a low-pass with a bandwidth; it is a notch or a
        # multi-band response, and the first edge is the STOP-BAND edge. There
        # is no single -3 dB bandwidth to report, so none is reported.
        recovery = None
        if out["f_3db_hi"] is not None:
            back_up = [fr for fr in all_crossings(f, g, ref - DB3, direction=1)
                       if fr > out["f_3db_hi"]]
            recovery = back_up[0] if back_up else None
        if recovery is not None:
            out["bandwidth_3db"] = None
            notes["bandwidth_3db"] = (
                f"refused: the gain falls {DB3:.4g} dB below the {ref_kind} "
                f"reference of {ref:.4g} dB at {out['f_3db_hi']:.6g} Hz but "
                f"climbs back above it at {recovery:.6g} Hz. That is a NOTCH "
                f"or a multi-band response: the first edge is the edge of a "
                f"stop band, not a -3 dB bandwidth, and reporting it as one is "
                f"how a band-stop filter comes to be scored as a low-pass. "
                f"f_3db_hi still carries the first edge."
            )
        elif out["f_3db_hi"] is None:
            notes["bandwidth_3db"] = (
                f"the gain never falls {DB3:.4g} dB below the {ref_kind} "
                f"reference of {ref:.4g} dB inside the sweep (last sample "
                f"{g[-1]:.4g} dB at {f[-1]:g} Hz)"
            )
        elif out["f_3db_lo"] is not None:
            span_3db = out["f_3db_hi"] - out["f_3db_lo"]
            notes["bandwidth_3db"] = (
                f"BAND-PASS response, referenced to the {ref_kind} gain "
                f"{ref:.4g} dB. bandwidth_3db is the UPPER edge "
                f"({out['f_3db_hi']:.6g} Hz); there is also a lower edge at "
                f"{out['f_3db_lo']:.6g} Hz, so the -3 dB SPAN is "
                f"{span_3db:.6g} Hz."
            )
        else:
            notes["bandwidth_3db"] = (
                f"upper -3 dB edge, referenced to the {ref_kind} gain "
                f"{ref:.4g} dB"
            )

    # -- unity gain ----------------------------------------------------------
    # The magnitude span of the whole sweep. A response that is flat to within
    # a millionth of a dB has no meaningful 0 dB crossing: whichever sample the
    # interpolation lands on is floating-point noise, and so is every metric
    # derived from it.
    finite_g = [v for v in g if math.isfinite(v)]
    g_span = (max(finite_g) - min(finite_g)) if finite_g else 0.0
    if peak > 0.0 and g_span <= 1e-6:
        ugb = None
        notes["ugb"] = (
            f"refused: the magnitude varies by only {g_span:.3g} dB across the "
            f"whole sweep (peak {peak:.6g} dB). A 0 dB crossing on a curve this "
            "flat is floating-point noise, and so is any phase margin taken at "
            "it."
        )
    elif peak > 0.0:
        ugb = closure_freq(f, g, 0.0)
        out["ugb"] = ugb
        downs = all_crossings(f, g, 0.0, direction=-1)
        if ugb is None:
            notes["ugb"] = (
                f"the gain reaches {peak:.4g} dB but never falls back through "
                f"0 dB inside the sweep (last sample {g[-1]:.4g} dB at "
                f"{f[-1]:g} Hz); extend the sweep upward"
            )
        elif len(downs) > 1:
            notes["ugb"] = (
                f"the gain crosses 0 dB {len(downs)} times downward "
                f"({', '.join(f'{c:.6g}' for c in downs)} Hz) and climbs back "
                f"above 0 dB in between. ugb is the LOOP CLOSURE, the last "
                f"frequency at which the gain is still above unity "
                f"({ugb:.6g} Hz); the phase margin is taken there. The earlier "
                f"crossings are not the closure and a margin taken at one of "
                f"them is optimistic."
            )
    else:
        ugb = None
        notes["ugb"] = (
            f"peak gain is {peak:.4g} dB, so the response never exceeds 0 dB "
            "and there is no unity-gain frequency"
        )

    if f[n - 1] > 0.0 and f[0] > 0.0:
        # Rolloff over the top decade of the sweep.
        f_hi = f[n - 1]
        f_lo = f_hi / 10.0
        if f_lo >= f[0]:
            g_hi = value_at_freq(f, g, f_hi)
            g_lo = value_at_freq(f, g, f_lo)
            if g_hi is not None and g_lo is not None:
                out["rolloff_db_per_dec"] = g_hi - g_lo

    if ph is None:
        return out

    if ugb is not None:
        p_at_ugb = value_at_freq(f, ph, ugb)
        if p_at_ugb is not None and math.isfinite(p_at_ugb):
            out["phase_margin"] = 180.0 + p_at_ugb
        else:
            notes["phase_margin"] = (
                f"the phase is not defined at the unity-gain frequency "
                f"{ugb:g} Hz"
            )
    elif "ugb" in notes:
        notes["phase_margin"] = "no unity-gain frequency: " + notes["ugb"]

    # -- gain margin ---------------------------------------------------------
    # Every -180 deg crossing is a candidate. The one that matters is the WORST
    # of them: the loop is unstable if the gain is above unity at ANY frequency
    # where the phase is -180, so the smallest margin is the honest one.
    crossings_180 = all_crossings(f, ph, -180.0, direction=-1)
    worst_f: Optional[float] = None
    worst_g: Optional[float] = None
    for fc in crossings_180:
        gc = value_at_freq(f, g, fc)
        if gc is None or not math.isfinite(gc):
            continue
        if worst_g is None or gc > worst_g:
            worst_g, worst_f = gc, fc
    if worst_f is not None:
        out["f_180"] = worst_f
        out["gain_margin"] = -float(worst_g)
        if len(crossings_180) > 1:
            notes["gain_margin"] = (
                f"the phase crosses -180 deg {len(crossings_180)} times; the "
                f"margin is reported at the WORST of them ({worst_f:.6g} Hz, "
                f"gain {worst_g:.4g} dB), because a loop is unstable if its "
                f"gain exceeds unity at ANY -180 deg crossing."
            )
    else:
        # No crossing. Either the phase stays above -180 for the whole sweep --
        # unconditionally stable, an infinite margin -- or it SITS at or below
        # -180 without ever crossing, which is the opposite situation and must
        # never be reported as infinite.
        first = next((ph[i] for i in range(n) if math.isfinite(ph[i])), None)
        at_or_below = [i for i in range(n)
                       if math.isfinite(ph[i]) and ph[i] <= -180.0 + 1e-9]
        if first is not None and first < -180.0 - 1e-9:
            # The phase is ALREADY past -180 at the bottom of the sweep, so the
            # crossing -- and the gain at it -- lie BELOW f_start and are not in
            # this data. The worst case inside the sweep would be optimistic by
            # exactly the gain the sweep cannot see, so nothing is reported.
            notes["gain_margin"] = (
                f"refused: the phase is already {first:.4g} deg at "
                f"f_start = {f[0]:g} Hz, past -180 deg, so the -180 deg "
                f"crossing is BELOW the sweep and the gain there is not in "
                f"this data. The worst case inside the sweep "
                f"({-max((g[i] for i in at_or_below if math.isfinite(g[i])), default=float('nan')):.4g} dB) "
                f"is optimistic by whatever the response does below f_start. "
                f"Extend the sweep downward. This is NOT an infinite margin."
            )
        elif at_or_below:
            gs = [g[i] for i in at_or_below if math.isfinite(g[i])]
            if gs:
                out["gain_margin"] = -max(gs)
                out["f_180"] = f[max(at_or_below, key=lambda i: g[i]
                                     if math.isfinite(g[i]) else -math.inf)]
            notes["gain_margin"] = (
                f"the phase never CROSSES -180 deg because it is already at or "
                f"below -180 deg at {len(at_or_below)} of {n} samples "
                f"(from {f[at_or_below[0]]:g} Hz). The margin is reported at "
                f"the highest gain in that region, which is the worst case; it "
                f"is emphatically NOT infinite."
            )
        else:
            notes["gain_margin"] = (
                "the phase never falls through -180 deg inside the sweep, so "
                "the gain margin is infinite; reported as None, never as 0.0"
            )
    return out


# ----------------------------------------------------------------------------
# Transient metrics
# ----------------------------------------------------------------------------

def settled_levels(y: Sequence[float], frac: float = 0.02,
                   tol: float = 0.002) -> tuple[float, float]:
    """Initial level and END-OF-RECORD level, averaged over the first/last `frac`.

    Deliberately not min()/max(): on a ringing waveform max() is the overshoot
    peak, which shrinks the apparent 90 pct level and understates rise time.

    The LEADING window is only averaged while it is quiescent. When the step
    happens at t = 0 -- no pre-step samples at all, which is what a .tran with
    a source that starts immediately produces -- the first 2 pct of the vector
    is already part of the response, and averaging it invents a non-zero
    initial level. That single error moved a series RLC overshoot from its
    analytic 16.3033 pct to 19.91 pct and cut its rise time by 20 pct, with
    nothing in the returned dict to distinguish the two cases. So the window is
    halved until it is quiescent, down to the single sample y[0], which is the
    true pre-step value in that case.

    Quiescence is judged by DRIFT, not by spread: the mean of the window's
    second half minus the mean of its first half, compared against `tol` times
    the PEAK-TO-PEAK excursion of the whole record. Zero-mean noise cancels in
    both halves, so a genuinely quiescent but noisy window keeps its full
    averaging, while a window that straddles an edge fails immediately. A
    spread test cannot tell those two apart and throws away the averaging
    exactly when it is needed. The scale has to be the peak-to-peak excursion
    rather than |y_end - y[0]|: on a PULSE the waveform ends where it started,
    that difference is ~0, and the drift test then passes whatever the leading
    window contains -- including the whole of the first edge.

    THE SECOND RETURN VALUE IS NOT NECESSARILY A FINAL LEVEL. It is the mean of
    the tail of the record, which is the settled final level only for a
    waveform that actually settles. For a pulse or a periodic waveform it is
    the level the signal happens to be at when the simulation stops. Use
    waveform_levels(), which detects that case and measures the first edge
    instead; every metric in this module does.
    """
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    k0 = max(1, int(n * frac))
    y1 = sum(float(v) for v in y[-k0:]) / k0
    finite = [float(v) for v in y if math.isfinite(float(v))]
    ptp = (max(finite) - min(finite)) if finite else 0.0
    k = k0
    while k > 1:
        window = [float(v) for v in y[:k]]
        drift = window_drift(window)
        excursion = max(ptp, abs(y1 - window[0]))
        if drift is None or excursion <= 0.0 or drift <= tol * excursion:
            break
        k //= 2
    y0 = sum(float(v) for v in y[:k]) / k
    return y0, y1


def window_drift(window: Sequence[float]) -> Optional[float]:
    """|mean(second half) - mean(first half)| of a window, or None if too short.

    The drift of a window, not its spread. Zero-mean noise cancels in both
    halves, so a genuinely quiescent but noisy window reads a drift of ~0 while
    a window that straddles an edge, or one sitting on a waveform that is still
    heading somewhere, reads the movement it actually contains. Needs at least
    two samples; a one-sample window has no halves and no drift to measure, and
    that is reported as None (cannot tell) rather than as 0.0 (settled).
    """
    k = len(window)
    if k < 2:
        return None
    vals = [float(v) for v in window]
    half = k // 2
    return abs(sum(vals[half:]) / (k - half) - sum(vals[:half]) / half)


def _segment_rate(vals: Sequence[float], t: Optional[Sequence[float]],
                  i: int, j: int) -> Optional[float]:
    """|dy/dt| across one segment. Index spacing stands in when `t` is absent.

    Returns None when either sample is missing or non-finite, or when the two
    samples share a time.
    """
    if i < 0 or j >= len(vals) or j <= i:
        return None
    a, b = float(vals[i]), float(vals[j])
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if t is None:
        dt = float(j - i)
    else:
        if j >= len(t):
            return None
        dt = float(t[j]) - float(t[i])
    if not (dt > 0.0):
        return None
    return abs(b - a) / dt


# A waveform whose end-of-record level is closer to its starting level than
# this fraction of its peak-to-peak excursion did not step anywhere: it went
# out and came back. A second-order step response approaches but never reaches
# 0.5 from above (100 pct overshoot is the limit), so 0.4 separates the two
# cases with margin at both ends.
PULSE_RETURN_FRAC = 0.4

# Below this fraction of the peak-to-peak excursion, a reversal is numerical
# noise rather than overshoot. 1 ppm of the excursion is far below any
# overshoot worth reporting and far above a simulator's integration jitter.
MONOTONE_TOL_FRAC = 1e-6


@dataclass
class WaveformLevels:
    """The 0 pct and 100 pct references for a transient measurement.

    `kind` is what the record actually contains:

      "step"  the waveform goes somewhere and stays. y1 is the settled level,
              the measurement region is the whole record.
      "pulse" the waveform goes out and comes back -- a PULSE source, a clock,
              a ring oscillator, any periodic drive. y1 is the level of the
              FIRST plateau (IEEE 181 calls it the top; a scope calls it the
              same), and the measurement region ends where the return edge
              starts, so nothing downstream measures across it.
      "unsettled"
              the waveform went somewhere and was STILL GOING when the record
              ended. There is no final level, so y1 is the last level reached
              and nothing may be referred to it. Every metric that needs a
              final level is refused.
      "flat"  no excursion at all; nothing is defined.
    """

    y0: float = 0.0
    y1: float = 0.0
    kind: str = "flat"
    i_edge: int = 0
    """First sample of the first edge."""
    i_end: int = 0
    """Last sample of the measurement region, inclusive."""
    monotone: bool = True
    """True when the region never reverses direction by more than 1 ppm."""
    note: str = ""


def waveform_levels(y: Sequence[float], frac: float = 0.02,
                    tol: float = 0.002,
                    t: Optional[Sequence[float]] = None) -> WaveformLevels:
    """The base and top levels a transient metric must be referenced to.

    settled_levels() alone is right only for a STEP. Fed a pulse it takes the
    mean of the tail as "the final level", and on any waveform that returns to
    where it started that is the STARTING level, so the whole 10/90 ladder is
    referenced to a span of nothing. Measured on this repo's own
    scripts/agent_ngspice.py deck (PULSE(0 1.8 0 1n 1n 5u 10u) into 1k/1n,
    .tran 0.1u 20u) the reported step was 14.3 mV instead of 1.8 V, and with it

        rise_time      6.39632e-09 s   against 2.19722e-06 s   (343x fast)
        overshoot_pct  12374.85 pct    against 0 pct
        slew_rate      1.793 V/us      against 0.655 V/us

    every one of them a confident number for a step that is not there. Every
    inverter, buffer, ring oscillator and clocked block in the eval set is
    driven by exactly such a source.

    So a returning waveform is DETECTED and the FIRST EDGE is measured instead:
    the base is the quiescent level before the edge, the top is the plateau the
    first excursion reaches, and the region ends where the waveform starts
    coming back. That is the standard pulse measurement, and for a testbench
    that drives an amplifier with a PULSE source it is also exactly the step
    response the caller wanted.

    A waveform that does NOT return is a step only if it actually settled. One
    that was still moving when the simulation stopped has no final level at
    all, and taking the mean of the tail as one is self-referential -- the tail
    is inside its own band by construction. An R = 1 Meg, C = 1 uF stage
    (tau = 1 s) run for 1 ms reaches 0.0999 pct of its final value, and that
    was reported as a rise time of 792 us against a true 2.197 s: 2773x fast,
    with an empty notes dict, so a circuit 1000x too slow passed a `max: 1 ms`
    spec cleanly. Such a record is `kind = "unsettled"` and every metric that
    needs a final level is refused.

    `t` is the time axis. It is optional only because a caller may not have
    one; pass it whenever you do. It is what tells a fast RETURN EDGE from a
    slow settling DROOP on the top of a pulse, and without it uniform sampling
    is assumed.
    """
    n = len(y)
    lv = WaveformLevels()
    if n == 0:
        return lv
    vals = [float(v) for v in y]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        lv.note = "no finite sample in the waveform"
        return lv
    ptp = max(finite) - min(finite)
    y0, y_tail = settled_levels(vals, frac, tol)
    lv.y0 = y0
    lv.i_end = n - 1

    if ptp <= 0.0:
        lv.y1 = y0
        lv.kind = "flat"
        lv.note = "the waveform is constant; no edge and no levels"
        return lv

    if abs(y_tail - y0) >= PULSE_RETURN_FRAC * ptp:
        lv.y1 = y_tail
        lv.kind = "step"
        lv.i_edge = 0
        lv.monotone = _is_monotone(vals, 0, n - 1, y_tail - y0, ptp)
        # ... but only if it actually STOPPED there. The tail window is judged
        # by the same drift test the leading window is: a waveform still moving
        # by more than `tol` of its own excursion across the last `frac` of the
        # record has not reached a final level, and every metric referred to
        # one would be referred to a level the circuit never reached.
        k_tail = max(1, int(n * frac))
        drift = window_drift(vals[n - k_tail:])
        if drift is not None and drift > tol * ptp:
            lv.kind = "unsettled"
            lv.note = (
                f"the waveform is STILL MOVING at the end of the record. Over "
                f"the last {100.0 * frac:g} pct of it the level drifts "
                f"{drift:.6g}, which is {100.0 * drift / ptp:.4g} pct of the "
                f"{ptp:.6g} peak-to-peak excursion (tolerance "
                f"{100.0 * tol:g} pct). It never settled, so the "
                f"{y_tail:.6g} at the end of the record is a point on the way "
                f"somewhere and NOT a final level. Referring the 10/50/90 pct "
                f"ladder to it measures a fraction of a level the circuit "
                f"never reached: on an R = 1 Meg, C = 1 uF stage run for 1 ms "
                f"that read a rise time of 792 us against a true 2.197 s. "
                f"rise_time, fall_time, settling_time, slew_rate, "
                f"overshoot_pct and the fractional times are refused. Extend "
                f"tstop past a few time constants."
            )
        return lv

    # -- returning waveform: measure the first edge --------------------------
    dev = [abs(v - y0) if math.isfinite(v) else 0.0 for v in vals]
    dmax = max(dev)
    if dmax <= 0.0:
        lv.y1 = y0
        lv.kind = "flat"
        return lv
    i_half = next(i for i in range(n) if dev[i] >= 0.5 * dmax)
    d = 1.0 if vals[i_half] > y0 else -1.0
    i_ret = n
    for i in range(i_half + 1, n):
        if math.isfinite(vals[i]) and d * (vals[i] - y0) < 0.5 * dmax:
            i_ret = i
            break
    j = min(i_ret - 1, n - 1)
    # Walk back off the RETURN EDGE, and off nothing else. i_ret is the first
    # sample below the 50 pct level, so j = i_ret - 1 may sit part-way down the
    # return edge, and averaging the top there biases it low. But a top that is
    # still SETTLING descends too, and the old test -- "is this sample lower
    # than the one before it" -- cannot tell the two apart, so on an ordinary
    # passive lead network (R1 10k shunted by Cf 1n into R2 10k, a 20 us pulse
    # sampled every 10 ns) it walked the whole 20 us plateau back to the 1 ns
    # feedthrough spike: the measurement region collapsed from 4027 samples to
    # samples 7..8, y_final read 1.798 V instead of 0.9205 V, a 95.3 pct
    # overshoot was reported as 0.0, and the slew rate came out 1798 V/us.
    # Every lead-compensated stage, every AC-coupled stage and every pulse with
    # top droop has that shape.
    #
    # So the walk-back is bounded by RATE, against the one segment that is
    # certainly on the return edge: the segment that crosses the 50 pct level.
    # On that deck the crossing segment runs at 1.8e8 V/s and the droop just
    # before it at 3.2e3 V/s, so the walk stops immediately and the plateau
    # survives; on a pulse whose fall really is slow the two rates are within a
    # factor of two and the walk proceeds to the top of the edge as before.
    ref_rate = _segment_rate(vals, t, j, i_ret) if i_ret < n else None
    if ref_rate:
        while j > i_half and d * (vals[j] - vals[j - 1]) < 0.0:
            rate = _segment_rate(vals, t, j - 1, j)
            if rate is None or rate < RETURN_EDGE_RATE_FRAC * ref_rate:
                break
            j -= 1
    i_end = max(j, i_half)
    m = max(1, int((i_end - i_half + 1) * frac))
    lv.y1 = sum(vals[i_end - m + 1:i_end + 1]) / m
    lv.kind = "pulse"
    lv.i_edge = i_half
    lv.i_end = i_end
    lv.monotone = _is_monotone(vals, 0, i_end, lv.y1 - y0, ptp)
    lv.note = (
        "the waveform RETURNS to its starting level, so it is a pulse or a "
        "periodic drive, not a step. Every level is taken from the FIRST edge: "
        f"base {lv.y0:.6g}, top {lv.y1:.6g} (sample {i_end} of {n}), and the "
        "return edge is outside the measurement region. The mean of the tail "
        "of the record, which a step measurement would use as the final level, "
        f"is {y_tail:.6g} -- within "
        f"{100.0 * abs(y_tail - y0) / ptp:.3g} pct of the peak-to-peak "
        "excursion of the starting level, which is what gives a pulse away."
    )
    return lv


def _is_monotone(y: Sequence[float], i0: int, i1: int, direction: float,
                 ptp: float) -> bool:
    """True when y never reverses `direction` by more than 1 ppm of `ptp`."""
    if direction == 0.0:
        return True
    d = 1.0 if direction > 0 else -1.0
    slack = MONOTONE_TOL_FRAC * ptp
    prev: Optional[float] = None
    for i in range(max(0, i0), min(i1, len(y) - 1) + 1):
        v = float(y[i])
        if not math.isfinite(v):
            continue
        if prev is not None and d * (v - prev) < -slack:
            return False
        prev = v
    return True


def crossing_time(t: Sequence[float], y: Sequence[float], level: float,
                  start_index: int = 0, direction: int = 0,
                  stop_index: Optional[int] = None) -> Optional[float]:
    """First time at or after start_index where y crosses level, interpolated.

    `stop_index` (inclusive) bounds the search, which is how the pulse metrics
    stay inside the first edge instead of finding the same level again on the
    way back down.
    """
    n = min(len(t), len(y))
    if stop_index is not None:
        n = min(n, int(stop_index) + 1)
    for i in range(max(0, start_index), n - 1):
        a, b = float(y[i]), float(y[i + 1])
        falling, rising = _seg_crosses(a, b, level)
        if direction > 0 and not rising:
            continue
        if direction < 0 and not falling:
            continue
        if direction == 0 and not (rising or falling):
            continue
        t0, t1 = float(t[i]), float(t[i + 1])
        if t1 == t0:
            continue
        return _interp(t0, a, t1, b, level)
    return None


def time_to_fraction(t: Sequence[float], y: Sequence[float],
                     frac: float) -> Optional[float]:
    """Time at which y first reaches `frac` of its total excursion.

    Measured from the time origin of the vector, using the base and top levels
    of the FIRST EDGE (see waveform_levels) as the 0 pct and 100 pct
    references.
    """
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled" or lv.y1 == lv.y0:
        return None
    level = lv.y0 + frac * (lv.y1 - lv.y0)
    direction = 1 if lv.y1 > lv.y0 else -1
    return crossing_time(t, y, level, direction=direction,
                         stop_index=lv.i_end)


def rise_time(t: Sequence[float], y: Sequence[float],
              lo: float = 0.1, hi: float = 0.9) -> Optional[float]:
    """10 pct to 90 pct rise time of the first rising edge."""
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled" or lv.y1 <= lv.y0:
        return None
    span = lv.y1 - lv.y0
    t_lo = crossing_time(t, y, lv.y0 + lo * span, direction=1,
                         stop_index=lv.i_end)
    t_hi = crossing_time(t, y, lv.y0 + hi * span, direction=1,
                         stop_index=lv.i_end)
    if t_lo is None or t_hi is None:
        return None
    return t_hi - t_lo


def fall_time(t: Sequence[float], y: Sequence[float],
              lo: float = 0.1, hi: float = 0.9) -> Optional[float]:
    """90 pct to 10 pct fall time of the first falling edge."""
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled" or lv.y1 >= lv.y0:
        return None
    span = lv.y0 - lv.y1
    t_hi = crossing_time(t, y, lv.y1 + hi * span, direction=-1,
                         stop_index=lv.i_end)
    t_lo = crossing_time(t, y, lv.y1 + lo * span, direction=-1,
                         stop_index=lv.i_end)
    if t_lo is None or t_hi is None:
        return None
    return t_lo - t_hi


def overshoot_pct(y: Sequence[float],
                  t: Optional[Sequence[float]] = None) -> Optional[float]:
    """Peak overshoot past the settled level, in percent of the step.

    A waveform that never reverses direction has NO overshoot, and this returns
    exactly 0.0 for it. It used to compare max(y) against the mean of the tail,
    which on any still-settling exponential is a little below the last sample,
    so a plain RC step was reported as overshooting by a few tenths of a
    percent that were never there.
    """
    if not y:
        return None
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled" or lv.y1 == lv.y0:
        return None
    if lv.monotone:
        return 0.0
    d = 1.0 if lv.y1 > lv.y0 else -1.0
    region = [float(v) for v in y[:lv.i_end + 1] if math.isfinite(float(v))]
    if not region:
        return None
    past = max(d * (v - lv.y1) for v in region)
    if past <= 0.0:
        return 0.0
    return 100.0 * past / abs(lv.y1 - lv.y0)


def settling_time(t: Sequence[float], y: Sequence[float],
                  tol: float = 0.02) -> Optional[float]:
    """Time after which y stays inside +/- tol of its final level, forever.

    Scans backwards. A forward scan returns the first entry into the band and
    terminates early on a ringing waveform that leaves the band again.

    A single in-band sample at the very end of the vector is NOT evidence of
    settling -- a square wave that is out of band at t[n-2] and in band at
    t[n-1] never settled. The old guard fired one sample too late and returned
    t[-1] for exactly that waveform, which reads as "settles at the end of the
    simulation" instead of "does not settle". At least two consecutive in-band
    samples are required.

    On a pulse the band, the level and the search all belong to the FIRST
    plateau. Referring them to the mean of the tail of the record instead is
    self-referential: the tail is inside its own band by construction, so a
    waveform that never settles at all still reports a settling time.
    """
    n = min(len(t), len(y))
    if n == 0:
        return None
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled":
        return None
    span = abs(lv.y1 - lv.y0)
    if span == 0.0:
        return None
    band = tol * span
    last = min(lv.i_end, n - 1)
    for i in range(last, -1, -1):
        if abs(float(y[i]) - lv.y1) > band:
            if i + 1 >= last:
                return None  # never settles inside the measured region
            return float(t[i + 1])
    return float(t[0])


def slew_rate(t: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """10-90 chord slew rate in volts per second (units per second)."""
    lv = waveform_levels(y, t=t)
    if lv.kind == "unsettled":
        return None
    if lv.y1 > lv.y0:
        tr = rise_time(t, y)
    else:
        tr = fall_time(t, y)
    if tr is None or tr <= 0.0:
        return None
    return 0.8 * abs(lv.y1 - lv.y0) / tr * (1.0 if lv.y1 > lv.y0 else -1.0)


def prop_delay(t: Sequence[float], y_in: Sequence[float],
               y_out: Sequence[float]) -> Optional[float]:
    """50 pct to 50 pct propagation delay between two waveforms on one time axis."""
    ti = time_to_fraction(t, y_in, 0.5)
    to = time_to_fraction(t, y_out, 0.5)
    if ti is None or to is None:
        return None
    return to - ti


def tran_metrics(t: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """All single-waveform transient metrics at once.

    "notes" carries the reason for anything the data does not define, and the
    reference levels every other number is relative to, the same way ac_metrics
    does. y_initial/y_final are the BASE and TOP of the first edge; on a pulse
    y_final is the plateau of that edge, NOT the level at the end of the
    simulation, and notes["levels"] says so.
    """
    lv = waveform_levels(y, t=t)
    st = settling_time(t, y)
    notes: dict[str, str] = {}
    if lv.note:
        notes["levels"] = lv.note
    if lv.kind == "unsettled":
        # Nothing here has a final level to be referred to. y_final is the last
        # level the record REACHED and is labelled as such, so a consumer that
        # reads it without reading notes still cannot mistake it for a settled
        # value: every metric derived from it is None.
        for refused in ("rise_time", "fall_time", "overshoot_pct",
                        "settling_time", "slew_rate", "t_50pct", "t_63pct",
                        "y_final"):
            notes[refused] = "refused: " + lv.note
    elif st is None and len(y) >= 2 and lv.y1 != lv.y0:
        notes["settling_time"] = (
            "the waveform is still outside the settling band at the end of the "
            + ("first pulse" if lv.kind == "pulse" else "simulated window")
            + ", so it never settles inside it; extend tstop"
        )
    return {
        "y_initial": lv.y0,
        "y_final": lv.y1,
        "waveform_kind": lv.kind,
        "rise_time": rise_time(t, y),
        "fall_time": fall_time(t, y),
        "overshoot_pct": overshoot_pct(y, t),
        "settling_time": st,
        "slew_rate": slew_rate(t, y),
        "t_50pct": time_to_fraction(t, y, 0.5),
        "t_63pct": time_to_fraction(t, y, 0.6321205588285577),
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# DC / supply metrics
# ----------------------------------------------------------------------------

# SPICE guarantees the FIRST LETTER of an element name is its device type, so
# '<name>#branch' with name[0] == 'v' really is an independent voltage source
# and name[0] == 'l' really is an inductor. That much is a language rule, not a
# guess. What is NOT knowable from the name is whether a given voltage source
# is a SUPPLY: a 0 V ammeter in series with a branch is spelled exactly the
# same way, and a current-source-biased block has no supply branch vector at
# all. See supply_current_report.
# "DC 1.8", "dc=1.8", "DC 0 AC 1". The value is whatever follows the keyword.
_DC_KEYWORD_RE = re.compile(r"\bdc\b", re.IGNORECASE)
_DC_VALUE_RE = re.compile(r"\bdc\b\s*=?\s*([-+0-9.][^\s]*)", re.IGNORECASE)

# ngspice engineering suffixes. 'meg' and 'mil' must be tried before 'm'.
_SPICE_SUFFIXES = (("meg", 1e6), ("mil", 25.4e-6), ("t", 1e12), ("g", 1e9),
                   ("k", 1e3), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9),
                   ("p", 1e-12), ("f", 1e-15))
_NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


@dataclass
class SupplyCurrentReport:
    """What supply_current measured, from which branches, and what it distrusts."""

    value: Optional[float] = None
    """Total supply current magnitude in amperes, or None when not identifiable."""

    sources: list[str] = field(default_factory=list)
    """Branch names actually summed, e.g. ['v1#branch']."""

    excluded: dict[str, str] = field(default_factory=dict)
    """Branch name -> why it was left out."""

    warnings: list[str] = field(default_factory=list)
    """Reasons the value may not be the supply current. NEVER ignore these."""

    @property
    def ambiguous(self) -> bool:
        return bool(self.warnings)


# A transient source with no explicit DC value operates, in .op and .dc, at the
# value its waveform has at t = 0. That value is a specific argument of each
# waveform function, and it is knowledge, not a guess. Anything not listed here
# is UNKNOWN (NaN), which keeps the source as a possible supply rather than
# excluding a rail on a guess.
_TRAN_FUNC_VALUE_INDEX = {
    "sin": 0,      # SIN(vo va freq td theta)      -> offset
    "sine": 0,
    "pulse": 0,    # PULSE(v1 v2 td tr tf pw per)  -> initial value
    "exp": 0,      # EXP(v1 v2 td1 tau1 td2 tau2)  -> initial value
    "sffm": 0,     # SFFM(vo va fc mdi fs)         -> offset
    "pwl": 1,      # PWL(t1 v1 t2 v2 ...)          -> the value at t1
}
_TRAN_FUNC_RE = re.compile(
    r"\b(sin|sine|pulse|exp|sffm|pwl)\b\s*\(?([^)]*)", re.IGNORECASE)


@dataclass
class DeckSources:
    """Every independent source in a deck, WITH ITS SUBCIRCUIT SCOPE.

    A .subckt body is not top level. Parsing it as if it were lets a 0 V
    ammeter inside a block overwrite a same-named rail at the top, and ngspice
    really does allow both to be called V1: the branch vectors are 'v1#branch'
    and 'v.x1.v1#branch', two different currents. Flattening them made the
    adapter exclude the REAL 1.8 mA rail as "declared DC 0" and keep the 1.8 uA
    ammeter, reporting a supply current 1000x low with an empty warning list.
    """

    top: dict[str, float] = field(default_factory=dict)
    """Top-level voltage source name -> DC value (NaN when unreadable)."""

    subckt_v: dict[str, dict[str, float]] = field(default_factory=dict)
    """subckt name -> {local voltage source name -> DC value}."""

    instances: dict[str, str] = field(default_factory=dict)
    """Top-level X instance name -> the subckt it instantiates."""

    subckt_x: dict[str, dict[str, str]] = field(default_factory=dict)
    """subckt name -> {local X instance name -> the subckt it instantiates}."""

    isources: list[str] = field(default_factory=list)
    """Independent current sources, top level and inside instantiated subckts."""

    elements: set[str] = field(default_factory=set)
    """Every element card name in the deck, at any level of hierarchy."""


def parse_deck_sources(netlist: str) -> DeckSources:
    """Read the independent sources out of a deck, keeping subcircuit scope.

    A deliberately small SPICE reader: it only needs the element letter, the
    name, the DC value of V cards, and which subcircuit each X card names.
    Comments, continuations and .control blocks are skipped.

    The DC value comes back as NaN only when the card names a value that
    cannot be read (an unusual unit suffix, a parameter expression). NaN means
    "unknown", and the caller never excludes an unknown source -- dropping a
    real supply is a much worse failure than keeping a stimulus source whose
    branch current is zero anyway. A card with no value at all ("V1 a b") or an
    AC-only stimulus ("Vin in 0 AC 1") has a DC value of exactly 0 by SPICE's
    own rules. A card carrying only a transient waveform ("Vdd vdd 0
    PULSE(1.8 0 ...)") operates at the value that waveform has at t = 0 --
    1.8 V, NOT 0 V. Reading those as 0 excluded real rails declared with SIN,
    PULSE, EXP or PWL as though they were sense sources.
    """
    deck = DeckSources()
    in_control = False
    scope: Optional[str] = None
    for raw in netlist.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(".control"):
            in_control = True
            continue
        if low.startswith(".endc"):
            in_control = False
            continue
        if in_control:
            continue
        if low.startswith(".subckt"):
            toks = line.split()
            scope = toks[1].lower() if len(toks) > 1 else "?"
            deck.subckt_v.setdefault(scope, {})
            deck.subckt_x.setdefault(scope, {})
            continue
        if low.startswith(".ends") or low.startswith(".eom"):
            scope = None
            continue
        if line[0] in "*+.;":
            continue
        head = line.split()[0]
        name = head.lower()
        kind = name[0]
        deck.elements.add(name)
        if kind == "x":
            toks = line.split()
            target = None
            for tok in reversed(toks[1:]):
                if "=" in tok or tok.lower() == "params:":
                    continue
                target = tok.lower()
                break
            if target:
                if scope is None:
                    deck.instances[name] = target
                else:
                    deck.subckt_x[scope][name] = target
            continue
        if kind == "i":
            deck.isources.append(name if scope is None else f"{scope}:{name}")
            continue
        if kind != "v":
            continue
        value = _v_card_dc_value(line[len(head):])
        if scope is None:
            deck.top[name] = value
        else:
            deck.subckt_v[scope][name] = value
    return deck


def _v_card_dc_value(rest: str) -> float:
    """The operating-point value of one V card, from everything after its name."""
    if _DC_KEYWORD_RE.search(rest):
        m = _DC_VALUE_RE.search(rest)
        # A 'DC' keyword whose value is a parameter expression ({vsup}) is
        # UNKNOWN, not zero. Never let an unreadable value exclude a rail.
        return _spice_float(m.group(1)) if m else math.nan
    toks = rest.split()
    if len(toks) < 3:
        return 0.0                          # "V1 a b" -> DC 0
    if toks[2][0].isalpha():
        m = _TRAN_FUNC_RE.search(rest)
        if m:
            idx = _TRAN_FUNC_VALUE_INDEX[m.group(1).lower()]
            args = m.group(2).replace(",", " ").split()
            if len(args) > idx:
                return _spice_float(args[idx])
            return math.nan
        if toks[2].lower() == "ac":
            return 0.0                      # "Vin in 0 AC 1" -> DC 0
        return math.nan                     # an unrecognised keyword: unknown
    return _spice_float(toks[2])


def _parse_source_cards(netlist: str) -> tuple[dict[str, float], list[str]]:
    """(top-level voltage source name -> DC value, current source names).

    Kept for callers that only care about the top level. New code should use
    parse_deck_sources(), which keeps the subcircuit hierarchy that a branch
    vector name like 'v.x1.vsense#branch' has to be resolved against.
    """
    deck = parse_deck_sources(netlist)
    return dict(deck.top), list(deck.isources)


def element_letter(branch_name: str) -> str:
    """The SPICE element letter of a (possibly hierarchical) branch name.

    'v1' -> 'v', and 'v.x1.vsense' -> 'v' as well: ngspice spells a branch
    inside a subcircuit instance '<letter>.<instance path>.<local name>'.
    """
    nm = str(branch_name).lower()
    return nm.split(".")[0][:1] if "." in nm else nm[:1]


def source_dc_value(branch_name: str, deck: DeckSources) -> float:
    """DC value of the source that produced '<branch_name>#branch'.

    Resolves a hierarchical name through the X instances, so the ammeter in
    'v.x1.vsense' is looked up in the subcircuit x1 instantiates and NOT in
    whatever top-level source happens to share its local name.
    """
    nm = str(branch_name).lower()
    if "." not in nm:
        return deck.top.get(nm, math.nan)
    parts = nm.split(".")
    local = parts[-1]
    scope_v = deck.top
    scope_x = deck.instances
    for inst in parts[1:-1]:
        sub = scope_x.get(inst)
        if sub is None:
            return math.nan
        scope_v = deck.subckt_v.get(sub, {})
        scope_x = deck.subckt_x.get(sub, {})
    return scope_v.get(local, math.nan)


def _spice_float(tok: Optional[str]) -> float:
    """Parse a SPICE number with an engineering suffix. NaN when unparseable.

    NaN means "unknown", and callers must treat unknown as "could be a supply".
    """
    if tok is None:
        return math.nan
    s = str(tok).strip().lower().rstrip(",")
    m = _NUMBER_RE.match(s)
    if m is None:
        return math.nan
    mantissa = float(m.group(0))
    tail = s[m.end():]
    if not tail:
        return mantissa
    for suf, mult in _SPICE_SUFFIXES:
        if tail.startswith(suf):
            return mantissa * mult
    # A trailing unit ("1.8V") or anything else unrecognised: the NUMBER is
    # still known, and a scale factor would have to start with a suffix letter.
    return mantissa


def supply_current_report(op_points: dict[str, float],
                          sources: Optional[Sequence[str]] = None,
                          netlist: Optional[str] = None) -> SupplyCurrentReport:
    """Total supply current, with every reason it might not be one.

    ngspice reports a voltage source branch current with the passive sign
    convention on the source, so a supply that delivers power reports a
    NEGATIVE branch current. Magnitudes are summed per source rather than the
    sum being taken and then abs()'d, so a sourcing rail and a sinking rail do
    not cancel.

    HOW THE SUPPLY IS IDENTIFIED, in order of preference:

      1. `sources`, an explicit list of source names (case insensitive, with or
         without the '#branch' suffix). Always prefer this. It is the only
         input that actually knows which sources are supplies.
      2. `netlist`, the deck that produced `op_points`. Independent voltage
         sources whose DC value is exactly 0 are 0 V AMMETERS, not supplies,
         and are excluded. Independent current sources are noted: ngspice
         writes no branch vector for them, so a current-source-biased block has
         its real supply current MISSING from op_points entirely, and whatever
         voltage-source branches remain are almost certainly not it.
      3. Neither: every '<name>#branch' entry whose element letter is 'v' is
         summed, and the result is flagged ambiguous whenever more than one
         source is involved or one of them carries less than a thousandth of
         the largest, because that is the shape of a sense source sitting next
         to a supply.

    This used to be rule 3 alone with no warning, which reported a 1 uA
    ammeter as the supply current of a 1 mA current-source-biased stage: wrong
    by 1000x, silently. It now returns 1e-3's absence instead of 1e-6's lie.
    """
    rep = SupplyCurrentReport()

    branches: dict[str, float] = {}
    for key, val in op_points.items():
        k = str(key).lower()
        if k.endswith("#branch"):
            branches[k.split("#")[0]] = float(val)

    if sources is not None:
        wanted = {str(s).lower().split("#")[0] for s in sources}
        missing = sorted(wanted - set(branches))
        if missing:
            rep.warnings.append(
                "named supply source(s) " + ", ".join(missing) +
                " have no '#branch' entry in this operating point"
            )
        chosen = [nm for nm in branches if nm in wanted]
        if not chosen:
            rep.warnings.append(
                "none of the named supply sources are present; supply current "
                "is not measurable from this operating point"
            )
            return rep
        rep.sources = sorted(chosen)
        rep.value = sum(abs(branches[nm]) for nm in chosen)
        return rep

    # -- no explicit list: classify what is there -----------------------------
    candidates: list[str] = []
    for nm, val in branches.items():
        if element_letter(nm) == "v":
            candidates.append(nm)
        else:
            rep.excluded[nm] = (
                f"element letter '{element_letter(nm)}' is not an independent "
                "voltage source, so SPICE cannot be reporting a supply rail here"
            )

    if netlist:
        deck = parse_deck_sources(netlist)
        isources = deck.isources
        for nm in list(candidates):
            if source_dc_value(nm, deck) == 0.0:
                candidates.remove(nm)
                rep.excluded[nm] = (
                    "declared DC 0 in the netlist: this is a 0 V sense source "
                    "or an AC-only stimulus, not a supply"
                )
        # A dual supply carries ONE current: it leaves the positive rail,
        # goes through the circuit, and comes back through the negative rail.
        # Summing both magnitudes counts it twice -- 1.8 V and -1.8 V across
        # 3.6 kOhm reported 2.000 mA against a true 1.000 mA -- and a design
        # drawing 1.8 mA against a 2 mA budget was scored at 3.6 mA and failed.
        #
        # But POLARITY ALONE DOES NOT ESTABLISH THAT. It is the return path of
        # the positive rails only when it carries the same current they do, and
        # the two branch currents are right here in `op_points`. Excluding on
        # polarity alone is the same defect sign-flipped: +1.8 V into 1.8 k and
        # -1.8 V into an independent 3.6 k are 1.0 mA and 0.5 mA to ground, a
        # true 1.5 mA, and the exclusion reported 1.0 mA -- 33 pct under
        # budget, with an empty warning list -- while the very text of the
        # exclusion ("its current is the same current") was refuted by the
        # numbers it was written next to. It also swallowed a -0.2 V bias
        # REFERENCE carrying 0.2 uA as "the return path of the positive rail".
        #
        # The magnitude test is the one the no-netlist path already makes on
        # its twins (below); it belongs on both paths.
        positive = [nm for nm in candidates if source_dc_value(nm, deck) > 0.0]
        negative = [nm for nm in candidates if source_dc_value(nm, deck) < 0.0]
        if positive and negative:
            pos_sum = sum(abs(branches[nm]) for nm in positive)
            neg_sum = sum(abs(branches[nm]) for nm in negative)
            returns = (pos_sum > 0.0 and abs(neg_sum - pos_sum)
                       <= RAIL_RETURN_MATCH_FRAC * pos_sum)
            if returns:
                for nm in negative:
                    candidates.remove(nm)
                    rep.excluded[nm] = (
                        f"declared DC {source_dc_value(nm, deck):g} in the "
                        f"netlist and carrying {abs(branches[nm]):.6g} A "
                        f"against {pos_sum:.6g} A on the positive rail(s), the "
                        f"SAME current to within "
                        f"{100.0 * RAIL_RETURN_MATCH_FRAC:g} pct: this is the return "
                        "path of the positive rail(s), not a second supply, "
                        "and adding it would count the supply twice"
                    )
            else:
                rep.warnings.append(
                    "negative rail(s) " + ", ".join(sorted(negative)) +
                    f" carry {neg_sum:.6g} A against {pos_sum:.6g} A on the "
                    "positive rail(s). That is NOT the same current, so they "
                    "are not the return path of the positive rails and are "
                    "summed as independent supplies. If one of them is in fact "
                    "a return path, or a bias reference rather than a supply, "
                    "name the real supplies via `sources`"
                )
        if isources:
            rep.warnings.append(
                "the deck contains independent current source(s) " +
                ", ".join(sorted(isources)) +
                "; ngspice writes no branch vector for a current source, so "
                "if one of them supplies the circuit its current is NOT in "
                "this operating point and cannot be summed here"
            )
    else:
        rep.warnings.append(
            "no netlist was supplied, so a 0 V sense source cannot be told "
            "apart from a supply and a current-source-biased block cannot be "
            "detected at all. Pass `netlist`, or name the supplies via "
            "`sources`"
        )

    if not candidates:
        rep.warnings.append(
            "no independent voltage source with a non-zero DC value remains, "
            "so no supply current can be identified from this operating point"
        )
        return rep

    if len(candidates) > 1:
        rep.warnings.append(
            f"{len(candidates)} voltage source branches were summed "
            f"({', '.join(sorted(candidates))}); which of them are supplies is "
            "not knowable from an operating point alone -- name them via "
            "`sources`"
        )
    biggest = max(abs(branches[nm]) for nm in candidates)
    small = sorted(nm for nm in candidates
                   if biggest > 0.0 and abs(branches[nm]) < biggest / 1000.0)
    if small:
        rep.warnings.append(
            "branch(es) " + ", ".join(small) + " carry less than a thousandth "
            "of the largest branch; that is the shape of a sense source, and a "
            "sense source is not a supply"
        )
    # The "much smaller than the largest" test only catches a sense source in
    # a branch of its own. The two shapes that DOUBLE the answer look nothing
    # like that: a supply ammeter in series with the rail carries the FULL rail
    # current, and the two halves of a dual supply carry the same current in
    # opposite directions. Both make the sum exactly 2x, and neither can be
    # told from two genuine rails WITHOUT THE DECK, so both are named here --
    # and only here. With a deck in hand the ammeter has already been excluded
    # by its DC 0 and the dual supply by the polarity-plus-magnitude test
    # above, so raising this on a deck that declares a 1.8 V rail and a 3.3 V
    # rail that happen to draw the same current says nothing true and ends by
    # advising the caller to pass the netlist they already passed.
    twins = sorted(
        nm for nm in candidates
        if biggest > 0.0 and nm != max(candidates, key=lambda x: abs(branches[x]))
        and abs(abs(branches[nm]) - biggest) <= 1e-3 * biggest
    )
    if twins and not netlist:
        rep.warnings.append(
            "branch(es) " + ", ".join(twins) + " carry the same current as the "
            "largest branch to within 0.1 pct. That is the shape of a supply "
            "AMMETER in series with the rail, or of the two halves of a dual "
            "supply -- in both cases it is ONE current and summing it twice "
            "reports exactly 2x the real supply current. Pass `netlist` so the "
            "polarities and the 0 V sense sources can be read, or name the "
            "supplies via `sources`"
        )

    rep.sources = sorted(candidates)
    rep.value = sum(abs(branches[nm]) for nm in candidates)
    return rep


def supply_current(op_points: dict[str, float],
                   sources: Optional[Sequence[str]] = None,
                   netlist: Optional[str] = None) -> Optional[float]:
    """Total supply current magnitude, or None when it is not identifiable.

    Thin wrapper over supply_current_report(); every warning that report
    carries is logged. Call supply_current_report() directly when the caller
    needs to act on the ambiguity rather than just record it.
    """
    rep = supply_current_report(op_points, sources, netlist)
    for w in rep.warnings:
        log.warning("supply_current: %s", w)
    return rep.value


def integrate_noise(freqs: Sequence[float], spectrum: Sequence[float]) -> Optional[float]:
    """RMS noise over the swept band from an amplitude spectral density.

    Trapezoid on power (spectrum**2) against linear frequency, then sqrt.
    ngspice's own inoise_total/onoise_total are grid dependent and read about
    12 pct high at the common 'dec 10' density even on a flat spectrum, so this
    recomputation is preferred. Use >= 100 points/decade for noise-scored decks.
    """
    n = min(len(freqs), len(spectrum))
    if n < 2:
        return None
    acc = 0.0
    for i in range(n - 1):
        f0, f1 = float(freqs[i]), float(freqs[i + 1])
        p0 = float(spectrum[i]) ** 2
        p1 = float(spectrum[i + 1]) ** 2
        acc += 0.5 * (p0 + p1) * (f1 - f0)
    if acc < 0.0:
        return None
    return math.sqrt(acc)
