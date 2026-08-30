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
    by the low-frequency SLOPE (see low_frequency_slope), not by comparing two
    adjacent samples. Otherwise it is None and notes["dc_gain_db"] says why;
    the gain at the bottom of the sweep is always available as
    low_freq_gain_db, under a name that does not claim to be a DC gain.
  - The -3 dB edges are referenced to the passband gain, which is the DC gain
    when the sweep reaches DC and the peak gain otherwise.
  - The unity-gain crossing is looked for whenever the response actually
    exceeds 0 dB anywhere, not only when it starts above 0 dB.
  - The inversion of an inverting loop is inferred from the phase in the
    MID-BAND (at the peak-gain frequency), which does not move when the sweep
    start moves.

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


def phase_inversion_shift(phase_unwrapped: Sequence[float], ref_index: int = 0,
                          tol_deg: float = PHASE_INVERSION_TOL_DEG) -> int:
    """How many whole 180 deg turns to remove, judged at ONE reference sample.

    Returns k, so that `phase - 180*k` puts the reference sample near 0 deg.
    Returns 0 when the reference phase is further than `tol_deg` from any
    multiple of 180 deg, because then there is no inversion to infer and
    guessing one would fabricate a 180 deg error rather than remove one.

    `ref_index` must point at a sample that is INVARIANT to where the sweep
    starts -- ac_metrics uses the peak-gain (mid-band) sample. Using sample 0
    is what made the phase margin depend on the sweep start.
    """
    n = len(phase_unwrapped)
    if n == 0:
        return 0
    i = min(max(0, ref_index), n - 1)
    p = float(phase_unwrapped[i])
    if not math.isfinite(p):
        return 0
    k = int(round(p / 180.0))
    if k == 0:
        return 0
    if abs(p - 180.0 * k) > tol_deg:
        return 0
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
    level is never crossed. `start_index` begins the scan at a later sample,
    which is how ac_metrics looks for the unity-gain crossing ABOVE the
    peak-gain frequency of a band-pass response.
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
    n = min(len(freqs), len(values))
    if n < 2:
        return None, None
    f0 = float(freqs[0])
    g0 = float(values[0])
    if f0 <= 0.0 or not math.isfinite(g0):
        return None, None
    j: Optional[int] = None
    for i in range(1, n):
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
      passband_gain_db     the level the -3 dB edges are referenced to: the DC
                           gain when the sweep reaches DC, else the peak gain.
      bandwidth_3db        the UPPER -3 dB edge, in Hz, referenced to
                           passband_gain_db. None when the passband itself is
                           not inside the sweep.
      f_3db_lo, f_3db_hi   both -3 dB edges. f_3db_lo is non-None only for a
                           band-pass response; then the -3 dB SPAN is
                           f_3db_hi - f_3db_lo and notes says so.
      ugb                  unity-gain frequency, looked for whenever the peak
                           gain exceeds 0 dB.
      phase_margin         180 + phase(ugb), after removing any whole inversion
                           inferred from the MID-BAND phase.
      gain_margin, f_180   -gain at the first -180 deg crossing.
      phase_inversion_k    whole 180 deg turns removed (0 for a non-inverting
                           response).
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

    # -- does this sweep actually reach DC? ---------------------------------
    slope, span = low_frequency_slope(f, g)
    out["low_slope_db_per_dec"] = slope
    if f[0] <= 0.0:
        reaches_dc: Optional[bool] = True
    elif slope is None:
        reaches_dc = None
    else:
        reaches_dc = abs(slope) <= DC_SLOPE_TOL_DB_PER_DEC

    if reaches_dc:
        out["dc_gain_db"] = g[0]
        out["dc_gain_valid"] = 1.0
    elif reaches_dc is None:
        out["dc_gain_valid"] = None
        notes["dc_gain_db"] = (
            "cannot tell whether the sweep reaches DC: no usable "
            "low-frequency span (need two finite samples above 0 Hz)"
        )
    else:
        out["dc_gain_valid"] = 0.0
        notes["dc_gain_db"] = (
            f"the sweep does not reach DC. At f_start = {f[0]:g} Hz the "
            f"response still slopes {slope:.4g} dB/decade (tolerance "
            f"{DC_SLOPE_TOL_DB_PER_DEC:g} dB/dec over {span:.3g} decades), so "
            f"the {g[0]:.4g} dB measured there is a point on a rolloff, not a "
            "DC gain. It is reported as low_freq_gain_db. Extend the sweep "
            "downward to measure the DC gain."
        )

    # -- passband reference for the -3 dB edges ------------------------------
    if out["dc_gain_db"] is not None:
        ref = float(out["dc_gain_db"])
        ref_kind = "DC"
    else:
        ref = peak
        ref_kind = "peak"
    out["passband_gain_db"] = ref

    if ref_kind == "peak" and peak_i == 0:
        notes["bandwidth_3db"] = (
            "refused: the peak gain is the FIRST sweep sample and the sweep "
            "does not reach DC, so the passband lies below f_start = "
            f"{f[0]:g} Hz and no -3 dB edge can be referenced to it. Extend "
            "the sweep to lower frequencies."
        )
    else:
        out["f_3db_hi"] = crossing_freq(f, g, ref - DB3, direction=-1,
                                        start_index=peak_i)
        if peak_i > 0:
            out["f_3db_lo"] = last_crossing_freq_below(
                f, g, ref - DB3, peak_i, direction=1)
        out["bandwidth_3db"] = out["f_3db_hi"]
        if out["f_3db_hi"] is None:
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
    if peak > 0.0:
        ugb = crossing_freq(f, g, 0.0, direction=-1, start_index=peak_i)
        out["ugb"] = ugb
        if ugb is None:
            notes["ugb"] = (
                f"the gain reaches {peak:.4g} dB but never falls back through "
                f"0 dB inside the sweep (last sample {g[-1]:.4g} dB at "
                f"{f[-1]:g} Hz); extend the sweep upward"
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

    if phase_deg is None:
        return out

    ph = unwrap_deg(list(phase_deg)[:n])
    k = 0
    if normalize_phase:
        k = phase_inversion_shift(ph, ref_index=peak_i)
        if k:
            ph = [p - 180.0 * k for p in ph]
    out["phase_inversion_k"] = float(k)
    if k:
        notes["phase"] = (
            f"an inversion of {180.0 * k:+g} deg was removed. It was inferred "
            f"from the mid-band phase at the peak-gain frequency "
            f"{f[peak_i]:g} Hz, which does not move when f_start moves."
        )

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

    f180 = crossing_freq(f, ph, -180.0, direction=-1)
    out["f_180"] = f180
    if f180 is not None:
        g_at_180 = value_at_freq(f, g, f180)
        if g_at_180 is not None and math.isfinite(g_at_180):
            out["gain_margin"] = -g_at_180
    else:
        # If the phase never reaches -180 the loop is unconditionally stable and
        # the gain margin is infinite. It is deliberately left as None rather
        # than 0.0, which would read as marginally unstable.
        notes["gain_margin"] = (
            "the phase never falls through -180 deg inside the sweep, so the "
            "gain margin is infinite; reported as None, never as 0.0"
        )
    return out


# ----------------------------------------------------------------------------
# Transient metrics
# ----------------------------------------------------------------------------

def settled_levels(y: Sequence[float], frac: float = 0.02,
                   tol: float = 0.002) -> tuple[float, float]:
    """Initial and final settled levels, averaged over the first/last `frac`.

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
    the total excursion. Zero-mean noise cancels in both halves, so a genuinely
    quiescent but noisy window keeps its full averaging, while a window that
    straddles an edge fails immediately. A spread test cannot tell those two
    apart and throws away the averaging exactly when it is needed.

    The TRAILING window is averaged as-is: on a waveform that is still ringing
    at the end of the simulation the mean of the tail is a better estimate of
    the final level than its last sample. Use settling_time() to find out
    whether the waveform settled at all -- it returns None when it did not.
    """
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    k0 = max(1, int(n * frac))
    y1 = sum(float(v) for v in y[-k0:]) / k0
    k = k0
    while k > 1:
        window = [float(v) for v in y[:k]]
        half = k // 2
        drift = abs(sum(window[half:]) / (k - half) - sum(window[:half]) / half)
        excursion = abs(y1 - window[0])
        if excursion <= 0.0 or drift <= tol * excursion:
            break
        k //= 2
    y0 = sum(float(v) for v in y[:k]) / k
    return y0, y1


def crossing_time(t: Sequence[float], y: Sequence[float], level: float,
                  start_index: int = 0, direction: int = 0) -> Optional[float]:
    """First time at or after start_index where y crosses level, interpolated."""
    n = min(len(t), len(y))
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

    Measured from the time origin of the vector, using the settled initial and
    final levels as the 0 pct and 100 pct references.
    """
    y0, y1 = settled_levels(y)
    if y1 == y0:
        return None
    level = y0 + frac * (y1 - y0)
    direction = 1 if y1 > y0 else -1
    return crossing_time(t, y, level, direction=direction)


def rise_time(t: Sequence[float], y: Sequence[float],
              lo: float = 0.1, hi: float = 0.9) -> Optional[float]:
    """10 pct to 90 pct rise time of the first rising edge."""
    y0, y1 = settled_levels(y)
    if y1 <= y0:
        return None
    span = y1 - y0
    t_lo = crossing_time(t, y, y0 + lo * span, direction=1)
    t_hi = crossing_time(t, y, y0 + hi * span, direction=1)
    if t_lo is None or t_hi is None:
        return None
    return t_hi - t_lo


def fall_time(t: Sequence[float], y: Sequence[float],
              lo: float = 0.1, hi: float = 0.9) -> Optional[float]:
    """90 pct to 10 pct fall time of the first falling edge."""
    y0, y1 = settled_levels(y)
    if y1 >= y0:
        return None
    span = y0 - y1
    t_hi = crossing_time(t, y, y1 + hi * span, direction=-1)
    t_lo = crossing_time(t, y, y1 + lo * span, direction=-1)
    if t_lo is None or t_hi is None:
        return None
    return t_lo - t_hi


def overshoot_pct(y: Sequence[float]) -> Optional[float]:
    """Peak overshoot past the settled final level, in percent of the step."""
    if not y:
        return None
    y0, y1 = settled_levels(y)
    if y1 == y0:
        return None
    if y1 > y0:
        peak = max(float(v) for v in y)
        return 100.0 * (peak - y1) / (y1 - y0)
    trough = min(float(v) for v in y)
    return 100.0 * (y1 - trough) / (y0 - y1)


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
    """
    n = min(len(t), len(y))
    if n == 0:
        return None
    y0, y1 = settled_levels(y)
    span = abs(y1 - y0)
    if span == 0.0:
        return None
    band = tol * span
    for i in range(n - 1, -1, -1):
        if abs(float(y[i]) - y1) > band:
            if i + 1 >= n - 1:
                return None  # never settles inside the simulated window
            return float(t[i + 1])
    return float(t[0])


def slew_rate(t: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """10-90 chord slew rate in volts per second (units per second)."""
    tr = rise_time(t, y)
    if tr is None or tr <= 0.0:
        return None
    y0, y1 = settled_levels(y)
    return 0.8 * (y1 - y0) / tr


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

    "notes" carries the reason for anything the data does not define, the same
    way ac_metrics does.
    """
    y0, y1 = settled_levels(y)
    st = settling_time(t, y)
    notes: dict[str, str] = {}
    if st is None and len(y) >= 2 and y1 != y0:
        notes["settling_time"] = (
            "the waveform is still outside the settling band at the end of the "
            "simulated window, so it never settles inside it; extend tstop"
        )
    return {
        "y_initial": y0,
        "y_final": y1,
        "rise_time": rise_time(t, y),
        "fall_time": fall_time(t, y),
        "overshoot_pct": overshoot_pct(y),
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


def _parse_source_cards(netlist: str) -> tuple[dict[str, float], list[str]]:
    """(voltage source name -> its DC value, independent current source names).

    A deliberately small SPICE reader: it only needs the element letter, the
    name, and the DC value of V cards. Comments, continuations and .control
    blocks are skipped.

    The DC value comes back as NaN only when the card names a value that
    cannot be read (an unusual unit suffix, a parameter expression). NaN means
    "unknown", and the caller never excludes an unknown source -- dropping a
    real supply is a much worse failure than keeping a stimulus source whose
    branch current is zero anyway. A card with no numeric value at all
    ("Vin in 0 AC 1", "Vin in 0 PULSE(0 1 ...)", "V1 a b") has a DC value of
    exactly 0 by SPICE's own rules, which is knowledge, not a guess.
    """
    vsources: dict[str, float] = {}
    isources: list[str] = []
    in_control = False
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
        if in_control or line[0] in "*+.;":
            continue
        head = line.split()[0]
        kind = head[0].lower()
        if kind not in ("v", "i"):
            continue
        name = head.lower()
        if kind == "i":
            isources.append(name)
            continue
        rest = line[len(head):]
        if _DC_KEYWORD_RE.search(rest):
            m = _DC_VALUE_RE.search(rest)
            # A 'DC' keyword whose value is a parameter expression ({vsup}) is
            # UNKNOWN, not zero. Never let an unreadable value exclude a rail.
            vsources[name] = _spice_float(m.group(1)) if m else math.nan
            continue
        toks = rest.split()
        if len(toks) < 3:
            vsources[name] = 0.0            # "V1 a b" -> DC 0
        elif toks[2][0].isalpha():
            vsources[name] = 0.0            # "Vin in 0 AC 1" -> DC 0
        else:
            vsources[name] = _spice_float(toks[2])
    return vsources, isources


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
        if nm[:1] == "v":
            candidates.append(nm)
        else:
            rep.excluded[nm] = (
                f"element letter '{nm[:1]}' is not an independent voltage "
                "source, so SPICE cannot be reporting a supply rail here"
            )

    if netlist:
        vsources, isources = _parse_source_cards(netlist)
        for nm in list(candidates):
            if vsources.get(nm, math.nan) == 0.0:
                candidates.remove(nm)
                rep.excluded[nm] = (
                    "declared DC 0 in the netlist: this is a 0 V sense source "
                    "or an AC-only stimulus, not a supply"
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
