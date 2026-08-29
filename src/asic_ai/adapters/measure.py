"""Scalar measurement helpers for simulation waveforms.

Pure Python. No ctypes, no simulator dependency, so every function here can be
unit tested against an analytic waveform. The ngspice adapter uses these to turn
raw vectors into the scalar metrics the reward function consumes
(dc_gain_db, ugb, phase_margin, gain_margin, bandwidth_3db, idd, rise_time, ...).

Conventions used throughout this module:

  - Frequency in Hz, time in seconds, gain in dB (20*log10), phase in degrees.
  - Frequency-domain interpolation is linear in (log10(f), value) space, which
    is exact for a single-pole rolloff and is what SPICE post-processors use.
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
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# 10*log10(2). The half-power point, exactly.
DB3 = 10.0 * math.log10(2.0)


# ----------------------------------------------------------------------------
# Generic numeric helpers
# ----------------------------------------------------------------------------

def db20(magnitude: float, floor: float = 1e-300) -> float:
    """20*log10(|magnitude|) with a floor so an exact zero cannot raise."""
    m = abs(magnitude)
    if m < floor:
        m = floor
    return 20.0 * math.log10(m)


def unwrap_deg(phase: Sequence[float]) -> list[float]:
    """Remove 2*pi wraps from a phase sequence in degrees.

    Exact no-op when the sequence never wraps, so it is always safe to apply.
    Assumes adjacent samples differ by less than 180 deg, which requires a
    reasonably dense sweep (>= 50 points/decade near a sharp pole pair).
    """
    if not phase:
        return []
    out = [float(phase[0])]
    for i in range(1, len(phase)):
        d = float(phase[i]) - float(phase[i - 1])
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        out.append(out[-1] + d)
    return out


def normalize_dc_phase(phase_unwrapped: Sequence[float]) -> list[float]:
    """Shift an unwrapped phase so the lowest-frequency point sits near 0 deg.

    A broken-loop response of a negative feedback amplifier starts at about
    +/-180 deg rather than 0 deg. Without this shift the phase margin comes out
    exactly 180 deg wrong. The shift is always a whole multiple of 180 deg, so
    it cannot invent phase that is not there.
    """
    if not phase_unwrapped:
        return []
    k = round(phase_unwrapped[0] / 180.0)
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
    """
    gain_db: list[float] = []
    phase_deg: list[float] = []
    for i, o in enumerate(out):
        h = complex(o)
        if inp is not None:
            d = complex(inp[i])
            if d == 0:
                h = complex(0.0, 0.0)
            else:
                h = h / d
        gain_db.append(db20(abs(h)))
        phase_deg.append(math.degrees(math.atan2(h.imag, h.real)))
    return gain_db, phase_deg


def _interp(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
    """Linear inverse interpolation: the x where the segment crosses y."""
    if y1 == y0:
        return x0
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0)


def crossing_freq(freqs: Sequence[float], values: Sequence[float], level: float,
                  direction: int = 0) -> Optional[float]:
    """First frequency where `values` crosses `level`, interpolated in log10(f).

    direction: -1 falling only, +1 rising only, 0 either. Returns None when the
    level is never crossed.
    """
    n = min(len(freqs), len(values))
    for i in range(n - 1):
        a, b = values[i], values[i + 1]
        if a == b:
            continue
        falling = a >= level > b
        rising = a <= level < b
        if direction <= 0 and falling:
            pass
        elif direction >= 0 and rising:
            pass
        else:
            continue
        f0, f1 = freqs[i], freqs[i + 1]
        if f0 <= 0.0 or f1 <= 0.0:
            return _interp(f0, a, f1, b, level)
        lf = _interp(math.log10(f0), a, math.log10(f1), b, level)
        return 10.0 ** lf
    return None


def value_at_freq(freqs: Sequence[float], values: Sequence[float],
                  f_target: float) -> Optional[float]:
    """Interpolate `values` at `f_target`, linear in log10(f)."""
    n = min(len(freqs), len(values))
    if n == 0 or f_target <= 0.0:
        return None
    if f_target <= freqs[0]:
        return float(values[0])
    if f_target >= freqs[n - 1]:
        return float(values[n - 1])
    lt = math.log10(f_target)
    for i in range(n - 1):
        f0, f1 = freqs[i], freqs[i + 1]
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

def ac_metrics(freqs: Sequence[float], gain_db: Sequence[float],
               phase_deg: Optional[Sequence[float]] = None,
               normalize_phase: bool = True) -> dict[str, Optional[float]]:
    """Standard small-signal metrics from one frequency response.

    Returns keys: dc_gain_db, bandwidth_3db, ugb, phase_margin, gain_margin,
    f_180, rolloff_db_per_dec. Any metric that the data does not define is None.

    dc_gain_db is the gain at the lowest swept frequency. That is only the true
    DC gain when the sweep starts well below the dominant pole; `dc_gain_valid`
    reports whether the first two points agree to within 0.01 dB.
    """
    out: dict[str, Optional[float]] = {
        "dc_gain_db": None, "dc_gain_valid": None, "bandwidth_3db": None,
        "ugb": None, "phase_margin": None, "gain_margin": None, "f_180": None,
        "rolloff_db_per_dec": None,
    }
    n = min(len(freqs), len(gain_db))
    if n < 2:
        return out

    dc_gain = float(gain_db[0])
    out["dc_gain_db"] = dc_gain
    out["dc_gain_valid"] = float(abs(gain_db[0] - gain_db[1]) < 0.01)
    out["bandwidth_3db"] = crossing_freq(freqs, gain_db, dc_gain - DB3, direction=-1)
    ugb = crossing_freq(freqs, gain_db, 0.0, direction=-1)
    if dc_gain <= 0.0:
        # No gain anywhere: unity gain bandwidth is undefined, not zero.
        ugb = None
    out["ugb"] = ugb

    if freqs[n - 1] > 0.0 and freqs[0] > 0.0:
        # Rolloff over the top decade of the sweep.
        f_hi = freqs[n - 1]
        f_lo = f_hi / 10.0
        g_hi = value_at_freq(freqs, gain_db, f_hi)
        g_lo = value_at_freq(freqs, gain_db, f_lo)
        if g_hi is not None and g_lo is not None and f_lo >= freqs[0]:
            out["rolloff_db_per_dec"] = g_hi - g_lo

    if phase_deg is None:
        return out
    ph = unwrap_deg(phase_deg)
    if normalize_phase:
        ph = normalize_dc_phase(ph)

    if ugb is not None:
        p_at_ugb = value_at_freq(freqs, ph, ugb)
        if p_at_ugb is not None:
            out["phase_margin"] = 180.0 + p_at_ugb

    f180 = crossing_freq(freqs, ph, -180.0, direction=-1)
    out["f_180"] = f180
    if f180 is not None:
        g_at_180 = value_at_freq(freqs, gain_db, f180)
        if g_at_180 is not None:
            out["gain_margin"] = -g_at_180
    # If the phase never reaches -180 the loop is unconditionally stable and
    # the gain margin is infinite. It is deliberately left as None rather than
    # 0.0, which would read as marginally unstable.
    return out


# ----------------------------------------------------------------------------
# Transient metrics
# ----------------------------------------------------------------------------

def settled_levels(y: Sequence[float], frac: float = 0.02) -> tuple[float, float]:
    """Initial and final settled levels, averaged over the first/last `frac`.

    Deliberately not min()/max(): on a ringing waveform max() is the overshoot
    peak, which shrinks the apparent 90 pct level and understates rise time.
    """
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    k = max(1, int(n * frac))
    y0 = sum(float(v) for v in y[:k]) / k
    y1 = sum(float(v) for v in y[-k:]) / k
    return y0, y1


def crossing_time(t: Sequence[float], y: Sequence[float], level: float,
                  start_index: int = 0, direction: int = 0) -> Optional[float]:
    """First time at or after start_index where y crosses level, interpolated."""
    n = min(len(t), len(y))
    for i in range(max(0, start_index), n - 1):
        a, b = float(y[i]), float(y[i + 1])
        if a == b:
            continue
        rising = a <= level < b
        falling = a >= level > b
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
            if i + 1 >= n:
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


def tran_metrics(t: Sequence[float], y: Sequence[float]) -> dict[str, Optional[float]]:
    """All single-waveform transient metrics at once."""
    y0, y1 = settled_levels(y)
    return {
        "y_initial": y0,
        "y_final": y1,
        "rise_time": rise_time(t, y),
        "fall_time": fall_time(t, y),
        "overshoot_pct": overshoot_pct(y),
        "settling_time": settling_time(t, y),
        "slew_rate": slew_rate(t, y),
        "t_50pct": time_to_fraction(t, y, 0.5),
        "t_63pct": time_to_fraction(t, y, 0.6321205588285577),
    }


# ----------------------------------------------------------------------------
# DC / supply metrics
# ----------------------------------------------------------------------------

def supply_current(op_points: dict[str, float],
                   sources: Optional[Sequence[str]] = None) -> Optional[float]:
    """Total supply current magnitude from operating point branch currents.

    ngspice reports a voltage source branch current with the passive sign
    convention on the source, so a supply that delivers power reports a
    NEGATIVE branch current. Magnitudes are summed per source rather than the
    sum being taken and then abs()'d, so a sourcing rail and a sinking rail do
    not cancel.

    `sources` names the voltage sources to include (case insensitive, with or
    without the '#branch' suffix). When omitted, every '*#branch' entry whose
    source name starts with 'v' is included.
    """
    total = 0.0
    found = False
    if sources is not None:
        wanted = {s.lower().split("#")[0] for s in sources}
    else:
        wanted = None
    for key, val in op_points.items():
        k = key.lower()
        if not k.endswith("#branch"):
            continue
        name = k.split("#")[0]
        if wanted is None:
            if not name.startswith("v"):
                continue
        elif name not in wanted:
            continue
        total += abs(float(val))
        found = True
    return total if found else None


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
