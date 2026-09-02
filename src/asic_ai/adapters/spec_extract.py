"""Turn simulation results into the spec-name-keyed scalars the reward wants.

This closes the gap between two layers that never met:

    adapters/*        -> DCResult / ACResult / TranResult / NoiseResult /
                         StabilityResult, keyed by SCHEMA FIELD names
                         (op_points, sweeps, frequencies, signals, time)

    reward/reward.py  -> RewardFunction.compute(results=...) does
                         results.get("dc_gain"), i.e. keyed by EVAL TASK
                         SPEC names, in the UNIT the task declares

Nothing converted between them, so `results.get("dc_gain")` was always None and
RewardFunction scored every spec at SCORE_CLIP_MIN (-1.0) no matter how good the
design was. A perfect adapter does not fix that on its own.

Two things make the conversion non-trivial, both established by reading all 77
eval tasks:

1. NAMING. 117 distinct spec names are in use for far fewer physical
   quantities. Gain alone appears as gain / dc_gain / gain_min / gain_max /
   conversion_gain; bandwidth as ugb / gbw / bw / bandwidth / cutoff_freq;
   phase margin as pm / phase_margin. ALIASES normalises them.

2. UNITS. The same spec name carries different units in different tasks: idd is
   uA in some and mA in others, gain is dB in some and V/V in others, delay is
   ps or ns, settling_time is ns or s. Measurements are computed in SI and
   converted to whatever unit the task declared, per task.

HONEST COVERAGE IS THE POINT
----------------------------
About a third of all spec names are digital/functional booleans (correct,
no_metastability, cdc_correct, gray_code_pointers, ...) that no SPICE analysis
can produce. Silently omitting them is the dangerous option, because
RewardFunction reads a missing spec as -1.0 -- indistinguishable from a design
that was measured and failed. So every spec this module cannot produce is
returned in `unmeasurable` with a reason, and callers are expected to act on it
(drop those specs from the reward, or route them to a testbench) rather than
letting them silently pin the score to the floor.

Usage:
    ext = extract_specs(task["specs"], ac=ac_result, dc=dc_result,
                        output_signal="out")
    ext.values          # {"dc_gain": 62.4, "idd": 87.3}  in the task's units
    ext.unmeasurable    # {"correct": "digital/functional spec ..."}
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from asic_ai.adapters import measure

log = logging.getLogger(__name__)

__all__ = [
    "SpecExtraction",
    "extract_specs",
    "canonical_metric",
    "convert_from_si",
    "ALIASES",
    "DIMENSIONS",
]


# ----------------------------------------------------------------------------
# Spec name -> canonical metric
# ----------------------------------------------------------------------------
# Keys are lowercased spec names as they appear in eval/tasks/**/*.yaml.
# Values are canonical metric names produced by _compute_metrics() below.
# A spec name absent from this table is reported as unmeasurable by NAME, which
# is deliberate: guessing at an unknown spec is how fabricated rewards start.

ALIASES: dict[str, str] = {
    # -- small-signal gain ---------------------------------------------------
    # "dc_gain" means the gain AT DC and maps to dc_gain_db, which measure.py
    # produces only when the sweep demonstrably reaches DC. Every other gain
    # spelling means "the gain of this amplifier", which for an AC-coupled or
    # band-pass stage is the MID-BAND gain, not whatever the response happens
    # to be at f_start. Those map to passband_gain_db, which is the DC gain
    # when the sweep reaches DC and the gain of the FLAT region otherwise.
    # Before this split an AC-coupled stage with 40 dB of mid-band gain
    # reported -24 dB; and while it was the PEAK gain rather than the flat
    # region, an under-damped 35 dB amplifier reported its 42 dB resonance and
    # passed a 40 dB spec, so every step toward a peakier design read better.
    "gain": "passband_gain_db",
    "dc_gain": "dc_gain_db",
    "gain_db": "passband_gain_db",
    "conversion_gain": "passband_gain_db",
    "gain_max": "passband_gain_db",
    "gain_min": "passband_gain_db",
    "linearity": "passband_gain_db",

    # -- bandwidth / frequency ----------------------------------------------
    # bandwidth_3db is the UPPER -3 dB edge in Hz, referenced to the passband
    # gain (see measure.ac_metrics). For a band-pass response the low-side edge
    # is reported separately as f_3db_lo and the -3 dB SPAN is the difference;
    # ac_metrics["notes"]["bandwidth_3db"] spells that out per measurement.
    "ugb": "ugb",
    "gbw": "ugb",
    "bw": "bandwidth_3db",
    "bandwidth": "bandwidth_3db",
    "cutoff_freq": "bandwidth_3db",

    # -- stability -----------------------------------------------------------
    "pm": "phase_margin",
    "phase_margin": "phase_margin",
    "gain_margin": "gain_margin",

    # -- supply current ------------------------------------------------------
    "idd": "idd",
    "iq": "idd",
    "idd_quiescent": "idd",
    "idd_1ghz": "idd",
    "i_supply": "idd",

    # -- dc output -----------------------------------------------------------
    "vout": "vout",
    "v_out": "vout",
    "vref": "vout",
    "output_swing": "output_swing",
    "vout_swing": "output_swing",
    "swing": "output_swing",

    # -- dc sweep shape ------------------------------------------------------
    # The grounded SFT generator (and any VTC-style task) names the maximum
    # small-signal slope of a monotone dc sweep "max_gain" in V/V. It is a
    # different quantity from the AC gains above: no frequency axis exists.
    "max_gain": "dc_max_slope",
    "dc_slope": "dc_max_slope",
    "vtc_gain": "dc_max_slope",
    "iout_max": "iout_max",
    "iout_swing": "iout_swing",
    "output_tc": "output_tc",

    # -- transient extras ----------------------------------------------------
    "vout_final": "vout_final",
    "final_value": "vout_final",
    "osc_freq": "osc_freq",
    "oscillation_frequency": "osc_freq",
    "fosc": "osc_freq",

    # -- transient -----------------------------------------------------------
    "slew_rate": "slew_rate",
    "settling_time": "settling_time",
    "settling": "settling_time",
    "delay": "prop_delay",
    "rise_time": "rise_time",
    "fall_time": "fall_time",
    "overshoot": "overshoot_pct",

    # -- noise ---------------------------------------------------------------
    "noise": "input_noise_density",
    "input_noise": "input_noise_density",
    "output_noise": "output_noise_density",
    "noise_rms": "input_noise_rms",
}


# Canonical metric -> physical dimension, used to pick the unit conversion.
DIMENSIONS: dict[str, str] = {
    "dc_gain_db": "gain",
    "passband_gain_db": "gain",
    "ugb": "frequency",
    "bandwidth_3db": "frequency",
    "phase_margin": "angle",
    "gain_margin": "gain_db_only",
    "idd": "current",
    "vout": "voltage",
    "output_swing": "voltage",
    "vout_final": "voltage",
    "dc_max_slope": "gain_vv",
    "iout_max": "current",
    "iout_swing": "current",
    "output_tc": "tc_voltage",
    "osc_freq": "frequency",
    "slew_rate": "slew",
    "settling_time": "time",
    "prop_delay": "time",
    "rise_time": "time",
    "fall_time": "time",
    "overshoot_pct": "percent",
    "input_noise_density": "noise_density",
    "output_noise_density": "noise_density",
    # The INTEGRATED input-referred noise is in the same family as the input
    # density it was integrated from: volts for '.noise v(out) Vin' and AMPERES
    # for '.noise v(out) Iin'. Declaring it "voltage" outright is the C6 defect
    # left in place one field over -- a nA spec was refused as "not a voltage
    # unit" and a uV spec silently rescaled amperes as volts.
    "input_noise_rms": "noise_rms",
}


# Unit string (lowercased) -> size of that unit in SI.
# value_in_unit = value_SI / UNIT_SCALE[unit]
_SCALES: dict[str, dict[str, float]] = {
    "frequency": {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12,
                  "sa/s": 1.0},
    "current": {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "na": 1e-9, "pa": 1e-12},
    "voltage": {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9},
    "time": {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
             "fs": 1e-15},
    "slew": {"v/s": 1.0, "v/us": 1e6, "v/ms": 1e3, "v/ns": 1e9},
    "angle": {"deg": 1.0, "degrees": 1.0, "degree": 1.0, "": 1.0},
    "percent": {"%": 1.0, "percent": 1.0, "": 1.0},
    "gain_db_only": {"db": 1.0, "": 1.0},
    "gain_vv": {"v/v": 1.0, "": 1.0},
    "tc_voltage": {"v/c": 1.0, "mv/c": 1e-3, "uv/c": 1e-6},
    # ngspice's inoise_spectrum is referred to the source named on the .noise
    # card, so its UNIT depends on that source: V/sqrt(Hz) for a voltage source
    # and A/sqrt(Hz) for a current source. One table holding both let a TIA
    # task declaring pA/sqrt(Hz) divide a V/sqrt(Hz) measurement by 1e-12 --
    # silently wrong by twelve orders of magnitude. The families are kept apart
    # and the .noise card decides which one applies; see noise_input_kind().
    "noise_density_v": {"v/sqrt(hz)": 1.0, "nv/sqrt(hz)": 1e-9,
                        "uv/sqrt(hz)": 1e-6},
    "noise_density_i": {"a/sqrt(hz)": 1.0, "pa/sqrt(hz)": 1e-12,
                        "na/sqrt(hz)": 1e-9, "fa/sqrt(hz)": 1e-15},
}
# The integrated RMS of each family is just that family's plain unit.
_NOISE_RMS_DIMENSION = {"v": "voltage", "i": "current"}

# The .noise card: '.noise v(out) V1 dec 100 1 1G'. The FIRST token is the
# output expression and the SECOND is the input source, whose first letter is
# its type -- a SPICE language rule.
_NOISE_CARD_RE = re.compile(
    r"^\s*\.noise\s+(\S+)\s+([a-zA-Z]\S*)", re.IGNORECASE | re.MULTILINE)


def noise_input_kind(netlist: Optional[str]) -> Optional[str]:
    """'v', 'i', or None: the type of the source a .noise run is referred to.

    This is what makes an INPUT-referred noise density a voltage density or a
    current density. It is not knowable from a NoiseResult, which carries
    numbers and no units.

    It says nothing whatever about the OUTPUT-referred density; use
    noise_output_kind() for that.
    """
    if not netlist:
        return None
    m = _NOISE_CARD_RE.search(netlist)
    if not m:
        return None
    letter = m.group(2)[0].lower()
    return letter if letter in ("v", "i") else None


def noise_output_kind(netlist: Optional[str]) -> Optional[str]:
    """'v', 'i', or None: the family of the OUTPUT-referred noise density.

    onoise_spectrum is the noise at the OUTPUT EXPRESSION of the .noise card,
    so its family comes from that expression and NOT from the input source.
    '.noise v(out) Iin dec 100 1 1G' measures a VOLTAGE density at out even
    though it refers the input-equivalent one to a current.

    Scaling the output density by the INPUT source's letter is the C6 defect
    moved one field over, and it fires on exactly the deck C6 was written for:
    on that TIA a `nV/sqrt(Hz)` output_noise spec was refused as "not a
    noise_density_i unit" while a `pA/sqrt(Hz)` one was ACCEPTED and a
    0.3 uV/sqrt(Hz) measurement came back as 300000 -- a factor of 1e12, in the
    direction that makes the design look good.

    ngspice's .noise output must be a voltage expression, so anything of the
    form 'v(...)' is a voltage density and anything else is unknown rather
    than guessed.
    """
    if not netlist:
        return None
    m = _NOISE_CARD_RE.search(netlist)
    if not m:
        return None
    expr = m.group(1).strip().lower()
    if expr.startswith("v(") or expr.startswith("vdb(") or expr.startswith("vm("):
        return "v"
    if expr.startswith("i("):
        return "i"
    return None

# Units that mark a spec as functional/digital rather than analog-measurable.
_NON_ANALOG_UNITS = {"bool", "bits", "lsb", "cycles", "years", "um2"}


class UnitError(ValueError):
    """A spec declared a unit that does not belong to its physical dimension."""


def convert_from_si(value: float, dimension: str, unit: str) -> float:
    """Convert an SI-valued measurement into the unit a task declared.

    Gain is special: it is measured in dB, so a 'db' unit passes through and a
    'v/v' unit is converted out of the log domain.
    """
    u = (unit or "").strip().lower()

    if dimension == "gain":
        if u in ("db", ""):
            return value
        if u in ("v/v", "v/v ", "vv", "ratio"):
            return 10.0 ** (value / 20.0)
        raise UnitError(f"unit {unit!r} is not a gain unit")

    table = _SCALES.get(dimension)
    if table is None:
        raise UnitError(f"no unit table for dimension {dimension!r}")
    if u not in table:
        raise UnitError(f"unit {unit!r} is not a {dimension} unit")
    return value / table[u]


def canonical_metric(spec_name: str) -> Optional[str]:
    """Canonical metric for a spec name, or None when the name is unknown."""
    return ALIASES.get(spec_name.strip().lower())


def spec_noise_freq(specs: Mapping[str, Any]) -> Optional[float]:
    """The frequency a task's noise-DENSITY spec is declared at, if it says.

    A noise density is a value AT A FREQUENCY, and _noise_density() rightly
    refuses to invent one: without a frequency it reports the bottom of the
    sweep only when the spectrum is demonstrably flat there, and on any real
    1/f-plus-thermal spectrum it refuses. That refusal was correct and it was
    also unreachable, because nothing ever passed `noise_freq`: rl_env did not
    plumb it and no eval task carried it. eval/tasks/analog/tia_001.yaml
    declares `noise: {max: 10, unit: pA/sqrt(Hz)}` and that spec could
    therefore NEVER be scored -- and an unmeasurable spec that the caller does
    not drop is a silent -1.0 on the task, which is exactly the failure the
    refusal was meant to avoid.

    So a density spec says where it is measured, on the spec itself:

        noise:
          max: 10
          unit: pA/sqrt(Hz)
          at_freq: 100e3        # Hz

    which is the right home for it, because the frequency is a property of the
    SPEC and not of the simulation. The first density spec that names one wins;
    an explicit `noise_freq=` argument still overrides.
    """
    for name, sdef in (specs or {}).items():
        metric = canonical_metric(name)
        if metric not in ("input_noise_density", "output_noise_density"):
            continue
        d = _as_dict(sdef) or {}
        raw = d.get("at_freq", d.get("noise_freq"))
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            log.warning("spec %r declares an unreadable at_freq %r; ignored",
                        name, raw)
            continue
        if value > 0.0:
            return value
    return None


# ----------------------------------------------------------------------------
# Result normalisation -- accept pydantic models or their model_dump() dicts
# ----------------------------------------------------------------------------

def _as_dict(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    raise TypeError(f"cannot normalise {type(obj).__name__} into a result dict")


def _signal_xy(sig: Any) -> tuple[list[float], list[float]]:
    d = _as_dict(sig) or {}
    return list(d.get("x_values") or []), list(d.get("y_values") or [])


def _is_monotone_axis(x: Sequence[float]) -> bool:
    """True when a sweep axis never doubles back on itself.

    False for the flattened NESTED '.dc' that ngspice writes as one vector with
    the inner sweep repeated per outer value.
    """
    signs = [1 if b > a else (-1 if b < a else 0)
             for a, b in zip(list(x), list(x)[1:])]
    signs = [v for v in signs if v]
    return all(a == b for a, b in zip(signs, signs[1:]))


def _pick_output(signals: Mapping[str, Any], preferred: Optional[str],
                 prefix: str = "", suffix: str = "") -> Optional[str]:
    """Choose the output signal key, preferring an explicit name.

    AC signals are keyed "vdb(<vector>)" / "vp(<vector>)", so callers pass
    prefix="vdb(" AND suffix=")". Omitting the suffix builds "vdb(out", which is
    never a key -- both the caller's explicit choice and the out/vout/output
    fallback then miss, and selection silently degrades to "whatever vector
    ngspice happened to list first". That is right by luck on a testbench whose
    output is the first vector, which is exactly why it went unnoticed.

    Falls back to a node literally called 'out', then to the first key that is
    not a branch current, so a bare testbench still works without configuration.
    """
    if not signals:
        return None

    def _key(name: str) -> str:
        return f"{prefix}{name}{suffix}" if (prefix or suffix) else name

    if preferred:
        key = _key(preferred)
        if key in signals:
            return key
        # The caller may already have passed a fully-formed key.
        if preferred in signals:
            return preferred
    for cand in ("out", "vout", "output"):
        key = _key(cand)
        if key in signals:
            return key
    for key in signals:
        if "#branch" not in key:
            return key
    return None


def _pick_input(signals: Mapping[str, Any],
                preferred: Optional[str] = None) -> Optional[str]:
    """Choose the INPUT/stimulus signal key, or None when there is not one.

    _pick_output() must never be used for this. It is an OUTPUT chooser: asked
    for "in" and given a result with no input at all it falls through to
    out / vout / output and then to the first non-branch vector, so a
    TranResult holding only `out` and `vout` produced a confident 40 ns
    "propagation delay" between two OUTPUTS, with nothing in `unmeasurable` to
    say so. A propagation delay with no input reference is not a small error,
    it is a different measurement.

    Only names that actually denote a stimulus are accepted, and when none is
    present the answer is None so the caller can refuse.
    """
    if not signals:
        return None
    for cand in (preferred, "in", "vin", "input", "v(in)", "in_p"):
        if cand and cand in signals:
            return cand
    return None


# ----------------------------------------------------------------------------
# Canonical metrics in SI
# ----------------------------------------------------------------------------

def _compute_metrics(dc: Any, ac: Any, tran: Any, noise: Any, stb: Any,
                     output_signal: Optional[str],
                     supply_sources: Optional[Sequence[str]],
                     netlist: Optional[str] = None,
                     unmeasurable: Optional[dict[str, str]] = None,
                     noise_freq: Optional[float] = None,
                     input_signal: Optional[str] = None
                     ) -> dict[str, float]:
    """Every canonical metric the supplied results support, in SI units.

    `unmeasurable`, when given, collects the CANONICAL metric names that the
    measurement layer explicitly refused, keyed to the reason it gave. Those
    reasons are far more useful than "not produced by the supplied results".
    """
    m: dict[str, float] = {}
    why: dict[str, str] = {} if unmeasurable is None else unmeasurable

    def put(name: str, value: Optional[float]) -> None:
        if value is None:
            return
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return
        m[name] = v

    # -- AC ------------------------------------------------------------------
    acd = _as_dict(ac)
    if acd:
        freqs = list(acd.get("frequencies") or [])
        signals = acd.get("signals") or {}
        key = _pick_output(signals, output_signal, prefix="vdb(", suffix=")")
        if key and key.startswith("vdb("):
            # The SIGNAL's own x axis, not the result's global frequency list.
            # A signal is defined only where its transfer function is defined,
            # so the two can differ in length, and pairing a short y with a
            # long x silently shifts every sample against its frequency.
            sig_f, gain_db = _signal_xy(signals[key])
            if len(sig_f) == len(gain_db) and sig_f:
                freqs = sig_f
            raw = key[4:-1]
            ph_sig = signals.get(f"vp({raw})")
            phase = _signal_xy(ph_sig)[1] if ph_sig else None
            if phase is not None and len(phase) != len(gain_db):
                phase = None
            am = measure.ac_metrics(freqs, gain_db, phase)
            for metric in ("dc_gain_db", "passband_gain_db", "ugb",
                           "bandwidth_3db", "phase_margin", "gain_margin"):
                put(metric, am.get(metric))
            # ac_metrics explains every None it returns. Carry the explanation
            # across rather than replacing it with a generic "not produced".
            notes = am.get("notes") or {}
            for metric, note in notes.items():
                if metric in ("dc_gain_db", "passband_gain_db",
                              "bandwidth_3db", "ugb", "phase_margin",
                              "gain_margin"):
                    why[metric] = note

    # -- stability overrides AC for margins, it is the purpose-built analysis -
    stbd = _as_dict(stb)
    if stbd:
        put("phase_margin", stbd.get("phase_margin"))
        put("gain_margin", stbd.get("gain_margin"))

    # -- DC ------------------------------------------------------------------
    dcd = _as_dict(dc)
    if dcd:
        op = dcd.get("op_points") or {}
        if op:
            idd = measure.supply_current_report(
                op, list(supply_sources) if supply_sources else None, netlist)
            put("idd", idd.value)
            if idd.value is None and idd.warnings:
                why["idd"] = "; ".join(idd.warnings)
            elif idd.warnings:
                log.warning("idd may not be the supply current: %s",
                            "; ".join(idd.warnings))
            node = output_signal or "out"
            for cand in (node, "out", "vout"):
                if cand in op:
                    put("vout", op[cand])
                    break
        sweeps = dcd.get("sweeps") or {}
        key = _pick_output(sweeps, output_signal)
        if key:
            xs, y = _signal_xy(sweeps[key])
            if y and not _is_monotone_axis(xs):
                # ngspice flattens a NESTED '.dc V1 ... V2 ...' into one vector
                # per signal, so the axis doubles back once per outer step:
                # [0, 0.5, 1, 0, 0.5, 1, ...]. max-minus-min across that is the
                # excursion over the whole 2-D GRID -- it includes whatever the
                # outer variable did -- and it is not the output swing of a
                # sweep. Nothing flagged it, and the number looked ordinary.
                why["output_swing"] = (
                    f"refused: the sweep axis of {key!r} is not monotonic "
                    f"({len(xs)} points that double back on themselves), which "
                    f"is how ngspice writes a NESTED '.dc' -- two swept sources "
                    f"flattened into one vector. max-minus-min across that grid "
                    f"is the excursion over BOTH sweeps, not the output swing "
                    f"of one. Run a single '.dc' per outer value."
                )
            elif y:
                put("output_swing", max(y) - min(y))
                if len(xs) == len(y) and len(xs) >= 3:
                    slopes = [abs((y[i + 1] - y[i]) / (xs[i + 1] - xs[i]))
                              for i in range(len(xs) - 1) if xs[i + 1] != xs[i]]
                    if slopes:
                        put("dc_max_slope", max(slopes))
                    # A voltage temperature coefficient only exists when the
                    # sweep variable IS temperature, and only the deck knows
                    # that. A least-squares slope, not endpoint difference:
                    # a bandgap's curvature would make the endpoints lie about
                    # the local behaviour around nominal.
                    if netlist and re.search(r"^\s*\.dc\s+temp\b", netlist,
                                             re.IGNORECASE | re.MULTILINE):
                        n = len(xs)
                        mx = sum(xs) / n
                        my_ = sum(y) / n
                        den = sum((a - mx) ** 2 for a in xs)
                        if den > 0:
                            put("output_tc",
                                sum((a - mx) * (b - my_)
                                    for a, b in zip(xs, y)) / den)

        # Output CURRENT of a swept dc run lives in a '#branch' vector. Which
        # one is the output is only unambiguous when the caller names it or
        # exactly one non-supply branch exists; guessing between two branch
        # currents is how a supply current becomes an "output" current.
        branch_named = (output_signal if output_signal
                        and "#branch" in output_signal.lower() else None)
        branches = {k: v for k, v in sweeps.items()
                    if "#branch" in k.lower()
                    and k.lower() not in ("vdd#branch", "vss#branch")}
        pick = None
        if branch_named and branch_named in sweeps:
            pick = branch_named
        elif len(branches) == 1:
            pick = next(iter(branches))
        elif len(branches) > 1:
            why["iout_max"] = why["iout_swing"] = (
                f"ambiguous: {sorted(branches)} are all branch currents; "
                f"name the output via `output_signal`")
        if pick:
            bxs, by = _signal_xy(sweeps[pick])
            if by and len(bxs) == len(by) and _is_monotone_axis(bxs):
                put("iout_max", max(abs(v) for v in by))
                put("iout_swing", max(by) - min(by))
            elif by:
                why["iout_max"] = why["iout_swing"] = (
                    f"refused: the sweep axis of {pick!r} is not monotonic "
                    f"(a nested '.dc' grid); an extremum across both sweeps "
                    f"is not this output's compliance behaviour")

    # -- transient -----------------------------------------------------------
    trd = _as_dict(tran)
    if trd:
        t = list(trd.get("time") or [])
        signals = trd.get("signals") or {}
        key = _pick_output(signals, output_signal)
        if key and t:
            sig_t, y = _signal_xy(signals[key])
            if len(sig_t) == len(y) and sig_t:
                t = sig_t
            # The STIMULUS is picked before the metrics are taken, not after:
            # it is what tells a settled response from one the drive abandoned
            # (measure.drive_truncation_note). _pick_input, NOT _pick_output --
            # the latter falls back to another OUTPUT.
            in_key = _pick_input(signals, input_signal)
            t_in: list[float] = []
            y_in: list[float] = []
            if in_key is not None and in_key != key:
                t_in, y_in = _signal_xy(signals[in_key])
            same_axis = bool(y_in) and len(y_in) == len(t) and t_in == t
            tm = measure.tran_metrics(t, y, y_in if same_axis else None)
            # tran_metrics explains every None it returns -- a record that
            # never settled refuses the whole 10/50/90 ladder. Carry the
            # explanation across instead of "not produced by the supplied
            # results", which reads as a missing analysis.
            for metric, note in (tm.get("notes") or {}).items():
                if (metric in ("rise_time", "fall_time", "settling_time",
                               "slew_rate", "overshoot_pct", "prop_delay")
                        and tm.get(metric) is None):
                    why[metric] = note
            put("rise_time", tm.get("rise_time"))
            put("fall_time", tm.get("fall_time"))
            put("settling_time", tm.get("settling_time"))
            # WHICH EDGE is measured belongs to the testbench, not to the
            # design. Taking abs() of the FIRST edge removed the sign
            # dependence and left the edge dependence untouched: the same deck,
            # the same source, only the `.tran` start moved, scored 0.045 V/us
            # opening on the slow edge and 0.450 V/us opening on the fast one.
            # That turned a deterministic false-FAIL into a stimulus-dependent
            # false-PASS, which is worse. A slew-rate spec is a `min:`, so the
            # number it is scored on is the SLOWEST complete edge in the
            # record -- the same number whichever edge the record opens on.
            # The signed first-edge value stays in tran_metrics, with a note
            # naming the edge and counting the others.
            put("slew_rate", tm.get("slew_rate_worst"))
            put("overshoot_pct", tm.get("overshoot_pct"))
            # A propagation delay needs an input reference AND an output that
            # is not that same reference. The two failures are different and
            # they used to share one message: a result holding ONLY a stimulus
            # (so _pick_output falls back onto it and in_key == key) was told
            # it "carries no input signal", which is the exact opposite of what
            # is wrong with it, and it sent the reader off to name an
            # `input_signal` that is already there.
            if in_key is None:
                why["prop_delay"] = (
                    f"a propagation delay is measured from an INPUT to an "
                    f"output, and this result carries no input signal "
                    f"(signals: {sorted(signals)}; output {key!r}). Name the "
                    f"stimulus vector via `input_signal`, or add it to the "
                    f"deck's save list. A 50 pct to 50 pct time between two "
                    f"OUTPUTS is not a propagation delay."
                )
            elif in_key == key:
                why["prop_delay"] = (
                    f"a propagation delay is measured from an INPUT to an "
                    f"output, and here they are the SAME vector: {key!r} was "
                    f"chosen as both the stimulus and the output (signals: "
                    f"{sorted(signals)}). The delay of a waveform against "
                    f"itself is zero and means nothing. Save the stage's "
                    f"output in the deck, or name it via `output_signal`."
                )
            elif same_axis:
                put("prop_delay", measure.prop_delay(t, y_in, y))
            else:
                why["prop_delay"] = (
                    f"the input signal {in_key!r} is sampled on a "
                    f"different time axis ({len(t_in)} points) from the "
                    f"output ({len(t)} points), so a 50 pct to 50 pct "
                    f"delay between them is not defined"
                )

    # -- transient extras: the last sample, the record's excursion, and the
    # oscillation frequency -- each refused rather than guessed when the
    # record does not support it.
    if trd:
        t = list(trd.get("time") or [])
        signals = trd.get("signals") or {}
        key = _pick_output(signals, output_signal)
        if key and t:
            sig_t, y = _signal_xy(signals[key])
            if len(sig_t) == len(y) and sig_t:
                t = sig_t
            if y:
                put("vout_final", y[-1])
                if "output_swing" not in m:
                    put("output_swing", max(y) - min(y))
            if len(y) >= 8 and len(t) == len(y):
                mid = (max(y) + min(y)) / 2.0
                cross = [t[i] for i in range(1, len(y))
                         if (y[i - 1] - mid) * (y[i] - mid) < 0]
                if len(cross) >= 4:
                    half_periods = [b - a for a, b in zip(cross, cross[1:])]
                    mean_hp = sum(half_periods) / len(half_periods)
                    # A frequency is only a frequency if the period repeats:
                    # a chirp or a settling wiggle also crosses the midline.
                    if mean_hp > 0 and all(
                            abs(hp - mean_hp) <= 0.35 * mean_hp
                            for hp in half_periods):
                        put("osc_freq", 1.0 / (2.0 * mean_hp))
                    else:
                        why["osc_freq"] = (
                            f"refused: {len(cross)} midline crossings with "
                            f"irregular spacing (half-periods vary more than "
                            f"35 pct); this record does not repeat, so it has "
                            f"no single oscillation frequency")
                elif cross:
                    why["osc_freq"] = (
                        f"only {len(cross)} midline crossing(s) in the "
                        f"record; at least 4 are needed to call it periodic")

    # -- noise ---------------------------------------------------------------
    nsd = _as_dict(noise)
    if nsd:
        nfreqs = list(nsd.get("frequencies") or [])
        for field_name, dens, rms in (
            ("input_noise", "input_noise_density", "input_noise_rms"),
            ("output_noise", "output_noise_density", None),
        ):
            sig = nsd.get(field_name)
            if not sig:
                continue
            xs, y = _signal_xy(sig)
            if len(xs) != len(y) or not xs:
                xs = nfreqs
            if not y:
                continue
            value, reason = _noise_density(xs, y, noise_freq)
            if value is None:
                why[dens] = reason
            else:
                put(dens, value)
            if rms:
                put(rms, measure.integrate_noise(xs, y))

    return m


def _noise_density(freqs: Sequence[float], spectrum: Sequence[float],
                   noise_freq: Optional[float]
                   ) -> tuple[Optional[float], str]:
    """A single noise-density number, or None and the reason there is not one.

    A noise density is a value AT A FREQUENCY. Taking spectrum[0] makes the
    answer a function of where the sweep starts, which is the same defect that
    made dc_gain_db depend on f_start: a .noise from 1 Hz reports the 1/f
    corner and one from 10 kHz reports the thermal floor, for the same circuit.

    So: with `noise_freq`, the density is interpolated there. Without it, the
    density at the bottom of the sweep is reported ONLY when the spectrum is
    demonstrably FLAT there -- the same low-frequency slope test ac_metrics
    uses, in dB/decade of density, where a 1/f region slopes -10 dB/dec and a
    thermal floor slopes 0.
    """
    n = min(len(freqs), len(spectrum))
    if n == 0:
        return None, "the spectrum is empty"
    if noise_freq is not None:
        v = measure.value_at_freq(freqs, spectrum, float(noise_freq))
        if v is None:
            return None, (
                f"the requested noise frequency {noise_freq:g} Hz is outside "
                f"the swept band {freqs[0]:g} .. {freqs[n - 1]:g} Hz"
            )
        return v, ""
    db = [measure.db20(v) for v in spectrum[:n]]
    slope, span = measure.local_slope(list(freqs[:n]), db, 0)
    if slope is None:
        return None, (
            "cannot tell whether the spectrum is flat at the bottom of the "
            "sweep: no usable low-frequency span"
        )
    if abs(slope) > measure.DC_SLOPE_TOL_DB_PER_DEC:
        return None, (
            f"refused: the spectrum still slopes {slope:.4g} dB/decade at "
            f"f_start = {freqs[0]:g} Hz (tolerance "
            f"{measure.DC_SLOPE_TOL_DB_PER_DEC:g} dB/dec over {span:.3g} "
            f"decades), so the {spectrum[0]:.4g} measured there is a point on "
            f"the 1/f corner, not a band noise density -- and which point it "
            f"is depends only on where the sweep was started. Pass "
            f"`noise_freq` to ask for the density at a stated frequency, or "
            f"use the integrated RMS over the band."
        )
    return float(spectrum[0]), ""


# ----------------------------------------------------------------------------
# Public result
# ----------------------------------------------------------------------------

@dataclass
class SpecExtraction:
    """What could be measured, what could not, and why."""

    values: dict[str, float] = field(default_factory=dict)
    """spec_name -> value, in the unit the task declared. Feed to RewardFunction."""

    unmeasurable: dict[str, str] = field(default_factory=dict)
    """spec_name -> reason. NEVER silently drop these; a missing spec scores -1.0."""

    metrics_si: dict[str, float] = field(default_factory=dict)
    """canonical metric -> SI value, for diagnostics and logging."""

    @property
    def coverage(self) -> float:
        """Fraction of the task's specs that were actually measured."""
        total = len(self.values) + len(self.unmeasurable)
        return len(self.values) / total if total else 0.0

    def measurable_specs(self, specs: Mapping[str, Any]) -> dict[str, Any]:
        """The task's spec definitions restricted to what was measured.

        Use this to build the RewardFunction so unmeasurable specs cannot pin
        the score to -1.0 for reasons that have nothing to do with the design.
        """
        return {k: v for k, v in specs.items() if k in self.values}


def extract_specs(specs: Mapping[str, Any],
                  *,
                  dc: Any = None,
                  ac: Any = None,
                  tran: Any = None,
                  noise: Any = None,
                  stb: Any = None,
                  output_signal: Optional[str] = None,
                  input_signal: Optional[str] = None,
                  supply_sources: Optional[Iterable[str]] = None,
                  netlist: Optional[str] = None,
                  noise_freq: Optional[float] = None) -> SpecExtraction:
    """Measure a task's specs from simulation results.

    Args:
        specs: the eval task's `specs` mapping, spec_name -> {min/max/target/unit}.
        dc, ac, tran, noise, stb: results from the adapter. Pydantic models or
            their model_dump() dicts; any subset may be omitted.
        output_signal: raw vector name of the circuit output (e.g. "out"). When
            omitted, a node called 'out' is preferred, else the first
            non-branch-current signal.
        input_signal: raw vector name of the STIMULUS, used for the propagation
            delay. When omitted, only names that genuinely denote an input
            ('in', 'vin', 'input', ...) are accepted; there is deliberately no
            fallback to another output, because a 50 pct to 50 pct time between
            two outputs is not a propagation delay.
        supply_sources: voltage source names to sum for idd. Defaults to every
            independent-voltage-source branch, which is a guess -- see
            measure.supply_current_report.
        netlist: the deck that produced these results. Pass it whenever you
            have it: it is the only way to tell a 0 V sense source from a
            supply rail, and the only way to notice that the block is biased
            by a current source whose branch current ngspice never reports.
        noise_freq: the frequency, in Hz, at which a noise DENSITY spec is
            declared. A density is a value at a frequency; without this the
            density is reported only when the spectrum is demonstrably flat at
            the bottom of the sweep, and refused otherwise, because
            spectrum[0] on a 1/f corner is a function of where the sweep was
            started and not of the circuit. When omitted it is read off the
            spec itself (`at_freq`); see spec_noise_freq.

    Returns:
        SpecExtraction. `values` holds only specs that were genuinely measured.
    """
    refusals: dict[str, str] = {}
    if noise_freq is None:
        noise_freq = spec_noise_freq(specs)
    si = _compute_metrics(dc, ac, tran, noise, stb, output_signal,
                          list(supply_sources) if supply_sources else None,
                          netlist, refusals, noise_freq, input_signal)

    out = SpecExtraction(metrics_si=si)

    for name, sdef in specs.items():
        d = _as_dict(sdef) or {}
        unit = str(d.get("unit") or "").strip()

        if unit.lower() in _NON_ANALOG_UNITS:
            out.unmeasurable[name] = (
                f"functional/digital spec (unit {unit!r}); needs a testbench or "
                f"formal check, not derivable from a SPICE analysis"
            )
            continue

        metric = canonical_metric(name)
        if metric is None:
            out.unmeasurable[name] = (
                f"no measurement is defined for spec name {name!r}; add it to "
                f"spec_extract.ALIASES once the measurement exists"
            )
            continue

        if metric not in si:
            refused = refusals.get(metric)
            out.unmeasurable[name] = (
                f"metric {metric!r} was refused by the measurement: {refused}"
                if refused else
                f"metric {metric!r} was not produced by the supplied results "
                f"(need the matching analysis, and a resolvable output signal)"
            )
            continue

        dimension = DIMENSIONS.get(metric)
        if dimension is None:
            out.unmeasurable[name] = f"metric {metric!r} has no declared dimension"
            continue

        if dimension in ("noise_density", "noise_rms"):
            # WHICH END of the .noise card decides the family depends on which
            # end the metric is referred to. The INPUT-referred density and the
            # integrated input-referred RMS follow the SOURCE named on the card;
            # the OUTPUT-referred density follows the card's OUTPUT EXPRESSION,
            # which for ngspice is always a voltage. Running both through
            # noise_input_kind() divided a V/sqrt(Hz) output measurement by 1e-12
            # on any current-driven deck.
            output_referred = metric.startswith("output_")
            kind = (noise_output_kind(netlist) if output_referred
                    else noise_input_kind(netlist))
            if kind is None:
                end = "output expression" if output_referred else "input source"
                out.unmeasurable[name] = (
                    f"the family of {metric!r} comes from the {end} of the "
                    f"'.noise' card -- a VOLTAGE quantity for 'v(out)' or a "
                    f"voltage source, a CURRENT quantity for a current source "
                    f"-- and the result carries no units to say which. Without "
                    f"the netlist a {unit!r} spec cannot be scaled, and getting "
                    f"it wrong is a factor of 1e12, not a rounding error. Pass "
                    f"`netlist`."
                )
                continue
            dimension = (_NOISE_RMS_DIMENSION[kind] if dimension == "noise_rms"
                         else f"noise_density_{kind}")

        try:
            out.values[name] = convert_from_si(si[metric], dimension, unit)
        except UnitError as exc:
            out.unmeasurable[name] = f"unit conversion failed: {exc}"

    return out
