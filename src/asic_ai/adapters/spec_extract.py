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

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from asic_ai.adapters import measure

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
    "gain": "dc_gain_db",
    "dc_gain": "dc_gain_db",
    "gain_db": "dc_gain_db",
    "conversion_gain": "dc_gain_db",
    "gain_max": "dc_gain_db",
    "gain_min": "dc_gain_db",
    "linearity": "dc_gain_db",

    # -- bandwidth / frequency ----------------------------------------------
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
    "ugb": "frequency",
    "bandwidth_3db": "frequency",
    "phase_margin": "angle",
    "gain_margin": "gain_db_only",
    "idd": "current",
    "vout": "voltage",
    "output_swing": "voltage",
    "slew_rate": "slew",
    "settling_time": "time",
    "prop_delay": "time",
    "rise_time": "time",
    "fall_time": "time",
    "overshoot_pct": "percent",
    "input_noise_density": "noise_density",
    "output_noise_density": "noise_density",
    "input_noise_rms": "voltage",
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
    "noise_density": {"v/sqrt(hz)": 1.0, "nv/sqrt(hz)": 1e-9,
                      "uv/sqrt(hz)": 1e-6, "pa/sqrt(hz)": 1e-12,
                      "a/sqrt(hz)": 1.0},
}

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


def _pick_output(signals: Mapping[str, Any], preferred: Optional[str],
                 prefix: str = "") -> Optional[str]:
    """Choose the output signal key, preferring an explicit name.

    Falls back to a node literally called 'out', then to the first key that is
    not a branch current, so a bare testbench still works without configuration.
    """
    if not signals:
        return None
    if preferred:
        key = f"{prefix}{preferred}" if prefix else preferred
        if key in signals:
            return key
    for cand in ("out", "vout", "output"):
        key = f"{prefix}{cand}" if prefix else cand
        if key in signals:
            return key
    for key in signals:
        if "#branch" not in key:
            return key
    return None


# ----------------------------------------------------------------------------
# Canonical metrics in SI
# ----------------------------------------------------------------------------

def _compute_metrics(dc: Any, ac: Any, tran: Any, noise: Any, stb: Any,
                     output_signal: Optional[str],
                     supply_sources: Optional[Sequence[str]]) -> dict[str, float]:
    """Every canonical metric the supplied results support, in SI units."""
    m: dict[str, float] = {}

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
        key = _pick_output(signals, output_signal, prefix="vdb(")
        if key and key.startswith("vdb("):
            _, gain_db = _signal_xy(signals[key])
            raw = key[4:-1]
            ph_sig = signals.get(f"vp({raw})")
            phase = _signal_xy(ph_sig)[1] if ph_sig else None
            am = measure.ac_metrics(freqs, gain_db, phase)
            put("dc_gain_db", am.get("dc_gain_db"))
            put("ugb", am.get("ugb"))
            put("bandwidth_3db", am.get("bandwidth_3db"))
            put("phase_margin", am.get("phase_margin"))
            put("gain_margin", am.get("gain_margin"))

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
            put("idd", measure.supply_current(
                op, list(supply_sources) if supply_sources else None))
            node = output_signal or "out"
            for cand in (node, "out", "vout"):
                if cand in op:
                    put("vout", op[cand])
                    break
        sweeps = dcd.get("sweeps") or {}
        key = _pick_output(sweeps, output_signal)
        if key:
            _, y = _signal_xy(sweeps[key])
            if y:
                put("output_swing", max(y) - min(y))

    # -- transient -----------------------------------------------------------
    trd = _as_dict(tran)
    if trd:
        t = list(trd.get("time") or [])
        signals = trd.get("signals") or {}
        key = _pick_output(signals, output_signal)
        if key and t:
            _, y = _signal_xy(signals[key])
            tm = measure.tran_metrics(t, y)
            put("rise_time", tm.get("rise_time"))
            put("fall_time", tm.get("fall_time"))
            put("settling_time", tm.get("settling_time"))
            put("slew_rate", tm.get("slew_rate"))
            put("overshoot_pct", tm.get("overshoot_pct"))
            # Propagation delay needs an input reference; use it when present.
            in_key = _pick_output(signals, "in")
            if in_key and in_key != key:
                _, y_in = _signal_xy(signals[in_key])
                put("prop_delay", measure.prop_delay(t, y_in, y))

    # -- noise ---------------------------------------------------------------
    nsd = _as_dict(noise)
    if nsd:
        freqs = list(nsd.get("frequencies") or [])
        for field_name, dens, rms in (
            ("input_noise", "input_noise_density", "input_noise_rms"),
            ("output_noise", "output_noise_density", None),
        ):
            sig = nsd.get(field_name)
            if not sig:
                continue
            _, y = _signal_xy(sig)
            if y:
                put(dens, y[0])
                if rms:
                    put(rms, measure.integrate_noise(freqs, y))

    return m


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
                  supply_sources: Optional[Iterable[str]] = None) -> SpecExtraction:
    """Measure a task's specs from simulation results.

    Args:
        specs: the eval task's `specs` mapping, spec_name -> {min/max/target/unit}.
        dc, ac, tran, noise, stb: results from the adapter. Pydantic models or
            their model_dump() dicts; any subset may be omitted.
        output_signal: raw vector name of the circuit output (e.g. "out"). When
            omitted, a node called 'out' is preferred, else the first
            non-branch-current signal.
        supply_sources: voltage source names to sum for idd. Defaults to every
            'v*#branch' entry.

    Returns:
        SpecExtraction. `values` holds only specs that were genuinely measured.
    """
    si = _compute_metrics(dc, ac, tran, noise, stb, output_signal,
                          list(supply_sources) if supply_sources else None)

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
            out.unmeasurable[name] = (
                f"metric {metric!r} was not produced by the supplied results "
                f"(need the matching analysis, and a resolvable output signal)"
            )
            continue

        dimension = DIMENSIONS.get(metric)
        if dimension is None:
            out.unmeasurable[name] = f"metric {metric!r} has no declared dimension"
            continue

        try:
            out.values[name] = convert_from_si(si[metric], dimension, unit)
        except UnitError as exc:
            out.unmeasurable[name] = f"unit conversion failed: {exc}"

    return out
