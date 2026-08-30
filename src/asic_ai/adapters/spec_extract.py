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
    # when the sweep reaches DC and the peak gain otherwise. Before this split
    # an AC-coupled stage with 40 dB of mid-band gain reported -24 dB.
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

# The .noise card: '.noise v(out) V1 dec 100 1 1G'. The second token is the
# input source, and its first letter is its type -- a SPICE language rule.
_NOISE_CARD_RE = re.compile(
    r"^\s*\.noise\s+\S+\s+([a-zA-Z]\S*)", re.IGNORECASE | re.MULTILINE)


def noise_input_kind(netlist: Optional[str]) -> Optional[str]:
    """'v', 'i', or None: the type of the source a .noise run is referred to.

    This is what makes an input-referred noise density a voltage density or a
    current density. It is not knowable from a NoiseResult, which carries
    numbers and no units.
    """
    if not netlist:
        return None
    m = _NOISE_CARD_RE.search(netlist)
    if not m:
        return None
    letter = m.group(1)[0].lower()
    return letter if letter in ("v", "i") else None

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


# ----------------------------------------------------------------------------
# Canonical metrics in SI
# ----------------------------------------------------------------------------

def _compute_metrics(dc: Any, ac: Any, tran: Any, noise: Any, stb: Any,
                     output_signal: Optional[str],
                     supply_sources: Optional[Sequence[str]],
                     netlist: Optional[str] = None,
                     unmeasurable: Optional[dict[str, str]] = None,
                     noise_freq: Optional[float] = None
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
                if metric in ("dc_gain_db", "bandwidth_3db", "ugb",
                              "phase_margin", "gain_margin"):
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
            sig_t, y = _signal_xy(signals[key])
            if len(sig_t) == len(y) and sig_t:
                t = sig_t
            tm = measure.tran_metrics(t, y)
            put("rise_time", tm.get("rise_time"))
            put("fall_time", tm.get("fall_time"))
            put("settling_time", tm.get("settling_time"))
            put("slew_rate", tm.get("slew_rate"))
            put("overshoot_pct", tm.get("overshoot_pct"))
            # Propagation delay needs an input reference; use it when present.
            in_key = _pick_output(signals, "in")
            if in_key and in_key != key:
                t_in, y_in = _signal_xy(signals[in_key])
                # Both waveforms must be on the SAME time axis for a 50-50
                # delay to mean anything.
                if len(y_in) == len(t) and t_in == t:
                    put("prop_delay", measure.prop_delay(t, y_in, y))
                else:
                    why["prop_delay"] = (
                        f"the input signal {in_key!r} is sampled on a "
                        f"different time axis ({len(t_in)} points) from the "
                        f"output ({len(t)} points), so a 50 pct to 50 pct "
                        f"delay between them is not defined"
                    )

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
            started and not of the circuit.

    Returns:
        SpecExtraction. `values` holds only specs that were genuinely measured.
    """
    refusals: dict[str, str] = {}
    si = _compute_metrics(dc, ac, tran, noise, stb, output_signal,
                          list(supply_sources) if supply_sources else None,
                          netlist, refusals, noise_freq)

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

        if dimension == "noise_density":
            kind = noise_input_kind(netlist)
            if kind is None:
                out.unmeasurable[name] = (
                    "an input-referred noise density is a VOLTAGE density when "
                    "the .noise card names a voltage source and a CURRENT "
                    "density when it names a current source, and the result "
                    "carries no units to say which. Without the netlist a "
                    f"{unit!r} spec cannot be scaled -- getting it wrong is a "
                    "factor of 1e12, not a rounding error. Pass `netlist`."
                )
                continue
            dimension = f"noise_density_{kind}"

        try:
            out.values[name] = convert_from_si(si[metric], dimension, unit)
        except UnitError as exc:
            out.unmeasurable[name] = f"unit conversion failed: {exc}"

    return out
