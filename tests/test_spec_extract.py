"""Tests for the results -> spec-name-keyed scalar converter.

The bug this module exists to prevent: RewardFunction looks up EVAL TASK SPEC
names ("dc_gain", "pm", "idd") in a dict that was keyed by SCHEMA FIELD names
("op_points", "sweeps", "frequencies"). Every lookup returned None and every
spec scored SCORE_CLIP_MIN = -1.0, regardless of the design.

The regression tests at the bottom fail if that wiring breaks again.
"""
from __future__ import annotations

import json
import math
import tempfile

import pytest

from asic_ai.adapters import spec_extract as sx
from asic_ai.adapters.spec_extract import UnitError
from asic_ai.reward.reward import RewardFunction, SpecTarget
from asic_ai.tool_interface.schema import (
    ACResult, DCResult, SignalData, SimParams, TranResult,
)

try:
    from asic_ai.adapters.ngspice_shared import find_ngspice_dll
    HAS_NGSPICE = find_ngspice_dll() is not None
except Exception:
    HAS_NGSPICE = False

skip_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE, reason="ngspice DLL not found")


# ---------------------------------------------------------------- aliases ---

@pytest.mark.parametrize("name,expected", [
    # A bare "gain" spec means the amplifier's gain, which for an AC-coupled
    # or band-pass stage is the MID-BAND gain. Only "dc_gain" asks for the
    # gain at DC, and measure.py refuses that unless the sweep reaches DC.
    ("gain", "passband_gain_db"),
    ("gain_db", "passband_gain_db"),
    ("dc_gain", "dc_gain_db"),
    ("DC_Gain", "dc_gain_db"),        # case insensitive
    ("  pm  ", "phase_margin"),       # whitespace tolerant
    ("phase_margin", "phase_margin"),
    ("ugb", "ugb"),
    ("gbw", "ugb"),
    ("bw", "bandwidth_3db"),
    ("bandwidth", "bandwidth_3db"),
    ("idd", "idd"),
    ("iq", "idd"),
    ("vout", "vout"),
    ("v_out", "vout"),
    ("settling", "settling_time"),
])
def test_alias_normalisation(name, expected):
    assert sx.canonical_metric(name) == expected


def test_unknown_spec_name_is_not_guessed():
    """Guessing at an unknown spec is how fabricated rewards start."""
    assert sx.canonical_metric("kvco") is None
    assert sx.canonical_metric("phase_noise") is None


def test_every_alias_target_has_a_dimension():
    missing = {m for m in sx.ALIASES.values() if m not in sx.DIMENSIONS}
    assert not missing, f"metrics with no declared dimension: {missing}"


# ------------------------------------------------------------ conversions ---

@pytest.mark.parametrize("si,dim,unit,expected", [
    (30e6, "frequency", "MHz", 30.0),
    (2.4e9, "frequency", "GHz", 2.4),
    (1e5, "frequency", "kHz", 100.0),
    (159154.9431, "frequency", "Hz", 159154.9431),
    (87.3e-6, "current", "uA", 87.3),
    (1.5e-3, "current", "mA", 1.5),
    (0.9, "voltage", "V", 0.9),
    (5e-5, "voltage", "mV", 0.05),
    (5e-12, "time", "ps", 5.0),
    (1e-8, "time", "ns", 10.0),
    (1e8, "slew", "V/us", 100.0),
    (62.0, "angle", "deg", 62.0),
    (62.0, "angle", "degrees", 62.0),
])
def test_unit_conversion(si, dim, unit, expected):
    assert sx.convert_from_si(si, dim, unit) == pytest.approx(expected, rel=1e-12)


def test_gain_db_passthrough_and_linear_conversion():
    assert sx.convert_from_si(60.0, "gain", "dB") == pytest.approx(60.0)
    assert sx.convert_from_si(60.0, "gain", "V/V") == pytest.approx(1000.0)
    assert sx.convert_from_si(20.0, "gain", "V/V") == pytest.approx(10.0)


def test_wrong_unit_for_dimension_raises():
    with pytest.raises(UnitError):
        sx.convert_from_si(1.0, "frequency", "uA")
    with pytest.raises(UnitError):
        sx.convert_from_si(1.0, "gain", "MHz")


def test_same_spec_different_units_across_tasks():
    """idd is uA in some eval tasks and mA in others; both must be right."""
    specs_ua = {"idd": {"max": 100, "unit": "uA"}}
    specs_ma = {"idd": {"max": 1, "unit": "mA"}}
    dc = DCResult(op_points={"v1#branch": -9e-05}, sweeps={})
    assert sx.extract_specs(specs_ua, dc=dc).values["idd"] == pytest.approx(90.0)
    assert sx.extract_specs(specs_ma, dc=dc).values["idd"] == pytest.approx(0.09)


# ---------------------------------------------------------- unmeasurable ----

def test_digital_boolean_spec_is_reported_not_dropped():
    """A silently dropped spec scores -1.0, indistinguishable from a failure."""
    ext = sx.extract_specs({"correct": {"target": 1, "unit": "bool"}})
    assert "correct" not in ext.values
    assert "correct" in ext.unmeasurable
    assert "testbench" in ext.unmeasurable["correct"]


@pytest.mark.parametrize("unit", ["bool", "bits", "LSB", "cycles", "years"])
def test_non_analog_units_are_reported(unit):
    ext = sx.extract_specs({"x": {"max": 1, "unit": unit}})
    assert "x" in ext.unmeasurable


def test_unknown_spec_name_is_reported():
    ext = sx.extract_specs({"kvco": {"min": 200, "unit": "MHz/V"}})
    assert "kvco" in ext.unmeasurable
    assert "ALIASES" in ext.unmeasurable["kvco"]


def test_missing_analysis_is_reported():
    """Asking for gain without an AC result must say so, not return a number."""
    ext = sx.extract_specs({"gain": {"min": 50, "unit": "dB"}})
    assert "gain" not in ext.values
    assert "gain" in ext.unmeasurable


def test_coverage_and_measurable_specs():
    specs = {
        "idd": {"max": 100, "unit": "uA"},
        "correct": {"target": 1, "unit": "bool"},
    }
    dc = DCResult(op_points={"v1#branch": -9e-05}, sweeps={})
    ext = sx.extract_specs(specs, dc=dc)
    assert ext.coverage == pytest.approx(0.5)
    assert set(ext.measurable_specs(specs)) == {"idd"}


# ----------------------------------------------------------------- input ----

def test_accepts_model_dump_dicts():
    """rl_env holds model_dump() dicts, not pydantic objects."""
    dc = DCResult(op_points={"v1#branch": -9e-05}, sweeps={})
    specs = {"idd": {"max": 100, "unit": "uA"}}
    from_obj = sx.extract_specs(specs, dc=dc).values
    from_dict = sx.extract_specs(specs, dc=dc.model_dump()).values
    assert from_obj == from_dict


def test_nan_and_inf_never_reach_the_reward():
    dc = DCResult(op_points={"v1#branch": float("nan"), "v2#branch": float("inf")},
                  sweeps={})
    ext = sx.extract_specs({"idd": {"max": 100, "unit": "uA"}}, dc=dc)
    for v in ext.values.values():
        assert math.isfinite(v)


def test_synthetic_ac_gives_exact_metrics():
    """Hand-built flat-then-rolling response with a known 0 dB crossing."""
    freqs = [10.0 ** (i / 10.0) for i in range(0, 81)]  # 1 Hz .. 100 MHz
    fp = 1.0e3
    gain_db = [40.0 - 10.0 * math.log10(1.0 + (f / fp) ** 2) for f in freqs]
    phase = [-math.degrees(math.atan(f / fp)) for f in freqs]
    ac = ACResult(
        frequencies=freqs,
        signals={
            "vdb(out)": SignalData(name="vdb(out)", x_values=freqs, y_values=gain_db),
            "vp(out)": SignalData(name="vp(out)", x_values=freqs, y_values=phase),
        },
    )
    specs = {"gain": {"min": 30, "unit": "dB"},
             "bw": {"min": 0.5, "unit": "kHz"},
             "ugb": {"min": 50, "unit": "kHz"}}
    ext = sx.extract_specs(specs, ac=ac, output_signal="out")
    assert ext.values["gain"] == pytest.approx(40.0, abs=0.01)
    assert ext.values["bw"] == pytest.approx(1.0, rel=0.02)      # 1 kHz pole
    assert ext.values["ugb"] == pytest.approx(100.0, rel=0.02)   # 40 dB * 1 kHz


# ------------------------------------------------------------- regression ---

def test_schema_field_names_are_not_mistaken_for_spec_values():
    """The original bug: schema field names leaking through as if measured."""
    dc = DCResult(op_points={"v1#branch": -9e-05}, sweeps={})
    ext = sx.extract_specs({"idd": {"max": 100, "unit": "uA"}}, dc=dc)
    for forbidden in ("op_points", "sweeps", "frequencies", "signals", "time"):
        assert forbidden not in ext.values


def test_reward_scores_positively_on_a_passing_design():
    """End to end: a design that meets its specs must NOT score -1.0.

    Before the converter existed, RewardFunction saw an empty results dict and
    clipped every spec to -1.0 no matter how good the circuit was.
    """
    dc = DCResult(op_points={"v1#branch": -9e-05, "out": 0.9}, sweeps={})
    specs = {"idd": {"max": 200, "unit": "uA"}}
    ext = sx.extract_specs(specs, dc=dc)

    rf = RewardFunction(specs=[SpecTarget(name="idd", max_val=200.0, unit="uA")])
    good = rf.compute(results=ext.values)
    assert good.spec_scores[0].met is True
    assert good.total_reward > 0.0

    # And the same reward on the un-converted dict is the -1.0 failure mode.
    bad = rf.compute(results=dc.model_dump())
    assert bad.spec_scores[0].met is False
    assert bad.total_reward == pytest.approx(-1.0)


@skip_no_ngspice
def test_live_ngspice_end_to_end():
    """Real simulation -> real spec values, checked against hand calculation."""
    from asic_ai.adapters import get_adapter

    ad = get_adapter("ngspice_shared", binary_path="", work_dir=tempfile.mkdtemp())
    ac = ad.ac(
        "* vcvs amp, one pole\n"
        "V1 in 0 AC 1 DC 0\n"
        "E1 amp 0 in 0 1000\n"
        "R1 amp out 1k\n"
        "C1 out 0 1n\n"
        ".ac dec 50 1 1G\n"
        ".end\n",
        SimParams(analysis_type="ac"),
    )
    dc = ad.dc(
        "* divider\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\nR2 out 0 10k\n.op\n.end\n",
        SimParams(analysis_type="dc"),
    )
    specs = {
        "gain": {"min": 50, "unit": "dB"},
        "ugb": {"min": 50, "unit": "MHz"},
        "pm": {"min": 60, "unit": "deg"},
        "idd": {"max": 100, "unit": "uA"},
        "vout": {"target": 0.9, "unit": "V"},
    }
    ext = sx.extract_specs(specs, ac=ac, dc=dc, output_signal="out")

    # E1 gain 1000 -> exactly 60 dB.
    assert ext.values["gain"] == pytest.approx(60.0, abs=0.05)
    # Single pole at 1/(2*pi*1k*1n) = 159.1549 kHz, so UGB = 1000 * that.
    assert ext.values["ugb"] == pytest.approx(159.1549, rel=0.01)
    # One pole -> 90 degrees of phase margin.
    assert ext.values["pm"] == pytest.approx(90.0, abs=1.0)
    # 1.8 V across 20 kOhm.
    assert ext.values["idd"] == pytest.approx(90.0, rel=1e-6)
    assert ext.values["vout"] == pytest.approx(0.9, abs=1e-9)
    assert not ext.unmeasurable


# ------------------------------------------------------- rl_env wiring ------

@skip_no_ngspice
def test_rl_env_spec_check_scores_a_real_design():
    """The whole point: spec.check in the RL loop must score real measurements.

    Before this wiring, _run_spec_check forwarded a SCHEMA-keyed dict to a
    reward that looks up SPEC names, so every spec scored -1.0; and it called a
    RewardFunction instance as if it were a function, raising TypeError into a
    bare `except: pass` that returned 0.0. Both paths are exercised here.
    """
    from asic_ai.adapters import get_adapter
    from asic_ai.training.rl_env import CircuitDesignEnv

    adapter = get_adapter("ngspice_shared", binary_path="",
                          work_dir=tempfile.mkdtemp())
    rf = RewardFunction(specs=[SpecTarget(name="idd", max_val=200.0, unit="uA")])
    env = CircuitDesignEnv(adapter, rf, max_steps=5)

    task = {
        "id": "divider",
        "specs": {"idd": {"max": 200, "unit": "uA"}},
    }
    env.reset(task)

    netlist = ("* divider\nV1 vdd 0 DC 1.8\n"
               "R1 vdd out 10k\nR2 out 0 10k\n.op\n.end\n")
    sim = env.step({"name": "sim.dc", "arguments": {"netlist": netlist}})
    assert sim.info["tool_name"] == "sim.dc"
    assert "dc" in env.state.analyses, "typed result must be kept for measurement"

    chk = env.step({"name": "spec.check", "arguments": {}})
    payload = json.loads(chk.observation)

    assert payload["measured"]["idd"] == pytest.approx(90.0, rel=1e-6)
    assert not payload["unmeasurable"]
    assert payload["score"] > 0.0, "a passing design must not score at the floor"
    assert payload["passed"] is True


def test_rl_env_reports_unmeasurable_instead_of_flooring():
    """A digital spec must be reported, not silently scored -1.0."""
    from asic_ai.adapters.mock import MockSimulatorAdapter
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.training.rl_env import CircuitDesignEnv

    adapter = MockSimulatorAdapter(AdapterConfig(binary_path="",
                                                 work_dir=tempfile.mkdtemp()))
    rf = RewardFunction(specs=[SpecTarget(name="idd", max_val=200.0, unit="uA")])
    env = CircuitDesignEnv(adapter, rf, max_steps=5)
    env.reset({"id": "t", "specs": {"correct": {"target": 1, "unit": "bool"}}})

    chk = env.step({"name": "spec.check", "arguments": {}})
    payload = json.loads(chk.observation)
    assert "correct" in payload["unmeasurable"]
    assert "correct" not in payload["measured"]


# ------------------------------------------------ N3 / C3 regressions -------

def test_output_signal_is_honoured_for_ac_signals():
    """N3: the AC key is "vdb(<vec>)"; building "vdb(out" silently missed.

    Both the explicit choice and the out/vout/output fallback missed, so
    selection degraded to "first non-branch vector". That is right by luck when
    the output happens to be listed first, which is why 49 tests passed over it.
    Measured cost on a real deck: 15.97 Hz reported for a 999.9 Hz node.
    """
    from asic_ai.adapters.spec_extract import _pick_output
    signals = {"vdb(mid)": 1, "vdb(out)": 2, "vdb(mon)": 3, "vdb(vo)": 4}
    assert _pick_output(signals, "vo", prefix="vdb(", suffix=")") == "vdb(vo)"
    assert _pick_output(signals, "out", prefix="vdb(", suffix=")") == "vdb(out)"
    # No preference: 'out' still wins over whatever is listed first.
    assert _pick_output(signals, None, prefix="vdb(", suffix=")") == "vdb(out)"


def test_ac_metrics_follow_the_requested_output_node():
    """End to end for N3: two nodes with very different corner frequencies."""
    freqs = [10.0 ** (i / 20.0) for i in range(0, 141)]   # 1 Hz .. 1 MHz

    def lowpass(fp):
        return [-10.0 * math.log10(1.0 + (f / fp) ** 2) for f in freqs]

    ac = ACResult(
        frequencies=freqs,
        signals={
            # 'mon' is listed FIRST, so a broken picker returns it.
            "vdb(mon)": SignalData(name="vdb(mon)", x_values=freqs,
                                   y_values=lowpass(16.0)),
            "vdb(vo)": SignalData(name="vdb(vo)", x_values=freqs,
                                  y_values=lowpass(1000.0)),
        },
    )
    specs = {"bw": {"min": 1, "unit": "Hz"}}
    got = sx.extract_specs(specs, ac=ac, output_signal="vo").values["bw"]
    assert got == pytest.approx(1000.0, rel=0.02), (
        f"followed the wrong node: {got:.2f} Hz instead of ~1000 Hz")


def test_rl_env_scores_only_measured_specs(monkeypatch):
    """C3: unmeasurable specs must not pin the reward to the floor.

    measurable_specs() was written for exactly this and then never called, so
    the same circuit scored +0.63 or -0.00 depending only on where the model
    started its AC sweep.
    """
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.mock import MockSimulatorAdapter
    from asic_ai.training.rl_env import CircuitDesignEnv

    adapter = MockSimulatorAdapter(AdapterConfig(binary_path="",
                                                 work_dir=tempfile.mkdtemp()))
    rf = RewardFunction(specs=[SpecTarget(name="idd", max_val=200.0, unit="uA")])
    env = CircuitDesignEnv(adapter, rf, max_steps=5)
    env.reset({"id": "t", "specs": {
        "idd": {"max": 200, "unit": "uA"},
        "correct": {"target": 1, "unit": "bool"},   # never measurable
    }})
    env.state.analyses["dc"] = DCResult(op_points={"v1#branch": -9e-05}, sweeps={})

    payload = json.loads(env.step({"name": "spec.check", "arguments": {}}).observation)
    assert payload["measured"]["idd"] == pytest.approx(90.0, rel=1e-6)
    assert payload["specs_measured"] == 1 and payload["specs_checked"] == 2
    assert payload["score"] > 0.0, "the measured spec passes; score must not be floored"
    # Success still requires full coverage -- a half-checked design is not done.
    assert payload["passed"] is False
    assert payload["coverage"] == pytest.approx(0.5)


def test_rl_env_reports_when_nothing_is_measurable():
    """No measurable spec must read as neither pass nor design failure."""
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.mock import MockSimulatorAdapter
    from asic_ai.training.rl_env import CircuitDesignEnv

    adapter = MockSimulatorAdapter(AdapterConfig(binary_path="",
                                                 work_dir=tempfile.mkdtemp()))
    rf = RewardFunction(specs=[SpecTarget(name="idd", max_val=200.0, unit="uA")])
    env = CircuitDesignEnv(adapter, rf, max_steps=5)
    env.reset({"id": "t", "specs": {"correct": {"target": 1, "unit": "bool"}}})

    payload = json.loads(env.step({"name": "spec.check", "arguments": {}}).observation)
    assert payload["passed"] is False
    assert payload["coverage"] == 0.0
    assert "no spec could be measured" in payload["error"]
