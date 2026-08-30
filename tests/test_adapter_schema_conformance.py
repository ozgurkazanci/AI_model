"""Every adapter must build results the frozen schema actually accepts.

Audited against src/asic_ai/tool_interface/schema.py, all SEVEN result
constructors in adapters/ngspice.py used field names that do not exist:

    DCResult          outputs / sweep_values / sweep_var
    ACResult          magnitudes / phases
    TranResult        outputs
    NoiseResult       node_noise / total_input_noise / total_output_noise
    StabilityResult   frequencies / loop_gain_mag / loop_gain_phase
    CornerResult      corners / results
    MonteCarloResult  iterations

So every method raised a pydantic ValidationError the moment it was called --
and `ngspice` was the DEFAULT backend of get_adapter(), so a caller who did not
name one got the adapter where nothing worked. adapters/spectre_wsl.py had the
same defect in noise() and stb(), passing a list and a dict where SignalData is
declared.

Nothing caught it because nothing called those paths. The audit test below is
static, so it does not need a simulator, a licence, or a binary: it reads every
result construction in every adapter and checks the field names against the
schema. It fails on the old code for all seven.
"""
from __future__ import annotations

import ast
import inspect
import struct
from pathlib import Path

import pytest

from asic_ai.tool_interface import schema as sch
from asic_ai.tool_interface.schema import (
    ACResult, DCResult, NoiseResult, SignalData, StabilityResult, TranResult,
)

REPO_ROOT = Path(__file__).parent.parent
ADAPTERS = REPO_ROOT / "src" / "asic_ai" / "adapters"

RESULT_MODELS = {
    name: getattr(sch, name) for name in dir(sch)
    if name.endswith("Result") and hasattr(getattr(sch, name), "model_fields")
}


def _constructions(path: Path):
    """(model name, keyword names, line) for every Result(...) call in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name in RESULT_MODELS:
            yield name, {k.arg for k in node.keywords if k.arg}, node.lineno


@pytest.mark.parametrize(
    "path", sorted(ADAPTERS.glob("*.py")), ids=lambda p: p.name)
def test_adapter_only_uses_fields_the_schema_declares(path: Path):
    """The static audit. Catches all seven original violations."""
    problems = []
    for model_name, kwargs, line in _constructions(path):
        model = RESULT_MODELS[model_name]
        declared = set(model.model_fields)
        bogus = kwargs - declared
        if bogus:
            problems.append(
                f"{path.name}:{line} {model_name} uses {sorted(bogus)}, "
                f"which the schema does not declare (it has {sorted(declared)})")
    assert not problems, problems


@pytest.mark.parametrize(
    "path", sorted(ADAPTERS.glob("*.py")), ids=lambda p: p.name)
def test_adapter_supplies_every_required_field(path: Path):
    problems = []
    for model_name, kwargs, line in _constructions(path):
        model = RESULT_MODELS[model_name]
        required = {f for f, info in model.model_fields.items() if info.is_required()}
        missing = required - kwargs
        if missing:
            problems.append(
                f"{path.name}:{line} {model_name} omits required {sorted(missing)}")
    assert not problems, problems


def test_the_audit_would_catch_the_original_defect():
    """Guard the guard: the checks above must actually reject the old shapes."""
    for bad in (
        lambda: NoiseResult(frequencies=[], node_noise={},
                            total_input_noise=0.0, total_output_noise=0.0),
        lambda: StabilityResult(frequencies=[], loop_gain_mag=[],
                                loop_gain_phase=[], phase_margin=0.0,
                                gain_margin=0.0),
        lambda: NoiseResult(frequencies=[], input_noise=[], output_noise=[]),
        lambda: StabilityResult(phase_margin=0.0, gain_margin=0.0, loop_gain={}),
    ):
        with pytest.raises(Exception):
            bad()


# ------------------------------------------------------------- factory -----

def test_default_backend_is_one_that_works(tmp_path):
    """get_adapter() with no backend used to return the broken subprocess one."""
    from asic_ai.adapters import get_adapter
    adapter = get_adapter(binary_path="", work_dir=str(tmp_path))
    assert type(adapter).__name__ == "NgspiceSharedAdapter"


def test_subprocess_adapter_refuses_without_a_binary(tmp_path):
    """KiCad ships only ngspice.dll, so this must say so instead of failing later."""
    from asic_ai.adapters import get_adapter
    with pytest.raises(FileNotFoundError, match="ngspice_shared"):
        get_adapter("ngspice", binary_path="", work_dir=str(tmp_path))


# ------------------------------------------------------- rawfile parsing ---

def _write_raw(path: Path, variables, points, complex_data=False):
    """A minimal ngspice binary rawfile, so the parser can be tested with no
    ngspice executable present (this machine has none)."""
    flags = "complex" if complex_data else "real"
    head = (
        "Title: test\nDate: now\nPlotname: test\n"
        f"Flags: {flags}\n"
        f"No. Variables: {len(variables)}\n"
        f"No. Points: {len(points)}\n"
        "Variables:\n")
    for i, v in enumerate(variables):
        head += f"\t{i}\t{v}\tvoltage\n"
    head += "Binary:\n"
    with open(path, "wb") as f:
        f.write(head.encode("utf-8"))
        for row in points:
            for value in row:
                if complex_data:
                    f.write(struct.pack("dd", value.real, value.imag))
                else:
                    f.write(struct.pack("d", float(value)))


@pytest.fixture
def adapter(tmp_path):
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.ngspice import NgspiceAdapter
    # A path that exists is enough; nothing is executed in these tests.
    return NgspiceAdapter(AdapterConfig(binary_path=str(tmp_path / "ngspice"),
                                        work_dir=str(tmp_path)))


def test_real_rawfile_round_trips(adapter, tmp_path):
    raw = tmp_path / "r.raw"
    _write_raw(raw, ["v-sweep", "out"], [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])
    variables, data = adapter._parse_raw(raw)
    assert variables == ["v-sweep", "out"]
    assert data["out"] == [0.0, 0.25, 0.5]


def test_complex_rawfile_reads_sixteen_bytes_per_point(adapter, tmp_path):
    """An AC plot writes two doubles per point.

    The previous reader always consumed eight, so it walked out of step through
    the entire file. It never showed because ac() discarded the parse result.
    """
    raw = tmp_path / "c.raw"
    rows = [[complex(1.0, 0.0), complex(0.0, -1.0)],
            [complex(10.0, 0.0), complex(-2.0, 0.0)]]
    _write_raw(raw, ["frequency", "out"], rows, complex_data=True)
    variables, data = adapter._parse_raw(raw)
    assert data["frequency"] == [complex(1, 0), complex(10, 0)]
    assert data["out"] == [complex(0, -1), complex(-2, 0)]


def test_build_dc_makes_a_valid_sweep(adapter):
    res = adapter.build_dc(["v-sweep", "out"],
                           {"v-sweep": [0.0, 0.5, 1.0], "out": [0.0, 0.25, 0.5]})
    assert isinstance(res, DCResult)
    assert res.sweeps["out"].x_values == [0.0, 0.5, 1.0]
    assert res.sweeps["out"].y_values == [0.0, 0.25, 0.5]


def test_build_dc_treats_a_single_point_as_an_operating_point(adapter):
    res = adapter.build_dc(["v-sweep", "out", "vdd"],
                           {"v-sweep": [0.0], "out": [0.9], "vdd": [1.8]})
    assert res.op_points == {"v-sweep": 0.0, "out": 0.9, "vdd": 1.8}
    assert res.sweeps == {}


def test_build_ac_names_signals_the_way_ngspice_shared_does(adapter):
    """spec_extract looks up vdb(<vec>) / vp(<vec>); both backends must agree."""
    res = adapter.build_ac(
        ["frequency", "out"],
        {"frequency": [complex(1, 0), complex(10, 0)],
         "out": [complex(10.0, 0.0), complex(0.0, -1.0)]})
    assert isinstance(res, ACResult)
    assert set(res.signals) == {"vdb(out)", "vp(out)"}
    # |10| -> 20 dB exactly, |-1j| -> 0 dB, phase -90 deg.
    assert res.signals["vdb(out)"].y_values == pytest.approx([20.0, 0.0])
    assert res.signals["vp(out)"].y_values == pytest.approx([0.0, -90.0])
    assert res.frequencies == pytest.approx([1.0, 10.0])


def test_build_tran_is_a_valid_result(adapter):
    res = adapter.build_tran(["time", "out"],
                             {"time": [0.0, 1e-6], "out": [0.0, 1.8]})
    assert isinstance(res, TranResult)
    assert res.time == [0.0, 1e-6]
    assert res.signals["out"].y_values == [0.0, 1.8]


@pytest.mark.parametrize("method,args", [
    ("noise", ("c.cir", None)),
    ("stb", ("c.cir", None)),
    ("corners", ("c.cir", [])),
    ("mc", ("c.cir", 4, 1)),
])
def test_unimplemented_analyses_refuse_and_name_the_alternative(adapter, method, args):
    """An empty NoiseResult reads as a successful measurement of no noise."""
    with pytest.raises(NotImplementedError, match="ngspice_shared"):
        getattr(adapter, method)(*args)


# ------------------------------------------------------------- spectre -----

def test_spectre_stability_result_is_constructible():
    """loop_gain was {}, which is a ValidationError, so the real PM/GM parsed
    just above it never reached a caller."""
    res = StabilityResult(phase_margin=62.0, gain_margin=12.0,
                          loop_gain=SignalData(name="loop_gain_not_parsed",
                                               x_values=[], y_values=[]))
    assert res.phase_margin == 62.0
    assert res.loop_gain.y_values == []
