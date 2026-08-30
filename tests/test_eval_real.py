"""The eval runner must report what happened, never a fabricated pass.

eval/runner.run_task previously returned, for every task without exception:

    passed = True
    score  = 85.5

The design document lists "progress without an eval set" as a pitfall because
progress cannot be measured. An eval set that always reports a pass is worse:
it reports progress that did not happen. Across 78 tasks it would have claimed
100 pct.

These tests pin the honesty properties rather than any particular score:
no engine is not a pass, a broken task is not a pass, a model that emits no tool
call is not a pass, and a real run is scored from the real spec check.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from eval.runner import EvalResult, format_task_prompt, load_task, run_task

REPO_ROOT = Path(__file__).parent.parent
TASKS = REPO_ROOT / "eval" / "tasks"


class _StubEngine:
    """A scripted model. Returns each canned reply in turn, then stops."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, **kwargs):
        from asic_ai.inference.engine import GenerationResult
        self.seen.append(messages[-1])
        text = self.replies.pop(0) if self.replies else "Done."
        return GenerationResult(text=text, prompt_tokens=1, completion_tokens=1)

    def get_token_count(self, text):
        return len(text) // 4


def _mock_adapter():
    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.mock import MockSimulatorAdapter
    return MockSimulatorAdapter(AdapterConfig(binary_path="",
                                              work_dir=tempfile.mkdtemp()))


@pytest.fixture
def a_task(tmp_path):
    task = {
        "id": "honesty_001",
        "category": "analog",
        "pdk": "sky130",
        "supply": 1.8,
        "description": "A divider used only to exercise the harness.",
        "specs": {"idd": {"max": 200, "unit": "uA"}},
        "corners": ["tt"],
        "pass_criteria": "typical_only",
    }
    p = tmp_path / "honesty_001.yaml"
    p.write_text(yaml.safe_dump(task), encoding="utf-8")
    return p


# ------------------------------------------------------------- honesty -----

def test_no_engine_is_not_a_pass(a_task):
    """The exact old failure mode: something missing must never score a pass."""
    result = run_task(a_task, engine=None, adapter=_mock_adapter())
    assert isinstance(result, EvalResult)
    assert result.passed is False
    assert result.final_score == 0.0
    assert result.error and "engine" in result.error


def test_unloadable_task_is_not_a_pass(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("specs: [this is not: a mapping", encoding="utf-8")
    result = run_task(bad)
    assert result.passed is False
    assert result.error


def test_a_model_that_calls_no_tools_does_not_pass(a_task):
    engine = _StubEngine(["I would design a bandgap reference. No tools needed."])
    result = run_task(a_task, engine=engine, adapter=_mock_adapter())
    assert result.passed is False
    assert result.final_score == 0.0
    assert result.trajectory and result.trajectory[0]["tool_calls"] == []


def test_a_hallucinated_tool_is_rejected_not_executed(a_task):
    engine = _StubEngine([
        '<tool_call>{"name": "report.generate", "arguments": {}}</tool_call>',
        "Giving up.",
    ])
    result = run_task(a_task, engine=engine, adapter=_mock_adapter())
    assert result.passed is False
    observations = result.trajectory[0].get("observations") or []
    assert observations and "Unknown tool" in observations[0]


def test_unparseable_tool_call_is_recorded_not_silently_dropped(a_task):
    engine = _StubEngine(['<tool_call>{"name": "sim.dc", oops}</tool_call>'])
    result = run_task(a_task, engine=engine, adapter=_mock_adapter())
    assert result.passed is False
    assert result.trajectory[0]["parse_errors"], "a broken call must be visible"
    assert result.error and "unparseable" in result.error


def test_a_real_spec_check_drives_the_score(a_task):
    """A measured, passing spec must produce a real score and a real pass."""
    netlist = "* divider\\nV1 vdd 0 DC 1.8\\nR1 vdd out 10k\\nR2 out 0 10k\\n.op\\n.end"
    engine = _StubEngine([
        '<tool_call>{"name": "sim.dc", "arguments": {"netlist": "' + netlist + '"}}</tool_call>',
        '<tool_call>{"name": "spec.check", "arguments": {}}</tool_call>',
    ])
    result = run_task(a_task, engine=engine, adapter=_mock_adapter(), max_steps=4)
    # The mock adapter's numbers are not the point; the wiring is. The score must
    # come from spec.check rather than from a constant.
    assert result.final_score != 85.5
    assert isinstance(result.final_score, float)
    assert result.steps >= 1


def test_result_is_json_serialisable(a_task):
    """Results are written to disk and read by eval/metrics.py."""
    engine = _StubEngine(["No tools."])
    result = run_task(a_task, engine=engine, adapter=_mock_adapter())
    json.dumps(result.model_dump(), allow_nan=False)


# --------------------------------------------------------------- prompt ----

def test_prompt_carries_the_specs_and_the_pdk():
    task = load_task(next(TASKS.glob("analog/*.yaml")))
    prompt = format_task_prompt(task)
    assert task.get("pdk", "sky130") in prompt
    for spec_name in task["specs"]:
        assert spec_name in prompt, f"{spec_name} missing from the task prompt"


def test_every_eval_task_produces_a_prompt():
    """A task the runner cannot describe can never be attempted."""
    for path in sorted(TASKS.rglob("*.yaml")):
        prompt = format_task_prompt(load_task(path))
        assert len(prompt) > 50, f"{path.name}: prompt is suspiciously short"


# ------------------------------------------------------------- baseline ----

def test_baseline_refuses_to_record_without_an_engine(tmp_path, monkeypatch):
    """A baseline of zeros would make every later run look like an improvement."""
    import eval.baseline as baseline
    monkeypatch.setattr(baseline, "_default_engine", lambda: None)
    out = tmp_path / "baseline.json"
    code = baseline.run_baseline(str(TASKS), "test-model", str(out), limit=1)
    assert code == 1
    assert not out.exists(), "nothing may be written when there is no model"


# ------------------------------------------------ measure_baseline ----------

def _load_measure_baseline():
    """Import scripts/measure_baseline.py as a module.

    It must be registered in sys.modules before exec, or @dataclass cannot
    resolve the module it belongs to.
    """
    import importlib.util as iu
    import sys

    path = REPO_ROOT / "scripts" / "measure_baseline.py"
    spec = iu.spec_from_file_location("_mb_under_test", path)
    module = iu.module_from_spec(spec)
    sys.modules["_mb_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_measure_baseline_without_an_engine_reports_no_score(monkeypatch):
    """It was honest as a placeholder; it must stay honest now that it is wired.

    A baseline of zeros recorded as if measured would make every later run look
    like an improvement, which is worse than recording nothing.
    """
    import asyncio

    import eval.runner as er

    mb = _load_measure_baseline()
    monkeypatch.setattr(er, "_default_engine", lambda: None)

    task = {"id": "t", "category": "analog", "difficulty": "easy",
            "specs": {"idd": {"max": 100, "unit": "uA"}}}
    result = asyncio.run(mb.run_task_with_model(task, mb.BaselineConfig()))

    assert result.passed is False
    assert result.final_score == 0.0
    assert result.error and "engine" in result.error


def test_measure_baseline_drives_the_shared_agent_loop(monkeypatch):
    """Four callers, one loop. A fifth implementation would be a fifth bug."""
    import asyncio

    import eval.runner as er
    from asic_ai.inference.engine import GenerationResult

    class _Engine:
        def generate(self, messages, **kwargs):
            return GenerationResult(
                text='<tool_call>{"name": "pdk.list_devices", "arguments": {}}</tool_call>',
                prompt_tokens=7, completion_tokens=3)

        def get_token_count(self, text):
            return len(text) // 4

    mb = _load_measure_baseline()
    monkeypatch.setattr(er, "_default_engine", lambda: _Engine())
    monkeypatch.setattr(er, "_default_adapter", _mock_adapter)

    task = {"id": "t", "category": "analog", "difficulty": "easy",
            "specs": {"idd": {"max": 100, "unit": "uA"}}}
    result = asyncio.run(mb.run_task_with_model(
        task, mb.BaselineConfig(max_steps=2)))

    assert result.error is None
    assert result.steps >= 1, "the loop must actually run"
    assert result.token_usage["prompt"] > 0, "real token counts, not zeros"
