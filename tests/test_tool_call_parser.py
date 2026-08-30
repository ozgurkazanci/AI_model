"""The parser must read the format the project actually trains and serves.

asic_ai.data.format defines one tool-call format and every training example uses
it. The parser implemented a different one -- `<function=name>...</function>` --
which occurs ZERO times in data/sft/, while `<tool_call>` occurs 4322 times. It
therefore returned no calls for real model output and no calls for the project's
own training data: the model could emit a perfect, in-contract tool call and the
inference loop would see nothing at all.

The corpus test at the bottom is the one that matters. It parses every tool call
in every SFT file, so the parser and the training format cannot drift apart
again without a failure.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from asic_ai.data.format import TOOL_DEFINITIONS
from asic_ai.inference.parser import ParsedToolCall, ToolCallParser

REPO_ROOT = Path(__file__).parent.parent

# Verbatim output from the fine-tuned 0.5B model running on the iGPU.
REAL_MODEL_OUTPUT = (
    "Let me start by checking available devices.\n"
    '<tool_call>{"name": "pdk.device_query", "arguments": '
    '{"model": "nfet_01v8", "W": 1e-05, "L": 1.8e-07, "VGS": 0.6, "VDS": 0.9}}'
    "</tool_call>"
)


@pytest.fixture
def parser():
    # Takes no constructor arguments; several call sites depend on that.
    return ToolCallParser()


def test_parses_real_model_output(parser):
    calls = parser.parse(REAL_MODEL_OUTPUT)
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "pdk.device_query"
    assert call.arguments["model"] == "nfet_01v8"
    assert call.arguments["W"] == pytest.approx(1e-05)
    assert call.thinking == "Let me start by checking available devices."
    assert call.parse_method == "chatml"


def test_every_tool_call_in_the_corpus_parses():
    """The regression guard: parser and training format must not drift apart."""
    parser = ToolCallParser()
    total = parsed = 0
    failures: list[str] = []

    for path in sorted(glob.glob(str(REPO_ROOT / "data" / "sft" / "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                for msg in (json.loads(line).get("messages") or []):
                    content = msg.get("content") or ""
                    blocks = content.count("<tool_call>")
                    if not blocks:
                        continue
                    got = parser.parse(content)
                    total += blocks
                    parsed += len(got)
                    if len(got) != blocks and len(failures) < 5:
                        failures.append(
                            f"{Path(path).name}:{line_no} "
                            f"{len(got)}/{blocks} -- {parser.parse_errors(content)[:1]}")

    assert total > 1000, f"corpus scan found only {total} tool calls; is the data there?"
    assert parsed == total, f"{total - parsed} unparsed of {total}: {failures}"


def test_multiple_calls_in_one_turn(parser):
    text = ('first\n<tool_call>{"name": "lint.check", "arguments": {"netlist": "a"}}</tool_call>\n'
            'second\n<tool_call>{"name": "sim.dc", "arguments": {"netlist": "b"}}</tool_call>')
    calls = parser.parse(text)
    assert [c.name for c in calls] == ["lint.check", "sim.dc"]


def test_newlines_inside_the_payload_survive(parser):
    """Netlists are multi-line; a non-DOTALL pattern would silently drop them."""
    netlist = "* amp\nV1 in 0 AC 1\nR1 in out 1k\n.end"
    text = "<tool_call>" + json.dumps(
        {"name": "sim.ac", "arguments": {"netlist": netlist}}) + "</tool_call>"
    call = parser.parse(text)[0]
    assert call.arguments["netlist"] == netlist
    assert call.arguments["netlist"].count("\n") == 3


def test_whitespace_around_the_body_is_tolerated(parser):
    text = ('<tool_call>\n  {"name": "pdk.list_devices", "arguments": {}}\n</tool_call>')
    assert parser.parse(text)[0].name == "pdk.list_devices"


def test_trailing_comma_is_recovered(parser):
    text = '<tool_call>{"name": "sim.dc", "arguments": {"netlist": "x",},}</tool_call>'
    assert parser.parse(text)[0].arguments == {"netlist": "x"}


# ------------------------------------------------------------- refusals -----

def test_malformed_body_is_not_turned_into_an_empty_call(parser):
    """An empty-argument call is plausible enough to run and return nonsense."""
    text = '<tool_call>{"name": "sim.ac", "arguments": {oops}}</tool_call>'
    assert parser.parse(text) == []
    assert parser.parse_errors(text), "a failure must be reported, not swallowed"


def test_missing_name_is_refused(parser):
    text = '<tool_call>{"arguments": {"netlist": "x"}}</tool_call>'
    assert parser.parse(text) == []
    assert "name" in parser.parse_errors(text)[0]


def test_non_object_arguments_are_refused(parser):
    text = '<tool_call>{"name": "sim.dc", "arguments": "netlist"}</tool_call>'
    assert parser.parse(text) == []


def test_no_tool_call_yields_no_calls_and_no_errors(parser):
    assert parser.parse("just some prose about a two-stage OTA") == []
    assert parser.parse_errors("just some prose") == []
    assert parser.parse("") == []


# ----------------------------------------------------------- validation -----

def test_validate_accepts_a_complete_contract_call(parser):
    call = ParsedToolCall(name="sim.ac", arguments={"netlist": "x"},
                          thinking="", raw_text="", parse_method="chatml")
    assert parser.validate_tool_call(call) == (True, "Valid")


def test_validate_rejects_a_hallucinated_tool(parser):
    call = ParsedToolCall(name="report.generate", arguments={},
                          thinking="", raw_text="", parse_method="chatml")
    ok, why = parser.validate_tool_call(call)
    assert ok is False
    assert "Unknown tool" in why


def test_validate_rejects_a_missing_required_argument(parser):
    call = ParsedToolCall(name="sim.ac", arguments={},
                          thinking="", raw_text="", parse_method="chatml")
    ok, why = parser.validate_tool_call(call)
    assert ok is False
    assert "netlist" in why


@pytest.mark.parametrize("name", sorted(t["function"]["name"] for t in TOOL_DEFINITIONS))
def test_every_contract_tool_validates_with_its_required_arguments(parser, name):
    spec = next(t["function"] for t in TOOL_DEFINITIONS if t["function"]["name"] == name)
    required = (spec.get("parameters", {}) or {}).get("required", []) or []
    call = ParsedToolCall(name=name, arguments={r: "x" for r in required},
                          thinking="", raw_text="", parse_method="chatml")
    assert parser.validate_tool_call(call)[0] is True


def test_legacy_function_tag_still_parses(parser):
    """Kept so a differently-templated base model does not break outright."""
    text = '<function=sim.dc>{"netlist": "x"}</function>'
    call = parser.parse(text)[0]
    assert call.name == "sim.dc" and call.parse_method == "xml"
