"""The GBNF grammar must be derived, valid, and actually sent.

Three ways this feature could rot silently, each pinned here:

  - the grammar drifts from TOOL_DEFINITIONS (a hand-edited name list is the
    parser-vs-corpus mismatch all over again);
  - a typo'd rule reference ships a grammar llama-server rejects with HTTP 400
    on EVERY request, which the eval would report as 77 engine errors;
  - the engine quietly stops attaching it, and hallucinated names come back
    while the config still says grammar_constrained: true.

The live test at the bottom proves the server ACCEPTS the grammar and that a
constrained generation cannot emit an out-of-contract name.
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import TOOL_DEFINITIONS, build_system_message
from asic_ai.inference import llama_server
from asic_ai.inference.grammar import (_grammar_for, contract_tool_names,
                                       tool_call_grammar)
from asic_ai.inference.llama_server import (GRAMMAR_FROM_CONFIG,
                                            LlamaServerEngine, ServerConfig,
                                            health)
from asic_ai.inference.parser import ToolCallParser

CONTRACT = {t["function"]["name"] for t in TOOL_DEFINITIONS}

# Names the 945ex eval actually hallucinated; none may be samplable.
HALLUCINATED = ["int.patch", "sim.dc_ac", "simulate_simulate", "monte_carrier",
                "lint_check", "corners", "sim.run", "pdk.config"]


# -------------------------------------------------------------- derivation ---

def test_grammar_names_come_from_the_contract():
    g = tool_call_grammar()
    assert contract_tool_names() == sorted(CONTRACT)
    for name in CONTRACT:
        assert f'"\\"{name}\\""' in g, f"contract tool {name} missing from grammar"


def test_hallucinated_names_are_not_in_the_grammar():
    g = tool_call_grammar()
    for bad in HALLUCINATED:
        assert f'"{bad}"' not in g and f'\\"{bad}\\"' not in g


def test_grammar_is_derived_not_hardcoded():
    """Dropping a name from the input must drop it from the output."""
    g = _grammar_for(["sim.dc"])
    assert '\\"sim.dc\\"' in g
    assert '\\"sim.ac\\"' not in g


def test_zero_tools_is_refused():
    with pytest.raises(ValueError):
        _grammar_for([])


# --------------------------------------------------------------- structure ---

def _rules(grammar: str) -> dict[str, str]:
    out = {}
    for line in grammar.splitlines():
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9-]*)\s*::=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_every_referenced_rule_is_defined():
    """A typo'd reference is a grammar the server rejects on every request."""
    g = tool_call_grammar()
    rules = _rules(g)
    assert "root" in rules
    for name, body in rules.items():
        # strip literals and char classes, what remains must be rule refs
        stripped = re.sub(r'"(?:[^"\\]|\\.)*"', " ", body)
        stripped = re.sub(r"\[(?:[^\]\\]|\\.)*\]", " ", stripped)
        for ref in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", stripped):
            assert ref in rules, f"rule {name} references undefined {ref}"


def test_no_fstring_artifacts_leaked():
    g = tool_call_grammar()
    assert "{{" not in g and "}}" not in g
    assert 'callobj ::= "{"' in g


def test_prose_cannot_open_a_stray_angle_bracket():
    rules = _rules(tool_call_grammar())
    assert rules["prose"].strip() == "[^<]*"
    assert '"<tool_call>"' in rules["toolcall"]


def test_json_strings_exclude_raw_control_characters():
    """Re-admitting raw newlines re-admits half the malformed bodies."""
    rules = _rules(tool_call_grammar())
    assert r"\x00-\x1F" in rules["jchar"]


# ------------------------------------------------------------------ wiring ---

class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        ).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_engine_sends_the_grammar_it_was_given(monkeypatch):
    captured = _capture_payload(monkeypatch)
    eng = LlamaServerEngine("http://127.0.0.1:9", grammar="root ::= [^<]*")
    eng.generate([{"role": "user", "content": "x"}])
    assert captured["payload"]["grammar"] == "root ::= [^<]*"


def test_explicit_none_means_unconstrained(monkeypatch):
    captured = _capture_payload(monkeypatch)
    eng = LlamaServerEngine("http://127.0.0.1:9", grammar=None)
    eng.generate([{"role": "user", "content": "x"}])
    assert "grammar" not in captured["payload"]


def test_config_default_attaches_the_contract_grammar(monkeypatch):
    monkeypatch.setattr(llama_server, "load_config",
                        lambda *a, **k: {"inference": {"grammar_constrained": True}})
    eng = LlamaServerEngine("http://127.0.0.1:9", grammar=GRAMMAR_FROM_CONFIG)
    assert eng.grammar == tool_call_grammar()


def test_config_off_means_no_grammar(monkeypatch):
    monkeypatch.setattr(llama_server, "load_config",
                        lambda *a, **k: {"inference": {"grammar_constrained": False}})
    eng = LlamaServerEngine("http://127.0.0.1:9", grammar=GRAMMAR_FROM_CONFIG)
    assert eng.grammar is None


def test_repo_config_actually_enables_it():
    """The eval and mikroelektronix constrain only if the real yaml says so."""
    inf = llama_server.load_config().get("inference") or {}
    assert inf.get("grammar_constrained") is True


# -------------------------------------------------------------------- live ---

_CFG = llama_server.load_config()
_MODEL = ServerConfig.from_config(_CFG).model
HAS_LIVE = (llama_server.available() and bool(_MODEL) and Path(_MODEL).exists())

skip_no_live = pytest.mark.skipif(
    not HAS_LIVE, reason="llama.cpp or GGUF model not present")


@pytest.fixture(scope="module")
def live_server():
    if not HAS_LIVE:
        pytest.skip("llama.cpp or GGUF model not present")
    cfg = ServerConfig.from_config()
    cfg.port = 8233  # own port: never collide with a user's server or 8232
    srv = llama_server.LlamaServer(cfg)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@skip_no_live
def test_live_server_accepts_the_grammar_and_stays_in_contract(live_server):
    """If llama-server rejected the GBNF this would be an HTTP 400, not a
    quiet degradation -- which is exactly why the eval needs this proven."""
    eng = LlamaServerEngine(live_server.base_url, grammar=tool_call_grammar())
    result = eng.generate(
        [{"role": "system", "content": build_system_message()},
         {"role": "user", "content":
          "Run a transient analysis of a synchronous FIFO and patch the "
          "netlist if it fails. Use your tools."}],
        temperature=0.0, max_new_tokens=200)

    parser = ToolCallParser()
    assert not parser.parse_errors(result.text), (
        "grammar-constrained output must always parse: " + result.text[:400])
    for call in parser.parse(result.text):
        assert call.name in CONTRACT, f"out-of-contract name {call.name!r}"
    # prose cannot contain '<' outside the tags, so a stray tag is impossible
    outside = re.sub(r"<tool_call>.*?</tool_call>", "", result.text,
                     flags=re.DOTALL)
    assert "<" not in outside
