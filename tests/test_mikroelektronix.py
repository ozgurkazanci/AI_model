"""mikroelektronix must not become a fourth private copy of the agent loop.

The desktop app is the easiest place in this repo for a reimplementation to go
unnoticed: it has its own UI, its own thread, and nobody runs its code in CI.
This repo has already had the agent loop written three times (two produced
nothing at all) and the tool-call parser written against a format that appears
nowhere in the training data.

So these tests pin two things:
  - it uses the shared pieces -- the canonical system message, the real parser,
    the real adapter -- rather than private copies;
  - it never fabricates. No model means an explicit error, not a chat that
    silently answers nothing.

Everything runs headless. pywebview is only needed to open a window.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from mikroelektronix.api import Api, DesignSession

REPO_ROOT = Path(__file__).parent.parent
WEB = REPO_ROOT / "mikroelektronix" / "web" / "index.html"


class _Recorder:
    """Collects emitted events and lets a test wait for the turn to end."""

    def __init__(self):
        self.events = []
        self._done = threading.Event()

    def __call__(self, kind, payload):
        self.events.append((kind, payload))
        if kind in ("done", "error", "cancelled"):
            self._done.set()

    def wait(self, timeout=20.0):
        assert self._done.wait(timeout), f"turn did not finish: {self.kinds}"

    @property
    def kinds(self):
        return [k for k, _ in self.events]

    def of(self, kind):
        return [p for k, p in self.events if k == kind]


class _Engine:
    """A scripted model, so the tests need no server and no GPU."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, **kwargs):
        from asic_ai.inference.engine import GenerationResult
        self.seen.append(list(messages))
        text = self.replies.pop(0) if self.replies else "Done."
        return GenerationResult(text=text, prompt_tokens=11, completion_tokens=3)

    def get_token_count(self, text):
        return len(text) // 4


def _session(replies):
    import tempfile

    from asic_ai.adapters.base import AdapterConfig
    from asic_ai.adapters.mock import MockSimulatorAdapter
    from asic_ai.inference.parser import ToolCallParser

    rec = _Recorder()
    s = DesignSession(rec)
    s._engine = _Engine(replies)
    s._adapter = MockSimulatorAdapter(
        AdapterConfig(binary_path="", work_dir=tempfile.mkdtemp()))
    s._parser = ToolCallParser()
    from asic_ai.data.format import build_system_message
    s._messages = [{"role": "system", "content": build_system_message()}]
    return s, rec


# --------------------------------------------------------------- honesty ---

def test_no_model_is_an_error_not_a_silent_chat():
    rec = _Recorder()
    s = DesignSession(rec)
    s._engine = None
    s.send("design an OTA")
    rec.wait()
    assert rec.kinds[-1] == "error"
    assert "No model connected" in rec.of("error")[0]["message"]


def test_connect_reports_what_is_missing_rather_than_guessing(monkeypatch):
    from asic_ai.inference import llama_server
    monkeypatch.setattr(llama_server, "available", lambda *a, **k: False)

    info = DesignSession(_Recorder()).connect()
    assert info["model"] is None
    assert info["errors"], "a missing model must be reported"
    assert "serve_local" in " ".join(info["errors"])


def test_connect_finds_the_simulator():
    info = DesignSession(_Recorder()).connect()
    assert info["simulator"], "the adapter should always be constructible"


# ---------------------------------------------------- uses the shared parts --

def test_the_system_message_is_the_canonical_one():
    """A privately assembled prompt is how tool calling silently stops working."""
    from asic_ai.data.format import build_system_message

    s, _ = _session(["nothing to do"])
    assert s._messages[0]["role"] == "system"
    assert s._messages[0]["content"] == build_system_message()


def test_a_tool_call_is_parsed_and_executed():
    s, rec = _session([
        '<tool_call>{"name": "pdk.list_devices", "arguments": {}}</tool_call>',
        "That is enough.",
    ])
    s.send("what devices are there?")
    rec.wait()

    calls = rec.of("tool_call")
    assert [c["name"] for c in calls] == ["pdk.list_devices"]
    assert calls[0]["valid"] is True
    assert rec.of("tool_result"), "the tool must actually run"


def test_a_hallucinated_tool_is_rejected_and_fed_back():
    """Recovering from a bad call is the behaviour worth showing the user."""
    s, rec = _session([
        '<tool_call>{"name": "report.generate", "arguments": {}}</tool_call>',
        "Giving up.",
    ])
    s.send("write a report")
    rec.wait()

    call = rec.of("tool_call")[0]
    assert call["valid"] is False
    assert "Unknown tool" in call["reason"]
    # and the rejection went back into the conversation
    assert any(m["role"] == "tool" and "Unknown tool" in m["content"]
               for m in s._messages)


def test_a_missing_required_argument_is_caught_before_the_simulator():
    s, rec = _session([
        '<tool_call>{"name": "sim.ac", "arguments": {}}</tool_call>',
        "Done.",
    ])
    s.send("run an ac sweep")
    rec.wait()
    call = rec.of("tool_call")[0]
    assert call["valid"] is False
    assert "netlist" in call["reason"]


def test_tool_tags_are_stripped_from_the_prose():
    """The tags get their own cards; showing them raw is noise."""
    s, rec = _session([
        'Checking the devices.\n<tool_call>{"name": "pdk.list_devices", "arguments": {}}</tool_call>',
        "Done.",
    ])
    s.send("go")
    rec.wait()
    first = rec.of("assistant")[0]["text"]
    assert "<tool_call>" not in first
    assert "Checking the devices." in first


def test_an_unparseable_call_is_surfaced_not_swallowed():
    s, rec = _session(['<tool_call>{"name": "sim.dc", oops}</tool_call>'])
    s.send("go")
    rec.wait()
    assert rec.of("assistant")[0]["parse_errors"], "the user must see it failed"


def test_a_turn_ends_when_the_model_stops_calling_tools():
    s, rec = _session(["A two-stage OTA would work here."])
    s.send("advise me")
    rec.wait()
    assert rec.of("done")[0]["reason"] == "no_tool_call"


def test_reset_restores_the_canonical_system_message_only():
    from asic_ai.data.format import build_system_message

    s, rec = _session(["ok"])
    s.send("hello")
    rec.wait()
    assert len(s._messages) > 1
    s.reset()
    assert s._messages == [{"role": "system", "content": build_system_message()}]


def test_events_are_json_serialisable():
    """They are pasted into the page through evaluate_js."""
    s, rec = _session([
        '<tool_call>{"name": "pdk.list_devices", "arguments": {}}</tool_call>',
        "Done.",
    ])
    s.send("go")
    rec.wait()
    for kind, payload in rec.events:
        json.dumps(payload, allow_nan=False, default=str)


# ------------------------------------------------------------------- ui ----

def test_the_ui_file_exists_and_defines_the_event_hook():
    html = WEB.read_text(encoding="utf-8")
    assert "window.onAgentEvent" in html
    for kind in ("assistant", "tool_call", "tool_result", "error", "done"):
        assert f"'{kind}'" in html, f"the UI ignores the {kind} event"


def test_the_ui_asks_python_only_for_what_the_api_exposes():
    html = WEB.read_text(encoding="utf-8")
    called = set(re.findall(r"pywebview\.api\.(\w+)", html))
    exposed = {n for n in dir(Api) if not n.startswith("_")} - {"bind"}
    assert called <= exposed, f"the UI calls {called - exposed}, which Api does not expose"


def test_the_ui_is_self_contained():
    """pywebview loads it from disk; a CDN would make the app need the network."""
    html = WEB.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html


# ------------------------------------------------------------- launching ---

def test_serve_reuses_a_running_server_instead_of_starting_a_second(monkeypatch,
                                                                    capsys):
    """Starting a second server cannot bind the port, but wait_until_healthy
    would find the EXISTING one and report success -- and then the finally block
    would stop our own failed child while the real server carried on."""
    import webview

    from asic_ai.inference import llama_server
    from mikroelektronix import app

    started = []

    class _NeverStarts:
        def __init__(self, cfg, binary=None):
            started.append(cfg)

        def start(self, **kw):
            raise AssertionError("a second server must not be started")

        def stop(self):
            pass

    monkeypatch.setattr(llama_server, "health", lambda url, timeout=5: True)
    monkeypatch.setattr(llama_server, "server_binary", lambda *a, **k: Path("x"))
    monkeypatch.setattr(llama_server, "LlamaServer", _NeverStarts)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(webview, "create_window", lambda *a, **k: object())
    monkeypatch.setattr(webview, "start", lambda **k: None)
    monkeypatch.setattr("sys.argv", ["app", "--serve"])

    assert app.main() == 0
    assert not started, "the app constructed a second server"
    assert "reusing it" in capsys.readouterr().out


def test_the_launcher_sets_what_a_double_click_does_not(tmp_path):
    """Explorer gives neither the repo root nor PYTHONPATH, and the import
    error that follows says nothing useful."""
    bat = (REPO_ROOT / "mikroelektronix" / "mikroelektronix.bat").read_text(
        encoding="utf-8", errors="replace")
    assert 'set "PYTHONPATH=src"' in bat
    assert 'cd /d' in bat and '%~dp0..' in bat, "must cd to the repository root"
    assert "--serve" in bat, "a double-click should bring its own model"
