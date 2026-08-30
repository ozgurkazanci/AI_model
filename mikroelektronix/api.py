"""The Python side of mikroelektronix: one design turn, driven from the UI.

Everything here is a thin shell over parts that already exist and are tested:

    asic_ai.inference.llama_server   the model on the iGPU (Vulkan)
    asic_ai.inference.parser         reads <tool_call> and validates the contract
    asic_ai.training.rl_env          executes a tool against a real simulator
    asic_ai.data.format              the canonical system message

Nothing about a design turn is reimplemented here. That matters: this repo has
already had the agent loop written three separate times, two of which produced
nothing at all, and the tool-call parser written against a format that appears
nowhere in the training data. A desktop app is exactly the place where a fourth
private copy would go unnoticed.

Threading: pywebview calls js_api methods on the UI thread, so a generation that
takes seconds would freeze the window. Each turn therefore runs on a worker
thread and pushes events back into the page, which is also what makes the tool
calls appear as they happen rather than all at the end.
"""
from __future__ import annotations

import json
import re
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

MAX_STEPS = 8


def _model_provenance(model_path: str) -> Optional[Dict[str, Any]]:
    """The MODEL_INFO.json scripts/rebuild_gguf.py writes next to a GGUF.

    Returns None when absent (a hand-built GGUF has no provenance to show),
    never a guess.
    """
    try:
        from pathlib import Path

        p = Path(model_path)
        info_path = p.with_name(p.stem + ".MODEL_INFO.json")
        if not info_path.exists():
            return None
        data = json.loads(info_path.read_text(encoding="utf-8"))
        t = data.get("training") or {}
        return {
            "built": data.get("built"),
            "examples": t.get("examples"),
            "epochs": t.get("epochs"),
            "sha": data.get("sha256_head", "")[:8],
        }
    except Exception:
        return None

_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _js_escape(value: Any) -> str:
    """JSON, safe to paste into a <script> evaluation."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class DesignSession:
    """One conversation, its tools, and the deck they run against."""

    def __init__(self, emit: Callable[[str, Dict[str, Any]], None]):
        self._emit = emit
        self._engine = None
        self._adapter = None
        self._parser = None
        self._env = None
        self._messages: List[Dict[str, Any]] = []
        self._busy = threading.Lock()
        self._cancel = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> Dict[str, Any]:
        """Find a model and a simulator. Reports what is missing, never guesses."""
        from asic_ai.data.format import build_system_message
        from asic_ai.inference.parser import ToolCallParser

        info: Dict[str, Any] = {"model": None, "simulator": None,
                                "device": None, "errors": []}

        try:
            from asic_ai.inference import llama_server
            self._engine = llama_server.available() and self._resolve_engine()
            if self._engine:
                cfg = llama_server.ServerConfig.from_config()
                info["model"] = cfg.model.split("/")[-1].split("\\")[-1]
                devices = llama_server.list_devices()
                info["device"] = devices[0] if devices else "CPU"
                # Provenance, if the build step wrote it. Without this the UI
                # cannot tell yesterday's weights from today's under the same
                # file name -- which is exactly how a stale model gets trusted.
                info["provenance"] = _model_provenance(cfg.model)
            else:
                info["errors"].append(
                    "No model server reachable. Start one with: "
                    "PYTHONPATH=src python scripts/serve_local.py")
        except Exception as exc:
            info["errors"].append(f"model: {exc}")

        try:
            import tempfile

            from asic_ai.adapters import get_adapter
            self._adapter = get_adapter(binary_path="",
                                        work_dir=tempfile.mkdtemp())
            info["simulator"] = type(self._adapter).__name__
        except Exception as exc:
            info["errors"].append(f"simulator: {exc}")

        self._parser = ToolCallParser()
        if not self._messages:
            self._messages = [{"role": "system", "content": build_system_message()}]
        return info

    def _resolve_engine(self):
        from asic_ai.inference import llama_server
        cfg = llama_server.ServerConfig.from_config()
        engine = llama_server.LlamaServerEngine(cfg.base_url)
        return engine if engine.healthy() else None

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        from asic_ai.reward.reward import RewardFunction, SpecTarget
        from asic_ai.training.rl_env import CircuitDesignEnv

        # A permissive placeholder reward: the chat has no task specs until the
        # user states some. spec.check reports what it could measure either way.
        reward_fn = RewardFunction(specs=[SpecTarget(name="idd", max_val=1.0)])
        self._env = CircuitDesignEnv(self._adapter, reward_fn, max_steps=MAX_STEPS)
        self._env.reset({"id": "chat", "specs": {}})

    # -- one turn ----------------------------------------------------------

    def send(self, text: str) -> None:
        """Run a turn on a worker thread. Returns immediately."""
        if not self._busy.acquire(blocking=False):
            self._emit("error", {"message": "A turn is already running."})
            return
        self._cancel.clear()
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def cancel(self) -> None:
        self._cancel.set()

    def _run_turn(self, text: str) -> None:
        try:
            if self._engine is None:
                self._emit("error", {"message":
                                     "No model connected. Start scripts/serve_local.py "
                                     "and press Reconnect."})
                return

            self._ensure_env()
            self._messages.append({"role": "user", "content": text})

            for step in range(MAX_STEPS):
                if self._cancel.is_set():
                    self._emit("cancelled", {})
                    return

                self._emit("thinking", {"step": step})
                gen = self._engine.generate(self._messages, temperature=0.2,
                                            max_new_tokens=768)
                reply = gen.text or ""
                self._messages.append({"role": "assistant", "content": reply})

                calls = self._parser.parse(reply)
                errors = self._parser.parse_errors(reply)

                # Tool calls get their own cards, so strip them from the prose
                # rather than showing the raw tags to the user.
                prose = _TOOL_CALL_RE.sub("", reply)
                self._emit("assistant", {
                    "text": prose.strip(),
                    "tokens": {"prompt": gen.prompt_tokens,
                               "completion": gen.completion_tokens},
                    "parse_errors": errors,
                })

                if not calls:
                    self._emit("done", {"reason": "no_tool_call"})
                    return

                for call in calls:
                    ok, why = self._parser.validate_tool_call(call)
                    self._emit("tool_call", {"name": call.name,
                                             "arguments": call.arguments,
                                             "valid": ok, "reason": why})
                    if not ok:
                        # Feed the rejection back: recovering from a bad call is
                        # the behaviour worth seeing.
                        observation = json.dumps({"error": why})
                    else:
                        result = self._env.step({"name": call.name,
                                                 "arguments": call.arguments})
                        observation = result.observation
                    self._emit("tool_result", {"name": call.name,
                                               "observation": observation[:4000]})
                    self._messages.append({"role": "tool",
                                           "content": observation[:4000]})

            self._emit("done", {"reason": "max_steps"})
        except Exception as exc:
            self._emit("error", {"message": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc()[-1500:]})
        finally:
            self._busy.release()

    def reset(self) -> None:
        from asic_ai.data.format import build_system_message
        self._messages = [{"role": "system", "content": build_system_message()}]
        self._env = None


class Api:
    """Exposed to JavaScript as window.pywebview.api."""

    def __init__(self) -> None:
        self._window = None
        self._session = DesignSession(self._emit)

    def bind(self, window) -> None:
        self._window = window

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._window is None:
            return
        try:
            self._window.evaluate_js(
                f"window.onAgentEvent({_js_escape(kind)}, {_js_escape(payload)})")
        except Exception:
            pass  # the window closed mid-turn

    # -- called from JS ----------------------------------------------------

    def connect(self) -> Dict[str, Any]:
        return self._session.connect()

    def send(self, text: str) -> None:
        self._session.send(text)

    def cancel(self) -> None:
        self._session.cancel()

    def reset(self) -> None:
        self._session.reset()
