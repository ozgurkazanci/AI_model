#!/usr/bin/env python3
"""mikroelektronix -- a desktop window over the local ASIC design model.

    PYTHONPATH=src python -m mikroelektronix.app
    PYTHONPATH=src python -m mikroelektronix.app --serve   # also start the model

Python + pywebview rather than Electron or Tauri: this machine has no Node and
no Rust, the backend is already Python, and pywebview renders through the
system WebView2 (the same Chromium family Electron bundles) instead of shipping
its own copy. That is roughly 15 MB of dependency against several hundred.

The window talks to `api.Api` directly through pywebview's js_api bridge, so
there is no HTTP server, no port to collide with, and no CORS. The one HTTP
connection in the system is the one to llama-server, which already exists.

--serve launches llama-server on the iGPU for the lifetime of the window. Left
off, the app expects one to be running already and says so plainly if it is not,
rather than opening a chat that silently answers nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mikroelektronix.api import Api  # noqa: E402

INDEX = Path(__file__).parent / "web" / "index.html"


def main() -> int:
    parser = argparse.ArgumentParser(description="mikroelektronix desktop app")
    parser.add_argument("--serve", action="store_true",
                        help="start llama-server on the iGPU for this session")
    parser.add_argument("--model", default=None, help="GGUF path override")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=780)
    parser.add_argument("--debug", action="store_true",
                        help="open the WebView developer tools")
    args = parser.parse_args()

    try:
        import webview
    except ImportError:
        print("pywebview is not installed. Install it with: pip install pywebview")
        return 1

    if not INDEX.exists():
        print(f"UI not found: {INDEX}")
        return 1

    server = None
    if args.serve:
        from asic_ai.inference import llama_server
        cfg = llama_server.ServerConfig.from_config(model=args.model)
        if llama_server.server_binary() is None:
            print("llama-server not found. Run scripts/gpu_probe.py for setup help.")
            return 1
        if not cfg.model or not Path(cfg.model).exists():
            print(f"GGUF model not found: {cfg.model!r}")
            return 1
        if llama_server.health(cfg.base_url, timeout=2):
            # A server is already there. Starting a second one cannot bind the
            # port, but wait_until_healthy would find the EXISTING server and
            # report success -- then the finally block would kill our own failed
            # child while the real server carried on. Reuse it instead.
            print(f"a model server is already running on {cfg.base_url}; reusing it")
        else:
            print(f"starting the model on {cfg.base_url} (ngl={cfg.n_gpu_layers})")
            server = llama_server.LlamaServer(cfg)
            try:
                server.start()
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"failed to start the model: {exc}")
                return 1

    api = Api()
    window = webview.create_window(
        "mikroelektronix", str(INDEX), js_api=api,
        width=args.width, height=args.height, min_size=(720, 520),
    )
    api.bind(window)

    try:
        webview.start(debug=args.debug)
    finally:
        # The window is closed by the time we get here; always stop the child.
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
