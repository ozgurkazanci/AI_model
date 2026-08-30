#!/usr/bin/env python3
"""Serve the fine-tuned model on the AMD 780M iGPU via llama.cpp + Vulkan.

Launches llama-server with the settings in configs/local_inference.yaml and
keeps it up, or runs a single prompt and exits.

Usage:
    # start the server and leave it running
    PYTHONPATH=src python scripts/serve_local.py

    # one prompt through the canonical system message, then exit
    PYTHONPATH=src python scripts/serve_local.py --prompt "Design a two-stage OTA in sky130."

    # force CPU to compare against the iGPU
    PYTHONPATH=src python scripts/serve_local.py --ngl 0 --prompt "..."

The system message is always build_system_message(). It must stay byte-identical
to what the model was trained on; a re-templated prompt is how tool calling
silently stops working.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import TOOL_DEFINITIONS, build_system_message
from asic_ai.inference import llama_server

SEP = "=" * 70


def show_tool_call(text: str) -> None:
    """Report whether the model emitted a call, and whether it is in contract."""
    if "<tool_call>" not in text:
        print("\n  no <tool_call> emitted")
        return
    known = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    start = text.index("<tool_call>") + len("<tool_call>")
    end = text.index("</tool_call>", start)
    try:
        call = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        print(f"\n  tool_call present but not valid JSON: {exc}")
        return
    name = call.get("name")
    ok = name in known
    print(f"\n  tool call: {name}   in frozen contract: {ok}")
    if not ok:
        print("  WARNING: this tool does not exist. The model is hallucinating"
              " a tool, which usually means a prompt or training-data problem.")
    print(f"  arguments: {json.dumps(call.get('arguments', {}))[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the model on the local iGPU")
    ap.add_argument("--model", default=None, help="GGUF path (default: config)")
    ap.add_argument("--ngl", type=int, default=None,
                    help="layers to offload; 0 = CPU, 99 = all (default: config)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--prompt", default=None,
                    help="run one prompt and exit instead of staying up")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--log", default=None, help="server log file")
    args = ap.parse_args()

    cfg = llama_server.ServerConfig.from_config(model=args.model)
    if args.ngl is not None:
        cfg.n_gpu_layers = args.ngl
    if args.port is not None:
        cfg.port = args.port

    if llama_server.server_binary() is None:
        print("llama-server not found. Run scripts/gpu_probe.py for setup help.")
        return 1
    if not cfg.model or not Path(cfg.model).exists():
        print(f"GGUF model not found: {cfg.model!r}")
        print("Build one: merge_lora.py -> convert_hf_to_gguf.py -> llama-quantize.exe")
        return 1

    where = "CPU" if cfg.n_gpu_layers == 0 else f"iGPU (ngl={cfg.n_gpu_layers})"
    print(f"\n{SEP}")
    print(f"  model:   {Path(cfg.model).name}")
    print(f"  compute: {where}")
    print(f"  url:     {cfg.base_url}")
    for k, v in cfg.env.items():
        print(f"  env:     {k}={v}")
    print(f"{SEP}\n")

    server = llama_server.LlamaServer(cfg)
    try:
        server.start(log_path=args.log)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"failed to start: {exc}")
        return 1
    print(f"  server healthy at {cfg.base_url}")

    engine = llama_server.LlamaServerEngine(cfg.base_url)

    try:
        if args.prompt is None:
            print("\n  Ready. OpenAI-compatible endpoint:")
            print(f"    POST {cfg.base_url}/v1/chat/completions")
            print("\n  Press Ctrl+C to stop.\n")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("\n  stopping")
            return 0

        messages = [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": args.prompt},
        ]
        t0 = time.time()
        result = engine.generate(messages, temperature=args.temperature,
                                 max_new_tokens=args.max_tokens)
        dt = time.time() - t0

        print(f"\n{SEP}")
        print(result.text)
        print(f"{SEP}")
        rate = result.completion_tokens / dt if dt > 0 else 0.0
        print(f"  {result.prompt_tokens} prompt + {result.completion_tokens} "
              f"completion tokens in {dt:.1f}s  ({rate:.1f} tok/s end to end)")
        show_tool_call(result.text)
        print()
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
