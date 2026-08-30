#!/usr/bin/env python3
"""Report which local inference backends actually work on this machine.

Written because "the GPU is available" is three different questions -- is the
device present, does the runtime load, and does a model actually run on it --
and on this hardware they had three different answers.

Usage:
    PYTHONPATH=src python scripts/gpu_probe.py
    PYTHONPATH=src python scripts/gpu_probe.py --bench   # also time CPU vs iGPU
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.inference import llama_server

SEP = "=" * 70
REPO_ROOT = Path(__file__).parent.parent


def _line(label: str, value: str) -> None:
    print(f"  {label:<26s} {value}")


def probe_torch() -> dict:
    """PyTorch, CUDA, and DirectML -- import success, not just installation."""
    out: dict = {"installed": False}
    try:
        import torch
    except ImportError:
        _line("torch", "not installed")
        return out

    out.update(installed=True, version=torch.__version__,
               cuda=bool(torch.cuda.is_available()))
    _line("torch", torch.__version__)
    _line("  cuda", "yes" if out["cuda"] else "no")

    try:
        import torch_directml  # noqa: F401
        out["directml"] = "ok"
        _line("  torch-directml", "importable")
    except ImportError as exc:
        # Installed-but-broken is the common case: the wheel is built against a
        # specific torch ABI and silently fails to load against another.
        out["directml"] = f"broken: {exc}"
        msg = str(exc)
        if "torch_directml_native" in msg:
            _line("  torch-directml", "INSTALLED BUT BROKEN (torch ABI mismatch)")
            _line("", "built for torch 2.4.1; downgrading would break")
            _line("", "transformers 5.16.1 -- use Vulkan instead")
        else:
            _line("  torch-directml", f"unavailable ({msg[:44]})")
    except Exception as exc:  # pragma: no cover - defensive
        out["directml"] = f"error: {exc}"
        _line("  torch-directml", f"error: {str(exc)[:44]}")
    return out


def probe_vulkan() -> dict:
    """Vulkan loader and device, via vulkaninfo when it is on PATH."""
    out: dict = {"loader": False}
    if not shutil.which("vulkaninfo"):
        _line("vulkaninfo", "not on PATH (llama.cpp probe below is definitive)")
        return out
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        _line("vulkaninfo", f"failed: {exc}")
        return out

    out["loader"] = True
    for line in (r.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("Vulkan Instance Version"):
            _line("vulkan instance", s.split(":", 1)[1].strip())
            out["instance"] = s.split(":", 1)[1].strip()
        elif s.startswith("deviceName"):
            name = s.split("=", 1)[1].strip()
            _line("vulkan device", name)
            out.setdefault("devices", []).append(name)
    return out


def probe_llama_cpp() -> dict:
    """The one that decides whether the iGPU is usable for inference."""
    cfg = llama_server.load_config()
    d = llama_server.find_llama_cpp_dir(cfg)
    if d is None:
        _line("llama.cpp", "NOT FOUND")
        _line("", "set ASIC_AI_LLAMA_CPP_DIR or llama_cpp.dir in")
        _line("", "configs/local_inference.yaml")
        return {"found": False}

    _line("llama.cpp dir", str(d))
    devices = llama_server.list_devices(cfg)
    for dev in devices:
        _line("  device", dev)
    if not devices:
        _line("  device", "none reported")

    env = (cfg.get("server") or {}).get("env") or {}
    if env.get("GGML_VK_DISABLE_COOPMAT") == "1":
        _line("  coopmat", "DISABLED by config")
        _line("", "the AMD driver does not expose")
        _line("", "VK_KHR_shader_bfloat16 / VK_EXT_pipeline_robustness,")
        _line("", "which llama.cpp's coopmat path requires. Without the")
        _line("", "flag every load dies with ErrorExtensionNotPresent.")
        _line("", "Matrix cores are therefore unused -- try removing the")
        _line("", "flag and re-benchmarking after a driver update.")
    return {"found": True, "dir": str(d), "devices": devices}


def probe_models() -> dict:
    cfg = llama_server.load_config()
    found = {}
    for name, rel in (cfg.get("models") or {}).items():
        p = Path(rel)
        if not p.is_absolute():
            p = REPO_ROOT / rel
        if p.exists():
            mb = p.stat().st_size / 1e6
            _line(f"model:{name}", f"{p.name}  ({mb:.0f} MB)")
            found[name] = str(p)
        else:
            _line(f"model:{name}", f"MISSING ({rel})")
    if not found:
        _line("", "build one with: merge_lora.py -> convert_hf_to_gguf.py")
        _line("", "                -> llama-quantize.exe")
    return found


def probe_server() -> bool:
    url = llama_server.ServerConfig.from_config().base_url
    ok = llama_server.health(url, timeout=3)
    _line("llama-server", f"{url} -- {'running' if ok else 'not running'}")
    return ok


def run_bench(models: dict) -> None:
    cfg = llama_server.load_config()
    d = llama_server.find_llama_cpp_dir(cfg)
    if d is None or not models:
        print("\n  bench skipped: need llama.cpp and at least one GGUF model")
        return
    bench = None
    for exe in ("llama-bench.exe", "llama-bench"):
        if (d / exe).exists():
            bench = d / exe
            break
    if bench is None:
        print("\n  bench skipped: llama-bench not found")
        return

    model = models.get("default") or next(iter(models.values()))
    env = dict((cfg.get("server") or {}).get("env") or {})
    import os
    full_env = {**os.environ, **{str(k): str(v) for k, v in env.items()}}

    print(f"\n{SEP}\n  Benchmark: CPU (ngl 0) vs iGPU (ngl 99)\n{SEP}")
    cmd = [str(bench), "-m", model, "-ngl", "0,99", "-p", "512", "-n", "128", "-r", "3"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                           env=full_env)
    except subprocess.SubprocessError as exc:
        print(f"  bench failed: {exc}")
        return
    for line in (r.stdout or "").splitlines():
        if line.startswith("|") or line.startswith("build:"):
            print("  " + line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe local inference backends")
    ap.add_argument("--bench", action="store_true",
                    help="also run llama-bench CPU vs iGPU (takes a few minutes)")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args()

    report: dict = {}

    print(f"\n{SEP}\n  ASIC-AI local inference backends\n{SEP}\n")
    print(" PyTorch")
    report["torch"] = probe_torch()
    print("\n Vulkan")
    report["vulkan"] = probe_vulkan()
    print("\n llama.cpp")
    report["llama_cpp"] = probe_llama_cpp()
    print("\n Models")
    report["models"] = probe_models()
    print("\n Server")
    report["server_running"] = probe_server()

    lc = report["llama_cpp"]
    gpu = lc.get("found") and any("vulkan" in d.lower() for d in lc.get("devices", []))
    print(f"\n{SEP}")
    if gpu and report["models"]:
        print("  VERDICT: iGPU inference available via llama.cpp + Vulkan.")
        print("  Run:  PYTHONPATH=src python scripts/serve_local.py")
    elif lc.get("found"):
        print("  VERDICT: llama.cpp present but no GPU device or no GGUF model.")
    else:
        print("  VERDICT: CPU only. See configs/local_inference.yaml for setup.")
    print(f"{SEP}\n")

    if args.bench:
        run_bench(report["models"])
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
