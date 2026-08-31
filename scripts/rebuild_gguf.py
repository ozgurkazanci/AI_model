#!/usr/bin/env python3
"""Rebuild the GGUF from a trained adapter: merge -> convert -> quantize.

    PYTHONPATH=src python scripts/rebuild_gguf.py --adapter outputs/sft_945_v1/final --tag 945ex

One command instead of three, because the three-step chain has a silent failure
mode: retrain the adapter, forget one step, and mikroelektronix happily serves
YESTERDAY'S weights under today's name. Nothing in the GGUF itself says which
training run produced it.

So this writes MODEL_INFO.json next to the GGUF, carrying the training
provenance (example count, epochs, LoRA config -- read from the adapter's own
training_info.json, written at training time) plus the build date and the
GGUF's hash. mikroelektronix reads it and shows which model is actually loaded.
The provenance travels with the artifact, not with anyone's memory.

Tool locations come from configs/local_inference.yaml (llama_cpp.dir and
llama_cpp.src_dir), same as everything else that touches llama.cpp.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEP = "=" * 70


def sha256_head(path: Path, mb: int = 64) -> str:
    """Hash of the first `mb` MB -- enough to tell two builds apart without
    reading a gigabyte twice."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(mb * 1024 * 1024))
    return h.hexdigest()[:16]


def run(cmd: list[str], what: str) -> None:
    print(f"  -> {what}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-8:]
        raise SystemExit(f"{what} FAILED:\n  " + "\n  ".join(tail))


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the GGUF from an adapter")
    ap.add_argument("--adapter", required=True,
                    help="trained LoRA dir, e.g. outputs/sft_945_v1/final")
    ap.add_argument("--tag", required=True,
                    help="name fragment for the artifacts, e.g. 945ex")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--keep-f16", action="store_true",
                    help="keep the intermediate f16 GGUF")
    ap.add_argument("--set-default", action="store_true",
                    help="point configs/local_inference.yaml models.default at the result")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = REPO_ROOT / adapter
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"not an adapter dir (no adapter_config.json): {adapter}")

    # Provenance is read from what TRAINING wrote, not reconstructed here.
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = adapter_cfg.get("base_model_name_or_path", "")
    training_info = {}
    ti_path = adapter / "training_info.json"
    if ti_path.exists():
        training_info = json.loads(ti_path.read_text(encoding="utf-8"))
    else:
        print("  NOTE: adapter has no training_info.json; provenance will be thin")

    from asic_ai.inference import llama_server
    cfg = llama_server.load_config()
    lc_dir = llama_server.find_llama_cpp_dir(cfg)
    src_dir = Path((cfg.get("llama_cpp") or {}).get("src_dir", ""))
    if lc_dir is None:
        raise SystemExit("llama.cpp binaries not found; run scripts/gpu_probe.py")
    quantize = next((lc_dir / n for n in ("llama-quantize.exe", "llama-quantize")
                     if (lc_dir / n).exists()), None)
    convert = src_dir / "convert_hf_to_gguf.py"
    if quantize is None or not convert.exists():
        raise SystemExit(f"missing tools: quantize={quantize}, convert={convert}")

    merged = REPO_ROOT / "models" / f"merged-{args.tag}"
    gguf_dir = REPO_ROOT / "models" / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16 = gguf_dir / f"asic-ai-0.5b-{args.tag}-f16.gguf"
    quant = gguf_dir / f"asic-ai-0.5b-{args.tag}-{args.quant.lower()}.gguf"

    print(f"\n{SEP}\n  GGUF rebuild: {adapter.name} ({args.tag})\n{SEP}")
    print(f"  base     : {base}")
    print(f"  trained  : {training_info.get('examples', '?')} examples, "
          f"{training_info.get('epochs', '?')} epochs")

    run([sys.executable, str(REPO_ROOT / "scripts" / "merge_lora.py"),
         "--base", base, "--adapter", str(adapter), "--output", str(merged)],
        f"merge LoRA -> {merged.name}")
    run([sys.executable, str(convert), str(merged),
         "--outfile", str(f16), "--outtype", "f16"],
        f"convert -> {f16.name}")

    # llama-quantize.exe reads its argv in the ANSI codepage, so a repo path
    # with a non-ASCII character (this one has an U-umlaut) arrives mangled and
    # the file "does not exist". Stage the work in the temp dir, which is
    # ASCII, and move the result back. Verified 2026-08-31: quantize failed in
    # the repo tree and succeeded unchanged from the staging dir.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gguf_quant_") as td:
        staged_f16 = Path(td) / f16.name
        staged_quant = Path(td) / quant.name
        shutil.copy2(f16, staged_f16)
        run([str(quantize), str(staged_f16), str(staged_quant), args.quant],
            f"quantize -> {quant.name}")
        shutil.move(str(staged_quant), quant)

    info = {
        "gguf": quant.name,
        "built": _dt.date.today().isoformat(),
        "quant": args.quant,
        "sha256_head": sha256_head(quant),
        "size_mb": round(quant.stat().st_size / 1e6, 1),
        "adapter": str(adapter.relative_to(REPO_ROOT)) if adapter.is_relative_to(REPO_ROOT) else str(adapter),
        "base_model": base,
        "training": training_info,
    }
    info_path = gguf_dir / f"{quant.stem}.MODEL_INFO.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    if not args.keep_f16:
        f16.unlink(missing_ok=True)

    if args.set_default:
        import re
        yml = REPO_ROOT / "configs" / "local_inference.yaml"
        text = yml.read_text(encoding="utf-8")
        new = re.sub(r'(default:\s*")[^"]*(")',
                     rf'\g<1>models/gguf/{quant.name}\g<2>', text, count=1)
        if new == text:
            print("  WARNING: could not rewrite models.default; edit the yaml by hand")
        else:
            yml.write_text(new, encoding="utf-8")
            print(f"  models.default -> models/gguf/{quant.name}")

    print(f"\n{SEP}")
    print(f"  {quant.name}  ({info['size_mb']} MB, {info['sha256_head']})")
    print(f"  provenance -> {info_path.name}")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
