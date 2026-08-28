#!/usr/bin/env python3
"""CPT corpus collection — downloads open-source data for Continued Pretraining.

Only APPROVED sources from data/corpus_registry.yaml are collected.

Usage:
    python scripts/collect_cpt_data.py --list
    python scripts/collect_cpt_data.py --source sky130_pdk --output data/cpt/
    python scripts/collect_cpt_data.py --all --output data/cpt/
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cpt_collector")


@dataclass
class CollectionResult:
    source_id: str
    files_collected: int = 0
    tokens_estimated: int = 0
    bytes_total: int = 0
    errors: list[str] = field(default_factory=list)


def _clone_repo(url: str, dest: Path, timeout: int = 300) -> bool:
    if dest.exists():
        log.info(f"Already cloned: {dest.name}")
        return True
    log.info(f"Cloning {url} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error(f"Clone failed: {e}")
        return False


def _count_files(directory: Path, extensions: set[str]) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            files += 1
            total_bytes += f.stat().st_size
    return files, total_bytes


TEXT_EXTS = {".rst", ".md", ".txt"}
SPICE_EXTS = {".spice", ".lib", ".spi", ".cdl", ".scs", ".sp"}
HDL_EXTS = {".sv", ".v", ".svh", ".vh", ".vhdl"}

SOURCES = {
    "sky130_pdk": {
        "url": "https://github.com/google/skywater-pdk.git",
        "extensions": TEXT_EXTS | SPICE_EXTS,
        "license": "Apache 2.0",
    },
    "gf180_pdk": {
        "url": "https://github.com/google/gf180mcu-pdk.git",
        "extensions": TEXT_EXTS | SPICE_EXTS,
        "license": "Apache 2.0",
    },
    "ihp_sg13g2": {
        "url": "https://github.com/IHP-GmbH/IHP-Open-PDK.git",
        "extensions": TEXT_EXTS | SPICE_EXTS,
        "license": "Apache 2.0",
    },
    "opentitan": {
        "url": "https://github.com/lowRISC/opentitan.git",
        "extensions": TEXT_EXTS | HDL_EXTS,
        "license": "Apache 2.0",
        "timeout": 600,
    },
    "openroad": {
        "url": "https://github.com/The-OpenROAD-Project/OpenROAD.git",
        "extensions": TEXT_EXTS | HDL_EXTS | {".tcl"},
        "license": "BSD 3-Clause",
    },
    "openfasoc": {
        "url": "https://github.com/idea-fasoc/OpenFASOC.git",
        "extensions": TEXT_EXTS | SPICE_EXTS | HDL_EXTS | {".py"},
        "license": "Apache 2.0",
    },
}


def collect_source(source_id: str, output_dir: Path) -> CollectionResult:
    result = CollectionResult(source_id=source_id)
    info = SOURCES[source_id]
    clone_dir = output_dir / source_id
    timeout = info.get("timeout", 300)

    if not _clone_repo(info["url"], clone_dir, timeout):
        result.errors.append("Clone failed")
        return result

    files, total_bytes = _count_files(clone_dir, info["extensions"])
    result.files_collected = files
    result.bytes_total = total_bytes
    result.tokens_estimated = total_bytes // 4

    log.info(f"{source_id}: {files} files, ~{result.tokens_estimated:,} tokens ({info['license']})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect CPT data from open sources")
    parser.add_argument("--source", choices=list(SOURCES.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="data/cpt")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable CPT sources:")
        for name, info in SOURCES.items():
            print(f"  {name:20s} | {info['license']:15s} | {info['url']}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(SOURCES.keys()) if args.all else ([args.source] if args.source else [])
    if not sources:
        parser.print_help()
        return

    results = []
    for s in sources:
        results.append(collect_source(s, output_dir))

    # Summary
    total_files = sum(r.files_collected for r in results)
    total_tokens = sum(r.tokens_estimated for r in results)
    print(f"\n{'='*60}\n{'Source':20s} | {'Files':>8s} | {'~Tokens':>14s} | Status")
    print(f"{'-'*60}")
    for r in results:
        status = "OK" if not r.errors else f"ERR({len(r.errors)})"
        print(f"{r.source_id:20s} | {r.files_collected:8d} | {r.tokens_estimated:14,} | {status}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':20s} | {total_files:8d} | {total_tokens:14,} |")

    # Manifest
    manifest = {"sources": [{"id": r.source_id, "files": r.files_collected,
                             "tokens_est": r.tokens_estimated, "errors": r.errors} for r in results]}
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
