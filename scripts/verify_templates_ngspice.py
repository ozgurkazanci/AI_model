#!/usr/bin/env python3
"""Verify circuit templates with real ngspice simulation.

Renders all templates, wraps subckt templates in testbenches,
and runs through ngspice to verify they simulate correctly.

Usage:
    PYTHONPATH=src python scripts/verify_templates_ngspice.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.templates import TEMPLATES, render_template
from asic_ai.adapters.ngspice_shared import NgspiceSharedAdapter, find_ngspice_dll
from asic_ai.adapters.base import AdapterConfig
from asic_ai.tool_interface.schema import SimParams

SEP = "=" * 60

# Testbench wrappers for subckt-based templates
TESTBENCHES = {
    "ota_2stage": """\
* OTA Two-Stage Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02
.model pfet_01v8 pmos level=1 vto=-0.45 kp=100u lambda=0.03

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VBIAS vbias 0 DC 0.6
VINP inp 0 DC 0.9
VINM inm 0 DC 0.9

X1 vdd vss inp inm out vbias ota_2stage

.dc VINP 0.5 1.3 0.005
.end
""",
    "diff_pair": """\
* Differential Pair Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VBIAS vbias 0 DC 0.6
VINP inp 0 DC 0.9
VINM inm 0 DC 0.9

X1 vdd vss inp inm outn outp vbias diff_pair

.dc VINP 0.5 1.3 0.005
.end
""",
    "current_mirror_cascode": """\
* Cascode Current Mirror Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VBIAS vbias 0 DC 0.6
Iref vdd iref DC 50u

X1 iref out vbias vss cascode_cm

Vout out 0 DC 0
.dc Vout 0 1.8 0.01
.end
""",
    "source_follower": """\
* Source Follower Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VBIAS vbias 0 DC 0.6
VIN inp 0 DC 0.9

X1 vdd vss inp out vbias source_follower

.dc VIN 0.5 1.5 0.005
.end
""",
    "comparator_basic": """\
* Comparator Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02
.model pfet_01v8 pmos level=1 vto=-0.45 kp=100u lambda=0.03

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VBIAS vbias 0 DC 0.6
VINP inp 0 DC 0.9
VINM inm 0 DC 0.9

X1 vdd vss inp inm out vbias comparator

.dc VINP 0.5 1.3 0.005
.end
""",
    "ring_osc": """\
* Ring Oscillator Testbench
{subckt}

.model nfet_01v8 nmos level=1 vto=0.45 kp=200u lambda=0.02
.model pfet_01v8 pmos level=1 vto=-0.45 kp=100u lambda=0.03

VDD vdd 0 DC 1.8
VSS vss 0 DC 0
VEN en 0 DC 1.8

X1 vdd vss en out ring_osc

.tran 0.1n 20n
.end
""",
}


def main():
    print(f"\n{SEP}")
    print("   Template Verification with Real ngspice")
    print(f"{SEP}\n")

    dll = find_ngspice_dll()
    if not dll:
        print("  [FAIL] ngspice not found!")
        return

    results = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        config = AdapterConfig(binary_path=dll, work_dir=td)
        adapter = NgspiceSharedAdapter(config)

        for tid, tmpl in TEMPLATES.items():
            netlist = render_template(tid)

            # Check if flat netlist or subckt
            if ".subckt" in netlist:
                if tid in TESTBENCHES:
                    full_netlist = TESTBENCHES[tid].format(subckt=netlist)
                else:
                    print(f"  [SKIP] {tid}: no testbench defined")
                    results[tid] = {"status": "skip", "reason": "no testbench"}
                    continue
            else:
                full_netlist = netlist

            cir = Path(td) / f"{tid}.cir"
            cir.write_text(full_netlist, encoding="utf-8")

            try:
                t0 = time.time()
                if ".tran" in full_netlist:
                    r = adapter.tran(str(cir), SimParams(analysis_type="tran"))
                    pts = len(r.time)
                elif ".ac" in full_netlist:
                    r = adapter.ac(str(cir), SimParams(analysis_type="ac"))
                    pts = len(r.frequencies)
                else:
                    r = adapter.dc(str(cir), SimParams(analysis_type="dc"))
                    pts = sum(len(s.x_values) for s in r.sweeps.values())
                sim_time = time.time() - t0

                status = "OK" if pts > 0 else "FAIL"
                print(f"  [{status}] {tid}: {pts} pts, {sim_time*1000:.0f}ms ({tmpl.category})")
                results[tid] = {"status": status, "points": pts, "time_ms": round(sim_time * 1000)}
            except Exception as e:
                print(f"  [ERR] {tid}: {e}")
                results[tid] = {"status": "error", "error": str(e)}

    passed = sum(1 for r in results.values() if r.get("status") == "OK")
    skipped = sum(1 for r in results.values() if r.get("status") == "skip")
    failed = len(results) - passed - skipped

    print(f"\n{SEP}")
    print(f"  Results: {passed} OK, {failed} FAIL, {skipped} SKIP / {len(TEMPLATES)} total")
    print(f"{SEP}\n")

    out_path = Path("eval_results/template_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
