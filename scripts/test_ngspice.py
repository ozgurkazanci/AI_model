#!/usr/bin/env python3
"""Test ngspice integration using KiCad's ngspice shared library.

Validates that we can run real SPICE simulations via the ngspice DLL.

Usage:
    PYTHONPATH=src python scripts/test_ngspice.py
    PYTHONPATH=src python scripts/test_ngspice.py --dll "C:/Program Files/KiCad/10.0/bin/ngspice.dll"
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60

# Default KiCad ngspice paths
NGSPICE_PATHS = [
    r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
    r"C:\Program Files (x86)\KiCad\bin\ngspice.dll",
]

# Simple test circuits
TEST_CIRCUITS = {
    "resistor_divider": """
* Simple Resistor Divider
V1 vdd 0 DC 1.8
R1 vdd out 10k
R2 out 0 10k
.dc V1 0 3.3 0.1
.end
""",
    "rc_filter": """
* RC Low-Pass Filter
V1 in 0 AC 1 DC 0
R1 in out 1k
C1 out 0 1n
.ac dec 10 1 1G
.end
""",
    "nmos_iv": """
* NMOS I-V Curve (behavioral)
.model nch nmos level=1 vto=0.5 kp=100u
M1 drain gate 0 0 nch W=10u L=1u
Vgs gate 0 DC 0.9
Vds drain 0 DC 0
.dc Vds 0 1.8 0.05 Vgs 0.5 1.0 0.1
.end
""",
}


# Callback types for ngspice shared library
NGSPICE_SEND_CHAR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
NGSPICE_SEND_STAT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
NGSPICE_EXIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))


class NgspiceShared:
    """Interface to ngspice shared library (DLL)."""

    def __init__(self, dll_path: str):
        self.dll_path = dll_path
        self.output_lines: list[str] = []
        self.status_lines: list[str] = []
        self.lib = None

        # Set up environment for ngspice plugins
        dll_dir = str(Path(dll_path).parent)
        ngspice_lib = str(Path(dll_path).parent.parent / "lib" / "ngspice")
        os.environ["SPICE_LIB_DIR"] = ngspice_lib
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

        # Load DLL
        self.lib = ctypes.CDLL(dll_path)

        # Set up callbacks
        @NGSPICE_SEND_CHAR
        def send_char(msg, id, ud):
            try:
                line = msg.decode("utf-8", errors="replace").strip()
                self.output_lines.append(line)
            except Exception:
                pass
            return 0

        @NGSPICE_SEND_STAT
        def send_stat(msg, id, ud):
            try:
                line = msg.decode("utf-8", errors="replace").strip()
                self.status_lines.append(line)
            except Exception:
                pass
            return 0

        @NGSPICE_EXIT
        def controlled_exit(status, immediate, quit_not_unload, id, ud):
            return 0

        # Keep references to prevent GC
        self._send_char = send_char
        self._send_stat = send_stat
        self._exit = controlled_exit

        # Initialize ngspice
        ret = self.lib.ngSpice_Init(send_char, send_stat, controlled_exit, None, None, None, None)
        if ret != 0:
            raise RuntimeError(f"ngSpice_Init failed with code {ret}")

    def command(self, cmd: str) -> int:
        """Send a command to ngspice."""
        return self.lib.ngSpice_Command(cmd.encode("utf-8"))

    def run_circuit(self, netlist: str) -> dict:
        """Run a circuit netlist and return output."""
        self.output_lines.clear()
        self.status_lines.clear()

        # Write netlist to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False, encoding="utf-8") as f:
            f.write(netlist)
            temp_path = f.name

        try:
            # Source the netlist
            self.command(f"source {temp_path}")
            # Run simulation
            self.command("run")

            return {
                "success": True,
                "output": self.output_lines.copy(),
                "status": self.status_lines.copy(),
                "output_count": len(self.output_lines),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": self.output_lines.copy(),
            }
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def find_ngspice_dll() -> str | None:
    """Find ngspice DLL on the system."""
    for path in NGSPICE_PATHS:
        if Path(path).exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Test ngspice integration")
    parser.add_argument("--dll", default=None, help="Path to ngspice.dll")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI ngspice Integration Test")
    print(f"{SEP}\n")

    # Find DLL
    dll_path = args.dll or find_ngspice_dll()
    if not dll_path:
        print("  [FAIL] ngspice.dll not found!")
        print("  Searched:", "\n           ".join(NGSPICE_PATHS))
        return

    print(f"  DLL: {dll_path}")
    dll_size = Path(dll_path).stat().st_size / 1024 / 1024
    print(f"  Size: {dll_size:.1f} MB")

    # Load ngspice
    print(f"\n  [1/4] Loading ngspice shared library...")
    try:
        ng = NgspiceShared(dll_path)
        print(f"  [OK] ngspice loaded successfully!")
    except Exception as e:
        print(f"  [FAIL] Failed to load: {e}")
        return

    # Run test circuits
    results = []
    for i, (name, netlist) in enumerate(TEST_CIRCUITS.items(), 2):
        print(f"\n  [{i}/4] Testing: {name}")
        result = ng.run_circuit(netlist)

        if result["success"]:
            print(f"  [OK] {name}: {result['output_count']} output lines")
            # Show some output
            for line in result["output"][:5]:
                if line.strip():
                    print(f"       {line[:80]}")
            if result["output_count"] > 5:
                print(f"       ... ({result['output_count'] - 5} more lines)")
        else:
            print(f"  [FAIL] {name}: {result.get('error', 'unknown')}")

        result["name"] = name
        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n{SEP}")
    print(f"   Results: {passed}/{total} circuits simulated")
    print(f"{SEP}")

    if passed == total:
        print(f"  [OK] ngspice is fully functional!")
        print(f"  DLL path: {dll_path}")
        print(f"  Ready for real circuit simulation!")
    else:
        print(f"  [WARN] Some tests failed. Check output above.")

    # Save results
    output_path = Path("eval_results/ngspice_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "dll_path": dll_path,
        "passed": passed,
        "total": total,
        "results": [{k: v for k, v in r.items() if k != "output"} for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved: {output_path}\n")


if __name__ == "__main__":
    import json
    main()
