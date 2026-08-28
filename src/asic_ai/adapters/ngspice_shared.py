"""ngspice shared library adapter using KiCad's DLL via ctypes.

Uses the ngspice shared library API for in-process simulation.
Works with KiCad's bundled ngspice.dll on Windows.
"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import tempfile
from pathlib import Path

from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.tool_interface.schema import (
    DCResult, ACResult, TranResult, NoiseResult, StabilityResult,
    CornerResult, MonteCarloResult, SimParams, PVTCorner, SignalData,
)

log = logging.getLogger(__name__)

_SEND_CHAR = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
)
_SEND_STAT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
)
_CTRL_EXIT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int)
)

DEFAULT_DLL_PATHS = [
    r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
]


def find_ngspice_dll() -> str | None:
    """Auto-detect ngspice DLL from KiCad installation."""
    for p in DEFAULT_DLL_PATHS:
        if Path(p).exists():
            return p
    return None


class NgspiceSharedAdapter(SimulatorAdapter):
    """ngspice adapter using shared library (DLL) via ctypes."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._output: list[str] = []
        self._lib = None

        dll_path = config.binary_path
        if not dll_path or not Path(dll_path).exists():
            dll_path = find_ngspice_dll()
        if not dll_path:
            raise FileNotFoundError("ngspice.dll not found. Install KiCad or set binary_path.")

        dll_dir = str(Path(dll_path).parent)
        lib_dir = str(Path(dll_path).parent.parent / "lib" / "ngspice")
        os.environ["SPICE_LIB_DIR"] = lib_dir
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

        self._lib = ctypes.CDLL(dll_path)
        self._init_callbacks()
        log.info(f"ngspice shared library loaded: {dll_path}")

    def _init_callbacks(self):
        @_SEND_CHAR
        def on_char(msg, id_, ud):
            try:
                self._output.append(msg.decode("utf-8", errors="replace").strip())
            except Exception:
                pass
            return 0

        @_SEND_STAT
        def on_stat(msg, id_, ud):
            return 0

        @_CTRL_EXIT
        def on_exit(status, immediate, quit_, id_, ud):
            return 0

        self._cbs = (on_char, on_stat, on_exit)
        ret = self._lib.ngSpice_Init(on_char, on_stat, on_exit, None, None, None, None)
        if ret != 0:
            raise RuntimeError(f"ngSpice_Init failed: {ret}")

    def _cmd(self, cmd: str) -> int:
        return self._lib.ngSpice_Command(cmd.encode("utf-8"))

    def _simulate(self, netlist: str) -> list[str]:
        self._output.clear()
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".cir", delete=False,
            dir=str(self.work_dir), encoding="utf-8",
        )
        tmp.write(netlist)
        tmp.close()
        try:
            self._cmd(f"source {tmp.name}")
            self._cmd("run")
            return self._output.copy()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _data_rows(self, output: list[str]) -> int:
        for line in output:
            m = re.search(r"No\. of Data Rows\s*:\s*(\d+)", line)
            if m:
                return int(m.group(1))
        return 0

    # ---- SimulatorInterface methods ----

    def dc(self, circuit_path: str, params: SimParams) -> DCResult:
        with open(circuit_path, "r", encoding="utf-8") as f:
            netlist = f.read()
        output = self._simulate(netlist)
        rows = self._data_rows(output)
        sig = SignalData(name="sweep", x_values=list(range(rows)), y_values=[0.0] * rows)
        return DCResult(op_points={}, sweeps={"sweep": sig})

    def ac(self, circuit_path: str, params: SimParams) -> ACResult:
        with open(circuit_path, "r", encoding="utf-8") as f:
            netlist = f.read()
        output = self._simulate(netlist)
        rows = self._data_rows(output)
        sig = SignalData(name="mag", x_values=list(range(rows)), y_values=[0.0] * rows)
        return ACResult(frequencies=list(range(rows)), signals={"mag": sig})

    def tran(self, circuit_path: str, params: SimParams) -> TranResult:
        with open(circuit_path, "r", encoding="utf-8") as f:
            netlist = f.read()
        output = self._simulate(netlist)
        rows = self._data_rows(output)
        sig = SignalData(name="out", x_values=list(range(rows)), y_values=[0.0] * rows)
        return TranResult(time=list(range(rows)), signals={"out": sig})

    def noise(self, circuit_path: str, params: SimParams) -> NoiseResult:
        with open(circuit_path, "r", encoding="utf-8") as f:
            netlist = f.read()
        self._simulate(netlist)
        return NoiseResult(frequencies=[], input_noise=[], output_noise=[])

    def stb(self, circuit_path: str, params: SimParams) -> StabilityResult:
        with open(circuit_path, "r", encoding="utf-8") as f:
            netlist = f.read()
        self._simulate(netlist)
        return StabilityResult(phase_margin=0.0, gain_margin=0.0, loop_gain={})

    def corners(self, circuit_path: str, pvt_list: list[PVTCorner]) -> list[CornerResult]:
        results = []
        with open(circuit_path, "r", encoding="utf-8") as f:
            base_netlist = f.read()
        for pvt in pvt_list:
            netlist = base_netlist
            if ".temp" not in netlist.lower():
                netlist = netlist.replace(".end", f".temp {pvt.temperature}\n.end")
            output = self._simulate(netlist)
            rows = self._data_rows(output)
            sig = SignalData(name="sweep", x_values=list(range(rows)), y_values=[0.0] * rows)
            results.append(CornerResult(
                corner=pvt, dc=DCResult(op_points={}, sweeps={"sweep": sig}),
            ))
        return results

    def mc(self, circuit_path: str, n: int, seed: int) -> MonteCarloResult:
        return MonteCarloResult(seed=seed, runs=n, results=[])
