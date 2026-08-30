"""ngspice via a subprocess and its binary rawfile.

Superseded on this machine by `ngspice_shared`, which drives KiCad's ngspice.dll
in-process. This adapter is kept for platforms that have an `ngspice` executable
but no usable shared library.

EVERY result constructor in this file was previously invalid. Audited against
the frozen schema, all seven violated it:

    DCResult          used outputs / sweep_values / sweep_var
    ACResult          used magnitudes / phases
    TranResult        used outputs
    NoiseResult       used node_noise / total_input_noise / total_output_noise
    StabilityResult   used frequencies / loop_gain_mag / loop_gain_phase
    CornerResult      used corners / results
    MonteCarloResult  used iterations

Not one of those field names exists on the model it was passed to, so every
method raised a pydantic ValidationError the moment it was called. `ngspice` was
also the DEFAULT backend of get_adapter(), so a caller who did not name a
backend got the adapter where nothing worked. Nothing caught it because nothing
called it.

The rawfile parsing was and is real. What follows fixes the shapes, handles the
complex rawfile that an AC analysis actually produces, and refuses -- loudly --
for the analyses whose parsing genuinely is not implemented, rather than
returning an empty structure that reads as a successful simulation of nothing.

The AC signal naming matches ngspice_shared exactly (`vdb(<vec>)` for dB
magnitude, `vp(<vec>)` for phase in degrees) so spec_extract and the measurement
helpers work against either backend without knowing which one produced the
result.
"""
from __future__ import annotations

import cmath
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

from asic_ai.adapters.base import AdapterConfig, SimulatorAdapter
from asic_ai.tool_interface.schema import (
    ACResult, CornerResult, DCResult, MonteCarloResult, NoiseResult, PVTCorner,
    SignalData, SimParams, StabilityResult, TranResult,
)

_USE_SHARED = ("This analysis is not implemented for the subprocess adapter. "
               "Use get_adapter('ngspice_shared'), which drives KiCad's "
               "ngspice.dll in-process and implements it.")


class NgspiceAdapter(SimulatorAdapter):
    """ngspice adapter driving the `ngspice` executable."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        if not config.binary_path:
            raise FileNotFoundError(
                "NgspiceAdapter needs binary_path pointing at an ngspice "
                "executable. KiCad ships only ngspice.dll, so on this machine "
                "use get_adapter('ngspice_shared').")

    # -- rawfile ----------------------------------------------------------

    def _parse_raw(self, raw_file: Path) -> Tuple[List[str], Dict[str, List[Any]]]:
        """Parse an ngspice binary rawfile.

        Returns (variable names in order, {name: values}). Values are complex
        for a complex plot and float otherwise.

        The complex case matters: an AC analysis writes TWO doubles per point,
        and the previous reader always consumed eight bytes, so it would have
        walked out of step through the whole file. It never showed because ac()
        discarded the parse result entirely.
        """
        with open(raw_file, "rb") as f:
            header: Dict[str, str] = {}
            while True:
                raw_line = f.readline()
                if not raw_line:
                    return [], {}
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line in ("Binary:", "Binary"):
                    break
                if line.lower().startswith("variables"):
                    header["__vars_started__"] = "1"
                    break
                if ":" in line:
                    key, val = line.split(":", 1)
                    header[key.strip()] = val.strip()

            if "No. Variables" not in header or "No. Points" not in header:
                return [], {}

            num_vars = int(header["No. Variables"])
            num_points = int(header["No. Points"])
            is_complex = "complex" in header.get("Flags", "").lower()

            # When the loop above stopped at "Variables:", the variable lines
            # follow it and then "Binary:".
            variables: List[str] = []
            if header.get("__vars_started__"):
                while len(variables) < num_vars:
                    line = f.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        variables.append(parts[1])
                marker = f.readline().decode("utf-8", errors="ignore").strip()
                if marker not in ("Binary:", "Binary"):
                    return [], {}
            else:
                for _ in range(num_vars):
                    line = f.readline().decode("utf-8", errors="ignore").strip()
                    parts = line.split()
                    if len(parts) >= 2:
                        variables.append(parts[1])

            if len(variables) != num_vars:
                return [], {}

            width = 16 if is_complex else 8
            data: Dict[str, List[Any]] = {v: [] for v in variables}
            for _ in range(num_points):
                for v in range(num_vars):
                    chunk = f.read(width)
                    if len(chunk) < width:
                        return variables, data
                    if is_complex:
                        re_, im_ = struct.unpack("dd", chunk)
                        data[variables[v]].append(complex(re_, im_))
                    else:
                        data[variables[v]].append(struct.unpack("d", chunk)[0])
            return variables, data

    def _run(self, circuit_path: str) -> Tuple[List[str], Dict[str, List[Any]]]:
        commands = ".control\nrun\nwrite sim.raw\n.endc"
        deck = self._write_deck(circuit_path, commands)
        cmd = [self.config.binary_path, "-b", "-r", "sim.raw", str(deck)]
        res = self._run_subprocess(cmd)
        if res.returncode != 0 or "failed" in (res.stderr or "").lower():
            raise RuntimeError(f"ngspice failed: {res.stderr}")
        return self._parse_raw(self.work_dir / "sim.raw")

    def _write_deck(self, circuit_path: str, commands: str) -> Path:
        deck_path = self.work_dir / "sim.sp"
        with open(circuit_path, "r", encoding="utf-8", errors="replace") as f:
            circuit_content = f.read()
        with open(deck_path, "w", encoding="utf-8") as f:
            f.write(circuit_content)
            f.write("\n")
            if self.config.seed is not None:
                f.write(f".set seed={self.config.seed}\n")
            f.write(commands)
            f.write("\n.end\n")
        return deck_path

    # -- result building --------------------------------------------------

    @staticmethod
    def _real(values: List[Any]) -> List[float]:
        return [float(v.real) if isinstance(v, complex) else float(v)
                for v in values]

    def build_dc(self, variables: List[str],
                 data: Dict[str, List[Any]]) -> DCResult:
        """DCResult from a parsed rawfile. Separated so it can be tested
        without an ngspice binary."""
        if not variables:
            return DCResult(op_points={}, sweeps={})

        axis_name = variables[0]
        axis = self._real(data.get(axis_name, []))

        # A single point is an operating point, not a sweep.
        if len(axis) <= 1:
            op = {v: self._real(data[v])[0] for v in variables
                  if data.get(v)}
            return DCResult(op_points=op, sweeps={})

        sweeps = {
            v: SignalData(name=v, x_values=axis, y_values=self._real(data[v]))
            for v in variables[1:] if len(data.get(v, [])) == len(axis)
        }
        return DCResult(op_points={}, sweeps=sweeps)

    def build_ac(self, variables: List[str],
                 data: Dict[str, List[Any]]) -> ACResult:
        """ACResult from a parsed rawfile, named as ngspice_shared names it."""
        if not variables:
            return ACResult(frequencies=[], signals={})

        freq_name = variables[0]
        freqs = self._real(data.get(freq_name, []))

        signals: Dict[str, SignalData] = {}
        for v in variables[1:]:
            vals = data.get(v, [])
            if len(vals) != len(freqs):
                continue
            mag, phase = [], []
            for z in vals:
                c = z if isinstance(z, complex) else complex(float(z), 0.0)
                a = abs(c)
                mag.append(20.0 * math.log10(a) if a > 0.0 else float("-inf"))
                phase.append(math.degrees(cmath.phase(c)))
            signals[f"vdb({v})"] = SignalData(name=f"vdb({v})",
                                              x_values=freqs, y_values=mag)
            signals[f"vp({v})"] = SignalData(name=f"vp({v})",
                                             x_values=freqs, y_values=phase)
        return ACResult(frequencies=freqs, signals=signals)

    def build_tran(self, variables: List[str],
                   data: Dict[str, List[Any]]) -> TranResult:
        """TranResult from a parsed rawfile."""
        if not variables:
            return TranResult(time=[], signals={})
        t = self._real(data.get(variables[0], []))
        signals = {
            v: SignalData(name=v, x_values=t, y_values=self._real(data[v]))
            for v in variables[1:] if len(data.get(v, [])) == len(t)
        }
        return TranResult(time=t, signals=signals)

    # -- SimulatorInterface ------------------------------------------------

    def dc(self, circuit_path: str, params: SimParams) -> DCResult:
        return self.build_dc(*self._run(circuit_path))

    def ac(self, circuit_path: str, params: SimParams) -> ACResult:
        return self.build_ac(*self._run(circuit_path))

    def tran(self, circuit_path: str, params: SimParams) -> TranResult:
        return self.build_tran(*self._run(circuit_path))

    def noise(self, circuit_path: str, params: SimParams) -> NoiseResult:
        raise NotImplementedError("noise(): " + _USE_SHARED)

    def stb(self, circuit_path: str, params: SimParams) -> StabilityResult:
        raise NotImplementedError("stb(): " + _USE_SHARED)

    def corners(self, circuit_path: str, pvt_list: List[PVTCorner]) -> List[CornerResult]:
        raise NotImplementedError("corners(): " + _USE_SHARED)

    def mc(self, circuit_path: str, n: int, seed: int) -> MonteCarloResult:
        raise NotImplementedError("mc(): " + _USE_SHARED)
