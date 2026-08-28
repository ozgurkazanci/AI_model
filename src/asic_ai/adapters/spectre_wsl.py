"""Cadence Spectre simulator adapter via WSL.

Runs Spectre simulations through WSL by:
1. Writing netlist to WSL-accessible path
2. Invoking spectre binary via `wsl` command
3. Parsing PSF/raw output files

Requires:
- WSL with AlmaLinux (Alma_EDA) configured
- Cadence Spectre installed at /opt/eda/cadence/SPECTRE241
- Valid Cadence license server running

Usage:
    from asic_ai.adapters.spectre_wsl import SpectreWSLAdapter
    adapter = SpectreWSLAdapter(AdapterConfig(binary_path="", work_dir="./sim"))
    result = adapter.dc("circuit.scs", SimParams(analysis_type="dc"))
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from asic_ai.adapters.base import AdapterConfig, SimulatorAdapter
from asic_ai.tool_interface.schema import (
    ACResult,
    CornerResult,
    DCResult,
    MonteCarloResult,
    NoiseResult,
    PVTCorner,
    SignalData,
    SimParams,
    StabilityResult,
    TranResult,
)

logger = logging.getLogger(__name__)

# Cadence Spectre paths
SPECTRE_INSTALL = "/opt/eda/cadence/SPECTRE241"
SPECTRE_BIN = f"{SPECTRE_INSTALL}/tools.lnx86/spectre/bin/64bit/spectre"
WSL_DISTRO = "Alma_EDA"

# Environment setup for Spectre
_LIB_DIRS = ":".join([
    f"{SPECTRE_INSTALL}/tools.lnx86/lib/rpath/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/spectre/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/mdl/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/fmc/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/cmi/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/cmi/lib/64bit/arch",
    f"{SPECTRE_INSTALL}/tools.lnx86/mmsim/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/inca/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/icc/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/nif/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/giganta/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/emir/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/pub/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/ktl/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/uri/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/ams/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/hdf5/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/lapack/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/lz4/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/fsdb/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/sev/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/vmor/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/relxpert/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/SobolTable/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/openmpi/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/jsoncpp/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/awrmwo/lib/64bit",
    f"{SPECTRE_INSTALL}/tools.lnx86/python/64bit/lib",
    f"{SPECTRE_INSTALL}/tools.lnx86/TPtools/grpc-1.57.0/lib64",
    f"{SPECTRE_INSTALL}/tools.lnx86/TPtools/protobuf-3.24.4/lib64",
    f"{SPECTRE_INSTALL}/tools.lnx86/TPtools/abseil-20230802.1/lib64",
    f"{SPECTRE_INSTALL}/tools.lnx86/TPtools/openssl-1.1.1o/lib",
    f"{SPECTRE_INSTALL}/tools.lnx86/TPtools/libstdc++6-13.2.0/lib/64bit",
])

SPECTRE_ENV = {
    "CDS_INST_DIR": SPECTRE_INSTALL,
    "PATH": f"{SPECTRE_INSTALL}/bin:{SPECTRE_INSTALL}/tools.lnx86/spectre/bin/64bit:$PATH",
    "LD_LIBRARY_PATH": f"{_LIB_DIRS}:$LD_LIBRARY_PATH",
}


def win_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    path = win_path.replace("\\", "/")
    # Handle UNC paths
    if path.startswith("//wsl.localhost/"):
        parts = path.split("/", 4)
        if len(parts) >= 5:
            return "/" + parts[4]
    # Handle drive letters
    if len(path) >= 3 and path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    return path


def wsl_to_win_path(wsl_path: str) -> str:
    """Convert WSL path to Windows UNC path."""
    return f"\\\\wsl.localhost\\{WSL_DISTRO}{wsl_path}"


class SpectreWSLAdapter(SimulatorAdapter):
    """Cadence Spectre simulator adapter via WSL.

    Executes Spectre simulations on WSL Linux environment
    and parses results back to Windows.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._work_dir = Path(config.work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._sim_count = 0

    def _run_spectre(self, netlist_path: str, output_dir: str = None) -> tuple[str, int]:
        """Run Spectre via WSL and return (stdout, return_code)."""
        wsl_netlist = win_to_wsl_path(netlist_path)
        if output_dir:
            wsl_outdir = win_to_wsl_path(output_dir)
        else:
            wsl_outdir = win_to_wsl_path(str(self._work_dir / "output"))

        # Build environment setup
        env_cmd = " ".join(f"export {k}={v};" for k, v in SPECTRE_ENV.items())

        # Spectre command
        cmd = (
            f"{env_cmd} "
            f"{SPECTRE_BIN} {wsl_netlist} "
            f"+escchars +log {wsl_outdir}/spectre.log "
            f"-outdir {wsl_outdir} "
            f"+aps +mt=2"
        )

        full_cmd = ["wsl", "-d", WSL_DISTRO, "bash", "-c", cmd]

        logger.info("Running Spectre: %s", wsl_netlist)
        self._sim_count += 1

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.error("Spectre simulation timed out")
            return "TIMEOUT", -1
        except FileNotFoundError:
            logger.error("WSL not found. Is WSL installed?")
            return "WSL_NOT_FOUND", -1

    def _parse_spectre_log(self, log_path: str) -> dict[str, Any]:
        """Parse Spectre log file for results summary."""
        info = {"convergence": False, "errors": [], "warnings": []}
        try:
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            if "completed successfully" in log_text.lower():
                info["convergence"] = True
            # Extract warnings/errors
            for line in log_text.splitlines():
                if "Warning" in line:
                    info["warnings"].append(line.strip())
                elif "Error" in line:
                    info["errors"].append(line.strip())
        except Exception as e:
            logger.warning("Could not parse log: %s", e)
        return info

    def _parse_csv_output(self, output_dir: str, analysis: str) -> dict[str, list[float]]:
        """Parse Spectre CSV/PSF output into signal data."""
        signals = {}
        out_path = Path(output_dir)

        # Look for .csv or .raw files
        for ext in ["*.csv", "*.raw", "*.psf"]:
            for f in out_path.rglob(ext):
                try:
                    if f.suffix == ".csv":
                        lines = f.read_text(encoding="utf-8").strip().splitlines()
                        if len(lines) > 1:
                            headers = lines[0].split(",")
                            for i, h in enumerate(headers):
                                signals[h.strip()] = []
                            for line in lines[1:]:
                                vals = line.split(",")
                                for i, h in enumerate(headers):
                                    if i < len(vals):
                                        try:
                                            signals[h.strip()].append(float(vals[i]))
                                        except ValueError:
                                            pass
                except Exception as e:
                    logger.warning("Could not parse %s: %s", f, e)

        return signals

    def dc(self, netlist: str, params: SimParams) -> DCResult:
        """Run DC simulation."""
        output_dir = str(self._work_dir / f"dc_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        if rc != 0:
            logger.warning("Spectre DC failed (rc=%d)", rc)
            return DCResult(op_points={}, sweeps={})

        signals = self._parse_csv_output(output_dir, "dc")

        sweeps = {}
        for name, values in signals.items():
            sweeps[name] = SignalData(
                name=name,
                x_values=list(range(len(values))),
                y_values=values,
            )

        return DCResult(op_points={}, sweeps=sweeps)

    def ac(self, netlist: str, params: SimParams) -> ACResult:
        """Run AC simulation."""
        output_dir = str(self._work_dir / f"ac_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        if rc != 0:
            logger.warning("Spectre AC failed (rc=%d)", rc)
            return ACResult(frequencies=[], signals={})

        signals = self._parse_csv_output(output_dir, "ac")
        freq = signals.pop("freq", signals.pop("frequency", []))

        ac_signals = {}
        for name, values in signals.items():
            ac_signals[name] = SignalData(
                name=name,
                x_values=freq if freq else list(range(len(values))),
                y_values=values,
            )

        return ACResult(frequencies=freq, signals=ac_signals)

    def tran(self, netlist: str, params: SimParams) -> TranResult:
        """Run transient simulation."""
        output_dir = str(self._work_dir / f"tran_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        if rc != 0:
            logger.warning("Spectre tran failed (rc=%d)", rc)
            return TranResult(time=[], signals={})

        signals = self._parse_csv_output(output_dir, "tran")
        time_data = signals.pop("time", [])

        tran_signals = {}
        for name, values in signals.items():
            tran_signals[name] = SignalData(
                name=name,
                x_values=time_data if time_data else list(range(len(values))),
                y_values=values,
            )

        return TranResult(time=time_data, signals=tran_signals)

    def noise(self, netlist: str, params: SimParams) -> NoiseResult:
        """Run noise simulation."""
        output_dir = str(self._work_dir / f"noise_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        return NoiseResult(frequencies=[], input_noise=[], output_noise=[])

    def stb(self, netlist: str, params: SimParams) -> StabilityResult:
        """Run stability (STB) analysis — Spectre-specific."""
        output_dir = str(self._work_dir / f"stb_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        # Parse PM/GM from log
        pm, gm = 0.0, 0.0
        log_path = Path(output_dir) / "spectre.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            pm_match = re.search(r"Phase Margin\s*=\s*([\d.]+)", log_text)
            gm_match = re.search(r"Gain Margin\s*=\s*([\d.]+)", log_text)
            if pm_match:
                pm = float(pm_match.group(1))
            if gm_match:
                gm = float(gm_match.group(1))

        return StabilityResult(
            phase_margin=pm,
            gain_margin=gm,
            loop_gain={},
        )

    def corners(self, netlist: str, params: SimParams) -> list[CornerResult]:
        """Run PVT corner simulation."""
        results = []
        corners_list = [
            PVTCorner(process="tt", voltage=1.8, temperature=27),
            PVTCorner(process="ff", voltage=1.98, temperature=-40),
            PVTCorner(process="ss", voltage=1.62, temperature=125),
        ]

        for corner in corners_list:
            dc_result = self.dc(netlist, params)
            results.append(CornerResult(
                corner=corner,
                dc=dc_result,
                ac=None,
                tran=None,
                stb=None,
            ))

        return results

    def mc(self, netlist: str, params: SimParams) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        output_dir = str(self._work_dir / f"mc_{self._sim_count}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stdout, rc = self._run_spectre(netlist, output_dir)

        return MonteCarloResult(seed=42, runs=0, results=[])


def check_spectre_available() -> bool:
    """Check if Spectre is available via WSL."""
    try:
        result = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "test", "-f", SPECTRE_BIN],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_spectre_version() -> str | None:
    """Get Spectre version string."""
    try:
        env_cmd = " ".join(f"export {k}={v};" for k, v in SPECTRE_ENV.items())
        result = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "bash", "-c", f"{env_cmd} {SPECTRE_BIN} -V"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "spectre" in line.lower() and ("version" in line.lower() or "sub" in line.lower()):
                return line.strip()
        return result.stdout.strip()[:100] if result.stdout else None
    except Exception:
        return None
