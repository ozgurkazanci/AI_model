import tempfile
import subprocess
import struct
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.tool_interface.schema import (
    DCResult, ACResult, TranResult, NoiseResult, StabilityResult,
    CornerResult, MonteCarloResult, SimParams, PVTCorner
)

class NgspiceAdapter(SimulatorAdapter):
    """ngspice adapter for circuit simulation."""
    
    def _write_deck(self, circuit_path: str, commands: str) -> Path:
        deck_path = self.work_dir / "sim.sp"
        with open(circuit_path, 'r') as f:
            circuit_content = f.read()
            
        with open(deck_path, 'w') as f:
            f.write(circuit_content)
            f.write("\n")
            if self.config.seed is not None:
                f.write(f".set seed={self.config.seed}\n")
            f.write(commands)
            f.write("\n.end\n")
            
        return deck_path
        
    def _parse_raw(self, raw_file: Path) -> Dict[str, Any]:
        """Parses ngspice binary raw format into structured data."""
        with open(raw_file, 'rb') as f:
            header = {}
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line == "Binary:":
                    break
                if ":" in line:
                    key, val = line.split(":", 1)
                    header[key.strip()] = val.strip()
                    
            if "No. Variables" not in header or "No. Points" not in header:
                return {}
                
            num_vars = int(header["No. Variables"])
            num_points = int(header["No. Points"])
            
            variables = []
            for i in range(num_vars):
                var_line = f.readline().decode('utf-8', errors='ignore').strip()
                parts = var_line.split()
                if len(parts) >= 3:
                    variables.append(parts[1]) # Name
                    
            data = {var: [] for var in variables}
            
            # Simple float64 parsing
            for p in range(num_points):
                for v in range(num_vars):
                    bytes_data = f.read(8)
                    if not bytes_data:
                        break
                    val = struct.unpack('d', bytes_data)[0]
                    data[variables[v]].append(val)
                    
            return data

    def dc(self, circuit_path: str, params: SimParams) -> DCResult:
        commands = ".control\nrun\nwrite sim.raw\n.endc"
        deck = self._write_deck(circuit_path, commands)
        
        cmd = [self.config.binary_path, "-b", "-r", "sim.raw", str(deck)]
        res = self._run_subprocess(cmd)
        
        if res.returncode != 0 or "failed" in res.stderr.lower():
            raise RuntimeError(f"ngspice failed: {res.stderr}")
            
        raw_data = self._parse_raw(self.work_dir / "sim.raw")
        return DCResult(sweep_var="v1", sweep_values=[], outputs=raw_data)
        
    def ac(self, circuit_path: str, params: SimParams) -> ACResult:
        commands = ".control\nrun\nwrite sim.raw\n.endc"
        deck = self._write_deck(circuit_path, commands)
        
        cmd = [self.config.binary_path, "-b", "-r", "sim.raw", str(deck)]
        res = self._run_subprocess(cmd)
        
        if res.returncode != 0 or "failed" in res.stderr.lower():
            raise RuntimeError(f"ngspice failed: {res.stderr}")
            
        raw_data = self._parse_raw(self.work_dir / "sim.raw")
        return ACResult(frequencies=[], magnitudes={}, phases={})
        
    def tran(self, circuit_path: str, params: SimParams) -> TranResult:
        commands = ".control\nrun\nwrite sim.raw\n.endc"
        deck = self._write_deck(circuit_path, commands)
        
        cmd = [self.config.binary_path, "-b", "-r", "sim.raw", str(deck)]
        res = self._run_subprocess(cmd)
        
        if res.returncode != 0 or "failed" in res.stderr.lower():
            raise RuntimeError(f"ngspice failed: {res.stderr}")
            
        raw_data = self._parse_raw(self.work_dir / "sim.raw")
        return TranResult(time=[], outputs=raw_data)

    def noise(self, circuit_path: str, params: SimParams) -> NoiseResult:
        return NoiseResult(frequencies=[], node_noise={}, total_input_noise=0.0, total_output_noise=0.0)
        
    def stb(self, circuit_path: str, params: SimParams) -> StabilityResult:
        return StabilityResult(frequencies=[], loop_gain_mag=[], loop_gain_phase=[], phase_margin=0.0, gain_margin=0.0)

    def corners(self, circuit_path: str, corners: List[PVTCorner], params: SimParams) -> CornerResult:
        return CornerResult(corners=corners, results={})

    def mc(self, circuit_path: str, iterations: int, params: SimParams) -> MonteCarloResult:
        return MonteCarloResult(iterations=iterations, results={})
