import random
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from asic_ai.adapters.base import SimulatorAdapter, AdapterConfig
from asic_ai.tool_interface.schema import (
    DCResult, ACResult, TranResult, NoiseResult, StabilityResult,
    CornerResult, MonteCarloResult, SimParams, PVTCorner, SignalData
)

class MockSimulatorAdapter(SimulatorAdapter):
    """Mock simulator adapter for end-to-end testing without ngspice.
    Returns realistic but synthetic simulation results based on circuit topology hints.
    """
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.random = random.Random(config.seed if config.seed is not None else 42)

    def _parse_netlist_hints(self, netlist: str) -> Dict[str, float]:
        """Simple parser to extract device counts and W/L ratios for synthetic results."""
        hints = {
            "transistors": len(re.findall(r'^[mM]\w+', netlist, re.MULTILINE)),
            "resistors": len(re.findall(r'^[rR]\w+', netlist, re.MULTILINE)),
            "capacitors": len(re.findall(r'^[cC]\w+', netlist, re.MULTILINE)),
            "w_sum": 0.0,
            "l_sum": 0.0
        }
        
        # Simple extraction of W and L if present
        w_matches = re.findall(r'[wW]\s*=\s*([\d\.]+)[um]', netlist)
        l_matches = re.findall(r'[lL]\s*=\s*([\d\.]+)[um]', netlist)
        
        if w_matches:
            hints["w_sum"] = sum(float(w) for w in w_matches)
        if l_matches:
            hints["l_sum"] = sum(float(l) for l in l_matches)
            
        return hints

    def dc(self, circuit_path: str, params: SimParams) -> DCResult:
        with open(circuit_path, 'r') as f:
            netlist = f.read()
            
        hints = self._parse_netlist_hints(netlist)
        base_current = 10e-6 * (hints["transistors"] + 1)
        # Random variation
        noise = self.random.gauss(0, 0.05) if self.config.seed is None else 0
        
        op_points = {
            "v(out)": 1.2 + noise,
            "i(vdd)": base_current * (1 + noise)
        }
        
        return DCResult(op_points=op_points, sweeps={})

    def ac(self, circuit_path: str, params: SimParams) -> ACResult:
        with open(circuit_path, 'r') as f:
            netlist = f.read()
            
        hints = self._parse_netlist_hints(netlist)
        
        # Determine performance based on W/L if available, else fallback
        w_eff = hints["w_sum"] if hints["w_sum"] > 0 else 10.0
        l_eff = hints["l_sum"] if hints["l_sum"] > 0 else 1.0
        
        # OTA: gain based on W/L ratio, UGB based on current (transistor count proxied)
        gain_db = 40.0 + 10.0 * (w_eff / l_eff)
        gain_db = min(gain_db, 100.0) # Cap at 100dB
        
        ugb_hz = 1e6 * hints["transistors"] * 10
        phase_margin = 60.0 - (hints["transistors"] * 2.0)
        
        # Noise
        noise = self.random.gauss(0, 0.02)
        gain_db *= (1 + noise)
        
        freqs = [1e3, 1e4, 1e5, 1e6, 1e7]
        mags = [gain_db, gain_db - 5, gain_db - 20, 0.0, -20.0]
        phases = [180.0, 135.0, 90.0, phase_margin, 10.0]
        
        signals = {
            "v(out)": SignalData(name="v(out)", x_values=freqs, y_values=mags),
            "phase(out)": SignalData(name="phase(out)", x_values=freqs, y_values=phases)
        }
        
        # Emulate returning the raw structured ACResult
        return ACResult(frequencies=freqs, signals=signals)

    def tran(self, circuit_path: str, params: SimParams) -> TranResult:
        time = [0.0, 1e-9, 2e-9, 3e-9, 4e-9]
        outs = [0.0, 0.5, 0.9, 1.1, 1.2]
        
        signals = {
            "v(out)": SignalData(name="v(out)", x_values=time, y_values=outs)
        }
        
        return TranResult(time=time, signals=signals)

    def noise(self, circuit_path: str, params: SimParams) -> NoiseResult:
        freqs = [1e3, 1e4, 1e5]
        inoise = [1e-8, 1e-9, 1e-10]
        onoise = [1e-6, 1e-7, 1e-8]
        
        return NoiseResult(
            frequencies=freqs,
            input_noise=SignalData(name="in", x_values=freqs, y_values=inoise),
            output_noise=SignalData(name="out", x_values=freqs, y_values=onoise)
        )

    def stb(self, circuit_path: str, params: SimParams) -> StabilityResult:
        return StabilityResult(
            phase_margin=65.0,
            gain_margin=12.0,
            loop_gain=SignalData(name="loop_gain", x_values=[1e6], y_values=[0.0])
        )

    def corners(self, circuit_path: str, pvt_list: List[PVTCorner], params: SimParams) -> List[CornerResult]:
        results = []
        for corner in pvt_list:
            # Apply PVT derating
            derating = 1.0
            if corner.process == "ss":
                derating = 0.8
            elif corner.process == "ff":
                derating = 1.2
            
            # Re-run AC with derating
            base_ac = self.ac(circuit_path, params)
            derated_signals = {}
            for name, sig in base_ac.signals.items():
                derated_signals[name] = SignalData(
                    name=sig.name,
                    x_values=sig.x_values,
                    y_values=[y * derating for y in sig.y_values]
                )
            
            derated_ac = ACResult(frequencies=base_ac.frequencies, signals=derated_signals)
            
            res = CornerResult(
                corner=corner,
                ac=derated_ac
            )
            results.append(res)
            
        return results

    def mc(self, circuit_path: str, iterations: int, params: SimParams) -> MonteCarloResult:
        runs = []
        for _ in range(iterations):
            runs.append({"gain": 60.0 + self.random.gauss(0, 2.0)})
        return MonteCarloResult(seed=self.random.randint(0, 1000), runs=iterations, results=runs)

    def spec_check(self, results: Dict[str, Any], specs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock check results against specifications."""
        checked = {}
        for spec_name, target in specs.items():
            if spec_name in results:
                val = results[spec_name]
                passed = val >= target.get("min", float('-inf')) and val <= target.get("max", float('inf'))
                checked[spec_name] = {"actual": val, "passed": passed}
        return checked
