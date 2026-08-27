import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

from asic_ai.tool_interface.schema import (
    DCResult, ACResult, TranResult, NoiseResult, StabilityResult,
    CornerResult, MonteCarloResult, SimParams, PVTCorner
)
from asic_ai.tool_interface.sim import SimulatorInterface
from asic_ai.tool_interface.pdk import PDKProvider

@dataclass
class AdapterConfig:
    binary_path: str
    work_dir: str
    timeout: int = 60
    seed: Optional[int] = None

class SimulatorAdapter(SimulatorInterface, ABC):
    """Abstract base class for all simulator adapters."""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def _run_subprocess(self, cmd: List[str], cwd: Optional[Path] = None, 
                        env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Utility for running subprocesses deterministically with timeout."""
        actual_cwd = cwd if cwd else self.work_dir
        actual_env = os.environ.copy()
        if env:
            actual_env.update(env)
            
        try:
            return subprocess.run(
                cmd,
                cwd=str(actual_cwd),
                env=actual_env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Simulation timed out after {self.config.timeout}s: {e}") from e

class PDKAdapter(PDKProvider, ABC):
    """Abstract base class for PDK providers."""
    pass
