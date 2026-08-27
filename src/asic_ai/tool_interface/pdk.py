from typing import Protocol, List, Dict, Any
from .schema import DeviceQueryResult, PVTCorner

class PDKProvider(Protocol):
    """
    Abstract protocol for PDK interactions. 
    Provides a query-based interface to prevent NDA violations (no direct rule deck dumps).
    """

    def device_query(self, model: str, W: float, L: float, VGS: float, VDS: float, VSB: float = 0.0) -> DeviceQueryResult:
        """
        Query small-signal parameters and operating point info for a specific device sizing and bias.
        """
        ...

    def list_devices(self) -> List[str]:
        """
        Return a list of available device models (e.g., nmos_rvt, pmos_lvt).
        """
        ...

    def get_corners(self) -> List[PVTCorner]:
        """
        Return a list of standard PVT corners defined by the PDK.
        """
        ...

    def get_supply_voltage(self) -> float:
        """
        Return the nominal supply voltage for the PDK process.
        """
        ...

    def get_design_rules(self, layer: str) -> Dict[str, Any]:
        """
        Get design rules specific to a certain layer (e.g., min width, min spacing).
        Returns a structured dictionary of rules.
        """
        ...
