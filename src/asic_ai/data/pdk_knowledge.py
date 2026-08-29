"""sky130 PDK knowledge base for SFT data generation.

Contains real device parameters, design rules, and process info
for Google/SkyWater sky130 open-source PDK.

Used by SFT generators to create process-aware training examples.
"""
from __future__ import annotations

# sky130 Process Parameters
SKY130_PARAMS = {
    "process_name": "sky130",
    "foundry": "SkyWater Technology",
    "node": "130nm",
    "supply_voltage": 1.8,
    "metal_layers": 5,  # li, m1, m2, m3, m4, m5
    "poly_pitch": 0.48e-6,  # um

    # NMOS (sky130_fd_pr__nfet_01v8)
    "nmos": {
        "model": "sky130_fd_pr__nfet_01v8",
        "vth0": 0.49,  # V (typical)
        "vth0_range": (0.42, 0.56),  # (ss, ff)
        "mu_n": 450,  # cm^2/Vs
        "cox": 8.6e-3,  # F/m^2
        "kp": 170e-6,  # A/V^2 (mu*Cox)
        "lambda_n": 0.1,  # 1/V (for L=0.15u)
        "w_min": 0.42e-6,
        "l_min": 0.15e-6,
        "avt": 4.5e-3,  # mV*um (threshold mismatch)
        "abeta": 0.8,  # %*um (beta mismatch)
    },

    # PMOS (sky130_fd_pr__pfet_01v8)
    "pmos": {
        "model": "sky130_fd_pr__pfet_01v8",
        "vth0": -0.54,  # V (typical)
        "vth0_range": (-0.62, -0.46),  # (ss, ff)
        "mu_p": 120,  # cm^2/Vs
        "cox": 8.6e-3,  # F/m^2
        "kp": 45e-6,  # A/V^2
        "lambda_p": 0.15,  # 1/V (for L=0.15u)
        "w_min": 0.42e-6,
        "l_min": 0.15e-6,
        "avt": 6.0e-3,  # mV*um
        "abeta": 1.2,  # %*um
    },

    # High-Vt NMOS (sky130_fd_pr__nfet_01v8_lvt)
    "nmos_lvt": {
        "model": "sky130_fd_pr__nfet_01v8_lvt",
        "vth0": 0.35,
        "w_min": 0.42e-6,
        "l_min": 0.15e-6,
    },

    # Resistors
    "resistors": {
        "poly_res": {"sheet_r": 48.2, "unit": "ohm/sq", "model": "sky130_fd_pr__res_xhigh_po"},
        "nwell_res": {"sheet_r": 900, "unit": "ohm/sq"},
        "metal1_res": {"sheet_r": 0.125, "unit": "ohm/sq"},
        "metal2_res": {"sheet_r": 0.125, "unit": "ohm/sq"},
    },

    # Capacitors
    "capacitors": {
        "mim_cap": {"density": 2.0, "unit": "fF/um^2", "model": "sky130_fd_pr__cap_mim_m3_1"},
        "mos_cap": {"density": 8.6, "unit": "fF/um^2"},
        "metal_cap": {"density": 0.04, "unit": "fF/um^2"},
    },

    # PVT Corners
    "corners": {
        "tt": {"process": "typical", "voltage": 1.8, "temperature": 27},
        "ss": {"process": "slow", "voltage": 1.62, "temperature": -40},
        "ff": {"process": "fast", "voltage": 1.98, "temperature": 125},
        "sf": {"process": "slow_n_fast_p", "voltage": 1.8, "temperature": 27},
        "fs": {"process": "fast_n_slow_p", "voltage": 1.8, "temperature": 27},
    },

    # Design Rules (key rules)
    "design_rules": {
        "poly_width_min": 0.15e-6,
        "poly_spacing_min": 0.21e-6,
        "metal1_width_min": 0.14e-6,
        "metal1_spacing_min": 0.14e-6,
        "via1_size": 0.15e-6,
        "nwell_spacing": 1.27e-6,
    },

    # ESD
    "esd": {
        "hbm": 2000,  # V
        "cdm": 500,  # V
    },
}

# gf180mcu Process Parameters
GF180MCU_PARAMS = {
    "process_name": "gf180mcu",
    "foundry": "GlobalFoundries",
    "node": "180nm",
    "supply_voltage": 3.3,
    "metal_layers": 5,

    "nmos": {
        "model": "nfet_03v3",
        "vth0": 0.56,
        "kp": 200e-6,
        "w_min": 0.22e-6,
        "l_min": 0.28e-6,
    },
    "pmos": {
        "model": "pfet_03v3",
        "vth0": -0.62,
        "kp": 65e-6,
        "w_min": 0.22e-6,
        "l_min": 0.28e-6,
    },

    "corners": {
        "tt": {"process": "typical", "voltage": 3.3, "temperature": 27},
        "ss": {"process": "slow", "voltage": 2.97, "temperature": -40},
        "ff": {"process": "fast", "voltage": 3.63, "temperature": 125},
    },
}


def get_pdk_params(pdk_name: str) -> dict:
    """Get PDK parameters by name."""
    pdks = {
        "sky130": SKY130_PARAMS,
        "gf180mcu": GF180MCU_PARAMS,
    }
    if pdk_name not in pdks:
        raise KeyError(f"Unknown PDK: {pdk_name}. Available: {list(pdks.keys())}")
    return pdks[pdk_name]


def calculate_gm(w: float, l: float, id: float, pdk: str = "sky130", device: str = "nmos") -> float:
    """Calculate transconductance: gm = 2*Id/Vov = sqrt(2*kp*W/L*Id)."""
    params = get_pdk_params(pdk)[device]
    kp = params["kp"]
    import math
    return math.sqrt(2 * kp * (w / l) * id)


def calculate_mismatch_sigma(w: float, l: float, pdk: str = "sky130", device: str = "nmos") -> float:
    """Calculate Vth mismatch sigma in volts: sigma_Vth = Avt / sqrt(W*L).

    Args:
        w: Width in meters (e.g., 10e-6 for 10 um)
        l: Length in meters (e.g., 0.5e-6 for 0.5 um)
    Returns:
        sigma_Vth in volts
    """
    params = get_pdk_params(pdk)[device]
    avt = params["avt"]  # in V*um (e.g., 4.5e-3 V*um)
    import math
    # Convert W, L from meters to microns for matching Avt units
    w_um = w * 1e6
    l_um = l * 1e6
    return avt / math.sqrt(w_um * l_um)  # result in volts
