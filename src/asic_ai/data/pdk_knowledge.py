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

    # PVT Corners.
    # SIGN-OFF CONVENTION: a corner is the worst case of ONE thing, so process,
    # supply and temperature must all push the same way.
    #   SS = slow silicon, LOW supply,  HOT  (125 C)  -> slowest / least drive
    #   FF = fast silicon, HIGH supply, COLD (-40 C)  -> fastest / most drive
    # SS and FF used to carry each other's temperature, which cancelled part of
    # the corner spread against itself and understated it.
    "corners": {
        "tt": {"process": "typical", "voltage": 1.8, "temperature": 27},
        "ss": {"process": "slow", "voltage": 1.62, "temperature": 125},
        "ff": {"process": "fast", "voltage": 1.98, "temperature": -40},
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
        "ss": {"process": "slow", "voltage": 2.97, "temperature": 125},
        "ff": {"process": "fast", "voltage": 3.63, "temperature": -40},
    },
}

# TSMC 65nm CRN65GPLUS PDK (real Cadence PDK on WSL)
TSMC65_PARAMS = {
    "process_name": "tsmc65",
    "foundry": "TSMC",
    "node": "65nm",
    "pdk_id": "CRN65GPLUS",
    "model_type": "BSIM4 v4.5",
    "supply_voltage": 1.0,
    "supply_voltages": {"core": 1.0, "io_25": 2.5, "io_33": 3.3},
    "metal_layers": 9,  # up to m9 (thick top metal)
    "variants": ["GP", "LP"],  # General Purpose, Low Power

    # WSL paths (Alma_EDA)
    "wsl_paths": {
        "pdk_root": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW",
        "models_spectre": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/models/spectre",
        "models_hspice": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/models/hspice",
        "techfile": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/techfile",
        "calibre": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/Calibre",
        "assura": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/Assura",
        "cds_lib": "/opt/eda/PDK/CRN65GPNEW/CRN65GPNEW/cds.lib",
        "alt_pdk": "/opt/eda/PDK/65/CMOS/GP/pdk",
        "alt_lp_pdk": "/opt/eda/PDK/65/CMOS/LP/pdk",
        "stclib_gp": "/opt/eda/PDK/65/CMOS/GP/stclib",
        "stclib_lp": "/opt/eda/PDK/65/CMOS/LP/stclib",
        "iolib": "/opt/eda/PDK/65/iolib",
    },

    # Spectre model include
    "spectre_model_file": "crn65gplus_2d5_lk_v1d0.scs",
    "spectre_corner_files": {
        "core": "cor_std_mos.scs",  # 1.0V core (sections tt, ss, ff, sf, fs)
        "io_18": "cor_18.scs",      # 1.8V thick oxide (sections tt_18 ...)
        "io_25": "cor_25.scs",      # 2.5V I/O
        "io_33": "cor_33.scs",      # 3.3V I/O
        "hvt": "cor_hvt.scs",       # High-Vt
        "lvt": "cor_lvt.scs",       # Low-Vt
        "std_mos": "cor_std_mos.scs",
        "res": "cor_res.scs",
        "mim": "cor_mim.scs",
        "dio": "cor_dio_18.scs",
        "bip": "cor_bip_npn.scs",
        "rfmos": "cor_rfmos.scs",
        "rfind": "cor_rfind.scs",
        "rfmim": "cor_rfmim.scs",
    },

    # NMOS core 1.0V.
    # "model" is the internal .model name; "subckt" is what designers actually
    # instantiate (X instance). The bare .model does resolve in ngspice, but it
    # bypasses the wrapper's computed ad/as/pd/ps and mismatch, so junction
    # caps and therefore every AC/transient result would be wrong.
    # NOTE: the vth0/kp/avt figures below are ENGINEERING APPROXIMATIONS for
    # analytic sizing, not values read out of the foundry deck. Do not treat
    # them as PDK data and do not replace them with deck values (NDA).
    "nmos": {
        "model": "nch",
        "subckt": "nch_mac",
        "vth0": 0.42,  # V (typical, SVT)
        "vth0_range": (0.36, 0.48),  # (ff, ss)
        "kp": 350e-6,  # A/V^2
        "mu_n": 400,  # cm^2/Vs
        "w_min": 0.12e-6,
        "l_min": 0.06e-6,
        "avt": 3.5e-3,  # V*um
        "abeta": 0.6,  # %*um
    },

    # PMOS core 1.0V (see the NMOS note above about approximate values).
    "pmos": {
        "model": "pch",
        "subckt": "pch_mac",
        "vth0": -0.39,  # V (typical, SVT)
        "vth0_range": (-0.45, -0.33),
        "kp": 120e-6,  # A/V^2
        "mu_p": 100,  # cm^2/Vs
        "w_min": 0.12e-6,
        "l_min": 0.06e-6,
        "avt": 4.5e-3,
        "abeta": 0.8,
    },

    # Threshold variants
    "nmos_hvt": {"model": "nch_hvt", "subckt": "nch_hvt_mac", "vth0": 0.55},
    "nmos_lvt": {"model": "nch_lvt", "subckt": "nch_lvt_mac", "vth0": 0.32},
    "pmos_hvt": {"model": "pch_hvt", "subckt": "pch_hvt_mac", "vth0": -0.52},
    "pmos_lvt": {"model": "pch_lvt", "subckt": "pch_lvt_mac", "vth0": -0.30},

    # I/O devices. These are thick-oxide and BIN OUT at core gate lengths:
    # nch_18_mac needs L >= ~0.28 um, nch_33_mac / nch_25od33_mac L >= ~0.5 um.
    # Handing an IO device a 60 nm gate is a hard parse failure.
    "nmos_25": {"model": "nch_25", "subckt": "nch_25_mac", "vth0": 0.55, "vdd": 2.5,
                "l_min": 0.28e-6, "w_min": 0.5e-6},
    "pmos_25": {"model": "pch_25", "subckt": "pch_25_mac", "vth0": -0.55, "vdd": 2.5,
                "l_min": 0.28e-6, "w_min": 0.5e-6},

    # RF devices
    "rf_devices": {
        "nmos_rf": "nmos_rf",
        "pmos_rf": "pmos_rf",
        "nmos_rf_25": "nmos_rf_25",
    },

    # Passive devices
    "resistors": {
        "rpoly": {"model": "rppoly", "sheet_r": 7.8, "unit": "ohm/sq"},
        "rpoly_wo": {"model": "rppolywo", "sheet_r": 325, "unit": "ohm/sq"},
        "rnpoly": {"model": "rnpoly", "sheet_r": 7.8, "unit": "ohm/sq"},
        "rnwell": {"model": "rnwsti", "sheet_r": 600, "unit": "ohm/sq"},
        "rnod": {"model": "rnod", "sheet_r": 85, "unit": "ohm/sq"},
        "rpod": {"model": "rpod", "sheet_r": 150, "unit": "ohm/sq"},
    },
    "capacitors": {
        "mim": {"model": "mimcap_um_rf", "density": 2.0, "unit": "fF/um^2"},
        "mim_udc": {"model": "mimcap_udc", "density": 1.0, "unit": "fF/um^2"},
        "moscap": {"model": "nmoscap", "density": 10.0, "unit": "fF/um^2"},
        "moscap_rf": {"model": "moscap_rf", "density": 10.0, "unit": "fF/um^2"},
    },
    "inductors": {
        "spiral_std": "spiral_std_mu_z",
        "spiral_sym": "spiral_sym_ct_mu_z",
    },
    "varactors": {
        "mosvar": "xjvar",
        "mosvar_nw": "xjvar_nw",
    },

    # Corners.
    # "section" is the HSPICE .lib section name for the 1.0V CORE devices,
    # which are the BARE corner names. The *_18 sections are the 1.8V
    # thick-oxide family, NOT the core: asking for tt_18 at 1.0V with a 60nm
    # gate is a hard parse failure ("could not find a valid modelname").
    "corners": {
        "tt": {"process": "typical", "voltage": 1.0, "temperature": 27, "section": "TT"},
        "ss": {"process": "slow", "voltage": 0.9, "temperature": 125, "section": "SS"},
        "ff": {"process": "fast", "voltage": 1.1, "temperature": -40, "section": "FF"},
        "sf": {"process": "slow_n_fast_p", "voltage": 1.0, "temperature": 27, "section": "SF"},
        "fs": {"process": "fast_n_slow_p", "voltage": 1.0, "temperature": 27, "section": "FS"},
    },

    # Sections for the thick-oxide families, kept separate so they are not
    # confused with the core corners above.
    "corner_sections_io": {
        "1v8": {"tt": "TT_18", "ss": "SS_18", "ff": "FF_18", "sf": "SF_18", "fs": "FS_18"},
        "2v5": {"tt": "TT_25", "ss": "SS_25", "ff": "FF_25", "sf": "SF_25", "fs": "FS_25"},
        "3v3": {"tt": "TT_33", "ss": "SS_33", "ff": "FF_33", "sf": "SF_33", "fs": "FS_33"},
    },
    # Statistical sections. "stat" defines the global process parameters that
    # "MC" consumes, so both are needed and stat must come first.
    "mc_sections": ["stat", "MC"],

    # ngspice HSPICE-compatibility mode required by this deck. "ps" must NOT
    # be used: it downgrades ".lib file section" to a plain include.
    "ngspice_ngbehavior": "hsa",

    # Design rules
    "design_rules": {
        "poly_width_min": 0.06e-6,
        "poly_spacing_min": 0.14e-6,
        "metal1_width_min": 0.09e-6,
        "metal1_spacing_min": 0.09e-6,
        "via1_size": 0.09e-6,
    },

    # Verification tools
    "drc_deck": "Calibre",
    "lvs_deck": "Calibre",
    "extraction": "Assura",
}


def get_pdk_params(pdk_name: str) -> dict:
    """Get PDK parameters by name."""
    pdks = {
        "sky130": SKY130_PARAMS,
        "gf180mcu": GF180MCU_PARAMS,
        "tsmc65": TSMC65_PARAMS,
        "tsmc65gp": TSMC65_PARAMS,  # alias
        "crn65gp": TSMC65_PARAMS,   # alias
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


# ==========================================
# Simulator deck plumbing (paths only, no model data)
# ==========================================
# The functions below are thin re-exports of asic_ai.adapters.pdk_deck, kept
# here so that PDK-related lookups have a single obvious entry point. The
# import is local to avoid a package-level cycle.
#
# NDA: these resolve a foundry model deck BY PATH. No parameter value from any
# proprietary deck is stored in this repository, and none may be added. When
# the deck is absent every function degrades to None / [] / False so the repo
# stays fully functional without it.

def get_pdk_lib_lines(pdk_name: str, corner: str = "tt", mc: bool = False) -> list:
    """Exact .lib/.param lines to prepend to a netlist for a PDK and corner.

    Returns [] when the model deck is not installed on this machine.

    Args:
        pdk_name: PDK id, e.g. "tsmc65".
        corner: process corner name, e.g. "tt", "ss", "ff", "sf", "fs".
        mc: when True, include the statistical sections instead of the corner
            section, for Monte Carlo.
    """
    from asic_ai.adapters.pdk_deck import lib_lines
    return lib_lines(pdk_name, corner=corner, mc=mc)


def get_pdk_corner_section(pdk_name: str, corner: str) -> str | None:
    """Library section name for a corner, e.g. ("tsmc65", "ss") -> "SS"."""
    from asic_ai.adapters.pdk_deck import corner_section
    return corner_section(pdk_name, corner)


def is_pdk_deck_available(pdk_name: str) -> bool:
    """True when this PDK's simulator model deck is installed and reachable."""
    from asic_ai.adapters.pdk_deck import pdk_available
    return pdk_available(pdk_name)


def describe_pdk_deck(pdk_name: str) -> dict:
    """Summary of a PDK's deck plumbing. Never includes model parameter values."""
    from asic_ai.adapters.pdk_deck import describe
    return describe(pdk_name)
