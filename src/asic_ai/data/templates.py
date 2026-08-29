"""Circuit topology template library.

Provides reusable SPICE netlist templates for common analog and digital
circuit topologies. Used by:
- SFT data generation (initial netlists)
- Agent loop (topology suggestion)
- Perturbation pipeline (base circuits to perturb)

Each template is parameterized with {variables} for device sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CircuitTemplate:
    """A parameterized circuit topology template."""
    id: str
    name: str
    category: str  # analog / digital
    description: str
    netlist: str
    parameters: dict[str, dict[str, Any]]  # name -> {default, min, max, unit}
    typical_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    design_notes: str = ""

    def render(self, **overrides: Any) -> str:
        """Render netlist with parameter values."""
        params = {k: v["default"] for k, v in self.parameters.items()}
        params.update(overrides)
        return self.netlist.format(**params)


# ============================================================
# Analog Templates
# ============================================================

OTA_TWO_STAGE = CircuitTemplate(
    id="ota_2stage",
    name="Two-Stage Miller OTA",
    category="analog",
    description="Two-stage operational transconductance amplifier with Miller compensation.",
    netlist="""\
* Two-Stage Miller OTA
.subckt ota_2stage VDD VSS INP INM OUT VBIAS

* Input differential pair
XM1 net1 INM tail VSS nfet_01v8 W={w1}u L={l1}n m={m1}
XM2 net2 INP tail VSS nfet_01v8 W={w1}u L={l1}n m={m1}

* PMOS active load
XM3 net1 net1 VDD VDD pfet_01v8 W={w3}u L={l3}n m={m3}
XM4 net2 net1 VDD VDD pfet_01v8 W={w3}u L={l3}n m={m3}

* Tail current source
XM5 tail VBIAS VSS VSS nfet_01v8 W={w5}u L={l5}n m={m5}

* Second stage (common-source)
XM6 OUT net2 VDD VDD pfet_01v8 W={w6}u L={l6}n m={m6}
XM7 OUT VBIAS VSS VSS nfet_01v8 W={w7}u L={l7}n m={m7}

* Miller compensation
Cc net2 OUT {cc}p
Rz net2 net_rz {rz}
Crz net_rz OUT 0.1p

.ends ota_2stage""",
    parameters={
        "w1": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "l1": {"default": 180, "min": 180, "max": 1000, "unit": "nm"},
        "m1": {"default": 4, "min": 1, "max": 16, "unit": ""},
        "w3": {"default": 20, "min": 4, "max": 100, "unit": "um"},
        "l3": {"default": 180, "min": 180, "max": 1000, "unit": "nm"},
        "m3": {"default": 4, "min": 1, "max": 16, "unit": ""},
        "w5": {"default": 5, "min": 1, "max": 30, "unit": "um"},
        "l5": {"default": 500, "min": 180, "max": 2000, "unit": "nm"},
        "m5": {"default": 2, "min": 1, "max": 8, "unit": ""},
        "w6": {"default": 40, "min": 10, "max": 200, "unit": "um"},
        "l6": {"default": 180, "min": 180, "max": 1000, "unit": "nm"},
        "m6": {"default": 8, "min": 1, "max": 32, "unit": ""},
        "w7": {"default": 20, "min": 5, "max": 100, "unit": "um"},
        "l7": {"default": 180, "min": 180, "max": 1000, "unit": "nm"},
        "m7": {"default": 8, "min": 1, "max": 32, "unit": ""},
        "cc": {"default": 2, "min": 0.5, "max": 10, "unit": "pF"},
        "rz": {"default": 500, "min": 0, "max": 5000, "unit": "Ohm"},
    },
    typical_specs={
        "dc_gain": {"min": 60, "unit": "dB"},
        "ugb": {"min": 50e6, "unit": "Hz"},
        "phase_margin": {"min": 60, "unit": "deg"},
        "idd": {"max": 500e-6, "unit": "A"},
    },
    design_notes="Start with gm/ID=10 for input pair. Cc ~ 0.25*CL. Rz cancels RHP zero.",
)

FOLDED_CASCODE = CircuitTemplate(
    id="folded_cascode",
    name="Folded Cascode OTA",
    category="analog",
    description="Single-stage folded cascode OTA for high gain with wide input range.",
    netlist="""\
* Folded Cascode OTA
.subckt folded_cascode VDD VSS INP INM OUT VBN VBP

* NMOS input pair
XM1 net1 INM tail VSS nfet_01v8 W={w_in}u L={l_in}n m={m_in}
XM2 net2 INP tail VSS nfet_01v8 W={w_in}u L={l_in}n m={m_in}

* Tail current source
XM_tail tail VBN VSS VSS nfet_01v8 W={w_tail}u L={l_tail}n m=2

* PMOS folding transistors
XM3 net1 VBP VDD VDD pfet_01v8 W={w_fold}u L=180n m={m_fold}
XM4 net2 VBP VDD VDD pfet_01v8 W={w_fold}u L=180n m={m_fold}

* NMOS cascode
XM5 net3 VBN net1 VSS nfet_01v8 W={w_cas}u L=180n m={m_cas}
XM6 OUT  VBN net2 VSS nfet_01v8 W={w_cas}u L=180n m={m_cas}

* PMOS cascode load
XM7 net3 net3 VDD VDD pfet_01v8 W={w_load}u L=180n m={m_cas}
XM8 OUT  net3 VDD VDD pfet_01v8 W={w_load}u L=180n m={m_cas}

.ends folded_cascode""",
    parameters={
        "w_in": {"default": 20, "min": 5, "max": 100, "unit": "um"},
        "l_in": {"default": 360, "min": 180, "max": 1000, "unit": "nm"},
        "m_in": {"default": 4, "min": 1, "max": 16, "unit": ""},
        "w_tail": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "l_tail": {"default": 500, "min": 180, "max": 2000, "unit": "nm"},
        "w_fold": {"default": 15, "min": 3, "max": 60, "unit": "um"},
        "m_fold": {"default": 4, "min": 1, "max": 16, "unit": ""},
        "w_cas": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "m_cas": {"default": 4, "min": 1, "max": 16, "unit": ""},
        "w_load": {"default": 15, "min": 3, "max": 60, "unit": "um"},
    },
    typical_specs={
        "dc_gain": {"min": 70, "unit": "dB"},
        "ugb": {"min": 100e6, "unit": "Hz"},
        "phase_margin": {"min": 65, "unit": "deg"},
    },
    design_notes="Single-stage -> inherently stable. Gain from cascode output impedance.",
)

BANDGAP_REFERENCE = CircuitTemplate(
    id="bandgap_brokaw",
    name="Brokaw Bandgap Reference",
    category="analog",
    description="Brokaw cell bandgap voltage reference generating ~1.2V.",
    netlist="""\
* Brokaw Bandgap Reference
.subckt bandgap VDD VSS VREF

* PNP pair (diode-connected)
Q1 VSS VSS net_e1 pnp_01v8 m=1
Q2 VSS VSS net_e2 pnp_01v8 m={n_ratio}

* Resistors
R1 net_e1 net_r1 {r1}k
R2 net_e2 net_r1 {r2}k
R3 net_r1 VSS {r3}k

* Error amplifier (drives PMOS mirrors)
XM1 net_d1 net_e1 VDD VDD pfet_01v8 W={w_amp}u L=500n m=2
XM2 net_d2 net_e2 VDD VDD pfet_01v8 W={w_amp}u L=500n m=2
XM3 net_d1 net_d1 VSS VSS nfet_01v8 W={w_load}u L=500n m=2
XM4 net_d2 net_d1 VSS VSS nfet_01v8 W={w_load}u L=500n m=2

* Output buffer
XM5 VREF net_d2 VDD VDD pfet_01v8 W={w_buf}u L=180n m=4

.ends bandgap""",
    parameters={
        "n_ratio": {"default": 8, "min": 4, "max": 16, "unit": ""},
        "r1": {"default": 10, "min": 1, "max": 100, "unit": "kOhm"},
        "r2": {"default": 10, "min": 1, "max": 100, "unit": "kOhm"},
        "r3": {"default": 30, "min": 5, "max": 200, "unit": "kOhm"},
        "w_amp": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "w_load": {"default": 5, "min": 1, "max": 30, "unit": "um"},
        "w_buf": {"default": 20, "min": 5, "max": 100, "unit": "um"},
    },
    typical_specs={
        "vref": {"min": 1.15, "max": 1.25, "unit": "V"},
        "tc": {"max": 20, "unit": "ppm/C"},
        "psrr": {"min": 40, "unit": "dB"},
    },
    design_notes="VPTAT = (kT/q)*ln(N)*R1/R2. VCTAT = VBE. Sum at ~1.2V.",
)

LDO_REGULATOR = CircuitTemplate(
    id="ldo_basic",
    name="Basic LDO Regulator",
    category="analog",
    description="Low-dropout voltage regulator with PMOS pass transistor.",
    netlist="""\
* Basic LDO Regulator
.subckt ldo VDD VSS VOUT VREF

* Error amplifier
XM1 net1 VREF net3 VSS nfet_01v8 W={w_ea}u L=360n m=2
XM2 net2 net_fb net3 VSS nfet_01v8 W={w_ea}u L=360n m=2
XM3 net1 net1 VDD VDD pfet_01v8 W={w_load}u L=360n m=2
XM4 net2 net1 VDD VDD pfet_01v8 W={w_load}u L=360n m=2
XM5 net3 Vbn VSS VSS nfet_01v8 W={w_tail}u L=500n m=1

* PMOS pass transistor
XMP VOUT net2 VDD VDD pfet_01v8 W={w_pass}u L={l_pass}n m={m_pass}

* Feedback resistor divider
Rfb1 VOUT net_fb {rfb1}k
Rfb2 net_fb VSS {rfb2}k

* Output cap (off-chip)
Cout VOUT VSS {cout}u

* Bias
Ibias VDD Vbn {ibias}u

.ends ldo""",
    parameters={
        "w_ea": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "w_load": {"default": 15, "min": 3, "max": 60, "unit": "um"},
        "w_tail": {"default": 5, "min": 1, "max": 30, "unit": "um"},
        "w_pass": {"default": 500, "min": 50, "max": 5000, "unit": "um"},
        "l_pass": {"default": 180, "min": 180, "max": 500, "unit": "nm"},
        "m_pass": {"default": 20, "min": 1, "max": 100, "unit": ""},
        "rfb1": {"default": 100, "min": 10, "max": 1000, "unit": "kOhm"},
        "rfb2": {"default": 100, "min": 10, "max": 1000, "unit": "kOhm"},
        "cout": {"default": 1, "min": 0.1, "max": 10, "unit": "uF"},
        "ibias": {"default": 10, "min": 1, "max": 100, "unit": "uA"},
    },
    typical_specs={
        "vout": {"min": 1.05, "max": 1.15, "unit": "V"},
        "dropout": {"max": 200e-3, "unit": "V"},
        "load_reg": {"max": 10e-3, "unit": "V/mA"},
        "psrr": {"min": 40, "unit": "dB"},
        "iq": {"max": 100e-6, "unit": "A"},
    },
    design_notes="Pass transistor W determines dropout and Iload_max. Cc for stability.",
)

CURRENT_MIRROR = CircuitTemplate(
    id="current_mirror_cascode",
    name="Cascode Current Mirror",
    category="analog",
    description="High-output-impedance cascode current mirror.",
    netlist="""\
* Cascode Current Mirror
.subckt cm_cascode VDD VSS IREF IOUT

* Reference side
XM1 net1 net1 VSS VSS nfet_01v8 W={w1}u L={l1}n m={m_ref}
XM3 IREF net_cas net1 VSS nfet_01v8 W={w1}u L={l1}n m={m_ref}

* Mirror side
XM2 net2 net1 VSS VSS nfet_01v8 W={w1}u L={l1}n m={m_out}
XM4 IOUT net_cas net2 VSS nfet_01v8 W={w1}u L={l1}n m={m_out}

* Cascode bias
Vbias_cas net_cas VSS 0.6

.ends cm_cascode""",
    parameters={
        "w1": {"default": 5, "min": 0.5, "max": 50, "unit": "um"},
        "l1": {"default": 500, "min": 180, "max": 2000, "unit": "nm"},
        "m_ref": {"default": 1, "min": 1, "max": 16, "unit": ""},
        "m_out": {"default": 1, "min": 1, "max": 64, "unit": ""},
    },
    typical_specs={
        "ratio_accuracy": {"max": 1, "unit": "%"},
        "output_impedance": {"min": 10e6, "unit": "Ohm"},
        "vmin_out": {"max": 0.4, "unit": "V"},
    },
    design_notes="Ratio = m_out/m_ref. Cascode adds ~gm*ro to output impedance.",
)

COMPARATOR = CircuitTemplate(
    id="comparator_basic",
    name="Basic Comparator",
    category="analog",
    description="Simple differential comparator with rail-to-rail output.",
    netlist="""\
* Basic Comparator
.subckt comparator VDD VSS INP INM OUT

* Differential pair
XM1 net1 INM tail VSS nfet_01v8 W={w_in}u L=180n m=2
XM2 net2 INP tail VSS nfet_01v8 W={w_in}u L=180n m=2

* Active load
XM3 net1 net1 VDD VDD pfet_01v8 W={w_load}u L=180n m=2
XM4 net2 net1 VDD VDD pfet_01v8 W={w_load}u L=180n m=2

* Tail
XM5 tail Vbn VSS VSS nfet_01v8 W={w_tail}u L=500n m=1

* Output inverter (gain stage)
XM6 OUT net2 VDD VDD pfet_01v8 W={w_inv}u L=180n m=4
XM7 OUT net2 VSS VSS nfet_01v8 W={w_inv_n}u L=180n m=2

Ibias VDD Vbn {ibias}u

.ends comparator""",
    parameters={
        "w_in": {"default": 5, "min": 1, "max": 30, "unit": "um"},
        "w_load": {"default": 8, "min": 2, "max": 40, "unit": "um"},
        "w_tail": {"default": 3, "min": 1, "max": 20, "unit": "um"},
        "w_inv": {"default": 10, "min": 2, "max": 50, "unit": "um"},
        "w_inv_n": {"default": 5, "min": 1, "max": 25, "unit": "um"},
        "ibias": {"default": 20, "min": 5, "max": 100, "unit": "uA"},
    },
    typical_specs={
        "offset": {"max": 10e-3, "unit": "V"},
        "delay": {"max": 5e-9, "unit": "s"},
        "idd": {"max": 100e-6, "unit": "A"},
    },
)


# ============================================================
# Source Follower (Buffer)
# ============================================================

SOURCE_FOLLOWER = CircuitTemplate(
    id="source_follower",
    name="Source Follower (Buffer)",
    category="analog",
    description="Unity-gain buffer with low output impedance. Used for driving capacitive loads.",
    netlist=""".subckt source_follower VDD VSS VIN VOUT
* Source Follower Buffer - sky130
.include "sky130.lib" tt

* Input NMOS (common-drain)
XM1 VDD VIN VOUT VSS sky130_fd_pr__nfet_01v8 W={W_M1}u L={L_M1}u m={M_M1}

* Current source NMOS
XM2 VOUT VBIAS VSS VSS sky130_fd_pr__nfet_01v8 W={W_M2}u L={L_M2}u m={M_M2}

* Bias generation
VBIAS VBIAS VSS {VBIAS_V}

* Load cap
CL VOUT VSS {CL_pF}p

.ends source_follower
""",
    parameters={
        "W_M1": {"default": 20.0, "min": 2.0, "max": 100.0, "unit": "um"},
        "L_M1": {"default": 0.18, "min": 0.15, "max": 1.0, "unit": "um"},
        "M_M1": {"default": 2, "min": 1, "max": 8, "unit": ""},
        "W_M2": {"default": 10.0, "min": 2.0, "max": 50.0, "unit": "um"},
        "L_M2": {"default": 0.5, "min": 0.18, "max": 2.0, "unit": "um"},
        "M_M2": {"default": 2, "min": 1, "max": 4, "unit": ""},
        "VBIAS_V": {"default": 0.6, "min": 0.4, "max": 0.8, "unit": "V"},
        "CL_pF": {"default": 5.0, "min": 1.0, "max": 20.0, "unit": "pF"},
    },
    typical_specs={
        "gain": {"min": -1.5, "max": 0.0, "unit": "dB", "description": "Near unity gain"},
        "bandwidth": {"min": 100e6, "unit": "Hz"},
        "output_impedance": {"max": 500.0, "unit": "ohm"},
        "idd": {"max": 300e-6, "unit": "A"},
    },
    design_notes="Source follower provides voltage buffering. Gain is slightly less than 1 due to body effect.",
)


# ============================================================
# Differential Pair
# ============================================================

DIFFERENTIAL_PAIR = CircuitTemplate(
    id="diff_pair",
    name="Differential Pair",
    category="analog",
    description="Basic differential input stage. Building block for OTAs and comparators.",
    netlist=""".subckt diff_pair VDD VSS VINP VINM VOUTP VOUTM
* Differential Pair - sky130
.include "sky130.lib" tt

* Input differential pair
XM1 VOUTP VINM VTAIL VSS sky130_fd_pr__nfet_01v8 W={W_IN}u L={L_IN}u m={M_IN}
XM2 VOUTM VINP VTAIL VSS sky130_fd_pr__nfet_01v8 W={W_IN}u L={L_IN}u m={M_IN}

* PMOS active loads
XM3 VOUTP VOUTP VDD VDD sky130_fd_pr__pfet_01v8 W={W_LOAD}u L={L_LOAD}u m={M_LOAD}
XM4 VOUTM VOUTP VDD VDD sky130_fd_pr__pfet_01v8 W={W_LOAD}u L={L_LOAD}u m={M_LOAD}

* Tail current source
XM5 VTAIL VBIAS VSS VSS sky130_fd_pr__nfet_01v8 W={W_TAIL}u L={L_TAIL}u m={M_TAIL}

* Bias
VBIAS VBIAS VSS {VBIAS_V}

.ends diff_pair
""",
    parameters={
        "W_IN": {"default": 10.0, "min": 2.0, "max": 50.0, "unit": "um"},
        "L_IN": {"default": 0.36, "min": 0.18, "max": 1.0, "unit": "um"},
        "M_IN": {"default": 2, "min": 1, "max": 8, "unit": ""},
        "W_LOAD": {"default": 5.0, "min": 1.0, "max": 30.0, "unit": "um"},
        "L_LOAD": {"default": 0.36, "min": 0.18, "max": 1.0, "unit": "um"},
        "M_LOAD": {"default": 2, "min": 1, "max": 8, "unit": ""},
        "W_TAIL": {"default": 10.0, "min": 2.0, "max": 40.0, "unit": "um"},
        "L_TAIL": {"default": 0.5, "min": 0.18, "max": 2.0, "unit": "um"},
        "M_TAIL": {"default": 2, "min": 1, "max": 4, "unit": ""},
        "VBIAS_V": {"default": 0.6, "min": 0.4, "max": 0.8, "unit": "V"},
    },
    typical_specs={
        "dc_gain": {"min": 20.0, "unit": "dB"},
        "input_offset": {"max": 5e-3, "unit": "V"},
        "cmrr": {"min": 40.0, "unit": "dB"},
        "idd": {"max": 200e-6, "unit": "A"},
    },
    design_notes="Core building block. Matched input pair critical for low offset.",
)


# ============================================================
# Ring Oscillator
# ============================================================

RING_OSCILLATOR = CircuitTemplate(
    id="ring_osc",
    name="Ring Oscillator",
    category="digital",
    description="3-stage CMOS ring oscillator. Process monitor and clock generator.",
    netlist=""".subckt ring_osc VDD VSS VOUT
* 3-Stage Ring Oscillator - sky130
.include "sky130.lib" tt

* Stage 1
XMP1 N1 N3 VDD VDD sky130_fd_pr__pfet_01v8 W={WP}u L={LP}u m={MP}
XMN1 N1 N3 VSS VSS sky130_fd_pr__nfet_01v8 W={WN}u L={LN}u m={MN}

* Stage 2
XMP2 N2 N1 VDD VDD sky130_fd_pr__pfet_01v8 W={WP}u L={LP}u m={MP}
XMN2 N2 N1 VSS VSS sky130_fd_pr__nfet_01v8 W={WN}u L={LN}u m={MN}

* Stage 3
XMP3 N3 N2 VDD VDD sky130_fd_pr__pfet_01v8 W={WP}u L={LP}u m={MP}
XMN3 N3 N2 VSS VSS sky130_fd_pr__nfet_01v8 W={WN}u L={LN}u m={MN}

* Output buffer
XMP4 VOUT N3 VDD VDD sky130_fd_pr__pfet_01v8 W={WP_BUF}u L={LP}u m=1
XMN4 VOUT N3 VSS VSS sky130_fd_pr__nfet_01v8 W={WN_BUF}u L={LN}u m=1

.ends ring_osc
""",
    parameters={
        "WP": {"default": 2.0, "min": 0.5, "max": 10.0, "unit": "um"},
        "LP": {"default": 0.18, "min": 0.15, "max": 0.5, "unit": "um"},
        "MP": {"default": 1, "min": 1, "max": 4, "unit": ""},
        "WN": {"default": 1.0, "min": 0.3, "max": 5.0, "unit": "um"},
        "LN": {"default": 0.18, "min": 0.15, "max": 0.5, "unit": "um"},
        "MN": {"default": 1, "min": 1, "max": 4, "unit": ""},
        "WP_BUF": {"default": 4.0, "min": 1.0, "max": 20.0, "unit": "um"},
        "WN_BUF": {"default": 2.0, "min": 0.5, "max": 10.0, "unit": "um"},
    },
    typical_specs={
        "frequency": {"min": 500e6, "max": 5e9, "unit": "Hz"},
        "duty_cycle": {"min": 45.0, "max": 55.0, "unit": "%"},
        "power": {"max": 500e-6, "unit": "W"},
    },
    design_notes="Frequency depends on inverter delay. Wp/Wn ratio sets duty cycle. Used as process monitor.",
)


# ============================================================
# ngspice-Verified Templates (level=1 models, work directly)
# ============================================================

CASCODE_CS_AMP = CircuitTemplate(
    id="cascode_cs",
    name="Cascode Common-Source Amplifier",
    category="analog",
    description="Cascode CS amplifier for high gain and output resistance. ngspice-verified.",
    netlist="""\
* Cascode Common-Source Amplifier
.model nch nmos level=1 vto={vth}  kp={kp}u lambda=0.02
VDD vdd 0 DC {vdd}
Vin gate1 0 DC 0.8
Vbias gate2 0 DC {vbias}
RD vdd out {rd}k
M2 out gate2 mid 0 nch W={w}u L={l}u
M1 mid gate1 0 0 nch W={w}u L={l}u
.dc Vin 0.3 1.5 0.01
.end
""",
    parameters={
        "vdd": {"default": 3.3, "min": 1.8, "max": 5.0, "unit": "V"},
        "vth": {"default": 0.5, "min": 0.3, "max": 0.7, "unit": "V"},
        "kp": {"default": 200, "min": 100, "max": 500, "unit": "uA/V^2"},
        "vbias": {"default": 1.5, "min": 1.0, "max": 2.5, "unit": "V"},
        "rd": {"default": 10, "min": 1, "max": 50, "unit": "kohm"},
        "w": {"default": 10, "min": 1, "max": 100, "unit": "um"},
        "l": {"default": 1, "min": 0.18, "max": 10, "unit": "um"},
    },
    typical_specs={
        "gain_db": {"min": 40, "typical": 60, "unit": "dB"},
        "rout": {"min": 1e6, "typical": 1e7, "unit": "ohm"},
    },
    design_notes="ngspice-verified. Gain = gm1 * (gm2*ro2*ro1). "
                 "Requires VDD > Vdsat1 + Vdsat2 + Vds_load.",
)

WIDLAR_CURRENT_SOURCE = CircuitTemplate(
    id="widlar_cs",
    name="Widlar Current Source",
    category="analog",
    description="Widlar current source for sub-microamp current generation. ngspice-verified.",
    netlist="""\
* Widlar Current Source
.model nch nmos level=1 vto={vth} kp={kp}u lambda=0.02
VDD vdd 0 DC {vdd}
Iref vdd drain1 DC {iref}u
M1 drain1 drain1 0 0 nch W={w}u L={l}u
M2 drain2 drain1 src2 0 nch W={w}u L={l}u
Rs src2 0 {rs}k
Vds drain2 0 DC 0
.dc Vds 0 {vdd} 0.01
.end
""",
    parameters={
        "vdd": {"default": 1.8, "min": 1.2, "max": 5.0, "unit": "V"},
        "vth": {"default": 0.5, "min": 0.3, "max": 0.7, "unit": "V"},
        "kp": {"default": 200, "min": 100, "max": 500, "unit": "uA/V^2"},
        "iref": {"default": 100, "min": 10, "max": 1000, "unit": "uA"},
        "rs": {"default": 20, "min": 1, "max": 100, "unit": "kohm"},
        "w": {"default": 10, "min": 1, "max": 100, "unit": "um"},
        "l": {"default": 2, "min": 0.5, "max": 10, "unit": "um"},
    },
    typical_specs={
        "iout": {"min": 1e-6, "typical": 10e-6, "unit": "A"},
        "rout": {"min": 100e3, "typical": 500e3, "unit": "ohm"},
    },
    design_notes="ngspice-verified. Iout << Iref due to source degeneration. "
                 "Iout set by Vgs1 - Vgs2 = Iout * Rs.",
)

CMOS_INVERTER = CircuitTemplate(
    id="cmos_inv",
    name="CMOS Inverter",
    category="digital",
    description="Standard CMOS inverter for VTC and delay analysis. ngspice-verified.",
    netlist="""\
* CMOS Inverter
.model nch nmos level=1 vto={vthn} kp={kpn}u
.model pch pmos level=1 vto=-{vthp} kp={kpp}u
VDD vdd 0 DC {vdd}
Vin in 0 DC 0
M1 out in 0 0 nch W={wn}u L={l}u
M2 out in vdd vdd pch W={wp}u L={l}u
.dc Vin 0 {vdd} 0.01
.end
""",
    parameters={
        "vdd": {"default": 1.8, "min": 0.8, "max": 5.0, "unit": "V"},
        "vthn": {"default": 0.5, "min": 0.3, "max": 0.7, "unit": "V"},
        "vthp": {"default": 0.5, "min": 0.3, "max": 0.7, "unit": "V"},
        "kpn": {"default": 200, "min": 100, "max": 500, "unit": "uA/V^2"},
        "kpp": {"default": 100, "min": 50, "max": 250, "unit": "uA/V^2"},
        "wn": {"default": 2, "min": 0.5, "max": 50, "unit": "um"},
        "wp": {"default": 4, "min": 1, "max": 100, "unit": "um"},
        "l": {"default": 0.18, "min": 0.09, "max": 2, "unit": "um"},
    },
    typical_specs={
        "vm": {"min": 0.7, "typical": 0.9, "unit": "V"},
        "delay": {"min": 10e-12, "typical": 50e-12, "unit": "s"},
    },
    design_notes="ngspice-verified. Wp/Wn = kpn/kpp for balanced switching threshold.",
)

RC_FILTER = CircuitTemplate(
    id="rc_filter",
    name="RC Low-Pass Filter",
    category="analog",
    description="First-order RC low-pass filter. ngspice-verified.",
    netlist="""\
* RC Low-Pass Filter
V1 in 0 AC 1 DC 0
R1 in out {r}
C1 out 0 {c}n
.ac dec 20 {fstart} {fstop}
.end
""",
    parameters={
        "r": {"default": 1000, "min": 10, "max": 1e6, "unit": "ohm"},
        "c": {"default": 1, "min": 0.1, "max": 1000, "unit": "nF"},
        "fstart": {"default": 100, "min": 1, "max": 1e6, "unit": "Hz"},
        "fstop": {"default": "100Meg", "min": 1e3, "max": 1e12, "unit": "Hz"},
    },
    typical_specs={
        "f3db": {"typical": 159e3, "unit": "Hz"},
        "rolloff": {"typical": -20, "unit": "dB/dec"},
    },
    design_notes="ngspice-verified. f3dB = 1/(2*pi*R*C). First-order -20 dB/dec rolloff.",
)

# --- New templates: PLL, ADC, DAC, LNA ---

CHARGE_PUMP_PLL = CircuitTemplate(
    id="charge_pump_pll",
    name="Charge Pump PLL",
    category="analog",
    description="Type-II charge pump PLL with loop filter for clock generation.",
    netlist="""\
.subckt charge_pump_pll vco_out ref_clk vdd vss
* Phase Frequency Detector (simplified behavioral)
* PFD outputs UP and DN pulses

* Charge Pump
M1 vdd up net_cp vdd pmos w={wp}u l={l}u
M2 net_cp dn vss vss nmos w={wn}u l={l}u

* Loop Filter (2nd order)
R1 net_cp net_r {r_lf}
C1 net_r vss {c1_lf}p
C2 net_cp vss {c2_lf}p

* VCO (voltage-controlled oscillator)
* Kvco = {kvco} MHz/V, center freq = {f_center} MHz
E_vco vco_out vss net_cp vss {kvco}

.ends charge_pump_pll
""",
    parameters={
        "wp": {"default": 10, "min": 2, "max": 100, "unit": "um"},
        "wn": {"default": 5, "min": 1, "max": 50, "unit": "um"},
        "l": {"default": 0.18, "min": 0.09, "max": 1, "unit": "um"},
        "r_lf": {"default": 10e3, "min": 1e3, "max": 100e3, "unit": "ohm"},
        "c1_lf": {"default": 10, "min": 1, "max": 100, "unit": "pF"},
        "c2_lf": {"default": 1, "min": 0.1, "max": 10, "unit": "pF"},
        "kvco": {"default": 100, "min": 10, "max": 1000, "unit": "MHz/V"},
        "f_center": {"default": 1000, "min": 100, "max": 10000, "unit": "MHz"},
    },
    typical_specs={
        "lock_time": {"typical": 10e-6, "unit": "s"},
        "phase_noise": {"typical": -100, "unit": "dBc/Hz@1MHz"},
        "jitter": {"typical": 5e-12, "unit": "s_rms"},
    },
    design_notes="Type-II PLL. C1/C2 ratio ~ 10 for stability. Loop BW ~ 1/10 of ref freq.",
)

FLASH_ADC = CircuitTemplate(
    id="flash_adc_3bit",
    name="3-bit Flash ADC",
    category="analog",
    description="3-bit flash ADC with resistor ladder and 7 comparators.",
    netlist="""\
.subckt flash_adc_3bit vin vref vdd vss d2 d1 d0
* Resistor ladder (7 reference levels)
R7 vref net6 {r_ladder}
R6 net6 net5 {r_ladder}
R5 net5 net4 {r_ladder}
R4 net4 net3 {r_ladder}
R3 net3 net2 {r_ladder}
R2 net2 net1 {r_ladder}
R1 net1 vss {r_ladder}

* Comparators (simplified)
* Each compares vin against a reference tap
* Thermometer to binary encoder not shown
.ends flash_adc_3bit
""",
    parameters={
        "r_ladder": {"default": 1e3, "min": 100, "max": 10e3, "unit": "ohm"},
    },
    typical_specs={
        "resolution": {"typical": 3, "unit": "bits"},
        "sampling_rate": {"typical": 1e9, "unit": "Sa/s"},
        "dnl": {"typical": 0.5, "unit": "LSB"},
        "inl": {"typical": 0.5, "unit": "LSB"},
    },
    design_notes="Flash ADC trades area (2^N-1 comparators) for speed. 3-bit = 7 comparators.",
)

R2R_DAC = CircuitTemplate(
    id="r2r_dac_4bit",
    name="4-bit R-2R DAC",
    category="analog",
    description="4-bit R-2R ladder DAC for digital-to-analog conversion.",
    netlist="""\
.subckt r2r_dac_4bit d3 d2 d1 d0 vout vss
* R-2R Ladder Network
* MSB (d3) to LSB (d0)
R_2r_d3 d3 net3 {r2}
R_r_3 net3 net2 {r}
R_2r_d2 d2 net2b {r2}
R_j2 net2b net2 0
R_r_2 net2 net1 {r}
R_2r_d1 d1 net1b {r2}
R_j1 net1b net1 0
R_r_1 net1 net0 {r}
R_2r_d0 d0 net0b {r2}
R_j0 net0b net0 0
R_term net0 vss {r2}

* Output
R_out net3 vout {r}
.ends r2r_dac_4bit
""",
    parameters={
        "r": {"default": 10e3, "min": 1e3, "max": 100e3, "unit": "ohm"},
        "r2": {"default": 20e3, "min": 2e3, "max": 200e3, "unit": "ohm"},
    },
    typical_specs={
        "resolution": {"typical": 4, "unit": "bits"},
        "dnl": {"typical": 0.2, "unit": "LSB"},
        "settling_time": {"typical": 100e-9, "unit": "s"},
    },
    design_notes="R-2R uses only 2 resistor values. DNL depends on matching. Good for moderate resolution.",
)

CG_LNA = CircuitTemplate(
    id="cg_lna",
    name="Common-Gate LNA",
    category="analog",
    description="Common-gate low-noise amplifier for RF front-end. Input matched to 50 ohm.",
    netlist="""\
.subckt cg_lna rf_in rf_out vdd vss
* Input matching: Zin = 1/gm ~ 50 ohm
* Bias
Vb bias vss {vbias}
M1 rf_out rf_in net_s vss nmos w={w}u l={l}u
R_source net_s vss {r_source}

* Load
R_load vdd rf_out {r_load}

* DC bias for input
L_choke rf_in net_bias {l_choke}n
R_bias net_bias bias {r_bias}

* AC coupling
C_in rf_in_ac rf_in {c_in}p
C_out rf_out rf_out_ac {c_out}p
.ends cg_lna
""",
    parameters={
        "w": {"default": 50, "min": 10, "max": 200, "unit": "um"},
        "l": {"default": 0.18, "min": 0.09, "max": 0.5, "unit": "um"},
        "vbias": {"default": 0.7, "min": 0.5, "max": 1.0, "unit": "V"},
        "r_load": {"default": 500, "min": 100, "max": 2000, "unit": "ohm"},
        "r_source": {"default": 100, "min": 10, "max": 1000, "unit": "ohm"},
        "r_bias": {"default": 10e3, "min": 1e3, "max": 100e3, "unit": "ohm"},
        "l_choke": {"default": 10, "min": 1, "max": 100, "unit": "nH"},
        "c_in": {"default": 1, "min": 0.1, "max": 10, "unit": "pF"},
        "c_out": {"default": 1, "min": 0.1, "max": 10, "unit": "pF"},
    },
    typical_specs={
        "gain": {"typical": 15, "unit": "dB"},
        "nf": {"typical": 3.0, "unit": "dB"},
        "s11": {"typical": -15, "unit": "dB"},
        "iip3": {"typical": 5, "unit": "dBm"},
    },
    design_notes="CG LNA: wideband input match (Zin=1/gm). Higher NF than CS LNA but better linearity.",
)


TEMPLATES: dict[str, CircuitTemplate] = {
    t.id: t for t in [
        OTA_TWO_STAGE,
        FOLDED_CASCODE,
        BANDGAP_REFERENCE,
        LDO_REGULATOR,
        CURRENT_MIRROR,
        COMPARATOR,
        SOURCE_FOLLOWER,
        DIFFERENTIAL_PAIR,
        RING_OSCILLATOR,
        CASCODE_CS_AMP,
        WIDLAR_CURRENT_SOURCE,
        CMOS_INVERTER,
        RC_FILTER,
        CHARGE_PUMP_PLL,
        FLASH_ADC,
        R2R_DAC,
        CG_LNA,
    ]
}


def get_template(template_id: str) -> CircuitTemplate:
    """Get a circuit template by ID."""
    if template_id not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Unknown template: {template_id}. Available: {available}")
    return TEMPLATES[template_id]


def list_templates(category: str | None = None) -> list[CircuitTemplate]:
    """List available templates, optionally filtered by category."""
    templates = list(TEMPLATES.values())
    if category:
        templates = [t for t in templates if t.category == category]
    return templates


def render_template(template_id: str, **params: Any) -> str:
    """Render a template with custom parameters."""
    return get_template(template_id).render(**params)
