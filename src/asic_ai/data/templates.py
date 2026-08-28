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
# Template Registry
# ============================================================

TEMPLATES: dict[str, CircuitTemplate] = {
    t.id: t for t in [
        OTA_TWO_STAGE,
        FOLDED_CASCODE,
        BANDGAP_REFERENCE,
        LDO_REGULATOR,
        CURRENT_MIRROR,
        COMPARATOR,
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
