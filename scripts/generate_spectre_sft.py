#!/usr/bin/env python3
"""Generate SFT data with Cadence Spectre netlist format.

Creates training examples using Spectre's native .scs format
alongside SPICE format to teach the model both formats.

Usage:
    PYTHONPATH=src python scripts/generate_spectre_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60

# Spectre SCS format examples with expert reasoning
SPECTRE_EXAMPLES = [
    {
        "id": "spectre_cs_amp",
        "task": "Design a common-source amplifier using Spectre netlist format (.scs). Target DC gain > 25 dB with 1.8V supply.",
        "reasoning": (
            "I'll design a CS amplifier using Cadence Spectre's native .scs format. "
            "Spectre uses a different syntax than SPICE:\n"
            "- Instance: `M0 (drain gate source bulk) nch w=10u l=180n`\n"
            "- Analysis: `dc1 dc param=vgs start=0.3 stop=1.2 step=0.01`\n"
            "- Models use `model nch nmos ...` instead of `.model`\n\n"
            "For 25 dB gain: Av = gm * RD = (2*Id)/(Vgs-Vth) * RD\n"
            "With Id=100uA, Vgs-Vth=0.2V: gm=1mA/V. Need RD > 17.8k."
        ),
        "spectre_netlist": """\
// Common-Source Amplifier - Spectre format
simulator lang=spectre

model nch nmos (version=1 vth0=0.5 u0=200)
model pch pmos (version=1 vth0=-0.5 u0=100)

// Supplies
V0 (vdd 0) vsource dc=1.8
Vin (gate 0) vsource dc=0.7

// Circuit
RD (vdd out) resistor r=20k
M1 (out gate 0 0) nch w=10u l=1u

// Load
CL (out 0) capacitor c=1p

// Analysis
dc1 dc param=Vin:dc start=0.3 stop=1.2 step=0.01
ac1 ac start=100 stop=10G dec=20
""",
        "analysis_text": (
            "The Spectre DC sweep shows the amplifier characteristics. "
            "Key differences from SPICE format:\n"
            "1. Instance syntax: `M1 (out gate 0 0) nch w=10u l=1u`\n"
            "2. Source syntax: `V0 (vdd 0) vsource dc=1.8`\n"
            "3. Analysis: `dc1 dc param=Vin:dc start=0.3 stop=1.2 step=0.01`\n"
            "4. Comments use `//` instead of `*`\n\n"
            "With RD=20k and gm=1mA/V, gain = 20 = 26 dB > 25 dB target."
        ),
    },
    {
        "id": "spectre_diff_pair",
        "task": "Design a NMOS differential pair in Spectre format for a 5-transistor OTA.",
        "reasoning": (
            "A 5-transistor OTA has:\n"
            "- M1/M2: Input differential pair (NMOS)\n"
            "- M3/M4: Active load (PMOS current mirror)\n"
            "- M5: Tail current source\n\n"
            "In Spectre format, I'll use `subckt`/`ends` for hierarchy:\n"
            "```\n"
            "subckt ota5 (vdd vss inp inm out vbias)\n"
            "  M1 (net1 inm tail vss) nch ...\n"
            "ends ota5\n"
            "```\n"
            "Target: Gain > 40 dB, GBW > 10 MHz."
        ),
        "spectre_netlist": """\
// 5-Transistor OTA - Spectre format
simulator lang=spectre

model nch nmos (version=1 vth0=0.45 u0=200 lambda=0.02)
model pch pmos (version=1 vth0=-0.45 u0=100 lambda=0.03)

subckt ota5 (vdd vss inp inm out vbias)
  // Input differential pair
  M1 (net1 inm tail vss) nch w=10u l=500n m=2
  M2 (net2 inp tail vss) nch w=10u l=500n m=2

  // PMOS active load (current mirror)
  M3 (net1 net1 vdd vdd) pch w=20u l=500n m=2
  M4 (net2 net1 vdd vdd) pch w=20u l=500n m=2

  // Tail current source
  M5 (tail vbias vss vss) nch w=5u l=1u m=2
ends ota5

// Testbench
V0 (vdd 0) vsource dc=1.8
V1 (vss 0) vsource dc=0
Vbias (vbias 0) vsource dc=0.6
Vip (inp 0) vsource dc=0.9 mag=1
Vim (inm 0) vsource dc=0.9

I0 (vdd out) isource dc=0  // dummy for output

X0 (vdd vss inp inm out vbias) ota5

// Analysis
dc1 dc param=Vip:dc start=0.5 stop=1.3 step=0.005
ac1 ac start=1 stop=10G dec=20
""",
        "analysis_text": (
            "Spectre subcircuit syntax: `subckt name (port_list)` ... `ends name`.\n"
            "The 5-T OTA shows:\n"
            "- DC gain: gm1,2 * (ro2 || ro4) ~ 50 dB\n"
            "- Gain-bandwidth: gm / CL ~ 30 MHz\n"
            "- CMRR limited by tail current source ro5\n\n"
            "Key Spectre conventions: ports in parentheses, "
            "multiplier is `m=2` (same as SPICE), "
            "subcircuit instance uses `X0 (...) ota5`."
        ),
    },
    {
        "id": "spectre_corners",
        "task": "Set up PVT corner simulation in Spectre for a bandgap reference circuit.",
        "reasoning": (
            "PVT corner analysis in Spectre uses `alter` blocks or the `corners` analysis.\n\n"
            "Spectre corner syntax:\n"
            "```\n"
            "corners corner_name {\n"
            "  corner tt { ... }\n"
            "  corner ff { ... }\n"
            "  corner ss { ... }\n"
            "}\n"
            "```\n\n"
            "Or using APS (Analog Performance Specification):\n"
            "```\n"
            "dc1 dc ...\n"
            "sweep1 paramset [ tt ff ss ] { ... }\n"
            "```"
        ),
        "spectre_netlist": """\
// Bandgap Reference with PVT Corners - Spectre
simulator lang=spectre

model nch nmos (version=1 vth0=0.45 u0=200)
model pch pmos (version=1 vth0=-0.45 u0=100)

// Bandgap core (simplified)
V0 (vdd 0) vsource dc=3.3
R1 (vdd out) resistor r=10k
R2 (out mid) resistor r=5k
M1 (mid mid 0 0) nch w=10u l=2u
I1 (vdd out) isource dc=50u

// Temperature sweep (corners)
dc_temp dc param=temp start=-40 stop=125 step=5

// Supply sweep
dc_supply dc param=V0:dc start=2.5 stop=5.5 step=0.1

// Corner definitions for APS
parameters supply_nom=3.3 temp_nom=27
""",
        "analysis_text": (
            "Spectre corner simulation setup:\n"
            "1. Temperature sweep: `dc_temp dc param=temp start=-40 stop=125`\n"
            "2. Supply sweep: `dc_supply dc param=V0:dc start=2.5 stop=5.5`\n"
            "3. For full PVT: use `corners` block with model variants\n\n"
            "The bandgap reference should maintain ~1.2V output across:\n"
            "- Temperature: -40C to 125C (< 50 ppm/C)\n"
            "- Supply: 2.5V to 5.5V (PSRR > 60 dB)\n"
            "- Process: SS to FF corners (< 2% variation)"
        ),
    },
    {
        "id": "spectre_monte_carlo",
        "task": "Run Monte Carlo analysis in Spectre to verify yield of an OTA design.",
        "reasoning": (
            "Monte Carlo in Spectre:\n"
            "```\n"
            "mc1 montecarlo numruns=1000 {\n"
            "  dc1 dc ...\n"
            "  ac1 ac ...\n"
            "  export gain=oceanEval(\"ymax(dB20(vf(\\\"/out\\\")))\")\n"
            "}\n"
            "```\n\n"
            "For yield analysis, we check:\n"
            "- DC offset: < 5mV (3-sigma)\n"
            "- DC gain: > 60 dB\n"
            "- GBW: > 10 MHz\n"
            "- Phase margin: > 60 degrees"
        ),
        "spectre_netlist": """\
// Monte Carlo Yield Analysis - Spectre
simulator lang=spectre

model nch nmos (version=1 vth0=0.45 u0=200)
model pch pmos (version=1 vth0=-0.45 u0=100)

// OTA instance (subcircuit assumed defined)
V0 (vdd 0) vsource dc=1.8
Vin (inp 0) vsource dc=0.9 mag=1
V1 (inm 0) vsource dc=0.9
Vb (vbias 0) vsource dc=0.6

// Monte Carlo setup
mc1 montecarlo numruns=500 seed=42 {
  dc1 dc param=Vin:dc start=0.5 stop=1.3 step=0.005
  ac1 ac start=1 stop=10G dec=20
}
""",
        "analysis_text": (
            "Monte Carlo analysis in Spectre:\n"
            "- `montecarlo numruns=500 seed=42` for reproducibility\n"
            "- Includes both process and mismatch variations\n"
            "- Nested analyses (dc + ac) run for each MC iteration\n"
            "- Results extracted with OCEAN expressions\n\n"
            "Yield = (passed / total) * 100%. Target > 99.7% (3-sigma).\n"
            "If yield is low, increase W/L or add trimming circuits."
        ),
    },
    {
        "id": "spectre_stb",
        "task": "Analyze loop stability of a feedback amplifier using Spectre STB analysis.",
        "reasoning": (
            "Spectre STB (stability) analysis is unique and powerful:\n"
            "```\n"
            "stb1 stb start=1 stop=10G dec=20 probe=I0\n"
            "```\n\n"
            "It automatically finds loop gain by breaking the loop at a probe.\n"
            "This is much more accurate than traditional methods.\n"
            "Phase margin = phase at 0dB gain crossing.\n"
            "Gain margin = gain at -180 degree phase crossing."
        ),
        "spectre_netlist": """\
// Stability Analysis - Spectre STB
simulator lang=spectre

model nch nmos (version=1 vth0=0.45 u0=200)
model pch pmos (version=1 vth0=-0.45 u0=100)

// Amplifier (unity-gain feedback config)
V0 (vdd 0) vsource dc=1.8
Vin (inp 0) vsource dc=0.9

// OTA with feedback
RD (vdd out) resistor r=20k
M1 (out inp 0 0) nch w=10u l=500n m=4
CL (out 0) capacitor c=5p

// Feedback network
Rf (out inm) resistor r=100k
Ri (inm 0) resistor r=100k

// STB probe
I0 (out inm) isource dc=0 type=iprobe

// Analysis
stb1 stb start=1 stop=10G dec=20 probe=I0
dc1 dc oppoint=logfile
""",
        "analysis_text": (
            "Spectre STB analysis results:\n"
            "- Phase margin (PM): target > 60 degrees\n"
            "- Gain margin (GM): target > 10 dB\n"
            "- Loop gain bandwidth\n\n"
            "STB analysis is a Spectre-specific feature that automates "
            "Middlebrook's method for loop gain extraction. It inserts "
            "test signals at the probe point and measures the loop response.\n\n"
            "If PM < 45: add Miller compensation (Cc between stages)\n"
            "If PM < 30: circuit is unstable, redesign required."
        ),
    },
    {
        "id": "spectre_pss_pnoise",
        "task": "Analyze phase noise of a VCO using Spectre PSS/PNoise analysis.",
        "reasoning": (
            "PSS (Periodic Steady-State) and PNoise are Spectre-specific analyses "
            "for RF/oscillator circuits:\n\n"
            "```\n"
            "pss1 pss fund=freq_est harms=10\n"
            "pnoise1 pnoise start=1k stop=100M dec=10 pssname=pss1\n"
            "```\n\n"
            "PSS finds the periodic operating point much faster than transient.\n"
            "PNoise then computes phase noise spectrum around harmonics."
        ),
        "spectre_netlist": """\
// VCO Phase Noise Analysis - Spectre PSS/PNoise
simulator lang=spectre

model nch nmos (version=1 vth0=0.45 u0=200)
model pch pmos (version=1 vth0=-0.45 u0=100)

// 3-stage ring oscillator VCO
V0 (vdd 0) vsource dc=1.8

// Stage 1
M1n (n1 n3 0 0) nch w=2u l=180n
M1p (n1 n3 vdd vdd) pch w=4u l=180n

// Stage 2
M2n (n2 n1 0 0) nch w=2u l=180n
M2p (n2 n1 vdd vdd) pch w=4u l=180n

// Stage 3
M3n (n3 n2 0 0) nch w=2u l=180n
M3p (n3 n2 vdd vdd) pch w=4u l=180n

// Output buffer
M4n (out n3 0 0) nch w=4u l=180n
M4p (out n3 vdd vdd) pch w=8u l=180n
CL (out 0) capacitor c=100f

// Initial condition
ic n1=0 n2=1.8 n3=0

// PSS - find periodic steady state
pss1 pss fund=1G harms=10 tstab=5n

// PNoise - phase noise spectrum
pnoise1 pnoise start=1k stop=100M dec=10 pssname=pss1 noiseout=n3
""",
        "analysis_text": (
            "PSS/PNoise is Spectre's most powerful RF analysis:\n"
            "- PSS finds the oscillation frequency directly (no long transient)\n"
            "- PNoise computes phase noise at offset frequencies\n"
            "- `fund=1G` initial frequency estimate, `harms=10` harmonics\n\n"
            "Expected results:\n"
            "- Phase noise @ 1MHz offset: ~ -100 dBc/Hz\n"
            "- Phase noise @ 10MHz: ~ -120 dBc/Hz\n"
            "- 1/f^3 region below 100kHz, 1/f^2 above\n\n"
            "These analyses are NOT available in ngspice or basic SPICE."
        ),
    },
]


def main():
    output_path = "data/sft/spectre_format_v1.jsonl"

    print(f"\n{SEP}")
    print("   Generate Spectre Format SFT Data")
    print(f"{SEP}\n")
    print(f"  Examples: {len(SPECTRE_EXAMPLES)}")

    examples = []
    for ex in SPECTRE_EXAMPLES:
        example = {
            "messages": [
                {"role": "system", "content": build_system_message()},
                {"role": "user", "content": ex["task"]},
                {"role": "assistant", "content": (
                    f"{ex['reasoning']}\n\n"
                    f"Here is the Spectre netlist:\n"
                    f"```spectre\n{ex['spectre_netlist']}```\n\n"
                    f"<tool_call>\n"
                    f'  {{"name": "sim.dc", "arguments": {{"analysis_type": "dc"}}}}\n'
                    f"</tool_call>"
                )},
                {"role": "tool", "content": json.dumps({"status": "success", "simulator": "spectre"})},
                {"role": "assistant", "content": ex["analysis_text"]},
            ],
            "source": "spectre_format_v1",
            "circuit_id": ex["id"],
            "simulator": "spectre",
            "verified": False,  # Not verified with real Spectre (needs license)
        }
        examples.append(example)
        print(f"  [{ex['id']}] {ex['task'][:55]}...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{SEP}")
    print(f"  Generated: {len(examples)} Spectre format examples")
    print(f"  Saved: {output_path}")
    print(f"  NOTE: Not verified with real Spectre (needs license + WSL)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
