#!/usr/bin/env python3
"""Generate SFT data for analog layout and floorplanning.

Covers layout techniques critical for analog IC design:
- Matching (interdigitated, common-centroid)
- Floorplanning strategies
- Guard rings and isolation
- Routing for low-noise analog

Usage:
    PYTHONPATH=src python scripts/generate_layout_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message
SEP = "=" * 60

EXAMPLES = [
    {
        "id": "layout_matching",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "How should I layout a differential pair for best matching in sky130?"},
            {"role": "assistant", "content": (
                "Differential pair matching is critical for offset voltage. Key techniques:\n\n"
                "**1. Common-Centroid Layout**\n"
                "For M1/M2 with multiplier m=4, arrange as:\n"
                "```\n"
                "  M1a  M2a  M2b  M1b\n"
                "  M2c  M1c  M1d  M2d\n"
                "```\n"
                "This ABBA/BAAB pattern cancels linear gradients in both X and Y.\n\n"
                "**2. Interdigitated Layout**\n"
                "For simpler routing:\n"
                "```\n"
                "  M1  M2  M1  M2  M1  M2  M1  M2\n"
                "```\n"
                "Cancels X-gradient but not Y. Simpler to route.\n\n"
                "**3. Orientation Rules**\n"
                "- All transistors must have SAME orientation (gate direction)\n"
                "- Current flow in same direction for all fingers\n"
                "- Don't mirror or rotate individual devices\n\n"
                "**4. Sky130-specific**\n"
                "- Use `sky130_fd_pr__nfet_01v8` with L >= 1um for better matching\n"
                "- Larger area = better matching: sigma(Vth) ~ Avt / sqrt(W*L)\n"
                "- sky130 Avt ~ 5-10 mV*um for NMOS\n"
                "- For 1mV offset (3-sigma): W*L >= 25 um^2\n\n"
                "**5. Dummy devices**\n"
                "- Add dummy transistors at edges to protect active devices\n"
                "- Dummy gates see same etch pattern as active gates\n"
                "```\n"
                "  D  M1  M2  M2  M1  D\n"
                "```\n"
                "Where D = dummy (gate tied to VSS)"
            )},
        ],
    },
    {
        "id": "layout_floorplan_ota",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "Create a floorplan for a two-stage OTA with Miller compensation."},
            {"role": "assistant", "content": (
                "Two-stage OTA floorplan:\n\n"
                "```\n"
                "  ┌─────────────────────────────────────────────┐\n"
                "  │                   VDD Rail (M3)              │\n"
                "  ├──────────┬──────────┬──────────┬────────────┤\n"
                "  │          │  PMOS    │  PMOS    │            │\n"
                "  │  Guard   │  Load    │  2nd     │   Guard    │\n"
                "  │  Ring    │  M3/M4   │  Stage   │   Ring     │\n"
                "  │  (NWELL) │ (matched)│  M7      │  (NWELL)   │\n"
                "  ├──────────┼──────────┼──────────┼────────────┤\n"
                "  │          │  NMOS    │  NMOS    │            │\n"
                "  │  Guard   │  Input   │  2nd     │   Guard    │\n"
                "  │  Ring    │  M1/M2   │  Stage   │   Ring     │\n"
                "  │  (PSUB)  │ (matched)│  M6      │  (PSUB)    │\n"
                "  ├──────────┼──────────┼──────────┼────────────┤\n"
                "  │          │   Tail   │   Cc     │            │\n"
                "  │          │   M5     │  (MIM)   │            │\n"
                "  ├──────────┴──────────┴──────────┴────────────┤\n"
                "  │                   VSS Rail (M3)              │\n"
                "  └─────────────────────────────────────────────┘\n"
                "```\n\n"
                "**Key placement rules:**\n\n"
                "1. **Input pair M1/M2**: Center of die, common-centroid\n"
                "   - Maximum distance from noisy digital blocks\n"
                "   - Same well, same orientation\n\n"
                "2. **PMOS load M3/M4**: Directly above input pair\n"
                "   - Minimizes routing between stages\n"
                "   - Matched: same common-centroid pattern\n\n"
                "3. **2nd stage M6/M7**: Adjacent to 1st stage\n"
                "   - Short path from 1st stage output to 2nd stage input\n\n"
                "4. **Compensation cap Cc**: Between stages\n"
                "   - MIM cap for linearity (not MOS cap)\n"
                "   - Route both plates on upper metals\n\n"
                "5. **Guard rings**: Surround all analog blocks\n"
                "   - NWELL ring for PMOS, PSUB ring for NMOS\n"
                "   - Prevents substrate noise injection\n\n"
                "6. **Power rails**: M3 horizontal straps, M4 vertical\n"
                "   - Star-point grounding for sensitive nodes"
            )},
        ],
    },
    {
        "id": "layout_current_mirror",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "How do I layout a 1:4 current mirror with good accuracy?"},
            {"role": "assistant", "content": (
                "1:4 current mirror layout for accuracy:\n\n"
                "**Approach: Unit-element matching**\n"
                "Use 1 unit for reference, 4 units for mirror output.\n"
                "Total 5 units, arranged in common-centroid:\n\n"
                "```\n"
                "  Row 1:  D  M4  M1  M4  D\n"
                "  Row 2:     M4  M1  M4\n"
                "```\n"
                "Wait -- with 1:4, better pattern:\n"
                "```\n"
                "  D  B  A  B  D     (A=ref, B=mirror, D=dummy)\n"
                "  D  B  A  B  D\n"
                "```\n"
                "This gives 2 ref units and 4 mirror units (1:2 per row, 2 rows).\n"
                "For 1:4 ratio, use 1 ref unit and 4 mirror units:\n"
                "```\n"
                "  D  B  B  A  B  B  D\n"
                "```\n"
                "With A centered and B symmetric around A.\n\n"
                "**Critical rules:**\n"
                "1. All units: same W, same L, same orientation\n"
                "2. Same current direction through all fingers\n"
                "3. Dummy transistors at edges (gate to VSS)\n"
                "4. Source/drain sharing between adjacent units\n"
                "5. Connect all sources on M1, all drains on M2\n"
                "6. Gate bus on poly or M1 (low resistance)\n\n"
                "**Expected accuracy:**\n"
                "- Random mismatch: sigma(Id)/Id ~ 1/(gm/Id) * Avt/sqrt(W*L)\n"
                "- For W=2um, L=1um: sigma ~ 0.5% (1-sigma)\n"
                "- Systematic: < 0.1% with proper layout"
            )},
        ],
    },
    {
        "id": "layout_noise_routing",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "What are the best practices for routing in a mixed-signal IC to minimize noise coupling?"},
            {"role": "assistant", "content": (
                "Mixed-signal routing noise isolation techniques:\n\n"
                "**1. Power Domain Separation**\n"
                "- Separate AVDD/AVSS from DVDD/DVSS\n"
                "- Single point connection (star ground) at pad ring\n"
                "- Dedicated power pads for analog\n"
                "- Wide power buses: R < 1 ohm from pad to circuit\n\n"
                "**2. Signal Routing**\n"
                "- Analog signals on upper metals (M4, M5) -- less coupling\n"
                "- Digital signals on lower metals (M1, M2)\n"
                "- NEVER route digital signals over analog blocks\n"
                "- Cross analog/digital signals at 90 degrees only\n"
                "- Shield sensitive nets with grounded metal on both sides:\n"
                "  `VSS | signal | VSS`\n\n"
                "**3. Substrate Isolation**\n"
                "- Deep N-well under analog NMOS (if available)\n"
                "- Triple-well for full isolation\n"
                "- Guard rings around all analog blocks\n"
                "- Substrate contacts every 25-50um\n\n"
                "**4. Decoupling**\n"
                "- MIM/MOM caps between AVDD and AVSS\n"
                "- Place adjacent to each analog block\n"
                "- Target: 10-100 pF per block\n"
                "- Use stacked MOM caps for density\n\n"
                "**5. Clock Distribution**\n"
                "- Route clock in dedicated channel\n"
                "- Shield on both sides and top/bottom\n"
                "- Minimize clock-to-analog coupling\n"
                "- Use differential clock if possible\n\n"
                "**6. Matching-Critical Nets**\n"
                "- Route symmetrically (same length, same layers)\n"
                "- Same number of vias on matched nets\n"
                "- Avoid bends near sensitive transistors"
            )},
        ],
    },
]


def main():
    output_path = "data/sft/layout_v1.jsonl"

    print(f"\n{SEP}")
    print("   Generate Layout / Floorplanning SFT Data")
    print(f"{SEP}\n")

    examples = []
    for ex in EXAMPLES:
        example = {
            "messages": ex["messages"],
            "source": "layout_v1",
            "circuit_id": ex["id"],
            "domain": "layout",
        }
        print(f"  [{ex['id']}]")
        examples.append(example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{SEP}")
    print(f"  Generated: {len(examples)} layout examples")
    print(f"  Covers: matching, floorplan, current mirror, noise routing")
    print(f"  Saved: {output_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
