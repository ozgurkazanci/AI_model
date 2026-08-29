#!/usr/bin/env python3
"""Generate SFT data for physical verification and parasitic extraction.

Covers the full IC design signoff flow:
- PVS: DRC (Design Rule Check), LVS (Layout vs Schematic)
- Quantus: Parasitic extraction (RC, RCC)
- Post-layout simulation with parasitics

Usage:
    PYTHONPATH=src python scripts/generate_signoff_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message

SEP = "=" * 60

EXAMPLES = [
    # --- DRC ---
    {
        "id": "pvs_drc_fix",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My layout has DRC errors after adding guard rings. Fix the violations."},
            {"role": "assistant", "content": (
                "Common guard ring DRC violations in sky130:\n\n"
                "1. **MIN_SPACING**: Guard ring too close to active device\n"
                "   - Fix: Increase spacing between guard ring and MOSFET to >= 0.27um\n\n"
                "2. **MIN_WIDTH**: Guard ring metal/diffusion too narrow\n"
                "   - Fix: Widen guard ring tap to >= 0.15um (nwell tap) or >= 0.17um (psub tap)\n\n"
                "3. **MIN_ENCLOSURE**: Well not fully enclosing the guard ring tap\n"
                "   - Fix: Extend nwell by >= 0.18um beyond the tap on all sides\n\n"
                "4. **DENSITY**: Metal density violation from guard ring metal\n"
                "   - Fix: Add dummy fill or adjust metal routing\n\n"
                "Let me run DRC to see the specific violations:\n\n"
                "<tool_call>\n"
                '  {"name": "lint.check", "arguments": {"check_type": "drc", "rule_deck": "sky130"}}\n'
                "</tool_call>"
            )},
            {"role": "tool", "content": json.dumps({
                "violations": [
                    {"rule": "licon.7", "description": "Minimum spacing of licon to diffusion edge", "count": 3},
                    {"rule": "nwell.5", "description": "Minimum nwell enclosure of nsd", "count": 2},
                ],
                "total": 5,
                "clean": False
            })},
            {"role": "assistant", "content": (
                "Found 5 DRC violations:\n\n"
                "**licon.7** (3 errors): Local interconnect contact (licon) too close to diffusion edge.\n"
                "- Fix: Move licon contacts >= 0.04um from diff edge\n"
                "- This often happens when guard ring taps are placed at minimum dimensions\n\n"
                "**nwell.5** (2 errors): N-well doesn't sufficiently enclose N+ diffusion.\n"
                "- Fix: Extend nwell by >= 0.18um beyond the N+ tap\n\n"
                "After fixing:\n"
                "1. Re-run DRC to verify clean\n"
                "2. Run LVS to ensure guard rings didn't change connectivity\n"
                "3. Extract parasitics to check guard ring effectiveness"
            )},
        ],
    },
    # --- LVS ---
    {
        "id": "pvs_lvs_debug",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "LVS reports 3 unmatched nets. How do I debug this?"},
            {"role": "assistant", "content": (
                "LVS debugging strategy:\n\n"
                "**Step 1: Check the LVS report** for unmatched nets\n"
                "Common causes of unmatched nets:\n"
                "1. Missing vias between metal layers\n"
                "2. Floating gates (no connection to gate poly)\n"
                "3. Swapped source/drain connections\n"
                "4. Missing substrate/well taps\n"
                "5. Short circuits from DRC-clean but electrically wrong routing\n\n"
                "**Step 2: Use the cross-reference viewer**\n"
                "- Highlight unmatched net in schematic\n"
                "- Find corresponding geometry in layout\n"
                "- Check for breaks in the path\n\n"
                "**Step 3: Common fixes**\n"
                "- Add via where metal layers should connect\n"
                "- Extend poly to reach gate of MOSFET\n"
                "- Add body tap for PMOS (nwell contact) or NMOS (psub contact)\n\n"
                "Let me run LVS to get the details:\n\n"
                "<tool_call>\n"
                '  {"name": "lint.check", "arguments": {"check_type": "lvs"}}\n'
                "</tool_call>"
            )},
            {"role": "tool", "content": json.dumps({
                "matched_devices": 12,
                "unmatched_nets": ["net_vbias", "net_cascode", "VDD_local"],
                "shorts": [],
                "opens": [
                    {"net": "net_vbias", "location": "via1 missing between M1 and M2"},
                    {"net": "VDD_local", "location": "power rail break at row 3"}
                ],
                "clean": False
            })},
            {"role": "assistant", "content": (
                "LVS found 3 unmatched nets:\n\n"
                "1. **net_vbias**: Missing via1 between M1 and M2\n"
                "   - Fix: Add VIA1 at the junction point\n"
                "   - Common when copying layout cells without vias\n\n"
                "2. **net_cascode**: Check if cascode gate poly reaches the device\n"
                "   - Likely a poly routing issue\n"
                "   - Extend poly or add poly-to-M1 contact\n\n"
                "3. **VDD_local**: Power rail break at row 3\n"
                "   - Fix: Connect the M1 power rail segments\n"
                "   - Add a metal strap across the break\n\n"
                "After fixes: re-run LVS, then DRC (vias might introduce new spacing violations)."
            )},
        ],
    },
    # --- Parasitic Extraction ---
    {
        "id": "quantus_extraction",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "How do I extract parasitics for post-layout simulation of my OTA?"},
            {"role": "assistant", "content": (
                "Parasitic extraction flow with Quantus:\n\n"
                "**1. Extraction modes:**\n"
                "- **RC**: Resistance + Capacitance (most common)\n"
                "- **RCC**: RC + Coupling capacitance (for sensitive analog)\n"
                "- **RLC**: With inductance (for RF, > 1 GHz)\n\n"
                "**2. Quantus setup:**\n"
                "```\n"
                "extract -type rc_coupled\n"
                "  -output_format spectre\n"
                "  -nets * \n"
                "  -reduce_parasitics true\n"
                "  -coupling_threshold 0.1fF\n"
                "```\n\n"
                "**3. Post-layout simulation:**\n"
                "After extraction, the netlist includes parasitic R and C.\n"
                "Simulate with Spectre to check if parasitics degrade:\n"
                "- Gain (resistive drops reduce headroom)\n"
                "- Bandwidth (parasitic C adds poles)\n"
                "- Stability (phase margin may decrease)\n"
                "- Noise (additional resistors add thermal noise)\n\n"
                "Let me run the extraction and then compare pre/post-layout:\n\n"
                "<tool_call>\n"
                '  {"name": "sim.ac", "arguments": {"analysis_type": "ac"}}\n'
                "</tool_call>"
            )},
            {"role": "tool", "content": json.dumps({
                "frequencies": [1e3, 1e6, 1e7, 1e8, 1e9],
                "signals": {
                    "pre_layout": {"name": "pre_layout", "x_values": [1e3, 1e6, 1e7, 1e8, 1e9], "y_values": [60.0, 59.5, 55.0, 35.0, 8.0]},
                    "post_layout": {"name": "post_layout", "x_values": [1e3, 1e6, 1e7, 1e8, 1e9], "y_values": [58.5, 57.0, 50.0, 28.0, 5.0]},
                }
            })},
            {"role": "assistant", "content": (
                "Pre vs post-layout comparison:\n\n"
                "| Metric | Pre-Layout | Post-Layout | Degradation |\n"
                "|--------|-----------|-------------|-------------|\n"
                "| DC Gain | 60 dB | 58.5 dB | -1.5 dB |\n"
                "| UGF | ~300 MHz | ~200 MHz | -33% |\n"
                "| BW (-3dB) | ~50 MHz | ~35 MHz | -30% |\n\n"
                "**Analysis**: Parasitic capacitances reduced bandwidth by ~30%.\n"
                "This is typical for analog circuits.\n\n"
                "**Mitigation strategies:**\n"
                "1. Reduce wire lengths (compact layout)\n"
                "2. Use higher metal layers for critical nets (lower Cp)\n"
                "3. Add shielding for sensitive nodes\n"
                "4. Increase bias current to compensate for extra load\n"
                "5. Re-optimize compensation capacitor for post-layout poles"
            )},
        ],
    },
    # --- Full Signoff Flow ---
    {
        "id": "signoff_full_flow",
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "What is the complete signoff flow before tapeout?"},
            {"role": "assistant", "content": (
                "Complete IC signoff flow (pre-tapeout checklist):\n\n"
                "**1. Design Rule Check (DRC)** -- PVS\n"
                "   - Verify all geometries meet foundry rules\n"
                "   - Min width, spacing, enclosure, density\n"
                "   - Must be 100% clean (zero violations)\n\n"
                "**2. Layout vs Schematic (LVS)** -- PVS\n"
                "   - Verify layout matches schematic connectivity\n"
                "   - All devices matched, no shorts/opens\n"
                "   - Must be 100% clean\n\n"
                "**3. Antenna Check** -- PVS\n"
                "   - Verify no excessive charge buildup on gates during fabrication\n"
                "   - Fix: add antenna diodes or break long metal runs\n\n"
                "**4. Electrical Rule Check (ERC)** -- PVS\n"
                "   - No floating gates, missing well taps\n"
                "   - Proper power/ground connections\n\n"
                "**5. Parasitic Extraction** -- Quantus\n"
                "   - RC or RCC extraction\n"
                "   - Generate post-layout netlist\n\n"
                "**6. Post-Layout Simulation** -- Spectre\n"
                "   - Re-simulate with parasitics\n"
                "   - Verify all specs still met\n"
                "   - PVT corner sweep with parasitics\n\n"
                "**7. IR Drop Analysis** -- Voltus\n"
                "   - Verify power grid adequacy\n"
                "   - Static + dynamic IR drop\n\n"
                "**8. EM (Electromigration) Check**\n"
                "   - Verify current density in all metal layers\n"
                "   - Critical for reliability\n\n"
                "**9. Formal Verification** -- Conformal\n"
                "   - Verify equivalence between RTL and gate-level netlist\n\n"
                "**10. Static Timing Analysis** -- Tempus/SSV\n"
                "   - Setup/hold timing across all corners\n"
                "   - Clock domain crossing checks\n\n"
                "Each step uses specific Cadence tools:\n"
                "DRC/LVS/ERC: PVS222 | Extraction: QUANTUS231 | Sim: SPECTRE241\n"
                "Formal: CONFRML232 | STA: SSV231 | EM: within Spectre/Voltus"
            )},
        ],
    },
]


def main():
    output_path = "data/sft/signoff_v1.jsonl"

    print(f"\n{SEP}")
    print("   Generate Physical Verification / Signoff SFT Data")
    print(f"{SEP}\n")

    examples = []
    for ex in EXAMPLES:
        example = {
            "messages": ex["messages"],
            "source": "signoff_v1",
            "circuit_id": ex["id"],
            "domain": "signoff",
        }
        n_turns = len([m for m in ex["messages"] if m["role"] == "assistant"])
        print(f"  [{ex['id']}] {n_turns} assistant turns")
        examples.append(example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{SEP}")
    print(f"  Generated: {len(examples)} signoff examples")
    print(f"  Covers: DRC fix, LVS debug, parasitic extraction, full flow")
    print(f"  Saved: {output_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
