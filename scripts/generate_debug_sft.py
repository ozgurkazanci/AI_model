#!/usr/bin/env python3
"""Generate SFT data for error recovery and debugging scenarios.

The model needs to learn how to:
1. Diagnose simulation failures
2. Fix convergence issues
3. Handle spec violations
4. Debug LVS/DRC errors
5. Recover from bad initial designs

Creates 10 training examples.

Usage:
    PYTHONPATH=src python scripts/generate_debug_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from asic_ai.data.format import build_system_message
def build_examples():
    examples = []

    # 1. Convergence failure diagnosis
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My OTA simulation fails with 'no convergence at DC operating point'. How do I fix it?"},
            {"role": "assistant", "content": """DC convergence failure is one of the most common simulation issues. Let me diagnose:

Common causes:
1. **Missing DC path to ground** - floating nodes
2. **Incorrect biasing** - transistors in wrong region
3. **Too tight tolerances** - simulator can't find solution
4. **Feedback loop instability** - positive feedback at DC

Let me check the netlist for floating nodes:

<tool_call>
  {"name": "lint.check", "arguments": {"netlist": ".subckt ota vin_p vin_n vout vdd vss\\nM1 net1 vin_p tail vss nmos w=10u l=0.5u\\nM2 net2 vin_n tail vss nmos w=10u l=0.5u\\nM3 net1 net1 vdd vdd pmos w=20u l=0.5u\\nM4 net2 net1 vdd vdd pmos w=20u l=0.5u\\nM5 tail vbias vss vss nmos w=20u l=1u\\n.ends"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"errors": [{"node": "vbias", "line": 6, "message": "Node 'vbias' has no DC path to ground: driven by no source, only connected to gate of M5.", "severity": "error"}], "passed": False})},
            {"role": "assistant", "content": """Found the problem! The `vbias` node has no DC path - it's floating.

The tail current source M5 has its gate connected to `vbias` but there's no voltage source or bias circuit driving it.

Fix options:
1. Add a bias voltage source: `Vbias vbias vss 0.7`
2. Connect to a bias generator circuit
3. Use ideal current source: `Itail tail vss 100u`

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Add Vbias vbias vss 0.7 for tail current bias"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 1})},
            {"role": "assistant", "content": """Now let me re-run DC simulation:

<tool_call>
  {"name": "sim.dc", "arguments": {"netlist": "ota_fixed", "analysis_type": "dc"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"op_points": {"vout": 0.92, "tail": 0.35, "net1": 1.15, "net2": 0.92, "id_m5": 98e-6}})},
            {"role": "assistant", "content": """DC simulation converges! Operating point looks good:
- Vout = 0.92V (near mid-rail for 1.8V supply)
- Tail current = 98 uA (close to target 100 uA)
- Both diff pair transistors in saturation (Vds > Vgs-Vth)

General convergence debugging checklist:
1. Check all nodes have DC path to ground
2. Verify all voltage sources are connected
3. Add `.option reltol=1e-3` to relax tolerance
4. Use `nodesets` for initial guess: `.nodeset vout=0.9`
5. Try `.dc` sweep instead of `.op` to trace the issue"""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_convergence_001", "domain": "analog", "complexity": "medium",
    })

    # 2. Gain too low - systematic debug
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My folded cascode OTA shows only 35 dB gain. I need 60 dB. What's wrong?"},
            {"role": "assistant", "content": """35 dB gain for a folded cascode is unusually low. A properly designed folded cascode should achieve 60-80 dB.

Let me analyze the gain bottleneck. Folded cascode gain = gm1 * (Rout_p || Rout_n) where:
- Rout_p = gm_cascode_p * ro_cascode_p * ro_load_p
- Rout_n = gm_cascode_n * ro_cascode_n * ro_tail_n

Let me check operating points:

<tool_call>
  {"name": "sim.dc", "arguments": {"netlist": "folded_cascode_ota", "analysis_type": "op"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"op_points": {"gm_m1": 500e-6, "gm_m3": 200e-6, "ro_m1": 50e3, "ro_m3": 20e3, "ro_m5": 30e3, "gm_m5": 150e-6, "vds_m3": 0.15, "vgs_m3": 0.6}})},
            {"role": "assistant", "content": """Found the issue! M3 (PMOS cascode) has Vds = 0.15V.

With Vgs_m3 = 0.6V and typical Vth_p ~ 0.5V, Vov = 0.1V.
Vds = 0.15V < 2*Vov = 0.2V, so **M3 is in triode region!**

In triode, ro drops dramatically (from ~200k to ~20k), killing the gain.

The fix is to adjust the bias to keep M3 in saturation:

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Reduce PMOS cascode gate bias by 100mV: Vcasc_p from 1.0V to 0.9V"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 1})},
            {"role": "assistant", "content": """With adjusted bias, M3 Vds should increase to ~0.25V (well in saturation).

Expected gain improvement:
- ro_m3 increases from 20k to ~200k (10x)
- Rout_p = gm3 * ro3 * ro_load = 200u * 200k * 100k = huge
- Total gain = gm1 * Rout ~ 500u * 2M = 1000 = 60 dB

Key lesson: In cascode circuits, **always verify all cascode devices are in saturation**. Even 50mV of headroom loss can drop gain by 20+ dB."""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_gain_low_001", "domain": "analog", "complexity": "hard",
    })

    # 3. Phase margin too low - compensation fix
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My two-stage OTA has 15 degrees phase margin. It oscillates in closed loop. Fix it."},
            {"role": "assistant", "content": """15 degrees phase margin will definitely cause oscillation. Two-stage OTAs need Miller compensation.

Let me check if compensation exists and analyze the loop:

<tool_call>
  {"name": "sim.stb", "arguments": {"netlist": "two_stage_ota", "probe": "vout", "feedback_break": "vfb"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"phase_margin": 15.2, "gain_margin": 3.5, "loop_gain": {"name": "loop_gain", "x_values": [1e3, 1e6, 1e8, 5e8, 1e9], "y_values": [62, 40, 10, -3, -15]}})},
            {"role": "assistant", "content": """STB analysis confirms:
- Phase margin: 15.2 degrees (need > 60)
- GBW: ~500 MHz
- Two gain crossings visible (potential for conditional stability)

The problem is likely: no Miller compensation capacitor, or Cc too small.

For a two-stage OTA, the compensation capacitor should set:
- Second pole p2 > 2.2 * GBW for 60 degrees PM
- Cc > gm1 / (2.2 * GBW) = gm1 * Cl / gm2

Let me add proper Miller compensation:

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Add Cc=2pF between first and second stage output with Rz=500 ohm nulling resistor"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 2})},
            {"role": "assistant", "content": """With Cc=2pF and Rz=500 ohm:
- Dominant pole moves to ~gm1/(Av2*Cc) = low frequency
- Non-dominant pole at ~gm2/CL
- RHP zero moved to LHP by Rz > 1/gm2

Expected phase margin: ~65 degrees.

Compensation sizing rules of thumb:
- Cc > 0.2 * CL (for adequate pole splitting)
- Rz = 1/gm2 (to cancel RHP zero exactly)
- If PM still low: increase Cc or add cascode compensation"""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_phase_margin_001", "domain": "analog", "complexity": "hard",
    })

    # 4. Power consumption too high
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My bandgap reference draws 500 uA. Budget is 50 uA. How to reduce power 10x?"},
            {"role": "assistant", "content": """10x power reduction requires fundamental changes. Let me analyze where the current goes:

<tool_call>
  {"name": "sim.dc", "arguments": {"netlist": "bandgap_ref", "analysis_type": "op"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"op_points": {"i_r1": 200e-6, "i_r2": 200e-6, "i_opamp": 80e-6, "i_startup": 20e-6, "total": 500e-6}})},
            {"role": "assistant", "content": """Power breakdown:
- R1 branch: 200 uA (40%) - PTAT current through R1
- R2 branch: 200 uA (40%) - Complementary current  
- Opamp: 80 uA (16%) - Error amplifier
- Startup: 20 uA (4%) - Startup circuit

To reach 50 uA total:
1. **Scale resistors 10x**: R1, R2 from 10k to 100k -> current drops 10x to 20 uA/branch
2. **Use subthreshold opamp**: Bias at 5 uA instead of 80 uA
3. **Disable startup after lock**: Save 20 uA

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "R1: 10k->100k, R2: 10k->100k, opamp bias: 80uA->5uA, add startup disable"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 4})},
            {"role": "assistant", "content": """Expected new power: 20 + 20 + 5 + 2 = 47 uA (within 50 uA budget!)

Trade-offs of low-power bandgap:
- Higher R -> more area, more noise (thermal noise ~ 4kTR)
- Subthreshold opamp -> lower GBW, slower settling
- TC may degrade slightly (opamp offset contributes more)
- PSRR drops at high frequency (slower opamp loop)

Mitigation:
- Use chopping in opamp to reduce offset
- Add output buffer cap for noise filtering
- Run corner simulations to verify TC across PVT"""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_power_high_001", "domain": "analog", "complexity": "medium",
    })

    # 5. DRC violations - metal density
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "PVS DRC shows 147 metal density violations. How to fix efficiently?"},
            {"role": "assistant", "content": """Metal density violations are the most common DRC errors. They come in two types:
1. **Min density**: Not enough metal (needs fill)
2. **Max density**: Too much metal (needs slots)

For 147 violations, manual fixing is impractical. Use automated fill:

<tool_call>
  {"name": "lint.check", "arguments": {"type": "drc", "rules": "metal_density", "cell": "top_level"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"violations": 147, "by_layer": {"M1": 12, "M2": 23, "M3": 45, "M4": 38, "M5": 18, "M6": 11}, "type": {"min_density": 142, "max_density": 5}})},
            {"role": "assistant", "content": """Analysis: 142 min-density + 5 max-density violations.

Strategy:
1. **Auto-fill** for 142 min-density violations:
   - Use `metalFill` command with foundry-approved fill rules
   - Set keepout regions around sensitive analog nets (clock, bias)
   - Fill layers M3-M6 first (most violations)

2. **Manual fix** for 5 max-density violations:
   - Add slots/holes to wide metal traces
   - Split wide power buses into parallel narrower traces

3. **Prevention**: Add density-aware routing constraints

DRC fix procedure:

```tcl
# Step 1: Auto metal fill (handles 142 violations)
metalFill -rules drc_fill_rules.rs \\
  -keepout {analog_region bias_nets} \\
  -layers {M1 M2 M3 M4 M5 M6} \\
  -minSpace 0.5u -maxWidth 2u

# Step 2: Fix max density (5 violations)
# Add slots to wide buses > 10um
addSlots -layer M3 -width 0.3u -spacing 2u -minBusWidth 10u

# Step 3: Verify
verify drc -rules density_only
```

Critical: Keep fill away from sensitive nodes (< 2um spacing).
Grounded fill adds parasitic capacitance to signal nets."""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_drc_density_001", "domain": "analog", "complexity": "medium",
    })

    # 6. Corner failure analysis
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "LDO passes at TT 27C but fails output regulation at SS 125C. Why?"},
            {"role": "assistant", "content": """SS corner at 125C is the worst case for LDO regulation because:
1. Lower gm (slow silicon, and mobility falls as temperature rises)
2. Lower supply (SS is signed off at the low VDD limit, less headroom)
3. Lower loop gain -> worse regulation

Sign-off convention: every axis of a corner pushes the SAME way. SS is slow
silicon at LOW VDD and HOT (125C); FF is fast silicon at HIGH VDD and COLD
(-40C). Pairing SS with -40C cancels part of the corner against itself and
understates the real spread.

Let me run corner analysis:

<tool_call>
  {"name": "sim.corners", "arguments": {"netlist": "ldo_regulator", "corners": ["tt_27", "ss_125", "ff_n40"], "analysis": "ac"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"corners": {"tt_27": {"loop_gain": 55, "ugb": 2e6, "phase_margin": 62}, "ss_125": {"loop_gain": 32, "ugb": 800e3, "phase_margin": 71}, "ff_n40": {"loop_gain": 65, "ugb": 5e6, "phase_margin": 48}}})},
            {"role": "assistant", "content": """Corner comparison:

| Parameter | TT 27C | SS 125C | FF -40C |
|-----------|--------|---------|---------|
| Loop gain | 55 dB | 32 dB | 65 dB |
| UGB | 2 MHz | 800 kHz | 5 MHz |
| Phase margin | 62 deg | 71 deg | 48 deg |

**Root cause at SS 125C**: Loop gain drops 23 dB (55->32 dB)!
- gm reduction: ~40% (mobility falls as temperature rises)
- Headroom loss: SS is signed off at the low VDD limit
- ro decrease: output impedance drops at high temperature

Fix: Increase error amp bias current at the hot corner using CTAT-compensated
bias, so gm does not track mobility down:

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Replace constant bias with a constant-gm bias so Ibias rises with temperature and compensates the gm drop at SS 125C"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 1})},
            {"role": "assistant", "content": """With constant-gm bias, gm stays roughly constant across temperature:
- gm = sqrt(2 * mu * Cox * (W/L) * Id)
- mu decreases with T, but the constant-gm bias raises Id with T
- Net effect: gm variation < 20% across corners

Also fix FF -40C phase margin (48 deg is marginal):
- FF is fast silicon at high VDD and cold, so gm and UGB are highest there
- Increase the compensation capacitor, or use adaptive Miller compensation

Design rule: Always simulate all corners BEFORE tapeout.
Min specs should be met at worst-case corner, not typical."""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_corner_fail_001", "domain": "analog", "complexity": "hard",
    })

    # 7. LVS mismatch debug
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "LVS reports 3 shorts and 2 opens. How do I find and fix them?"},
            {"role": "assistant", "content": """LVS shorts and opens are layout-vs-schematic mismatches. Let me analyze:

<tool_call>
  {"name": "lint.check", "arguments": {"type": "lvs", "cell": "amplifier_top"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "INCORRECT", "shorts": [{"net1": "VDD", "net2": "net_bias", "layer": "M2"}, {"net1": "OUT", "net2": "VREF", "layer": "M3"}, {"net1": "VSS", "net2": "GND", "layer": "M1"}], "opens": [{"net": "feedback", "missing": "via12 at (45.2, 23.1)"}, {"net": "clk", "missing": "M1-M2 connection at (12.5, 67.3)"}]})},
            {"role": "assistant", "content": """LVS errors analyzed:

**SHORTS (3):**
1. VDD-net_bias on M2: Metal2 routing overlaps bias wire with power rail
   - Fix: Reroute bias wire on M3, add spacing to VDD
2. OUT-VREF on M3: Output and reference traces too close
   - Fix: Add 0.5um spacing or move VREF to M4
3. VSS-GND on M1: These might be intentionally connected
   - Check: Are VSS and GND separate in schematic? If same net, this is not an error

**OPENS (2):**
4. feedback: Missing via12 at (45.2, 23.1)
   - Fix: Add via1-via2 stack at the transition point
5. clk: M1-M2 connection missing at (12.5, 67.3)
   - Fix: Add via1 to connect M1 clk to M2 clk

Most common causes:
- Shorts: Insufficient spacing between dense routes
- Opens: Missing vias at layer transitions
- Both: Manual edits that break connectivity

After fixes, re-run LVS to verify clean."""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_lvs_mismatch_001", "domain": "analog", "complexity": "medium",
    })

    # 8. Noise too high - systematic reduction
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My sensor readout amplifier has 50 uV RMS input noise. Need < 10 uV. How?"},
            {"role": "assistant", "content": """5x noise reduction requires understanding the noise sources. Let me analyze:

<tool_call>
  {"name": "sim.noise", "arguments": {"netlist": "readout_amp", "output_node": "vout", "input_source": "vin", "freq_start": "0.1", "freq_stop": "100k"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"frequencies": [0.1, 1, 10, 100, 1e3, 1e4, 1e5], "input_noise": {"name": "input_noise_density", "x_values": [0.1, 1, 10, 100, 1e3, 1e4, 1e5], "y_values": [1e-6, 3e-7, 1e-7, 5e-8, 3e-8, 2e-8, 2e-8]}, "output_noise": {"name": "output_noise", "x_values": [0.1, 1, 10, 100, 1e3, 1e4, 1e5], "y_values": [1e-4, 3e-5, 1e-5, 5e-6, 3e-6, 2e-6, 2e-6]}})},
            {"role": "assistant", "content": """Noise analysis shows:
- 1/f corner at ~1 kHz
- White noise floor: 20 nV/sqrt(Hz)
- 1/f noise dominates below 1 kHz

Integrated noise (0.1 Hz to 100 kHz) = 50 uV RMS

Noise breakdown by source (typical):
- Input pair 1/f: 60% -> 30 uV
- Input pair thermal: 20% -> 10 uV
- Load/cascode: 15% -> 7.5 uV
- Bias circuits: 5% -> 2.5 uV

To reach < 10 uV, attack the dominant source (1/f noise):

Strategy 1: **Increase input pair W*L** (1/f noise ~ 1/(W*L))
- 5x area increase -> sqrt(5) = 2.2x noise reduction
- Combined with chopping -> 10x reduction possible

Strategy 2: **Chopper stabilization**
- Modulate input to high frequency (above 1/f corner)
- Demodulate at output -> moves 1/f noise out of band

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Add chopper modulation at 10 kHz, increase M1/M2 W from 10u to 50u"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 3})},
            {"role": "assistant", "content": """With chopping at 10 kHz + 5x larger input pair:
- 1/f noise eliminated from DC-10kHz band
- Thermal noise: 20/sqrt(5) = 9 nV/sqrt(Hz)
- Integrated (0.1-100kHz): ~8 uV RMS - meets target!

Trade-offs:
- 5x larger input pair = more area and Cgs
- Chopper adds clock feedthrough spikes
- Need low-pass filter after demodulation
- Ripple at 2*fchop = 20 kHz (filter with 10 kHz LPF)"""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_noise_high_001", "domain": "analog", "complexity": "hard",
    })

    # 9. Simulation setup error
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "AC simulation shows flat 0 dB gain. The circuit definitely has gain. What's wrong?"},
            {"role": "assistant", "content": """Flat 0 dB usually means the AC stimulus isn't reaching the circuit properly. Common mistakes:

1. AC source magnitude = 0 (forgot `AC 1`)
2. Wrong node connections
3. DC bias missing (transistors off)
4. Output measured at wrong node

Let me check the testbench:

<tool_call>
  {"name": "lint.check", "arguments": {"netlist": "Vin in 0 DC 0.9\\n.ac dec 20 1 1G\\nX1 in out vdd vss amp\\nVdd vdd 0 1.8\\nCL out 0 1p"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"errors": [{"node": "Vin", "line": 1, "message": "Source 'Vin' has no AC magnitude but an .ac analysis is requested: AC stimulus is zero.", "severity": "error"}], "passed": False})},
            {"role": "assistant", "content": """Found it! The voltage source `Vin` has `DC 0.9` but **no AC component**.

Should be: `Vin in 0 DC 0.9 AC 1`

Without `AC 1`, the AC stimulus is zero, so all AC voltages are zero (0 dB when measuring vout/vin = 0/0, simulator returns 1).

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Fix: Vin in 0 DC 0.9 AC 1"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 1})},
            {"role": "assistant", "content": """Other common AC simulation gotchas:

1. **Missing AC keyword**: `V1 in 0 DC 0.9` -> add `AC 1`
2. **Wrong probe**: `.print ac v(out)` not `.print dc v(out)`
3. **Coupling cap**: If input is AC-coupled, check cap value isn't too small
4. **Frequency range**: `.ac dec 20 1 1G` - make sure range covers BW
5. **Ground reference**: AC gain = V(out)/V(in) - both must reference ground

After fixing, expect to see proper frequency response with gain rolloff."""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_ac_setup_001", "domain": "analog", "complexity": "easy",
    })

    # 10. Matching and mismatch effects
    examples.append({
        "messages": [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": "My current mirror has 5% mismatch. How to improve to < 1%?"},
            {"role": "assistant", "content": """5% current mirror mismatch is typical for minimum-size devices. Sources:

1. **Vth mismatch**: sigma(Vth) = Avt / sqrt(W*L)
2. **Beta mismatch**: sigma(beta)/beta = Abeta / sqrt(W*L)
3. **Channel length modulation**: if Vds1 != Vds2

For sky130: Avt ~ 5 mV*um, Abeta ~ 1%*um

Let me run Monte Carlo:

<tool_call>
  {"name": "sim.mc", "arguments": {"netlist": "current_mirror", "num_runs": 500, "vary": "mismatch", "param": "imirror_ratio"}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"monte_carlo": {"num_runs": 500, "param": "imirror_ratio", "mean": 1.002, "std": 0.051, "min": 0.87, "max": 1.15, "yield": 0.68}})},
            {"role": "assistant", "content": """MC confirms: sigma = 5.1%, 3-sigma = 15.3% (bad!)

Current mirror W=2u, L=0.5u -> W*L = 1 um^2

To reach < 1% (sigma):
- Need 5x reduction -> 25x area increase (sigma ~ 1/sqrt(WL))
- W=10u, L=2.5u -> W*L = 25 um^2 -> sigma = 1%

But 25x area is expensive. Better approaches:

1. **Cascode mirror**: Reduces Vds sensitivity (CLM effect)
2. **Longer L**: L=2u instead of 0.5u gives 4x area improvement
   - W=5u, L=2u -> sigma ~ 1.6%
3. **Common-centroid layout**: Cancels systematic gradients
4. **Dummy devices**: Reduces edge effects on outer devices

<tool_call>
  {"name": "netlist.patch", "arguments": {"changes": "Increase L from 0.5u to 2u, W from 2u to 5u. Add cascode devices."}}
</tool_call>"""},
            {"role": "tool", "content": json.dumps({"status": "applied", "changes": 4})},
            {"role": "assistant", "content": """With L=2u, W=5u, cascode:
- Mismatch (Vth): sigma ~ 5mV/sqrt(10) = 1.58 mV
- Current mismatch: sigma ~ 1.58mV / (2*Vov) ~ 1.6% (Vov=50mV)
- Cascode further reduces CLM contribution

Combined with common-centroid layout: achievable < 1%.

Layout rules for matching:
- Same orientation (both vertical or both horizontal)
- Common-centroid (ABBA pattern)
- Dummy devices on edges
- Same metal routing for drain connections
- Keep devices in same well"""}
        ],
        "source": "debug_v1", "circuit_id": "dbg_mismatch_001", "domain": "analog", "complexity": "medium",
    })

    return examples


def main():
    output_path = "data/sft/debug_v1.jsonl"
    examples = build_examples()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} debug/error-recovery SFT examples -> {output_path}")
    for ex in examples:
        print(f"  {ex['circuit_id']}: {ex['messages'][1]['content'][:60]}...")


if __name__ == "__main__":
    main()
