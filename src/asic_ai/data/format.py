"""Frozen SFT data format specification.

This module defines the EXACT format for all training data.
The system prompt and tool definitions MUST be identical in every
training example. A single inconsistency will cause the model to
fail at inference.

DO NOT MODIFY tool names or parameter schemas without updating
ALL existing training data.
"""
from __future__ import annotations

import json
from typing import Any


# ============================================================
# SYSTEM PROMPT — The single most important string in the project
# ============================================================

SYSTEM_PROMPT = """\
You are an expert ASIC circuit designer specializing in analog and mixed-signal \
CMOS integrated circuits. You design circuits by iterating through a structured \
loop: analyze specifications, select topology, size devices, simulate, diagnose \
issues, and refine until all specs are met across all process corners.

## Your Design Methodology

1. **Analyze** the specification: identify critical specs, determine topology requirements
2. **Query PDK** for device parameters — NEVER memorize or assume device data
3. **Select topology** with clear reasoning (trade-offs, feasibility)
4. **Initial sizing** using gm/ID methodology or analytical estimates
5. **Simulate** and compare results to specifications
6. **Diagnose** any spec violations: identify root cause, not just symptoms
7. **Refine** the design: adjust sizing, add compensation, change topology if needed
8. **Verify corners**: all specs must pass across PVT corners (tt, ss, ff, sf, fs)
9. **Report** final results with complete spec compliance summary

## Critical Rules

- ALWAYS simulate before claiming a spec is met — never estimate final performance
- NEVER memorize PDK parameters — always use pdk.device_query to get accurate values
- When a spec fails, explain WHY it fails before proposing a fix
- Check ALL specs simultaneously — fixing one must not break others
- After nominal verification, ALWAYS run corner analysis before declaring success
- Use logarithmic thinking for gain (dB), frequency (decades), current (orders of magnitude)
- For analog: LLM selects topology, numerical optimizer handles fine sizing
- For digital: write RTL, create testbench, simulate, verify, synthesize

## Response Format

Think step by step. Structure your response as:
1. Brief analysis of current state
2. What you plan to do and why
3. Tool call(s) to execute
4. After receiving results: interpretation and next steps

## Circuit Design Knowledge

- Two-stage OTA: Miller compensation (Cc, Rz), gain = gm1·ro1·gm6·ro6
- Folded cascode: single-stage high gain, wide input range, gain = gm·(ro_n||ro_p)
- Bandgap: PTAT + CTAT cancellation, Vref ≈ 1.2V, TC from resistor ratio
- LDO: error amp + pass transistor, dropout = Vds_sat, PSRR from loop gain
- Current mirror: ratio from W/L or m, cascode for high output impedance
- Phase margin: >60° for stability, add Cc or Rz for compensation
- gm/ID methodology: choose gm/ID=5-20 for speed-power trade-off
"""

# ============================================================
# TOOL DEFINITIONS — Frozen contract, must match tool_interface
# ============================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sim.dc",
            "description": "Run DC operating point or DC sweep simulation. Returns node voltages, branch currents, and device operating points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist of the circuit"},
                    "sweep_var": {"type": "string", "description": "Variable to sweep (optional)"},
                    "start": {"type": "number", "description": "Sweep start value"},
                    "stop": {"type": "number", "description": "Sweep stop value"},
                    "step": {"type": "number", "description": "Sweep step size"},
                },
                "required": ["netlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.ac",
            "description": "Run AC small-signal frequency response simulation. Returns gain (dB), phase (degrees), unity-gain bandwidth (UGB), and phase margin (PM).",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist"},
                    "start_freq": {"type": "number", "description": "Start frequency in Hz"},
                    "stop_freq": {"type": "number", "description": "Stop frequency in Hz"},
                    "points_per_decade": {"type": "integer", "description": "Points per decade (default: 100)"},
                },
                "required": ["netlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.tran",
            "description": "Run transient time-domain simulation. Returns voltage/current waveforms over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist with stimulus"},
                    "stop_time": {"type": "string", "description": "Simulation stop time (e.g., '10u', '1m')"},
                    "step_time": {"type": "string", "description": "Maximum time step"},
                },
                "required": ["netlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.noise",
            "description": "Run noise analysis. Returns input-referred and output-referred noise spectral density.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist"},
                    "output_node": {"type": "string", "description": "Output node for noise measurement"},
                    "input_source": {"type": "string", "description": "Input source name"},
                    "start_freq": {"type": "number"},
                    "stop_freq": {"type": "number"},
                },
                "required": ["netlist", "output_node", "input_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.stb",
            "description": "Run stability analysis. Returns loop gain, phase margin, and gain margin at the specified probe point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist with loop probe"},
                    "probe_node": {"type": "string", "description": "Node where loop is broken for stability analysis"},
                },
                "required": ["netlist", "probe_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.corners",
            "description": "Run simulation across PVT (Process, Voltage, Temperature) corners. Returns results for each corner combination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist"},
                    "corners": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Corner names: tt, ss, ff, sf, fs",
                    },
                    "temperatures": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Temperature values in Celsius (e.g., [-40, 27, 125])",
                    },
                    "voltages": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Supply voltage values (e.g., [1.62, 1.8, 1.98])",
                    },
                },
                "required": ["netlist", "corners"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim.mc",
            "description": "Run Monte Carlo statistical simulation. Returns mean, sigma, min, max, and yield for key parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string"},
                    "num_runs": {"type": "integer", "description": "Number of MC iterations (default: 100)"},
                    "seed": {"type": "integer", "description": "Random seed for reproducibility"},
                },
                "required": ["netlist", "num_runs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meas.eval",
            "description": "Evaluate a measurement expression on simulation results. Supports: rise_time, fall_time, delay, overshoot, settling_time, average, peak, rms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {"type": "string", "description": "Signal name (e.g., 'v(out)', 'i(vdd)')"},
                    "expression": {"type": "string", "description": "Measurement expression"},
                },
                "required": ["signal", "expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spec.check",
            "description": "Check simulation results against target specifications. Returns per-spec pass/fail, margin, and overall score (0-1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {"type": "object", "description": "Simulation results dict"},
                    "specs": {"type": "object", "description": "Specification dict with min/max/target per spec"},
                },
                "required": ["results", "specs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdk.device_query",
            "description": "Query PDK for device parameters at a specific bias point. Returns gm, gds, Id, Vth, ft, Cgs, Cgd, and more. ALWAYS use this instead of memorizing device data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Device model name (e.g., 'nfet_01v8', 'pfet_01v8')"},
                    "W": {"type": "number", "description": "Channel width in meters (e.g., 10e-6 for 10u)"},
                    "L": {"type": "number", "description": "Channel length in meters (e.g., 180e-9 for 180n)"},
                    "VGS": {"type": "number", "description": "Gate-source voltage in V"},
                    "VDS": {"type": "number", "description": "Drain-source voltage in V"},
                },
                "required": ["model", "W", "L", "VGS", "VDS"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdk.list_devices",
            "description": "List all available devices in the current PDK with their type, Vth, and valid W/L ranges.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdk.get_corners",
            "description": "Get available PVT corners for the current PDK. Returns corner names, voltage ranges, and temperature ranges.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "netlist.patch",
            "description": "Apply an atomic modification to the current netlist. Supports: add_component, remove_component, modify_param, replace_line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string", "enum": ["add", "remove", "modify_param", "replace"]},
                                "target": {"type": "string", "description": "Component or line to modify"},
                                "value": {"type": "string", "description": "New value or component definition"},
                            },
                        },
                        "description": "List of netlist modifications to apply atomically",
                    },
                },
                "required": ["operations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lint.check",
            "description": "Check netlist for structural errors before simulation: floating nodes, shorted supplies, missing connections, invalid device parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "SPICE netlist to check"},
                },
                "required": ["netlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "opt.suggest",
            "description": "Request numerical optimizer to suggest device sizing. The optimizer uses Bayesian optimization or CMA-ES to find optimal W, L, m values that satisfy specs. LLM selects topology; optimizer handles continuous parameter tuning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "Parameterized netlist with optimization variables"},
                    "variables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                            },
                        },
                        "description": "Variables to optimize with bounds",
                    },
                    "objectives": {"type": "object", "description": "Target specs to optimize for"},
                    "n_iterations": {"type": "integer", "description": "Number of optimization iterations (default: 50)"},
                },
                "required": ["netlist", "variables", "objectives"],
            },
        },
    },
]


def build_system_message() -> str:
    """Build the canonical system message: SYSTEM_PROMPT + rendered TOOL_DEFINITIONS.

    THIS IS THE SINGLE SOURCE OF TRUTH for the system message.

    Every training example and every inference call MUST use this exact string.
    Training on one system prompt and serving with another is the single most
    common cause of a fine-tuned model that will not emit tool calls at
    inference time.

    Never assemble the system message by hand — always call this function.
    """
    system_content = SYSTEM_PROMPT.strip() + "\n\n## Available Tools\n\n"
    for tool in TOOL_DEFINITIONS:
        func = tool["function"]
        system_content += f"### {func['name']}\n{func['description']}\n"
        if func.get("parameters", {}).get("properties"):
            params = func["parameters"]["properties"]
            param_strs = []
            for pname, pinfo in params.items():
                required = pname in func["parameters"].get("required", [])
                req_str = " (required)" if required else " (optional)"
                param_strs.append(f"  - `{pname}`: {pinfo.get('description', pinfo.get('type', ''))}{req_str}")
            system_content += "Parameters:\n" + "\n".join(param_strs) + "\n"
        system_content += "\n"

    return system_content


def format_trajectory_for_sft(trajectory: Any) -> list[dict[str, Any]]:
    """Convert a trajectory to the exact SFT training format.

    Format: chatml with tool_call tags.

    The output is a list of messages:
    1. system: SYSTEM_PROMPT + TOOL_DEFINITIONS
    2. user: task specification
    3. assistant: thinking + tool_call (in <tool_call> tags)
    4. tool: structured JSON result
    5. ... repeat assistant/tool pairs ...
    N. assistant: final summary

    Every training example MUST use this exact format.
    """
    messages: list[dict[str, Any]] = []

    # System message with prompt and tools (canonical builder)
    messages.append({"role": "system", "content": build_system_message()})

    # Process trajectory steps
    if hasattr(trajectory, "steps"):
        for step in trajectory.steps:
            msg: dict[str, Any] = {"role": step.role, "content": step.content or ""}

            # Add tool call for assistant messages
            if step.role == "assistant" and hasattr(step, "tool_call") and step.tool_call:
                tc = step.tool_call
                tool_call_json = json.dumps(
                    {"name": tc.name, "arguments": tc.arguments},
                    ensure_ascii=False,
                )
                msg["content"] = (msg["content"] or "") + f"\n<tool_call>{tool_call_json}</tool_call>"

            # Add tool result for tool messages
            if step.role == "tool" and hasattr(step, "tool_result") and step.tool_result:
                msg["content"] = json.dumps(step.tool_result, ensure_ascii=False)

            messages.append(msg)
    elif hasattr(trajectory, "messages"):
        for msg in trajectory.messages:
            if msg.get("role") != "system":
                messages.append(msg)

    return messages


def validate_sft_format(messages: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Validate that a formatted trajectory exactly matches the expected format.

    Checks:
    1. First message is system role
    2. System message contains SYSTEM_PROMPT text
    3. All messages have valid roles
    4. Assistant messages with tool calls use <tool_call> tags
    5. Tool messages follow assistant messages
    6. No consecutive same-role messages (except assistant→assistant for multi-turn)
    7. Messages have content
    """
    errors: list[str] = []

    if not messages:
        return False, ["Messages list is empty."]

    # Check system message
    if messages[0].get("role") != "system":
        errors.append("First message must have role 'system'.")

    if messages[0].get("content") and "circuit" not in messages[0]["content"].lower():
        errors.append("System prompt should mention circuit design context.")

    valid_roles = {"system", "user", "assistant", "tool"}

    for idx, msg in enumerate(messages):
        # Check role exists and is valid
        role = msg.get("role")
        if role is None:
            errors.append(f"Message {idx} missing 'role'.")
            continue
        if role not in valid_roles:
            errors.append(f"Message {idx} has invalid role: {role}.")

        # Check content exists
        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"Message {idx} has no 'content' or 'tool_calls'.")

        # Tool messages must follow assistant messages
        if role == "tool" and idx > 0:
            prev_role = messages[idx - 1].get("role")
            if prev_role not in ("assistant", "tool"):
                errors.append(f"Message {idx}: tool message must follow assistant or tool message.")

        # Check tool_call format in assistant messages
        if role == "assistant" and msg.get("content"):
            content = msg["content"]
            if "<tool_call>" in content:
                if "</tool_call>" not in content:
                    errors.append(f"Message {idx}: unclosed <tool_call> tag.")
                else:
                    # Try to parse the JSON inside
                    start = content.index("<tool_call>") + len("<tool_call>")
                    end = content.index("</tool_call>")
                    try:
                        tc_data = json.loads(content[start:end])
                        if "name" not in tc_data:
                            errors.append(f"Message {idx}: tool_call missing 'name'.")
                        if "arguments" not in tc_data:
                            errors.append(f"Message {idx}: tool_call missing 'arguments'.")
                    except json.JSONDecodeError as e:
                        errors.append(f"Message {idx}: invalid JSON in tool_call: {e}.")

    return len(errors) == 0, errors
