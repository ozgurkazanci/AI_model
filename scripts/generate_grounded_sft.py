#!/usr/bin/env python3
"""Generate SFT data by DRIVING THE REAL ENVIRONMENT, not by imagining it.

Why this generator exists -- the 945ex eval, measured on 77 tasks:
the model passed a real netlist in 5 of 357 sim calls, called spec.check
with its required arguments 0 times out of 5, and retried the byte-identical
failing call up to 11 times. Every one of those behaviours traces to the
training data: batch_v1/v2 (600 of 945 examples) taught sim.* WITHOUT a
netlist argument and answered it with fabricated success data, and the
corpus contained 13 error->recovery sequences against 56 spec.check calls.
The model did exactly what it was taught.

So here every trajectory is executed against CircuitDesignEnv with the real
ngspice adapter before it is written:

  - every sim.* call carries the FULL deck inline (or the deck minus its
    analysis card plus the contract's parameter spelling -- both flavours);
  - every tool observation is the byte-exact string env.step() returned,
    including every error message in the recovery examples;
  - every number the assistant "reads" is computed from that observation;
  - spec.check is called with results= those measured values and specs= the
    task's specs, and its observation is the env's real verdict;
  - a trajectory whose sim fails unexpectedly is DROPPED, not patched.

Nothing is imagined. If ngspice is not installed this script refuses to run.

    PYTHONPATH=src python scripts/generate_grounded_sft.py \
        --output data/sft/grounded_v1.jsonl --per-circuit 40
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asic_ai.data.format import build_system_message, validate_sft_format

SEP = "=" * 70


# ------------------------------------------------------------ circuit bank ---

def _load_circuits() -> list[dict]:
    """The runnable decks from the two real-ngspice generators, normalised.

    Imported from the scripts that own them (single source); each has a
    complete deck: .model cards with public level-1 parameters, supplies,
    stimulus, and an analysis card that real ngspice executes.
    """
    bank = []
    for fname, attr in (("generate_ngspice_sft.py", "CIRCUIT_LIBRARY"),
                        ("generate_ngspice_sft_v2.py", "CIRCUITS_V2")):
        spec = importlib.util.spec_from_file_location(fname[:-3],
                                                      REPO_ROOT / "scripts" / fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for c in getattr(mod, attr):
            if not (c.get("netlist") and c.get("tool_name", "").startswith("sim.")):
                continue
            # A nested .dc (two swept sources) flattens every vector into a
            # 2-D grid: slope/swing along that axis measures across the outer
            # sweep too, and split_card cannot represent the second sweep in
            # contract params. The adapter itself warns about this; skip.
            m = _CARD_RE.search(c["netlist"])
            if m and m.group(1).lower() == "dc" and len(m.group(2).split()) > 4:
                continue
            # Ring-oscillator .tran decks are excluded: even mild perturbation
            # can stop the oscillation, and ngspice then grinds the timestep
            # down for CPU-minutes while the in-process DLL accumulates the
            # vectors (observed: 14 GB RSS on one variant, and a 45-example
            # trial that had not finished after 25 minutes). sim.tran coverage
            # comes from RC_STEP below, which settles in milliseconds.
            if m and m.group(1).lower() == "tran":
                continue
            bank.append({"id": c["id"], "task": c["task"],
                         "netlist": _inject_save(
                             _compact_sweep(c["netlist"]), c["id"]),
                         "tool": c["tool_name"]})
    bank.append(RC_STEP)
    return bank


# The vectors worth keeping, per deck -- probed from a real run of each.
# Without .save ngspice returns EVERY node and branch: 5-8 vectors of full
# double-precision floats per analysis, ~12 KB per observation, which pushed
# the median training example to 6039 tokens (the system message is 1750).
# Saving the 1-3 informative vectors keeps the whole observation readable
# inside the serving loop's 4000-char cut -- and teaches the model to write
# .save lines itself.
SAVES = {
    "cs_amp_sizing": "v(out) i(vdd)",
    "inv_vtc": "v(out)",
    "rc_bw_check": "v(out)",
    "diff_pair_offset": "v(outp) v(outn)",
    "bandgap_temp": "v(col1) v(col2)",
    "current_mirror": "i(vds) v(drain2)",
    "cascode_cs": "v(out) i(vdd)",
    "source_follower": "v(out) i(vdd)",
    "pmos_cs_load": "v(out) i(vdd)",
    "rc_integrator": "v(out)",
    "widlar_mirror": "i(vds) v(drain1)",
    "voltage_divider_precision": "v(out) i(vdd)",
}


def _inject_save(netlist: str, circuit_id: str) -> str:
    vectors = SAVES.get(circuit_id)
    if not vectors or re.search(r"^\s*\.save\b", netlist,
                                re.IGNORECASE | re.MULTILINE):
        return netlist
    return re.sub(r"^\s*\.end\s*$", f".save {vectors}\n.end", netlist,
                  count=1, flags=re.IGNORECASE | re.MULTILINE)


def _compact_sweep(netlist: str) -> str:
    """Thin dense sweeps so a full observation stays under the 4000-char
    truncation the serving loop applies (runner.py cuts every tool message at
    observation[:4000]). A 20-points-per-decade AC sweep serialises to ~15 KB
    -- the model would be trained to read vectors it can never see whole at
    eval time. 6 points/decade and <=40 DC steps keep the physics (the -3 dB
    crossing and the sweep slope survive) and the whole vector visible."""
    def thin_ac(m):
        return f"{m.group(1)}6{m.group(3)}"

    out = re.sub(r"(^\s*\.ac\s+dec\s+)(\d+)(\s)", thin_ac, netlist,
                 flags=re.IGNORECASE | re.MULTILINE)

    def widen_dc(m):
        head, start, stop, step = m.group(1), _num(m.group(2)), _num(m.group(3)), _num(m.group(4))
        min_step = abs(stop - start) / 40.0
        if step >= min_step:
            return m.group(0)
        return f"{head}{m.group(2)} {m.group(3)} {min_step:g}"

    out = re.sub(r"(^\s*\.dc\s+\S+\s+)(\S+)\s+(\S+)\s+(\S+)\s*$", widen_dc,
                 out, flags=re.IGNORECASE | re.MULTILINE)
    return out


# A transient deck that is fast BY CONSTRUCTION: single pole, 20 time
# constants, ~100 points. Same deck family as the live adapter test that
# asserts the step settles to 1 V.
RC_STEP = {
    "id": "rc_step_response",
    "task": ("Verify the step response of an RC low-pass driving a 1 nF "
             "load through 1 kOhm: the output must settle to the 1 V input "
             "step with plenty of margin inside 20 us."),
    "tool": "sim.tran",
    "netlist": (
        "* RC step response\n"
        "V1 in 0 PULSE(0 1 1u 1n 1n 1m 2m)\n"
        "R1 in out 1k\n"
        "C1 out 0 1n\n"
        ".tran 0.2u 20u\n"
        ".save v(out) v(in)\n"
        ".end\n"
    ),
}


_CARD_RE = re.compile(r"^\s*\.(dc|ac|tran)\s+(.*)$", re.IGNORECASE | re.MULTILINE)


def split_card(netlist: str) -> tuple[str, str | None, dict | None]:
    """(deck without its analysis card, kind, contract params for the card).

    Turning the deck's own card into the CONTRACT's parameter spelling is what
    teaches the model the second calling convention -- the one the 945ex model
    reached for 214 times with the wrong argument names.
    """
    m = _CARD_RE.search(netlist)
    if not m:
        return netlist, None, None
    kind, rest = m.group(1).lower(), m.group(2).split()
    stripped = netlist[:m.start()] + netlist[m.end():]
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    try:
        if kind == "ac":         # .ac dec N fstart fstop
            return stripped, kind, {"points_per_decade": int(rest[1]),
                                    "start_freq": _num(rest[2]),
                                    "stop_freq": _num(rest[3])}
        if kind == "tran":       # .tran tstep tstop
            return stripped, kind, {"step_time": _num(rest[0]),
                                    "stop_time": _num(rest[1])}
        if kind == "dc":         # .dc SRC start stop step
            return stripped, kind, {"sweep_var": rest[0],
                                    "start": _num(rest[1]),
                                    "stop": _num(rest[2]),
                                    "step": _num(rest[3])}
    except (IndexError, ValueError):
        return netlist, None, None
    return netlist, None, None


_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3,
           "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


def _num(tok: str) -> float:
    tok = tok.lower()
    m = re.match(r"^([-+]?[0-9.]+(?:e[-+]?\d+)?)(meg|[tgkmunpf])?", tok)
    if not m:
        raise ValueError(tok)
    return float(m.group(1)) * _SUFFIX.get(m.group(2) or "", 1.0)


# -------------------------------------------------------- variant synthesis --

_W_RE = re.compile(r"(W=)([0-9.]+)(u)", re.IGNORECASE)
_RES_RE = re.compile(r"^(R\w+\s+\S+\s+\S+\s+)([0-9.]+)(k?)\s*$",
                     re.IGNORECASE | re.MULTILINE)


def perturb(netlist: str, rng: random.Random) -> str:
    """A variant with scaled device widths / resistors. The env decides
    whether it still simulates; a variant that fails is dropped upstream.

    Transient decks are perturbed GENTLY: a ring oscillator whose device
    widths are scaled 1.6x can stop oscillating cleanly, and ngspice then
    grinds the timestep down and accumulates gigabytes of internal vectors
    (observed: 14 GB RSS and 10 CPU-minutes on one variant of
    ring_osc_3stage). W stays fixed there; only resistors move, mildly.
    """
    is_tran = bool(re.search(r"^\s*\.tran\b", netlist,
                             re.IGNORECASE | re.MULTILINE))

    def scale_w(m):
        return f"{m.group(1)}{float(m.group(2)) * rng.uniform(0.7, 1.6):.2g}{m.group(3)}"

    def scale_r(m):
        lo, hi = (0.9, 1.15) if is_tran else (0.75, 1.4)
        return f"{m.group(1)}{float(m.group(2)) * rng.uniform(lo, hi):.3g}{m.group(3)}"

    out = netlist if is_tran else _W_RE.sub(scale_w, netlist)
    if is_tran or rng.random() < 0.5:
        out = _RES_RE.sub(scale_r, out)
    return out


def strengthen(netlist: str, rng: random.Random) -> str:
    """The 'improve the design' edit for iterate trajectories: wider devices."""
    return _W_RE.sub(lambda m: f"{m.group(1)}{float(m.group(2)) * rng.uniform(1.3, 1.8):.2g}{m.group(3)}",
                     netlist)


# ------------------------------------------------------------- measurement ---

def measure(kind: str, observation: str,
            sweep_var: str | None = None) -> dict[str, float]:
    """Spec-keyed metrics computed from the REAL observation JSON.

    These are the numbers the assistant 'reads' in its prose and passes to
    spec.check as results=. They come from the simulator's own vectors --
    the generator never invents a value the deck did not produce.
    """
    try:
        data = json.loads(observation)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict) or data.get("error"):
        return {}
    out: dict[str, float] = {}

    def pick_signal(signals: dict) -> tuple[str, list, list] | None:
        for name, s in signals.items():
            low = name.lower()
            if "out" in low or low.startswith("v("):
                return name, s.get("x_values") or [], s.get("y_values") or []
        for name, s in signals.items():
            return name, s.get("x_values") or [], s.get("y_values") or []
        return None

    if kind == "ac" and data.get("frequencies"):
        freqs = data["frequencies"]
        signals = data.get("signals") or {}
        # The adapter names AC outputs vdb(<node>)/vp(<node>): the values ARE
        # decibels already. Taking 20*log10 of a dB value once turned a clean
        # 0 dB passband into -147 dB here; read vdb directly.
        dbs = None
        for name, s in signals.items():
            low = name.lower()
            if low.startswith("vdb(") and "#branch" not in low and "out" in low:
                dbs = s.get("y_values") or []
                break
        else:
            for name, s in signals.items():
                low = name.lower()
                if low.startswith("vdb(") and "#branch" not in low:
                    dbs = s.get("y_values") or []
                    break
        if dbs and len(dbs) == len(freqs):
            g0 = float(dbs[0])
            out["dc_gain"] = round(g0, 2) + 0.0   # +0.0 kills negative zero
            bw = next((f for f, g in zip(freqs, dbs) if g < g0 - 3.0), None)
            if bw is not None and bw > freqs[0]:
                out["bandwidth"] = float(f"{bw:.4g}")
        else:
            sig = pick_signal(signals)
            if sig:
                _, _, y = sig
                mags = [abs(v) if not isinstance(v, list) else math.hypot(*v[:2])
                        for v in y]
                if mags and mags[0] > 0:
                    out["dc_gain"] = round(20.0 * math.log10(mags[0]), 2)
                    target = mags[0] / math.sqrt(2.0)
                    bw = next((f for f, m in zip(freqs, mags) if m < target),
                              None)
                    if bw is not None and bw > freqs[0]:
                        out["bandwidth"] = float(f"{bw:.4g}")
    elif kind == "dc":
        sweeps = data.get("sweeps") or {}
        # Not every deck has an "out" node: mirrors put the story in a branch
        # current, a bandgap in internal nodes. Pick the most ACTIVE signal --
        # largest swing -- after discarding supply rails, sweep echoes (y == x)
        # and near-constant vectors; prefer voltage nodes, fall back to branch
        # currents with current-named metrics.
        def candidates(want_branch: bool):
            for name, s in sweeps.items():
                low = name.lower()
                if ("#branch" in low) != want_branch:
                    continue
                if not want_branch and low in ("vdd", "vss"):
                    continue
                x, y = s.get("x_values") or [], s.get("y_values") or []
                if len(y) < 3 or len(x) != len(y):
                    continue
                if not all(isinstance(v, (int, float)) for v in y[:5]):
                    continue
                swing = max(y) - min(y)
                if max(abs(a - b) for a, b in zip(x, y)) < 1e-9:
                    continue  # the sweep variable echoed back
                yield swing, name, x, y

        volt = max(candidates(False), default=None)
        curr = max(candidates(True), default=None)
        slope_name = "output_tc" if (sweep_var or "").lower() == "temp" \
            else "max_gain"
        if volt and volt[0] > 1e-3:
            _, _, x, y = volt
            slopes = [abs((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
                      for i in range(len(x) - 1) if x[i + 1] != x[i]]
            if slopes:
                out[slope_name] = float(f"{max(slopes):.4g}")
            out["vout_swing"] = round(max(y) - min(y), 3)
        elif curr and curr[0] > 1e-9:
            _, _, x, y = curr
            out["iout_max"] = float(f"{max(abs(v) for v in y):.4g}")
            out["iout_swing"] = float(f"{max(y) - min(y):.4g}")
        ops = data.get("op_points") or {}
        for k, v in ops.items():
            if "#branch" in k.lower() and isinstance(v, (int, float)):
                out["idd"] = round(abs(v), 9)
                break
    elif kind == "tran" and data.get("time"):
        sig = pick_signal(data.get("signals") or {})
        if sig:
            _, _, y = sig
            t = data["time"]
            if len(y) > 4:
                out["vout_final"] = round(y[-1], 4)
                out["vout_swing"] = round(max(y) - min(y), 4)
                mid = (max(y) + min(y)) / 2.0
                crossings = [t[i] for i in range(1, len(y))
                             if (y[i - 1] - mid) * (y[i] - mid) < 0]
                if len(crossings) >= 4:
                    period = 2.0 * (crossings[-1] - crossings[0]) / (len(crossings) - 1)
                    if period > 0:
                        out["osc_freq"] = float(f"{1.0 / period:.4g}")
    return out


UNITS = {"dc_gain": "dB", "bandwidth": "Hz", "max_gain": "V/V",
         "vout_swing": "V", "vout_final": "V", "idd": "A", "osc_freq": "Hz",
         "output_tc": "V/C", "iout_max": "A", "iout_swing": "A"}


def make_specs(measured: dict[str, float], rng: random.Random,
               meet: bool) -> dict:
    """Spec targets placed relative to what the deck ACTUALLY measures.

    meet=True places them below the measurement (an honest pass); meet=False
    above it (an honest fail, for the iterate pattern). Either way the later
    spec.check verdict is the env's real arithmetic, not this function's.
    The pass margin is wide (factor <= 0.5) because the env's early-stop
    threshold is 0.95: a spec met by only 20 pct scores ~0.5 and reads as NOT
    passed, which is the env being honest about thin margins.
    """
    specs = {}
    for name, val in list(measured.items())[:3]:
        if val <= 0:
            continue
        factor = rng.uniform(0.30, 0.50) if meet else rng.uniform(1.15, 1.6)
        if name == "idd":  # a current budget is a max, not a min
            factor = rng.uniform(2.0, 3.5) if meet else rng.uniform(0.4, 0.8)
            specs[name] = {"max": float(f"{val * factor:.3g}"),
                           "unit": UNITS.get(name, "")}
        else:
            specs[name] = {"min": float(f"{val * factor:.3g}"),
                           "unit": UNITS.get(name, "")}
    return specs


# ------------------------------------------------------------------- prose ---

OPENERS = [
    "Let me start from a complete deck and simulate it as-is.",
    "I will write the full netlist first, then run {analysis} on it.",
    "Starting with a known-good topology; the deck below goes straight to the simulator.",
    "First pass: simulate the actual netlist, then judge it against the specs.",
    "I'll run {analysis} on the complete circuit and read the numbers before deciding anything.",
    "Approach: full deck, real {analysis} run, then spec check against measurements.",
    "The netlist includes models, supplies and stimulus, so the simulator sees the whole circuit.",
    "Simulating the deck exactly as written -- every judgment afterwards comes from these vectors.",
]

READS = [
    "The simulation returned real vectors. Reading them: {vals}.",
    "From the waveforms: {vals}. Now checking these against the targets.",
    "Measured from the run: {vals}. Let me verify the specs formally.",
    "The vectors give {vals}. Passing exactly these numbers to the spec checker.",
    "Extracting from the result: {vals}.",
]

RECOVER = [
    "The simulator rejected that call -- {err}. Correcting it and rerunning.",
    "That failed: {err}. The fix is to supply what was missing, with the full deck.",
    "Error received: {err}. Retrying with the complete arguments.",
    "My call was wrong -- {err}. I will not repeat it; here is the corrected one.",
]

CONCLUDE_PASS = [
    "All checked specs are met; the design stands as simulated.",
    "The measured values satisfy every target. Done.",
    "Specs verified against real measurements -- the deck is good.",
]

CONCLUDE_ITER = [
    "The first sizing missed the target, the widened devices recovered it. Final deck above.",
    "After the resize the measurement clears the spec. Keeping the improved netlist.",
]

CONCLUDE_SHORT = [
    "Even after the resize the target is not met; the honest answer is that this topology "
    "needs {short} beyond what W scaling gives. Reporting the measured shortfall as-is.",
    "The improved deck still falls short on {short}. Rather than claim success, I report "
    "the real numbers; a topology change would be the next step.",
]


def fmt_vals(vals: dict[str, float]) -> str:
    parts = []
    for k, v in vals.items():
        u = UNITS.get(k, "")
        parts.append(f"{k} = {v:g} {u}".strip())
    return ", ".join(parts)


def call_block(name: str, args: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args},
                                        ensure_ascii=False) + "\n</tool_call>"


def obs_msg(observation: str) -> dict:
    """A tool message EXACTLY as the serving loop would deliver it: the shared
    agent loop appends observation[:4000] (runner.py line 180), so training
    must show the model the same truncated view it will read at eval time."""
    return {"role": "tool", "content": observation[:4000]}


# -------------------------------------------------------------- generation ---

def fresh_env(specs: dict):
    from asic_ai.adapters import get_adapter
    from asic_ai.reward.reward import RewardFunction
    from asic_ai.training.rl_env import CircuitDesignEnv

    adapter = get_adapter("ngspice_shared", binary_path="",
                          work_dir=tempfile.mkdtemp())
    task = {"id": "gen", "specs": specs}
    env = CircuitDesignEnv(adapter, RewardFunction.from_eval_task(task),
                           max_steps=12)
    env.reset(task)
    return env


def gen_one(circuit: dict, rng: random.Random, pattern: str) -> dict | None:
    """One executed trajectory, or None when the variant does not simulate.

    One env, one real run per sim call in the trajectory. The spec targets
    are derived from the measurement, so the sim runs FIRST and the messages
    are assembled afterwards from the recorded observations -- every tool
    message below is a string env.step() actually returned.
    """
    kind = circuit["tool"].split(".")[1]
    deck = perturb(circuit["netlist"], rng)
    stripped, card_kind, card_params = split_card(deck)

    # Two calling conventions, both real: the deck with its card, or the
    # stripped deck plus the contract's parameter names.
    use_params = card_params is not None and rng.random() < 0.5
    sim_args: dict = {"netlist": stripped if use_params else deck}
    if use_params:
        sim_args.update(card_params)

    env = fresh_env({})

    # -- run everything first ---------------------------------------------
    bad_args, bad_obs = None, None
    if pattern == "recovery":
        # A deliberate, common mistake -- then the env's REAL error text, then
        # the corrected call. The error string in the corpus is the error
        # string the model will see at eval time, byte for byte.
        if kind == "dc" or (use_params and rng.random() < 0.5):
            # A stripped dc deck still runs .op successfully, so the only
            # dc mistake that honestly FAILS is omitting the netlist.
            bad_args = {k: v for k, v in sim_args.items() if k != "netlist"}
        else:
            bad_args = {"netlist": stripped} if card_kind else \
                       {k: v for k, v in sim_args.items() if k != "netlist"}
        bad = env.step({"name": circuit["tool"], "arguments": bad_args})
        if '"error"' not in bad.observation:
            return None                   # not actually a failing call; drop
        bad_obs = bad.observation

    sweep_var = (card_params or {}).get("sweep_var")
    r1 = env.step({"name": circuit["tool"], "arguments": dict(sim_args)})
    obs1 = r1.observation
    vals1 = measure(kind, obs1, sweep_var)
    if not vals1:
        return None                       # variant did not produce usable data

    meet = pattern != "iterate"
    specs = make_specs(vals1, rng, meet=meet)
    if not specs:
        return None

    # spec.check runs in an env RESET WITH these specs: the env scores through
    # the RewardFunction built at reset time, so an env reset with {} scores
    # everything 0.0 no matter what the call's specs argument says. With
    # explicit results= no simulation happens, so this env is cheap.
    check_env = fresh_env(specs)
    check1 = check_env.step({"name": "spec.check",
                             "arguments": {"results": vals1, "specs": specs}})
    verdict1 = json.loads(check1.observation)
    if pattern == "iterate" and verdict1.get("passed"):
        return None                       # meant to fail first; it did not
    if pattern != "iterate" and not verdict1.get("passed"):
        return None                       # meant to pass; the env disagreed

    pdk_step = None
    if pattern == "direct" and rng.random() < 0.25:
        # A quarter of the direct examples open with a PDK enquiry -- the two
        # catalogue tools the rest of the corpus never demonstrates.
        pdk_tool = rng.choice(["pdk.list_devices", "pdk.get_corners"])
        pdk_step = (pdk_tool,
                    env.step({"name": pdk_tool, "arguments": {}}).observation)

    # -- assemble the messages --------------------------------------------
    msgs = [{"role": "system", "content": build_system_message()},
            {"role": "user",
             "content": f"{circuit['task']} Specs: {json.dumps(specs)}"}]
    analysis_name = {"dc": "a DC analysis", "ac": "an AC analysis",
                     "tran": "a transient analysis"}[kind]
    opener = rng.choice(OPENERS).format(analysis=analysis_name)

    if pdk_step:
        pdk_tool, pdk_obs = pdk_step
        lead = ("Checking what the PDK offers before simulating."
                if pdk_tool == "pdk.list_devices" else
                "Noting the available corners before the nominal run.")
        msgs += [{"role": "assistant",
                  "content": lead + "\n\n" + call_block(pdk_tool, {})},
                 obs_msg(pdk_obs)]

    if pattern == "recovery":
        err = json.loads(bad_obs).get("error", "")[:140].rstrip(".")
        msgs += [
            {"role": "assistant",
             "content": opener + "\n\n" + call_block(circuit["tool"], bad_args)},
            obs_msg(bad_obs),
            {"role": "assistant",
             "content": rng.choice(RECOVER).format(err=err) + "\n\n"
                        + call_block(circuit["tool"], sim_args)},
        ]
    else:
        msgs.append({"role": "assistant",
                     "content": opener + "\n\n"
                                + call_block(circuit["tool"], sim_args)})

    msgs += [
        obs_msg(obs1),
        {"role": "assistant",
         "content": rng.choice(READS).format(vals=fmt_vals(vals1))
                    + "\n\n" + call_block("spec.check",
                                          {"results": vals1, "specs": specs})},
        obs_msg(check1.observation),
    ]

    if pattern == "iterate":
        better = strengthen(stripped if use_params else deck, rng)
        args2 = dict(sim_args, netlist=better)
        r2 = env.step({"name": circuit["tool"], "arguments": dict(args2)})
        vals2 = measure(kind, r2.observation, sweep_var)
        if not vals2:
            return None
        check2 = check_env.step({"name": "spec.check",
                                 "arguments": {"results": vals2, "specs": specs}})
        verdict2 = json.loads(check2.observation)
        shortfall = [k for k in specs
                     if k in vals2 and not _meets(specs[k], vals2[k])]
        msgs += [
            {"role": "assistant",
             "content": "Below target. Widening the devices and re-simulating "
                        "the modified deck.\n\n"
                        + call_block(circuit["tool"], args2)},
            obs_msg(r2.observation),
            {"role": "assistant",
             "content": rng.choice(READS).format(vals=fmt_vals(vals2))
                        + "\n\n" + call_block("spec.check",
                                              {"results": vals2, "specs": specs})},
            obs_msg(check2.observation),
            {"role": "assistant",
             "content": rng.choice(CONCLUDE_ITER) if verdict2.get("passed")
                        else rng.choice(CONCLUDE_SHORT).format(
                            short=", ".join(shortfall) or "the remaining spec")},
        ]
        success = bool(verdict2.get("passed"))
    else:
        msgs.append({"role": "assistant",
                     "content": rng.choice(CONCLUDE_PASS)})
        success = True

    ok, errors = validate_sft_format(msgs)
    if not ok:
        raise RuntimeError(f"generated an invalid example: {errors}")
    return {
        "id": f"grounded_{circuit['id']}_{rng.randrange(16**6):06x}",
        "task_id": circuit["id"],
        "messages": msgs,
        "score": round(float(verdict1.get("score", 0.0)), 4),
        "success": success,
        "source": "grounded_v1",
        "pattern": pattern,
    }


def _meets(spec: dict, val: float) -> bool:
    if "min" in spec and val < spec["min"]:
        return False
    if "max" in spec and val > spec["max"]:
        return False
    return True


# --------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description="Grounded SFT generation")
    ap.add_argument("--output", default="data/sft/grounded_v1.jsonl")
    ap.add_argument("--per-circuit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from asic_ai.adapters.ngspice_shared import find_ngspice_dll
    if find_ngspice_dll() is None:
        print("REFUSED: ngspice DLL not found -- this generator only writes "
              "what it has actually simulated.")
        return 1

    rng = random.Random(args.seed)
    bank = _load_circuits()
    print(f"\n{SEP}\n  Grounded SFT generation: {len(bank)} runnable circuits, "
          f"target {args.per_circuit}/circuit\n{SEP}")

    patterns = ["direct"] * 5 + ["recovery"] * 3 + ["iterate"] * 2
    examples, dropped = [], 0
    for circuit in bank:
        made = 0
        attempts = 0
        while made < args.per_circuit and attempts < args.per_circuit * 4:
            attempts += 1
            pattern = rng.choice(patterns)
            try:
                ex = gen_one(circuit, rng, pattern)
            except Exception as exc:
                print(f"  {circuit['id']}: generation error: {exc}")
                dropped += 1
                continue
            if ex is None:
                dropped += 1
                continue
            examples.append(ex)
            made += 1
        print(f"  {circuit['id']:24s} {made:>3} examples "
              f"({attempts - made} variants dropped)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    pat = Counter(e["pattern"] for e in examples)
    print(f"\n  {len(examples)} examples -> {out}  (dropped {dropped} variants "
          f"the simulator rejected)")
    print(f"  patterns: {dict(pat)}")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
