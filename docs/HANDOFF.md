# ASIC-AI: Claude Code Handoff Guide

> This document is the complete guide for continuing development with Claude Code
> (or any AI coding agent). Read this FIRST before making any changes.

## Quick Start

```bash
cd "C:\Users\ozgur\OneDrive - Akdeniz Üniversitesi\CV_akademik_01192017\Arastirmalarim\AI_model"

# Set up environment
pip install -r requirements.txt

# Run tests (must pass)
PYTHONPATH=src python -m pytest tests/ -v --tb=short
# Expected: 766 passed (758 passed / 8 skipped when the TSMC PDK is absent)

# Run full pipeline demo
PYTHONPATH=src python scripts/demo_full_pipeline.py

# E2E AI + ngspice demo
PYTHONPATH=src python scripts/demo_ai_ngspice.py

# Check training status (if training is running)
PYTHONPATH=src python scripts/training_monitor.py

# Post-training validation (run after training completes)
PYTHONPATH=src python scripts/post_training_pipeline.py --model outputs/sft_local/final
```

## Critical Rules

> **DO NOT** modify anything in `src/asic_ai/tool_interface/` — this is a FROZEN CONTRACT.

> **DO NOT** break the SFT format in `src/asic_ai/data/format.py` — it's the MOST CRITICAL file.

> **ALWAYS** set `PYTHONPATH=src` before running any script.

> **ALWAYS** run `python -m pytest tests/ -v --tb=short` after changes.

> **ALWAYS** push to GitHub after completing a phase.

> **NO Unicode emoji** in scripts — Windows cp1252 breaks them.

> **NEVER build a system message by hand.** Call `build_system_message()` from
> `asic_ai.data.format` — it renders SYSTEM_PROMPT + the tool list and is the
> single source of truth for training data *and* inference. A model trained on
> one system prompt and served with another silently stops emitting tool calls.

> **NEVER emit a tool call outside `TOOL_DEFINITIONS`** in training data — the
> model learns to hallucinate tools that do not exist.

### System prompt invariant

Every SFT example and every inference call carries the byte-identical output of
`build_system_message()` (7003 chars). Enforced by
`tests/test_system_prompt_consistency.py`.

```bash
PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --check  # verify
PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --write  # repair sources
PYTHONPATH=src python scripts/prepare_training_data.py                # regenerate train/val
```

## Read This First: Prior Numbers Are Not Trustworthy

Several layers of this project reported success without doing their job. The
full table is in `CLAUDE.md` under "What Was Fabricated". In short: the ngspice
adapter returned arrays of zeros, the GRPO reward was a constant 0.1 regardless
of design, the tool-call parser matched a format that appears nowhere in the
training data (0 of 4322 calls parsed), the eval runner returned
`passed=True, score=85.5` for every task, and the numerical optimizer never
called its objective function.

Every one of those is fixed and guarded by a test that has been proven to fail
against the old behaviour. But any measurement recorded before Phase 69 was
taken through at least one of them, so treat historical results as unverified
rather than as a baseline.

## What Is Real Now

| Capability | Where | Evidence |
|---|---|---|
| Real SPICE vectors | `adapters/ngspice_shared.py` | divider `.op` v(out) = 0.9 exactly; RC f_3dB 159154.88 Hz vs analytic 159154.94 |
| TSMC CRN65GPLUS | `adapters/pdk_deck.py` | inverter trip point moves with corner (tt .4941 / ss .4986 / ff .4878 / sf .5243 / fs .4642); seeded statistical MC, sd 6.44 pct |
| Metrics | `adapters/measure.py` | ac/tran/noise/supply metrics, `None` rather than a wrong number |
| Results -> specs | `adapters/spec_extract.py` | 117 spec names normalised, per-task unit conversion, unmeasurable reported not dropped |
| Tool-call parsing | `inference/parser.py` | 4322/4322 corpus calls parsed; contract validation |
| Eval | `eval/runner.py`, `eval/baseline.py` | real agent loop; refuses to score without a model |
| Device sizing | `optimizer/scipy_opt.py`, `optimizer/circuit.py` | finds R2 = 10.0115k vs analytic 10k from simulation alone, 40 evaluations |
| iGPU inference | `inference/llama_server.py` | AMD 780M via Vulkan, 74.7 tok/s Q4_K_M vs 49.0 on CPU |
| HF inference | `inference/engine.py` | `TransformersEngine` emits the same in-contract tool call as the GGUF path on the same prompt |
| Tool-call parsing | `inference/parser.py` | 4322/4322 corpus calls parsed; contract and required-argument validation |
| Device sizing | `optimizer/scipy_opt.py`, `circuit.py` | finds R2 = 10.0115k against an analytic 10k from simulation alone, in 40 evaluations |
| Eval | `eval/runner.py`, `eval/baseline.py`, `scripts/measure_baseline.py` | one shared agent loop; all three refuse to score without a model |
| Adapter conformance | `tests/test_adapter_schema_conformance.py` | static audit of every `Result(...)` in every adapter against the frozen schema |

### The agent loop lives in ONE place

`asic_ai.inference.runner.run_agent_loop`. Four callers use it: `eval/runner.py`,
`scripts/measure_baseline.py`, `agent/loop.py` and `InferenceRunner`. There were
once three implementations and two of them produced nothing. Do not add a fifth;
extend the shared one.

## Local Inference on the iGPU

```bash
PYTHONPATH=src python scripts/gpu_probe.py            # what actually works here
PYTHONPATH=src python scripts/serve_local.py          # serve on the 780M
PYTHONPATH=src python scripts/serve_local.py --prompt "Design a two-stage OTA in sky130."
```

`GGML_VK_DISABLE_COOPMAT=1` in `configs/local_inference.yaml` is load-bearing:
this AMD driver does not expose the Vulkan extensions llama.cpp's coopmat path
requires, and without the flag every model load dies with
`ErrorExtensionNotPresent`. It costs the matrix cores, so re-benchmark and try
removing it after a driver update.

## Architecture

```
Agent Loop: plan → act → observe → decide → repeat
     ↓
LLM (Qwen3.6-35B-A3B) → generates tool calls
     ↓
Tool Interface (FROZEN) → 15 tools: sim.dc/ac/tran, pdk.*, spec.*, netlist.*, etc.
     ↓
Adapter Layer → ngspice (DLL) | Spectre (WSL) | mock
     ↓
Results → back to LLM for next step
```

## Simulator Adapters

| Adapter | Backend Key | Method | Status |
|---------|-------------|--------|--------|
| `ngspice_shared.py` | `ngspice_shared` | KiCad DLL via ctypes | **real vector data** (ngGet_Vec_Info) |
| `spectre_wsl.py` | `spectre` | WSL subprocess | **24.1.0 binary working** (needs license) |
| `mock.py` | `mock` | In-memory mock | Always available |

## Training Pipeline

```
CPT (optional) → SFT (DONE!) → RL/GRPO (ready)
                   ↑ 1050 examples    ↑ see the warning below
                   loss: 2.18 → 0.005
```

### Training Results (0.5B local test)

- **Loss**: 2.176 → 0.00462 (99.8% reduction)
- **Duration**: 8h46m on CPU (267 steps, 3 epochs)
- **Validation**: 3/5 pass (60%) — measured before the parser was fixed; re-measure.
- **Benchmark**: 5/12 pass (47.2%) — same caveat.

### SFT Data (1040 examples, 16 data files)

| File | Examples | Domain |
|------|----------|--------|
| `batch_v1.jsonl` | 500 | 20 topologies (programmatic) |
| `batch_v2.jsonl` | 100 | 20 topologies (programmatic) |
| `mock_analog_v1.jsonl` | 108 | Analog simulation |
| `augmented_v2.jsonl` | 90 | Augmented analog |
| `diverse_tools_v1.jsonl` | 71 | 14 tools coverage |
| `augmented_v1.jsonl` | 60 | Augmented analog |
| `mock_digital_v1.jsonl` | 54 | Digital simulation |
| `ngspice_real_v1.jsonl` | 8 | Real ngspice results |
| `ngspice_real_v2.jsonl` | 8 | Real ngspice results |
| `pdk.get_corners` | 10 | PDK corner data |
| `mixedsignal_v1.jsonl` | 8 | PLL/ADC/DAC/LNA/VCO |
| `spectre_format_v1.jsonl` | 6 | Spectre .scs format |
| `reasoning_v1.jsonl` | 5 | Multi-step reasoning |
| `digital_rtl_v1.jsonl` | 4 | Verilog/SV RTL |
| `signoff_v1.jsonl` | 4 | DRC/LVS/extraction |
| `layout_v1.jsonl` | 4 | Layout/floorplanning |

**Train/Val split**: 936 train + 104 val (curriculum ordered: easy->hard)

### Fine-Tuning

```bash
# Local 0.5B (8-10 hours CPU) — COMPLETED
PYTHONPATH=src python scripts/finetune_local.py --epochs 3

# Cloud 35B (8-12 hours A100, ~$14)
bash scripts/cloud/train_35b.sh

# Validate trained model
PYTHONPATH=src python scripts/validate_trained_model.py --model outputs/sft_local/final
PYTHONPATH=src python scripts/benchmark_model.py --model outputs/sft_local/final
```

### RL/GRPO

```bash
# GRPO with real ngspice rewards
PYTHONPATH=src python scripts/grpo_ngspice.py --episodes 100

# RL environment demo
PYTHONPATH=src python scripts/demo_rl_ngspice.py
```

## Known API Mismatches (DO NOT REVERT)

- `ToolCallParser()`: takes NO constructor arguments
- `ParsedToolCall`: Pydantic model with `.name`, `.arguments` (NOT dict)
- `PerturbedCircuit.perturbations_applied` (NOT `perturbation_types`)
- `TrajectoryStep` requires `step_index` field
- `Trajectory` requires `id`, `task_id`, `success`, `final_score`, `duration_seconds`
- `ToolCall` requires `call_id` field
- `ActionType` enum values UPPERCASE: `"SIMULATE"` not `"simulate"`
- `transformers 5.16.1`: `warmup_steps` NOT `warmup_ratio`, `use_cpu` NOT `no_cuda`
- `torch.load` CVE-2025-32434: requires torch >= 2.6 (checkpoint resume broken on 2.5.1)
- `SignalData`: `name`, `x_values`, `y_values` (NOT `values`, `unit`)
- `DCResult`: `op_points` and `sweeps` (Dict[str, SignalData])
- `PVTCorner`: `voltage` (NOT `supply_voltage`)
- `RewardFunction(specs=[SpecTarget(...)])` (NOT `RewardFunction()`)
- `SpecTarget`: `min_val`, `max_val`, `target_val` (NOT `direction`, `target`)

## Environment

- **OS**: Windows 11, Python 3.11.9
- **PyTorch**: 2.5.1+cpu
- **GPU**: AMD Radeon 780M (iGPU, 4GB shared VRAM)
- **ngspice**: KiCad 10.0 DLL (`C:\Program Files\KiCad\10.0\bin\ngspice.dll`)
  - real vector extraction since Phase 69
- **Cadence EDA** (WSL `Alma_EDA` at `/opt/eda/cadence/`):
  - SPECTRE241 (270MB), IC231, PVS222, QUANTUS231, XCELUMMAIN2309
  - DDI251, CONFRML232, MODUS231, SSV231, EMX20251, IC618
  - Full config: `configs/eda_tools.yaml`
- **PYTHONPATH**: Always set `PYTHONPATH=src`
- **git push**: Returns exit code 1 on PowerShell (check `main -> main`)

## Project Stats

```
v0.4.0 | 641 tests | 0 skip | 26 test files
1050 SFT examples | 77 eval tasks (53 analog + 24 digital)
13 templates | 16 ngspice circuits | 3 adapters
~40 scripts | ~47 modules | ~17,000 lines
```

## What's Next (Priority Order)

### High Priority
1. **Full 35B SFT** — Run on cloud A100 (~$14, 8 hours)
2. **More SFT data** — Generate 1000+ examples via API (GPT-4/Claude)
3. **RL/GRPO training** — Run on cloud A100 with ngspice rewards (~$42)
4. **Eval with trained model** — Run all 74 tasks, measure improvement
5. **Spectre license** — Connect license server for real Spectre simulation

### Medium Priority
6. RAG system for PDK parameters
7. More circuit templates (PLL, ADC, DAC, LNA)
8. HuggingFace model upload
9. Ablation study with real results
10. Nabla simulator adapter (for user's custom EDA tool)

### Low Priority
11. Verilator digital adapter
12. Multi-agent collaboration
13. Web UI for design visualization

## Git History (Phase 25-49)

```
8cf9cff Phase 25: AI agent rules (.cursorrules, CLAUDE.md)
f564c06 Phase 26: Training profiles, architecture dashboard
873d3e7 Phase 27: ngspice WORKING! Ablation study script
5b5f456 Phase 28: Cloud deploy script
d58cfb4 Phase 29: ngspice shared library adapter + 5 tests
0bc0eba Phase 30: AI + ngspice E2E demo (3 circuits)
728938d Phase 31: Real ngspice SFT data (8 circuits, 1751 pts)
6fa67d4 Phase 32: 175 tests ALL PASS, 401 training examples
4c52b54 Phase 33: Model validation + docs update
e1af617 Phase 34: ngspice agent loop + extended SFT (409 examples)
096b322 Phase 35: v0.3.0, CLAUDE.md + cursorrules final
b77a641 Phase 36: 4 ngspice-verified circuit templates (13 total)
69f9663 Phase 37: Template verification + training loss analysis
8e57195 Phase 38: RL environment + real ngspice demo
713da9f Phase 39: Post-training automation pipeline
f91bb3b Phase 40: Cadence EDA integration + Spectre SFT data
9b786de Phase 41: Cadence Spectre WSL adapter (270MB binary)
0ea434f Phase 42: Spectre eval tasks (74 total)
7a0fe61 Phase 43: Multi-step reasoning SFT data
fb53ccd Phase 44: Spectre tests (187 total) + CLAUDE.md update
e5227cd Phase 45: GRPO training script with ngspice rewards
2c52c31 Phase 46: README update + GRPO results
8a2e9c2 Phase 47: Digital RTL SFT (Verilog/SystemVerilog)
319a691 Phase 48: Physical verification + signoff SFT
ce1f190 Phase 49: Layout + floorplanning SFT
```

## Contact

- **Owner**: Ozgur Kazanci (`ozgurkazanci@gmail.com`)
- **GitHub**: `https://github.com/ozgurkazanci/AI_model`
- **University**: Akdeniz Universitesi
