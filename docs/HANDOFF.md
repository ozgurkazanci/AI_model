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
# Expected: 147 passed, 4 skipped

# Run full pipeline demo
PYTHONPATH=src python scripts/demo_full_pipeline.py

# Analyze training data
PYTHONPATH=src python scripts/analyze_sft_data.py
```

## Critical Rules

> **DO NOT** modify anything in `src/asic_ai/tool_interface/` — this is a FROZEN CONTRACT.
> All adapters, training data, and inference depend on this exact interface.

> **DO NOT** break the SFT format in `src/asic_ai/data/format.py` — it's the MOST CRITICAL file.
> A single format inconsistency will break all training data.

> **ALWAYS** run `python -m pytest tests/ -v --tb=short` after changes.

> **ALWAYS** push to GitHub after completing a phase.

## Architecture

```
Agent Loop: plan → act → observe → decide → repeat
     ↓
LLM (Qwen3.6-35B-A3B or smaller) → generates tool calls
     ↓
Tool Interface (FROZEN) → 15 tools: sim.dc/ac/tran, pdk.*, spec.*, netlist.*, etc.
     ↓
Adapter Layer → ngspice, nabla, Verilator, or MockSimulator
     ↓
Results → back to LLM for next step
```

## Training Pipeline

```
CPT (optional) → SFT (critical) → RL/GRPO (game-changer)
```

### SFT Data (Current: 223 examples, ~483K tokens)
- `data/sft/mock_analog_v1.jsonl` — 108 analog examples
- `data/sft/mock_digital_v1.jsonl` — 54 digital examples
- `data/sft/augmented_v1.jsonl` — 60 template-augmented examples
- Format: ChatML with `<tool_call>` tags, validated by `validate_sft_format()`

### Fine-Tuning (Proven)
```bash
# Quick test (15 min CPU)
PYTHONPATH=src python scripts/finetune_local.py --quick-test

# Full local (5-8 hours CPU)
PYTHONPATH=src python scripts/finetune_local.py --epochs 3

# Cloud (8 hours A100)
bash scripts/cloud/train_sft_large_lambda.sh
```

### RL/GRPO (Ready but not run)
- `src/asic_ai/training/rl_grpo.py` — Full GRPO trainer
- `src/asic_ai/training/rl_env.py` — Gym-like RL environment (tested)
- Reward from simulator, not humans

## Key Files Reference

### Source Code (src/asic_ai/)
| File | Lines | Purpose |
|------|-------|---------|
| `tool_interface/schema.py` | 212 | All Pydantic v2 models. **FROZEN.** |
| `data/format.py` | ~300 | System prompt + 15 tool definitions. **MOST CRITICAL.** |
| `data/templates.py` | ~250 | 6 parameterized SPICE circuit templates |
| `data/trajectory.py` | ~120 | Trajectory/TrajectoryStep/ToolCall models |
| `training/rl_env.py` | ~280 | Gym-like RL environment |
| `training/rl_grpo.py` | 313 | Full GRPO trainer with TRL |
| `inference/runner.py` | ~100 | Inference pipeline |
| `inference/parser.py` | 54 | Tool call parser (multi-format) |
| `adapters/mock.py` | 163 | Mock simulator (no ngspice needed) |
| `reward/partial_credit.py` | ~150 | Reward with logarithmic distance |
| `tokenizer/domain_tokens.py` | ~200 | 195 domain-specific tokens |

### Scripts (17 total)
| Script | Purpose |
|--------|---------|
| `demo_full_pipeline.py` | E2E pipeline demo (no deps) |
| `run_agent.py` | Multi-step agent with real LLM |
| `compare_models.py` | Base vs fine-tuned comparison |
| `finetune_local.py` | Local SFT with LoRA (CPU/GPU) |
| `merge_lora.py` | Merge LoRA into base model |
| `validate_with_real_model.py` | Pipeline validation with real LLM |
| `local_inference.py` | GGUF/DirectML inference |
| `serve_model.py` | FastAPI inference server |
| `cloud_train.py` | Cloud GPU cost estimator + script gen |
| `augment_from_templates.py` | Template-based SFT augmentation |
| `analyze_sft_data.py` | SFT data quality analysis |
| `generate_mock_sft.py` | Mock SFT data generation |
| `generate_sft_data.py` | Full SFT data generation (needs API) |
| `validate_sft_data.py` | SFT format validation |
| `measure_baseline.py` | Baseline measurement |
| `extend_tokenizer.py` | Domain tokenizer extension |
| `collect_cpt_data.py` | CPT corpus downloader |

### Tests (13 files, 147 passed, 4 skipped)
```
test_schema.py          20 tests  — Pydantic models
test_perturbation.py    15 tests  — Perturbation pipeline
test_reward.py          18 tests  — Reward function
test_tokenizer.py       12 tests  — Tokenizer extension
test_trajectory.py      14 tests  — Trajectory models
test_validator.py       10 tests  — Data validation
test_format.py           7 tests  — SFT format
test_inference.py        7 tests  — Inference pipeline
test_e2e_agent.py        7 tests  — E2E agent
test_eval_runner.py     10 tests  — Eval runner
test_rl_env.py          11 tests  — RL environment
test_templates.py       16 tests  — Circuit templates
test_ngspice_smoke.py    4 tests  — ngspice (auto-skip)
```

## Known API Mismatches (DO NOT REVERT)

These were fixed during development. If you see different behavior, check these:

- `PerturbedCircuit.perturbations_applied` (NOT `perturbation_types`)
- `Perturbation.apply(self, netlist, seed)` returns `Tuple[str, str]`
- `TrajectoryStep` requires `step_index` field
- `Trajectory` requires `id`, `task_id`, `success`, `final_score`, `duration_seconds`
- `ToolCall` requires `call_id` field
- `ActionType` enum values are UPPERCASE: `"SIMULATE"` not `"simulate"`
- `BaselineReport` does NOT have `total_tokens` field
- `validate_sft_format()` checks for "circuit" keyword in system prompt
- `ToolCallParser()` takes NO constructor arguments
- Unicode emoji cause UnicodeEncodeError on Windows cp1252 — use ASCII

## Environment Notes

- **OS**: Windows 11, Python 3.11.9
- **PyTorch**: 2.5.1+cpu
- **GPU**: AMD Radeon 780M (iGPU, 4GB shared VRAM)
  - DirectML works for small inference
  - Training must be on cloud GPU (A100/H100)
- **ngspice**: NOT installed
- **Git**: GitHub CLI authenticated as `ozgurkazanci`
- **PYTHONPATH**: Always set `PYTHONPATH=src` before running

## What's Next (Priority Order)

### High Priority
1. **More SFT data** — Generate 1000+ examples with stronger model (GPT-4/Claude via API)
2. **Full 35B SFT** — Run on cloud A100 (~$14, 8 hours)
3. **RL/GRPO training** — Run on cloud A100 (~$42, 24 hours)
4. **ngspice integration** — Install and test with real simulator
5. **Eval with trained model** — Run all 54 tasks, measure improvement

### Medium Priority
6. More circuit templates (source follower, PLL, ADC)
7. RAG system for PDK parameters
8. Curriculum learning (easy → hard task ordering)
9. Model merging utilities
10. HuggingFace model upload

### Low Priority
11. Nabla simulator adapter
12. Verilator digital adapter
13. Multi-agent collaboration
14. Web UI for design visualization

## Git History

```
ac61f3c Phase 1:  Infrastructure (10 modules)
20f7ec8 Phase 2:  54 eval tasks
e042b89 Phase 3:  SFT + inference pipeline
655918a Phase 4:  Tests + gold trajectories
bafc639 Phase 5:  Mock simulator + E2E
403b6dc Phase 6:  System prompt + Dockerfile
265d6bc Phase 7:  SFT data + eval report
978ba0b Phase 7b: Mock SFT data files
909db0e Phase 8:  RL environment + pipeline demo
63a2aed Phase 9:  Circuit templates + model card
87bf5b6 Phase 10: Inference server + augmentation
2284d5f Phase 11: AMD 780M + cloud training
24fabe1 Phase 12: Real LLM validation
9bc84c7 Phase 13: First fine-tuned model
13d3929 Phase 14: Agent runner + comparison
```

## Contact

- **Owner**: Ozgur Kazanci (`ozgurkazanci@gmail.com`)
- **GitHub**: `https://github.com/ozgurkazanci/AI_model`
- **University**: Akdeniz Universitesi
