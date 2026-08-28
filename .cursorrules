# ASIC-AI Project Rules for AI Coding Agents

## Critical Constraints

1. **NEVER modify** `src/asic_ai/tool_interface/` -- FROZEN CONTRACT
2. **NEVER break** SFT format in `src/asic_ai/data/format.py` -- MOST CRITICAL FILE
3. **ALWAYS** set `PYTHONPATH=src` before running any script
4. **ALWAYS** run `python -m pytest tests/ -v --tb=short` after changes
5. **ALWAYS** push to GitHub after completing a phase
6. **NO Unicode emoji** in scripts -- Windows cp1252 breaks them

## API Gotchas (transformers 5.16.1)

- `TrainingArguments`: use `warmup_steps` NOT `warmup_ratio`, use `use_cpu` NOT `no_cuda`
- `from_pretrained()`: use `dtype=` NOT `torch_dtype=`
- `ToolCallParser()`: takes NO constructor arguments
- `ParsedToolCall`: Pydantic model with `.name`, `.arguments` (NOT dict)
- `torch.load` requires torch >= 2.6 for safe loading (CVE-2025-32434)
- Cannot resume from checkpoint with torch 2.5.1

## Pydantic Schema Field Names (EXACT -- DO NOT GUESS)

- `DCResult`: fields = `op_points`, `sweeps` (Dict[str, SignalData])
- `ACResult`: fields = `frequencies`, `signals` (Dict[str, SignalData])
- `TranResult`: fields = `time`, `signals` (Dict[str, SignalData])
- `SignalData`: fields = `name`, `x_values`, `y_values` (NOT `values`, `unit`)
- `PVTCorner`: fields = `process`, `voltage`, `temperature` (NOT `supply_voltage`)
- `CornerResult`: fields = `corner`, `dc`, `ac`, `tran`, `stb`
- `NoiseResult`: fields = `frequencies`, `input_noise`, `output_noise`
- `StabilityResult`: fields = `phase_margin`, `gain_margin`, `loop_gain`
- `MonteCarloResult`: fields = `seed`, `runs`, `results`
- `PerturbedCircuit.perturbations_applied` (NOT `perturbation_types`)
- `TrajectoryStep` requires `step_index`
- `Trajectory` requires `id`, `task_id`, `success`, `final_score`, `duration_seconds`
- `ToolCall` requires `call_id`
- `ActionType` enum values UPPERCASE: `"SIMULATE"` not `"simulate"`

## ngspice Integration

- **DLL**: `C:\Program Files\KiCad\10.0\bin\ngspice.dll` (KiCad bundled)
- **Adapter**: `src/asic_ai/adapters/ngspice_shared.py` (ctypes, NOT subprocess)
- **Factory**: `get_adapter("ngspice_shared", binary_path="", work_dir="...")`
- **16 verified circuits**: CS amp, inverter, RC filter, NMOS I-V, diff pair, ring osc (3+5 stage), bandgap, current mirror, cascode, source follower, active load, integrator, Widlar, temp sweep, voltage divider
- **SFT data**: `data/sft/ngspice_real_v1.jsonl` + `ngspice_real_v2.jsonl` (16 examples from real simulation)

## Environment

- Python 3.11.9 on Windows 11
- PyTorch 2.5.1+cpu (DirectML incompatible with 2.5)
- transformers 5.16.1
- KiCad 10.0 (ngspice.dll)
- git push returns exit code 1 on PowerShell (check for `main -> main` in output)

## Project Commands

```bash
PYTHONPATH=src python -m pytest tests/ -v          # Run tests (175 passed)
PYTHONPATH=src python scripts/project_stats.py     # Show stats
PYTHONPATH=src python scripts/demo_ai_ngspice.py   # E2E AI+ngspice demo
PYTHONPATH=src python scripts/test_ngspice.py      # ngspice DLL test
PYTHONPATH=src python scripts/agent_ngspice.py     # Agent with real sim
PYTHONPATH=src python scripts/training_monitor.py  # Check training
PYTHONPATH=src python scripts/validate_trained_model.py  # Post-train validation
PYTHONPATH=src python scripts/chat.py --model outputs/sft_local/final  # Chat
```

## File Organization

- Source: `src/asic_ai/` (45 modules)
- Scripts: `scripts/` (33 CLI tools)
- Tests: `tests/` (16 files, 175 passed)
- Eval: `eval/tasks/` (70 tasks)
- Data: `data/sft/` (train_final.jsonl = 369 train, canonical dataset)
- Configs: `configs/training_profiles.yaml` (8 profiles)
- Cloud: `scripts/cloud/deploy_and_train.sh` (one-command deploy)
- Docs: `docs/HANDOFF.md` is the comprehensive guide
