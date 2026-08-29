# ASIC-AI Project Rules for AI Coding Agents

## Critical Constraints

1. **NEVER modify** `src/asic_ai/tool_interface/` -- FROZEN CONTRACT
2. **NEVER break** SFT format in `src/asic_ai/data/format.py` -- MOST CRITICAL FILE
3. **ALWAYS** set `PYTHONPATH=src` before running any script
4. **ALWAYS** run `python -m pytest tests/ -v --tb=short` after changes
5. **ALWAYS** push to GitHub after completing a phase
6. **NO Unicode emoji** in scripts -- Windows cp1252 breaks them
7. **NEVER build a system message by hand** -- call `build_system_message()` from
   `asic_ai.data.format`. It is the single source of truth for training AND
   inference. Never write `{"role": "system", "content": SYSTEM_PROMPT}`.
8. **NEVER emit a tool call outside `TOOL_DEFINITIONS`** in training data --
   the model learns to hallucinate tools that do not exist.

## System Prompt Invariant

Every SFT example and every inference call must carry the byte-identical output
of `build_system_message()` (SYSTEM_PROMPT + rendered tool list, 7003 chars).
Training on one prompt and serving with another silently kills tool calling.

```bash
PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --check  # verify
PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --write  # repair
PYTHONPATH=src python scripts/prepare_training_data.py                # regenerate split
```

Guarded by `tests/test_system_prompt_consistency.py` (43 tests): one prompt
variant across the corpus, contract-only tool names, and no module referencing
`SYSTEM_PROMPT` outside `format.py`.

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

## Cadence EDA Suite (WSL)

- **WSL distro**: `Alma_EDA` (AlmaLinux)
- **Base path**: `\\wsl.localhost\Alma_EDA\opt\eda\cadence\`  (Linux: `/opt/eda/cadence/`)
- **Spectre 24.1**: `/opt/eda/cadence/SPECTRE241/tools.lnx86/spectre/bin/64bit/spectre` (270MB)
- **Adapter**: `src/asic_ai/adapters/spectre_wsl.py` (WSL subprocess)
- **Factory**: `get_adapter("spectre", binary_path="", work_dir="...")`
- **Available tools**: SPECTRE241, IC231 (Virtuoso), PVS222 (DRC/LVS), QUANTUS231 (extraction), XCELUMMAIN2309 (Xcelium), DDI251, CONFRML232, MODUS231, SSV231, EMX20251, IC618
- **Config**: `configs/eda_tools.yaml` (all paths + capabilities)
- **Spectre-specific analyses**: STB (stability), PSS (periodic SS), PNoise, dcmatch
- **Note**: Requires Cadence license for simulation; binary loads but lib deps need full LD_LIBRARY_PATH

## Environment

- Python 3.11.9 on Windows 11
- PyTorch 2.5.1+cpu (DirectML incompatible with 2.5)
- transformers 5.16.1
- KiCad 10.0 (ngspice.dll)
- WSL Alma_EDA (Cadence EDA)
- git push returns exit code 1 on PowerShell (check for `main -> main` in output)

## Project Commands

```bash
PYTHONPATH=src python -m pytest tests/ -v          # Run tests (242 passed)
PYTHONPATH=src python scripts/project_stats.py     # Show stats
PYTHONPATH=src python scripts/demo_ai_ngspice.py   # E2E AI+ngspice demo
PYTHONPATH=src python scripts/demo_rl_ngspice.py   # RL env + ngspice
PYTHONPATH=src python scripts/grpo_ngspice.py      # GRPO with ngspice rewards
PYTHONPATH=src python scripts/benchmark_model.py   # 12-prompt benchmark
PYTHONPATH=src python scripts/post_training_pipeline.py    # Post-train automation
PYTHONPATH=src python scripts/validate_trained_model.py    # Post-train validation
PYTHONPATH=src python scripts/chat.py --model outputs/sft_local/final  # Chat
```

## File Organization

- Source: `src/asic_ai/` (47 modules)
- Scripts: `scripts/` (44 CLI tools)
- Tests: `tests/` (17 files, 242 passed)
- Eval: `eval/tasks/` (74 tasks: 50 analog + 24 digital)
- Data: `data/sft/` (15 files, 1032 total: 929 train + 103 val)
- Configs: `configs/eda_tools.yaml`, `configs/training_profiles.yaml`
- Cloud: `scripts/cloud/train_35b.sh` (35B training)
- Docs: `docs/HANDOFF.md` is the comprehensive guide
- Adapters: ngspice_shared (DLL), spectre_wsl (WSL, 24.1.0), mock (test)

