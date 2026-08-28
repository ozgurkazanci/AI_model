# ASIC-AI Project Rules for AI Coding Agents

## Critical Constraints

1. **NEVER modify** `src/asic_ai/tool_interface/` — FROZEN CONTRACT
2. **NEVER break** SFT format in `src/asic_ai/data/format.py` — MOST CRITICAL FILE
3. **ALWAYS** set `PYTHONPATH=src` before running any script
4. **ALWAYS** run `python -m pytest tests/ -v --tb=short` after changes
5. **ALWAYS** push to GitHub after completing a phase
6. **NO Unicode emoji** in scripts — Windows cp1252 breaks them

## API Gotchas (transformers 5.16.1)

- `TrainingArguments`: use `warmup_steps` NOT `warmup_ratio`, use `use_cpu` NOT `no_cuda`
- `from_pretrained()`: use `dtype=` NOT `torch_dtype=`
- `ToolCallParser()`: takes NO constructor arguments
- `ParsedToolCall`: Pydantic model with `.name`, `.arguments` (NOT dict)
- `torch.load` requires torch >= 2.6 for safe loading (CVE-2025-32434)

## Pydantic Model Field Names (EXACT)

- `PerturbedCircuit.perturbations_applied` (NOT `perturbation_types`)
- `Perturbation.apply(self, netlist, seed)` returns `Tuple[str, str]`
- `TrajectoryStep` requires `step_index` field
- `Trajectory` requires `id`, `task_id`, `success`, `final_score`, `duration_seconds`
- `ToolCall` requires `call_id` field
- `ActionType` enum values are UPPERCASE: `"SIMULATE"` not `"simulate"`

## Environment

- Python 3.11.9 on Windows 11
- PyTorch 2.5.1+cpu (DirectML incompatible with 2.5)
- transformers 5.16.1
- git push returns exit code 1 on PowerShell (check for `main -> main` in output)

## Project Commands

```bash
PYTHONPATH=src python -m pytest tests/ -v          # Run tests
PYTHONPATH=src python scripts/project_stats.py     # Show stats
PYTHONPATH=src python scripts/demo_full_pipeline.py # E2E demo
PYTHONPATH=src python scripts/analyze_sft_data.py  # Data analysis
PYTHONPATH=src python scripts/training_monitor.py  # Check training
```

## File Organization

- Source: `src/asic_ai/` (45 modules)
- Scripts: `scripts/` (26 CLI tools)
- Tests: `tests/` (15 files, 163+ passed)
- Eval: `eval/tasks/` (70 tasks)
- Data: `data/sft/` (train_final.jsonl is the canonical dataset)
- Docs: `docs/HANDOFF.md` is the comprehensive guide
