# Contributing to ASIC-AI

Thank you for your interest in contributing to the ASIC-AI project!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ozgurkazanci/AI_model.git
cd AI_model

# Install dependencies
pip install -r requirements.txt
pip install pytest pyyaml ruff

# Run tests
PYTHONPATH=src python -m pytest tests/ -v

# Run the full pipeline demo
PYTHONPATH=src python scripts/demo_full_pipeline.py
```

## Project Rules

### CRITICAL: Do Not Modify

1. **Tool Interface** (`src/asic_ai/tool_interface/`) — This is a FROZEN CONTRACT.
   All adapters, training data, and inference depend on this exact interface.

2. **SFT Format** (`src/asic_ai/data/format.py`) — The system prompt and tool definitions
   are the MOST CRITICAL files. A single format inconsistency will break all training data.

### Before Submitting Changes

1. **Run all tests**: `PYTHONPATH=src python -m pytest tests/ -v`
2. **Validate SFT data**: `PYTHONPATH=src python scripts/validate_sft_data.py --input data/sft/train_final.jsonl --report /dev/null`
3. **Run the demo**: `PYTHONPATH=src python scripts/demo_full_pipeline.py`

### Code Style

- Python 3.11+ features are welcome (match statements, type unions with `|`)
- Use `from __future__ import annotations` for forward references
- Use Pydantic v2 for data models
- ASCII-only in scripts (no emoji — Windows cp1252 breaks them)
- Always set `PYTHONPATH=src` when running scripts

## Areas to Contribute

### High Priority
- More SFT training data (especially multi-turn tool calling trajectories)
- ngspice adapter testing and validation
- Circuit template additions (PLL, ADC, DAC, etc.)
- RL/GRPO training experiments

### Medium Priority
- Nabla simulator adapter
- RAG system for PDK parameters
- Training data quality improvements
- Performance benchmarks

### How to Add a New Circuit Template

1. Add template to `src/asic_ai/data/templates.py`
2. Include: id, name, category, description, netlist, parameters, typical_specs
3. Run: `python -m pytest tests/test_templates.py -v`
4. Generate augmented data: `python scripts/augment_from_templates.py`
5. Re-prepare training data: `python scripts/prepare_training_data.py`

### How to Add a New Tool

> **Warning**: Adding tools requires updating the FROZEN interface.
> This should only be done with extreme care and full backward compatibility.

1. Add to `src/asic_ai/tool_interface/schema.py`
2. Add to `src/asic_ai/data/format.py` TOOL_DEFINITIONS
3. Update all adapters
4. Generate training data for the new tool
5. Run ALL tests
6. Update docs/HANDOFF.md

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
