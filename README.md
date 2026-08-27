# ASIC-AI: Domain-Specialized AI Model for Circuit Design

**A domain-specialized language model for ASIC analog and digital CMOS circuit design.**

The model is an **agent** — not a one-shot netlist generator. It operates in a loop:

```
spec → topology → netlist → simulate → read result → diagnose → fix → re-simulate → until spec met
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Agent Loop                      │
│  plan → act → observe → decide → repeat          │
├─────────────┬───────────────┬───────────────────┤
│   LLM       │  Optimizer    │   Memory/RAG      │
│  (Qwen3.6)  │  (BoTorch)   │   (PDK, designs)  │
├─────────────┴───────────────┴───────────────────┤
│           Tool Interface (Frozen Contract)       │
│  sim.dc/ac/tran | spec.check | pdk.query | ...  │
├─────────────┬───────────────┬───────────────────┤
│  Adapter    │   Adapter     │   Adapter          │
│  (ngspice)  │   (nabla)     │   (Verilator)     │
└─────────────┴───────────────┴───────────────────┘
```

## Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Domain-specialize existing model | From-scratch pretraining is out of budget |
| 2 | Agent role (tool-calling, looping) | One-shot netlist generation is insufficient |
| 3 | Qwen3.6-35B-A3B base | MoE, agentic-coding tuned, Apache 2.0 |
| 4 | Simulator = reward source | Verifiable reward; no human labeling needed |
| 5 | LLM + numerical optimizer hybrid | LLM is weak at continuous numerical search |
| 6 | PDK via retrieval, not memorization | NDA compliance + PDK independence |
| 7 | Model-agnostic code | Base model is a config variable |

## Project Structure

```
AI_model/
├── src/asic_ai/
│   ├── tool_interface/    # Frozen contract (DO NOT MODIFY)
│   ├── adapters/          # Simulator backends (ngspice, nabla, ...)
│   ├── agent/             # Agent loop, strategy, memory
│   ├── optimizer/         # Numerical optimizer (Bayesian, CMA-ES)
│   ├── reward/            # Reward function (partial credit, corners)
│   ├── data/              # Data pipeline (trajectories, perturbation, SFT format)
│   ├── tokenizer/         # Tokenizer extension (SI units, devices)
│   ├── inference/         # Inference pipeline (runner, parser, engine)
│   └── training/          # Training launchers (CPT, SFT, RL)
├── eval/                  # 54 eval tasks (36 analog + 18 digital)
├── configs/               # All configuration (model, training, eval)
├── data/
│   ├── examples/          # Gold-standard trajectories (OTA, LDO, bandgap)
│   └── corpus_registry.yaml  # CPT source tracking with licenses
├── scripts/               # CLI tools (baseline, SFT generation, validation)
├── tests/                 # 113 tests (99 unit + 10 format/inference + 4 ngspice)
└── docs/                  # Design docs (TR), tool contract, ngspice setup
```

## Training Pipeline

```
CPT (domain knowledge) → SFT (agent behavior) → RL/GRPO (design skill)
      ↑ optional              ↑ critical              ↑ game-changer
```

### Stage 1: Continued Pretraining (CPT)
- Inject circuit design literature (1-5B tokens)
- Low learning rate + 15% general code mix (prevent catastrophic forgetting)
- **Least critical** — can be skipped

### Stage 2: Supervised Fine-Tuning (SFT)
- Train on agent **trajectories** (spec → think → netlist → simulate → diagnose → fix → ...)
- 20,000-50,000 trajectories from distillation + synthetic perturbation
- **Most distinctive** — format consistency is critical

### Stage 3: Reinforcement Learning (GRPO)
- Reward from simulator, not humans
- Partial credit with logarithmic distance
- Corner + Monte Carlo inclusion in reward
- Feasibility constraints prevent reward hacking
- **Game-changer** — where real design capability emerges

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/ -v

# List all eval tasks
PYTHONPATH=src python scripts/measure_baseline.py --dry-run

# Generate SFT data (requires API key + ngspice)
PYTHONPATH=src python scripts/generate_sft_data.py --mode distillation --tasks eval/tasks/analog/

# Validate SFT data
PYTHONPATH=src python scripts/validate_sft_data.py --input data/sft/output.jsonl

# Launch training (requires GPU)
python -m asic_ai.training.cpt --config configs/training/cpt_axolotl.yaml
python -m asic_ai.training.sft --config configs/training/sft_axolotl.yaml
python -m asic_ai.training.rl_grpo --config configs/training/rl_grpo.yaml --dry-run
```

## Build Order

> **Warning:** Narrative order ≠ build order. Build the system interface first.

| Step | Task | Status |
|------|------|--------|
| 1 | Tool interface schema (frozen) | Done |
| 2 | Eval set (54 tasks: 36 analog + 18 digital) | Done |
| 3 | Corpus list + license audit | Done |
| 4 | Baseline measurement | Ready (needs API key) |
| 5 | Adapter layer (ngspice/Verilator) | Done |
| 6 | Agent loop + RL env | Done |
| 7 | Synthetic perturbation pipeline (real SPICE parsing) | Done |
| 8 | SFT data generation + inference pipeline | Done |
| 9 | Training: CPT → SFT → RL | Ready (needs GPU) |
| 10 | Numerical optimizer integration | Done |

## For Claude Code Continuation

This project is designed for handoff between AI coding agents:

1. **Read** `docs/design_document_tr.md` — the single source of truth
2. **Don't modify** `src/asic_ai/tool_interface/` — frozen contract
3. **Config, not code** — base model is in `configs/model_config.yaml`
4. **Validate data** before any training — `python -m asic_ai.data.validator`
5. **Measure first** — run eval set before and after any change

## License

Apache 2.0

## Related Projects

- **nabla**: Circuit simulator engine (separate project)
- **EDA tool suite**: Cadence-like analog/digital tools (separate project)
