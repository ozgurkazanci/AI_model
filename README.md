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
├──────────┬──────────┬───────────┬───────────────┤
│ ngspice  │ Spectre  │   nabla   │  Verilator    │
│ (DLL)    │ (WSL)    │ (future)  │  (digital)    │
└──────────┴──────────┴───────────┴───────────────┘
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
│   ├── adapters/          # Simulator backends (ngspice, Spectre, mock)
│   ├── agent/             # Agent loop, strategy, memory
│   ├── optimizer/         # Numerical optimizer (Bayesian, CMA-ES)
│   ├── reward/            # Reward function (partial credit, corners)
│   ├── data/              # Data pipeline, trajectories, templates, SFT format
│   ├── tokenizer/         # Tokenizer extension (195 domain tokens)
│   ├── inference/         # Inference pipeline (runner, parser, engine)
│   └── training/          # Training launchers (CPT, SFT, RL/GRPO) + RL environment
├── eval/                  # 74 eval tasks (50 analog + 24 digital)
├── configs/               # EDA tools, training profiles
├── data/
│   ├── sft/               # 420 SFT training examples
│   │   ├── train_final.jsonl  # 378 train (curriculum ordered)
│   │   └── val_final.jsonl    # 42 validation
│   ├── examples/          # Gold-standard trajectories (OTA, LDO, bandgap, counter)
│   └── corpus_registry.yaml  # CPT source tracking with licenses
├── scripts/               # 38 CLI tools
├── tests/                 # 187 passed, 0 skipped (17 test files)
└── docs/                  # Design docs, handoff guide, tool contract
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
pip install -r requirements.txt

# Run tests (175 passed, 0 skipped)
PYTHONPATH=src pytest tests/ -v

# Run full pipeline demo (no GPU, no simulator needed)
PYTHONPATH=src python scripts/demo_full_pipeline.py

# Test ngspice integration (requires KiCad with ngspice)
PYTHONPATH=src python scripts/test_ngspice.py

# E2E demo: AI model + real ngspice simulation
PYTHONPATH=src python scripts/demo_ai_ngspice.py

# Fine-tune locally (CPU, ~15 min quick test)
PYTHONPATH=src python scripts/finetune_local.py --quick-test

# Run agent with fine-tuned model
PYTHONPATH=src python scripts/run_agent.py --model outputs/sft_local/final

# Interactive chat with model
PYTHONPATH=src python scripts/chat.py --model outputs/sft_local/final

# Run evaluation on all 70 tasks
PYTHONPATH=src python scripts/run_eval.py --model outputs/sft_local/final --limit 5

# Cloud training (single command on GPU instance)
bash scripts/cloud/deploy_and_train.sh
```

## ngspice Integration

Real SPICE simulation via KiCad's bundled ngspice shared library:

```python
from asic_ai.adapters import get_adapter

adapter = get_adapter("ngspice_shared", binary_path="", work_dir="/tmp/sim")
result = adapter.dc("circuit.cir", SimParams(analysis_type="dc"))
print(f"Data points: {sum(len(s.x_values) for s in result.sweeps.values())}")
```

**Verified circuits**: Common-source amp, CMOS inverter, RC filter, NMOS I-V,
differential pair, ring oscillator, bandgap reference, current mirror.

**Requirements**: [KiCad](https://www.kicad.org/) (ngspice.dll is bundled).

## Build Order

> **Warning:** Narrative order ≠ build order. Build the system interface first.

| Step | Task | Status |
|------|------|--------|
| 1 | Tool interface schema (frozen) | Done |
| 2 | Eval set (54 tasks: 36 analog + 18 digital) | Done |
| 3 | Corpus list + license audit | Done |
| 4 | Baseline measurement | Done (mock) |
| 5 | Adapter layer (ngspice/Verilator) | Done (mock) |
| 6 | Agent loop + RL env | Done |
| 7 | Synthetic perturbation pipeline | Done |
| 8 | SFT data generation (393 examples, 15 tools) | Done |
| 9 | Local fine-tuning (LoRA, CPU) | Done |
| 10 | Real LLM validation + agent loop | Done |
| 11 | Training: Cloud SFT → RL | Ready (needs GPU) |
| 12 | Numerical optimizer integration | Done |

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
