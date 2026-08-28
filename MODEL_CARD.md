---
license: apache-2.0
language:
  - en
tags:
  - asic
  - eda
  - vlsi
  - cmos
  - circuit-design
  - analog
  - digital
  - spice
  - agent
  - tool-calling
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
---

# ASIC-AI: Domain-Specialized Circuit Design Agent

A domain-specialized language model fine-tuned for **ASIC analog and digital CMOS circuit design**.
Unlike one-shot generators, this model operates as an **agent** — iterating through a structured design loop with real simulator feedback.

## Model Description

- **Base Model**: Qwen3.6-35B-A3B (MoE, 35B total / 3B active)
- **Training**: CPT (domain knowledge) → SFT (agent trajectories) → RL/GRPO (simulator reward)
- **Architecture**: Mixture of Experts, agentic tool-calling
- **License**: Apache 2.0

## How It Works

```
spec → topology → netlist → simulate → diagnose → fix → re-simulate → until spec met
```

The model uses 15 specialized tools:

| Tool | Purpose |
|------|---------|
| `sim.dc/ac/tran/noise/stb` | Run SPICE simulations |
| `sim.corners/mc` | PVT corner and Monte Carlo analysis |
| `spec.check` | Verify against specifications |
| `pdk.device_query/list_devices/get_corners` | PDK parameter retrieval |
| `netlist.patch` | Modify circuit netlist |
| `lint.check` | Structural error checking |
| `opt.suggest` | Numerical optimization suggestions |
| `meas.eval` | Measurement evaluation |

## Intended Use

- Analog circuit design: OTA, LDO, bandgap, comparator, current mirror, etc.
- Digital circuit design: counters, FSMs, FIFOs, ALUs, serial interfaces
- Circuit debugging and optimization
- Design space exploration

## Training Data

- **SFT**: 20,000-50,000 agent trajectories (distillation + synthetic perturbation)
- **RL**: Simulator-in-the-loop GRPO with verifiable reward
- **Format**: ChatML with `<tool_call>` tags

## Evaluation

Evaluated on 54 circuit design tasks (36 analog + 18 digital) across easy/medium/hard difficulties.

| Category | Tasks | Baseline | After SFT | After RL |
|----------|-------|----------|-----------|----------|
| Analog   | 36    | TBD      | TBD       | TBD      |
| Digital  | 18    | TBD      | TBD       | TBD      |

## Limitations

- Requires a SPICE simulator (ngspice/nabla) for actual design work
- PDK parameters are retrieved, not memorized — needs PDK access
- Numerical optimization (W/L sizing) delegated to Bayesian optimizer
- Not suitable for layout or physical design tasks

## Citation

```bibtex
@software{asic_ai_2026,
  title = {ASIC-AI: Domain-Specialized Circuit Design Agent},
  author = {Kazanci, Ozgur},
  year = {2026},
  url = {https://github.com/ozgurkazanci/AI_model},
  license = {Apache-2.0}
}
```
