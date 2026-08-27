# Build Order Reference

> **Warning:** Narrative order ≠ build order. The system interface is described in
> Section 5 but built FIRST.

## Order

| Step | Task | Agent? | Status | Notes |
|------|------|--------|--------|-------|
| 0 | Design document | No | ✅ | This document |
| 1 | Tool interface schema (JSON) | No — decision | ✅ | Frozen contract |
| 2 | Eval set (50-200 tasks, YAML) | No — manual | ✅ 9 tasks | Expand to 50+ |
| 3 | Corpus list + license audit | No | ✅ Framework | Populate sources |
| 4 | Baseline: measure existing models | Partial | 🔲 | Need API keys |
| 5 | Adapter layer (ngspice/Verilator) | Yes | ✅ | Test with real sims |
| 6 | Agent loop + RL env wrapper | Yes | ✅ | Integrate with model |
| 7 | Synthetic perturbation pipeline | Yes | ✅ | Need working circuits |
| 8 | SFT data generation | Yes (auto) | 🔲 | Needs step 5-7 |
| 9 | Training: CPT → SFT → RL | Yes | 🔲 | Needs GPU |
| 10 | Numerical optimizer integration | Yes | ✅ | Test end-to-end |

## Key Dependency Chain

```
Tool Interface (frozen)
    ↓
Adapters (ngspice → nabla)
    ↓
Eval Framework ←── Eval Tasks (YAML)
    ↓
Agent Loop + RL Environment
    ↓
Data Pipeline (trajectories + perturbation)
    ↓
Training (CPT → SFT → RL)
```

## Step 2 Emphasis: Eval Before Training

> Any training without a machine-checkable eval set is flying blind.
> You cannot know if you're making progress. Many projects skip this and
> waste months rowing with no compass.

## Proxy Strategy

```
model → tool interface (our schema) → adapter → [ngspice   | nabla      ]
                                              → [Verilator | our sim    ]
                                              → [OpenSTA   | our STA   ]
```

Train with open-source tools today. Swap adapters when proprietary tools are ready.
The model always sees the same schema.
