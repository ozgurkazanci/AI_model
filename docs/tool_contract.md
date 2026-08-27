# Tool Interface Contract — Frozen Specification

> **WARNING**: This contract is FROZEN. Changing it requires re-generating all training data
> and re-training the model. Do not modify without explicit decision.

## Overview

This is the API contract between the AI model and the EDA tools. The model is trained
to call these functions in exactly this format. Any deviation breaks inference.

## API Reference

### Simulation

```python
sim.dc(netlist: str, params: SimParams) -> DCResult
sim.ac(netlist: str, params: SimParams) -> ACResult
sim.tran(netlist: str, params: SimParams) -> TranResult
sim.noise(netlist: str, params: SimParams) -> NoiseResult
sim.stb(netlist: str, params: SimParams) -> StabilityResult
sim.corners(netlist: str, pvt_list: list[PVTCorner]) -> list[CornerResult]
sim.mc(netlist: str, n: int, seed: int) -> MonteCarloResult
```

### Measurement

```python
meas.eval(signal: SignalData, expr: str) -> MeasResult
```

Supported expressions: `max`, `min`, `rise_time`, `fall_time`, `settling_time`,
`overshoot`, `cross`, `avg`, `rms`, `pp`

### Spec Checking & Reward

```python
spec.check(results: dict, spec: SpecDefinition) -> SpecCheckResult
# Returns: {score: float, breakdown: {spec_name: {target, actual, met, score}}}
```

Score computation: logarithmic distance for partial credit.

### PDK Query (NOT memorization)

```python
pdk.device_query(model: str, W: float, L: float, VGS: float, VDS: float) -> DeviceQueryResult
# Returns: {gm, gds, ID, ft, Cgs, Cgd, Cdb, Vth, region, ...}
```

The model never memorizes PDK data. It queries at runtime.

### Netlist Editing (diff-based, NOT full rewrite)

```python
netlist.patch(netlist: str, diff: NetlistPatch) -> str
lint.check(netlist: str) -> LintResult
```

Patch operations: `add_instance`, `remove_instance`, `modify_param`, `add_net`, `remove_net`

### RL Environment

```python
env.reset(task: EvalTask) -> AgentObservation
env.step(action: AgentAction) -> (obs, reward, done, truncated, info)
```

## Two Critical Points

1. **`pdk.device_query`** — Model uses gm/ID data without memorization. This is the NDA solution.
   The model never sees NDA'd model cards; it queries them.

2. **`netlist.patch`** — Model produces diffs, not full netlists. This saves tokens and
   prevents regression. The model must learn to generate diffs.

## Format Rules

- All tool calls use structured JSON arguments
- All results are structured data (never parsed text/logs)
- Errors are machine-readable: which node, what time, how much deviation
- Everything is callable from Python API; GUI wraps API, not the other way around
- Deterministic and seedable
