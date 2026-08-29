---
language:
- en
license: apache-2.0
tags:
- circuit-design
- eda
- asic
- analog
- vlsi
- spice
- tool-calling
base_model: Qwen/Qwen3.6-35B-A3B
datasets:
- custom
pipeline_tag: text-generation
---

# ASIC-AI: Circuit Design Agent

An AI agent fine-tuned for ASIC/VLSI circuit design tasks. The model can:

- **Design analog circuits** (OTA, bandgap, LDO, PLL, ADC, DAC, LNA)
- **Design digital circuits** (FSM, FIFO, SPI, counters, SRAM)
- **Run simulations** via tool calls (ngspice, Cadence Spectre)
- **Analyze results** (gain, bandwidth, phase margin, noise)
- **Physical verification** (DRC fix, LVS debug, parasitic extraction)
- **Layout guidance** (matching, floorplanning, noise routing)

## Model Details

- **Base Model**: Qwen3.6-35B-A3B (MoE, 3B active params)
- **Fine-tuning**: SFT with LoRA (r=64, alpha=128)
- **Training Data**: 1032 examples covering full IC design flow
- **Tool Interface**: 15 specialized EDA tools
- **Simulators**: ngspice (verified), Cadence Spectre 24.1.0
- **Circuit Templates**: 17 parameterized topologies

## Training Data

| Domain | Examples | Topics |
|--------|----------|--------|
| Analog | 258 | CS amp, OTA, bandgap, LDO |
| Digital | 58 | FSM, FIFO, SPI, counters |
| Diverse tools | 71 | All 14 tool types |
| Batch | 600 | 20 topologies x 5 analyses x 4 PDKs |
| Real ngspice | 16 | Verified simulation results |
| Spectre | 6 | .scs netlist format |
| Reasoning | 5 | Multi-step design iteration |
| Signoff | 4 | DRC, LVS, extraction |
| Layout | 4 | Matching, floorplan, routing |
| RTL | 4 | Verilog/SystemVerilog |
| PDK | 10 | PVT corner analysis |
| **Total** | **1032** | **929 train + 103 val** |

## Tool Interface

```
sim.dc, sim.ac, sim.tran, sim.noise, sim.stb,
sim.corners, sim.mc, pdk.device_query, pdk.get_corners,
spec.check, spec.optimize, netlist.get, netlist.patch,
lint.check, report.generate
```

## License

Apache 2.0

## Citation

```bibtex
@software{asic_ai_2026,
  title={ASIC-AI: Circuit Design Agent},
  author={Kazanci, Ozgur},
  year={2026},
  url={https://github.com/ozgurkazanci/AI_model}
}
```
