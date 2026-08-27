"""ASIC-AI: Domain-specialized AI model for ASIC circuit design.

This package provides the complete infrastructure for training and running
a domain-specialized language model for analog and digital CMOS circuit design.

Key components:
- tool_interface: Frozen contract between model and EDA tools
- adapters: Simulator backends (ngspice, nabla, Verilator, OpenSTA)
- agent: Agent loop, strategy, and memory
- optimizer: Numerical optimization (Bayesian, CMA-ES)
- reward: Reward function for RL training
- data: Data pipeline (trajectories, perturbation, validation)
- tokenizer: Tokenizer extension for circuit design domain
- training: Training launchers (CPT, SFT, RL/GRPO)
"""

__version__ = "0.1.0"
__author__ = "Ozgur Kazanci"
__license__ = "Apache-2.0"
