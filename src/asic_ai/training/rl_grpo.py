"""GRPO Reinforcement Learning launcher.

Uses TRL (Transformer Reinforcement Learning) for Group Relative Policy Optimization.
Reward comes from the SIMULATOR, not humans.

Key insight from design doc:
    "nabla taç mücevherdir, model değil."
    The reward signal comes from the simulator and the model's capability
    ceiling is determined by the simulator.

Design Doc Reference: Section 4, Stage 3
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "training" / "rl_grpo.yaml"
)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load RL training configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def estimate_training_time(config: dict[str, Any]) -> dict[str, float]:
    """Estimate RL training time based on configuration.

    The CRITICAL bottleneck is simulation speed.

    Returns:
        Estimated times in hours.
    """
    grpo = config.get("grpo", {})
    sim = config.get("simulator", {})

    num_episodes = config.get("num_episodes", 10000)
    num_generations = grpo.get("num_generations", 8)
    avg_steps_per_episode = 15
    sim_time_sec = 3.0  # ngspice average
    sim_parallel = sim.get("max_parallel", 1)

    total_sims = num_episodes * num_generations * avg_steps_per_episode
    sequential_hours = (total_sims * sim_time_sec) / 3600
    parallel_hours = sequential_hours / sim_parallel

    # With nabla (estimated 300ms/sim)
    nabla_sim_time = 0.3
    nabla_total_hours = (total_sims * nabla_sim_time) / 3600
    nabla_parallel_hours = nabla_total_hours / sim_parallel

    return {
        "total_simulations": total_sims,
        "ngspice_sequential_hours": sequential_hours,
        "ngspice_parallel_hours": parallel_hours,
        "nabla_sequential_hours": nabla_total_hours,
        "nabla_parallel_hours": nabla_parallel_hours,
        "sim_parallel_workers": sim_parallel,
    }


def create_reward_fn(config: dict[str, Any]) -> Callable:
    """Create the reward function from config.

    The reward function wraps spec.check with:
    - Partial credit (logarithmic distance)
    - Corner inclusion
    - Feasibility constraints
    - Non-convergence penalty
    """
    from asic_ai.reward.reward import (
        FeasibilityConstraint,
        RewardFunction,
        RewardMode,
        SpecTarget,
    )

    reward_config = config.get("reward", {})

    # Map mode string to enum
    mode_map = {
        "nominal_only": RewardMode.NOMINAL_ONLY,
        "worst_corner": RewardMode.WORST_CORNER,
        "all_corners_weighted": RewardMode.ALL_CORNERS_WEIGHTED,
        "monte_carlo": RewardMode.MONTE_CARLO,
    }
    mode = mode_map.get(reward_config.get("mode", "worst_corner"), RewardMode.WORST_CORNER)

    # Build feasibility constraints
    constraints = []
    for fc in reward_config.get("feasibility_constraints", []):
        constraints.append(FeasibilityConstraint(
            name=fc["name"],
            parameter=fc["parameter"],
            max_val=fc.get("max"),
            min_val=fc.get("min"),
            unit=fc.get("unit", ""),
        ))

    def reward_fn(task_specs: dict[str, Any], results: dict[str, Any]) -> float:
        """Compute reward for a design attempt."""
        specs = []
        for name, spec_def in task_specs.items():
            specs.append(SpecTarget(
                name=name,
                min_val=spec_def.get("min"),
                max_val=spec_def.get("max"),
                target_val=spec_def.get("target"),
                weight=spec_def.get("weight", 1.0),
                unit=spec_def.get("unit", ""),
            ))

        rf = RewardFunction(
            specs=specs,
            feasibility_constraints=constraints,
            mode=mode,
            step_penalty=reward_config.get("step_penalty", 0.0),
        )

        reward_result = rf.compute(
            results=results.get("measurements", {}),
            step=results.get("step", 0),
            corner_results=results.get("corner_results"),
            design_params=results.get("design_params"),
            convergence_failed=results.get("convergence_failed", False),
        )

        return reward_result.total_reward

    return reward_fn


def validate_prerequisites(config: dict[str, Any]) -> list[str]:
    """Check RL prerequisites."""
    issues: list[str] = []

    # Check SFT checkpoint
    base_model = config.get("base_model", "")
    if base_model.startswith("./") and not Path(base_model).exists():
        issues.append(f"SFT checkpoint not found: {base_model}. Run SFT first.")

    # Check task source
    task_source = config.get("task_source", "")
    if task_source and not Path(task_source).exists():
        issues.append(f"Task source not found: {task_source}")

    # Check simulator
    sim_config = config.get("simulator", {})
    backend = sim_config.get("backend", "ngspice")
    if backend == "nabla":
        issues.append("nabla backend selected but may not be available yet.")

    # Estimate time
    time_est = estimate_training_time(config)
    if time_est["ngspice_parallel_hours"] > 168:  # > 1 week
        issues.append(
            f"Estimated training time: {time_est['ngspice_parallel_hours']:.0f} hours "
            f"({time_est['ngspice_parallel_hours']/24:.1f} days) with ngspice. "
            "Consider reducing num_episodes or increasing parallel workers."
        )

    return issues


def run_grpo(
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Main entry point for GRPO RL training.

    Note: This uses TRL's GRPOTrainer which handles:
    - Generating multiple candidate solutions per problem
    - Computing group-relative advantages
    - Policy gradient update with clipping
    - No separate value network needed

    Args:
        config_path: Path to config.
        dry_run: Validate and estimate only.

    Returns:
        0 on success.
    """
    config = load_config(config_path)
    issues = validate_prerequisites(config)

    if issues:
        for issue in issues:
            logger.warning("RL issue: %s", issue)

    # Time estimate
    time_est = estimate_training_time(config)
    logger.info(
        "RL training estimate: %d total simulations, "
        "%.1f hours with ngspice (%d workers), "
        "%.1f hours with nabla (%d workers)",
        time_est["total_simulations"],
        time_est["ngspice_parallel_hours"],
        time_est["sim_parallel_workers"],
        time_est["nabla_parallel_hours"],
        time_est["sim_parallel_workers"],
    )

    if dry_run:
        logger.info("Dry run complete.")
        print(json.dumps(time_est, indent=2))
        return 0

    # Actual training would use TRL's GRPOTrainer
    # This is a launcher that configures and calls TRL
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
    except ImportError as e:
        logger.error(
            "Required packages not installed. Install with: "
            "pip install trl transformers peft torch"
        )
        return 1

    logger.info("Loading model: %s", config["base_model"])

    # Configure LoRA
    adapter_config = config.get("adapter", {})
    lora_config = LoraConfig(
        r=adapter_config.get("lora_r", 32),
        lora_alpha=adapter_config.get("lora_alpha", 64),
        lora_dropout=adapter_config.get("lora_dropout", 0.0),
        target_modules=adapter_config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type="CAUSAL_LM",
    )

    # Configure GRPO
    grpo_config = config.get("grpo", {})
    training_config = GRPOConfig(
        output_dir=config.get("output_dir", "./outputs/rl"),
        num_generations=grpo_config.get("num_generations", 8),
        temperature=grpo_config.get("temperature", 0.8),
        max_new_tokens=config.get("max_new_tokens", 4096),
        max_prompt_length=config.get("max_prompt_length", 4096),
        learning_rate=config.get("learning_rate", 5e-6),
        per_device_train_batch_size=config.get("micro_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        logging_steps=config.get("logging_steps", 1),
        save_steps=config.get("save_steps", 100),
        bf16=config.get("bf16", True),
        report_to="wandb" if config.get("wandb_project") else "none",
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )

    # Create reward function
    reward_fn = create_reward_fn(config)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=training_config,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    logger.info("GRPO training complete. Saving model...")
    trainer.save_model()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch GRPO RL training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sys.exit(run_grpo(args.config, args.dry_run))
