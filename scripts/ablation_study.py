#!/usr/bin/env python3
"""Run ablation experiments to compare training configurations.

Compares curriculum vs random ordering, different LoRA ranks, etc.

Usage:
    PYTHONPATH=src python scripts/ablation_study.py --experiments curriculum,lora_rank
    PYTHONPATH=src python scripts/ablation_study.py --all --max-examples 50 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEP = "=" * 60


def run_experiment(name: str, config: dict, data_dir: str = "data/sft") -> dict:
    """Run a single ablation experiment and return metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print(f"\n  [{name}] Starting...")

    # Load data
    from finetune_local import load_sft_data, format_for_training
    examples = load_sft_data(data_dir, max_examples=config.get("max_examples"))

    if config.get("shuffle", False):
        random.shuffle(examples)

    # Load model
    model_name = config.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", lora_r * 2)
    targets = config.get("lora_target_modules", ["q_proj", "v_proj"])
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # Format data
    formatted = format_for_training(examples, tokenizer)
    ds = Dataset.from_list(formatted)

    # Train
    output_dir = f"outputs/ablation/{name}"
    epochs = config.get("epochs", 1)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 0.0002),
        warmup_steps=config.get("warmup_steps", 2),
        logging_steps=5,
        save_strategy="no",
        use_cpu=True,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    # Extract metrics
    log_history = trainer.state.log_history
    losses = [e["loss"] for e in log_history if "loss" in e]

    metrics = {
        "name": name,
        "config": config,
        "trainable_params_m": round(trainable, 1),
        "examples": len(examples),
        "epochs": epochs,
        "train_time_s": round(train_time, 1),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "loss_reduction_pct": round((1 - losses[-1] / losses[0]) * 100, 1) if len(losses) >= 2 else 0,
        "all_losses": losses,
    }

    print(f"  [{name}] Done in {train_time:.0f}s | Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({metrics['loss_reduction_pct']}%)")
    return metrics


EXPERIMENTS = {
    "curriculum": {
        "description": "Curriculum ordering (easy -> hard)",
        "shuffle": False,
    },
    "random": {
        "description": "Random ordering",
        "shuffle": True,
    },
    "lora_r8": {
        "description": "LoRA rank 8",
        "lora_r": 8,
    },
    "lora_r16": {
        "description": "LoRA rank 16 (default)",
        "lora_r": 16,
    },
    "lora_r32": {
        "description": "LoRA rank 32",
        "lora_r": 32,
    },
    "lr_low": {
        "description": "Low learning rate (1e-4)",
        "learning_rate": 0.0001,
    },
    "lr_high": {
        "description": "High learning rate (5e-4)",
        "learning_rate": 0.0005,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--experiments", default="curriculum,random", help="Comma-separated experiment names")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--max-examples", type=int, default=30, help="Max examples per experiment")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per experiment")
    parser.add_argument("--output", default="eval_results/ablation.json")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Ablation Study")
    print(f"{SEP}")

    if args.all:
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e.strip() for e in args.experiments.split(",")]

    print(f"  Experiments: {', '.join(exp_names)}")
    print(f"  Max examples: {args.max_examples}")
    print(f"  Epochs: {args.epochs}")

    results = []
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"  [SKIP] Unknown experiment: {name}")
            continue
        config = EXPERIMENTS[name].copy()
        config["max_examples"] = args.max_examples
        config["epochs"] = args.epochs
        try:
            metrics = run_experiment(name, config)
            results.append(metrics)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            results.append({"name": name, "error": str(e)})

    # Summary
    print(f"\n{SEP}")
    print("   Results Summary")
    print(f"{SEP}")
    print(f"  {'Experiment':<15} {'Init Loss':>10} {'Final Loss':>11} {'Reduction':>10} {'Time':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*11} {'-'*10} {'-'*8}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<15} {'FAILED':>10}")
        else:
            print(f"  {r['name']:<15} {r['initial_loss']:>10.4f} {r['final_loss']:>11.4f} {r['loss_reduction_pct']:>9.1f}% {r['train_time_s']:>7.0f}s")
    print(f"{SEP}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
