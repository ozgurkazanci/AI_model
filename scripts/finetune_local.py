#!/usr/bin/env python3
"""Local SFT fine-tuning on CPU/DirectML.

Fine-tunes a small model (0.5B/1.5B) with LoRA on the generated SFT data.
Works on CPU (slow but functional) or AMD GPU via DirectML.

This validates the ENTIRE training pipeline locally before spending
money on cloud GPU for the 35B model.

Usage:
    # Quick test (1 epoch, small subset)
    PYTHONPATH=src python scripts/finetune_local.py --quick-test

    # Full training (CPU, will take hours)
    PYTHONPATH=src python scripts/finetune_local.py --epochs 3

    # Resume from checkpoint
    PYTHONPATH=src python scripts/finetune_local.py --resume outputs/sft_local/checkpoint-100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("finetune")

SEP = "=" * 70


def load_sft_data(data_dir: str = "data/sft", max_examples: int | None = None) -> list[dict]:
    """Load and merge all SFT JSONL files."""
    data_path = Path(data_dir)
    all_examples = []

    for f in sorted(data_path.glob("*.jsonl")):
        if f.name == "demo_output.jsonl":
            continue  # Skip demo
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line.strip())
                if "messages" in ex:
                    all_examples.append(ex)

    log.info(f"Loaded {len(all_examples)} examples from {data_path}")

    if max_examples and len(all_examples) > max_examples:
        all_examples = all_examples[:max_examples]
        log.info(f"Truncated to {max_examples} examples")

    return all_examples


def format_for_training(examples: list[dict], tokenizer) -> list[dict]:
    """Convert SFT examples to training format."""
    formatted = []
    for ex in examples:
        messages = ex["messages"]
        # Use tokenizer's chat template to format
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            # Manual chatml fallback
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        formatted.append({"text": text, "id": ex.get("id", "unknown")})

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Local SFT fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="outputs/sft_local")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--quick-test", action="store_true", help="1 epoch, 20 examples")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 1
        max_examples = 20
        args.save_steps = 10
        log.info("Quick test mode: 1 epoch, 20 examples")
    else:
        max_examples = None

    print(f"\n{SEP}")
    print("   ASIC-AI Local SFT Fine-Tuning")
    print(f"{SEP}\n")

    # =============================================
    # Step 1: Check dependencies
    # =============================================
    print("[1/6] Checking dependencies...")
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        log.error(f"Missing: {e}. Install: pip install peft datasets transformers torch")
        sys.exit(1)

    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: CPU (AMD 780M DirectML not yet compatible with PyTorch 2.5)")

    # =============================================
    # Step 2: Load data
    # =============================================
    print(f"\n[2/6] Loading training data...")
    examples = load_sft_data(args.data_dir, max_examples)
    print(f"  Examples: {len(examples)}")

    # =============================================
    # Step 3: Load model + tokenizer
    # =============================================
    print(f"\n[3/6] Loading model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {param_count:.0f}M")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # =============================================
    # Step 4: Apply LoRA
    # =============================================
    print(f"\n[4/6] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({trainable/total*100:.2f}%)")

    # =============================================
    # Step 5: Prepare dataset
    # =============================================
    print(f"\n[5/6] Preparing dataset...")

    formatted = format_for_training(examples, tokenizer)

    def tokenize_fn(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_list(formatted)
    tokenized = dataset.map(tokenize_fn, remove_columns=["text", "id"])

    avg_len = sum(len(t) for t in tokenized["input_ids"]) / len(tokenized)
    print(f"  Tokenized: {len(tokenized)} examples")
    print(f"  Avg length: {avg_len:.0f} tokens")
    print(f"  Max seq len: {args.max_seq_len}")

    # =============================================
    # Step 6: Train
    # =============================================
    print(f"\n[6/6] Starting training...")

    total_steps = (len(tokenized) * args.epochs) // (args.batch_size * args.grad_accum)
    est_time_min = total_steps * 2.0  # ~2s per step on CPU (rough estimate)

    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Est. time: ~{est_time_min:.0f} min ({est_time_min/60:.1f} hours)")
    print(f"  Output: {args.output}")
    print(f"\n  Training started at: {time.strftime('%H:%M:%S')}")
    print(f"{SEP}\n")

    warmup_steps = max(1, total_steps // 10)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=False,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    if args.resume:
        log.info(f"Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    print(f"\n{SEP}")
    print("   Training Complete!")
    print(f"{SEP}")

    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"  Model saved to: {final_path}")
    print(f"  Finished at: {time.strftime('%H:%M:%S')}")

    # Save training info
    info = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "examples": len(examples),
        "trainable_params_m": round(trainable / 1e6, 1),
        "total_params_m": round(total / 1e6),
        "avg_token_length": round(avg_len),
    }
    info_path = final_path / "training_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  Training info: {info_path}")

    print(f"\n  Next: Test with local_inference.py or validate_with_real_model.py")
    print(f"  python scripts/validate_with_real_model.py --model {final_path} --cpu")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
