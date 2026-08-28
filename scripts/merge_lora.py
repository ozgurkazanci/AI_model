#!/usr/bin/env python3
"""Merge LoRA adapter weights into base model.

After SFT fine-tuning with LoRA, this script merges the adapter
into the base model for faster inference and deployment.

Usage:
    # Merge and save
    PYTHONPATH=src python scripts/merge_lora.py \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/sft_local/final \
        --output models/asic-ai-0.5b-merged

    # Merge and push to HuggingFace
    PYTHONPATH=src python scripts/merge_lora.py \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/sft_local/final \
        --output models/asic-ai-0.5b-merged \
        --push ozgurkazanci/asic-ai-0.5b-v1
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
log = logging.getLogger("merge")

SEP = "=" * 70


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name or path")
    parser.add_argument("--adapter", default="outputs/sft_local/final", help="LoRA adapter path")
    parser.add_argument("--output", default="models/asic-ai-merged", help="Output directory")
    parser.add_argument("--push", default=None, help="HuggingFace repo to push to")
    parser.add_argument("--test", action="store_true", help="Run inference test after merge")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI LoRA Merge Tool")
    print(f"{SEP}\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Step 1: Load base model
    print("[1/4] Loading base model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.float32,
    )
    base_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"  Base: {args.base} ({base_params:.0f}M params)")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # Step 2: Load LoRA adapter
    print(f"\n[2/4] Loading LoRA adapter: {args.adapter}")
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        log.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  LoRA params: {lora_params:.1f}M")

    # Load training info
    info_path = adapter_path / "training_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"  Training: {info.get('epochs', '?')} epochs, {info.get('examples', '?')} examples")
        print(f"  LoRA config: r={info.get('lora_r', '?')}, alpha={info.get('lora_alpha', '?')}")

    # Step 3: Merge
    print(f"\n[3/4] Merging LoRA into base model...")
    t0 = time.time()
    merged_model = model.merge_and_unload()
    merged_params = sum(p.numel() for p in merged_model.parameters()) / 1e6
    print(f"  Merged: {merged_params:.0f}M params")
    print(f"  Merge time: {time.time() - t0:.1f}s")

    # Step 4: Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] Saving merged model to: {output_path}")
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save merge info
    merge_info = {
        "base_model": args.base,
        "adapter_path": str(adapter_path),
        "merged_params_m": round(merged_params),
        "lora_params_m": round(lora_params, 1),
        "merge_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_path / "merge_info.json").write_text(json.dumps(merge_info, indent=2))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Total size: {size_mb:.0f} MB")

    # Optional: test inference
    if args.test:
        print(f"\n  Running inference test...")
        from asic_ai.data.format import SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Design a simple OTA for sky130. Start with a PDK query."},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs, max_new_tokens=64, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  Response: {response[:200]}")

    # Optional: push to HuggingFace
    if args.push:
        print(f"\n  Pushing to HuggingFace: {args.push}")
        merged_model.push_to_hub(args.push)
        tokenizer.push_to_hub(args.push)
        print(f"  Pushed successfully!")

    print(f"\n{SEP}")
    print(f"  Merge complete: {output_path}")
    print(f"  Use for inference: python scripts/run_agent.py --model {output_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
