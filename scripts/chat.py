#!/usr/bin/env python3
"""Interactive CLI chat with the ASIC-AI model.

Chat with your fine-tuned model in the terminal. The model has access
to the full system prompt and tool definitions.

Usage:
    PYTHONPATH=src python scripts/chat.py --model outputs/sft_local/final
    PYTHONPATH=src python scripts/chat.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import TOOL_DEFINITIONS, build_system_message

SEP = "=" * 60


def main():
    parser = argparse.ArgumentParser(description="Chat with ASIC-AI model")
    parser.add_argument("--model", default="outputs/sft_local/final")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("   ASIC-AI Interactive Chat")
    print(f"{SEP}")
    print(f"  Model: {args.model}")
    print(f"  Loading...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {params:.0f}M params")

    # Must be byte-identical to the training-time system message.
    system_content = build_system_message()

    messages = [{"role": "system", "content": system_content}]

    print(f"\n  Type your circuit design questions. Type 'quit' to exit.")
    print(f"  Type 'reset' to start a new conversation.")
    print(f"  Type 'tools' to list available tools.")
    print(f"{SEP}\n")

    while True:
        try:
            user_input = input("  You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("  Bye!")
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": system_content}]
            print("  [Conversation reset]\n")
            continue
        if user_input.lower() == "tools":
            for t in TOOL_DEFINITIONS:
                print(f"    {t['function']['name']:20s} {t['function'].get('description', '')[:60]}")
            print()
            continue

        messages.append({"role": "user", "content": user_input})

        # Generate
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        t0 = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                temperature=args.temperature, do_sample=True, top_p=0.9,
                repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - t0
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        tok_s = len(new_tokens) / gen_time if gen_time > 0 else 0

        messages.append({"role": "assistant", "content": response})

        print(f"\n  AI> {response}")
        print(f"  [{len(new_tokens)} tok, {gen_time:.1f}s, {tok_s:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
