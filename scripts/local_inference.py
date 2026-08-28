#!/usr/bin/env python3
"""Local inference using llama-cpp-python or DirectML.

Supports two backends:
1. llama-cpp-python: GGUF quantized models (CPU or Vulkan)
2. DirectML: PyTorch models via AMD GPU acceleration

For AMD Radeon 780M (4GB shared VRAM):
- Use GGUF Q4_K_M quantization (fits in RAM)
- Vulkan backend for GPU acceleration
- CPU fallback always available

Usage:
    # CPU inference with GGUF model
    PYTHONPATH=src python scripts/local_inference.py --model models/qwen2.5-3b-q4.gguf --task ota_2stage_001

    # DirectML inference
    PYTHONPATH=src python scripts/local_inference.py --backend directml --model Qwen/Qwen2.5-3B-Instruct --task ota_2stage_001

    # Interactive chat mode
    PYTHONPATH=src python scripts/local_inference.py --model models/qwen2.5-3b-q4.gguf --interactive
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from asic_ai.data.format import SYSTEM_PROMPT, TOOL_DEFINITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("local_inference")


def format_tools_for_prompt() -> str:
    """Format tool definitions into system prompt section."""
    tool_lines = []
    for tool in TOOL_DEFINITIONS:
        func = tool["function"]
        tool_lines.append(f"- {func['name']}: {func['description']}")
    return "\n".join(tool_lines)


def build_prompt(task: dict) -> str:
    """Build the full prompt for a design task."""
    specs = json.dumps(task.get("specs", {}), indent=2)
    tools = format_tools_for_prompt()

    return f"""<|im_start|>system
{SYSTEM_PROMPT}

Available Tools:
{tools}
<|im_end|>
<|im_start|>user
Design task: {task.get('description', task.get('id', 'unknown'))}
PDK: {task.get('pdk', 'sky130')}
Supply: {task.get('supply', 1.8)}V

Specifications:
{specs}

Design a circuit meeting ALL specifications. Use tools to simulate and verify.
<|im_end|>
<|im_start|>assistant
"""


class LlamaCppEngine:
    """Inference engine using llama-cpp-python (GGUF models)."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        try:
            from llama_cpp import Llama
        except ImportError:
            log.error("llama-cpp-python not installed. Install: pip install llama-cpp-python")
            sys.exit(1)

        log.info(f"Loading GGUF model: {model_path}")
        log.info(f"GPU layers: {n_gpu_layers}, Context: {n_ctx}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        log.info("Model loaded successfully")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        start = time.time()
        result = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        elapsed = time.time() - start
        text = result["choices"][0]["text"]
        tokens = result["usage"]["completion_tokens"]
        speed = tokens / elapsed if elapsed > 0 else 0
        log.info(f"Generated {tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
        return text


class DirectMLEngine:
    """Inference engine using PyTorch + DirectML (AMD GPU)."""

    def __init__(self, model_name: str, max_length: int = 4096):
        try:
            import torch
            import torch_directml
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            log.error(f"Missing package: {e}. Install: pip install torch-directml transformers")
            sys.exit(1)

        self.device = torch_directml.device()
        log.info(f"DirectML device: {self.device}")

        log.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self.device)
        log.info("Model loaded on DirectML")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        elapsed = time.time() - start
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        speed = len(new_tokens) / elapsed if elapsed > 0 else 0
        log.info(f"Generated {len(new_tokens)} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
        return text


def load_task(task_id: str) -> dict:
    """Load an eval task by ID."""
    for f in Path("eval/tasks").rglob("*.yaml"):
        with open(f) as fh:
            task = yaml.safe_load(fh)
            if task.get("id") == task_id:
                return task
    raise FileNotFoundError(f"Task not found: {task_id}")


def run_interactive(engine):
    """Interactive chat with the model."""
    print("\n=== ASIC-AI Interactive Mode ===")
    print("Type 'quit' to exit, 'task:<id>' to load an eval task\n")

    history = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.startswith("task:"):
            task_id = user_input[5:].strip()
            try:
                task = load_task(task_id)
                user_input = f"Design: {task.get('description', task_id)}. Specs: {json.dumps(task.get('specs', {}))}"
                print(f"[Loaded task: {task_id}]")
            except FileNotFoundError:
                print(f"[Task not found: {task_id}]")
                continue

        history += f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
        response = engine.generate(history, max_tokens=1024)
        history += f"{response}\n<|im_end|>\n"
        print(f"\nAssistant> {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Local ASIC-AI inference")
    parser.add_argument("--model", required=True, help="Model path (GGUF file or HF model name)")
    parser.add_argument("--backend", choices=["gguf", "directml"], default="gguf")
    parser.add_argument("--task", help="Eval task ID to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layers for GGUF (0=CPU only)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Create engine
    if args.backend == "gguf":
        engine = LlamaCppEngine(args.model, n_gpu_layers=args.gpu_layers)
    else:
        engine = DirectMLEngine(args.model)

    # Run mode
    if args.interactive:
        run_interactive(engine)
    elif args.task:
        task = load_task(args.task)
        prompt = build_prompt(task)
        print(f"\n=== Task: {task['id']} ===\n")
        response = engine.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        print(f"\n=== Response ===\n{response}\n")
    else:
        log.error("Specify --task <id> or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
