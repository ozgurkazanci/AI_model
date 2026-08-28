#!/usr/bin/env python3
"""Multi-step agent runner with real LLM.

Actually runs the model in an agent loop:
1. Send task to model
2. Parse tool call from response
3. Execute tool via RL environment
4. Feed result back to model
5. Model generates next action
6. Repeat until done or max steps

This is the REAL agent — not scripted, the model decides what to do.

Usage:
    PYTHONPATH=src python scripts/run_agent.py --model outputs/sft_local/final --task ota_2stage_001
    PYTHONPATH=src python scripts/run_agent.py --model Qwen/Qwen2.5-0.5B-Instruct --task ota_2stage_001
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from asic_ai.data.format import SYSTEM_PROMPT, TOOL_DEFINITIONS
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall
from asic_ai.data.format import format_trajectory_for_sft, validate_sft_format
from asic_ai.training.rl_env import CircuitDesignEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent")

SEP = "=" * 70


def load_model(model_path: str):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_tool_call(text: str) -> dict | None:
    """Extract tool call from model response using multiple patterns."""
    # Pattern 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if tc_match:
        try:
            return json.loads(tc_match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: JSON block with name/arguments
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([\w.]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})', text, re.DOTALL)
    if json_match:
        name = json_match.group(1)
        try:
            args = json.loads(json_match.group(2))
            return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            return {"name": name, "arguments": {}}

    # Pattern 3: Mentions of tool names in text
    tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    for tool_name in tool_names:
        if tool_name in text:
            return {"name": tool_name, "arguments": {}}

    return None


def generate_response(model, tokenizer, messages: list[dict], max_tokens: int = 256) -> str:
    """Generate model response from message history."""
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_task(task_id: str) -> dict:
    """Load an eval task by ID."""
    for f in Path("eval/tasks").rglob("*.yaml"):
        with open(f) as fh:
            task = yaml.safe_load(fh)
            if task.get("id") == task_id:
                return task

    # Fallback
    return {
        "id": task_id,
        "description": f"Design task: {task_id}",
        "specs": {"gain": {"min": 60, "unit": "dB"}},
        "pdk": "sky130",
        "supply": 1.8,
    }


def run_agent(model, tokenizer, task: dict, max_steps: int = 8, max_tokens: int = 256):
    """Run the agent loop."""
    print(f"\n{SEP}")
    print(f"   Agent Run: {task.get('id', 'unknown')}")
    print(f"{SEP}\n")

    # Setup environment
    def reward_fn(specs, results):
        measurements = results.get("measurements", {})
        if not measurements:
            return 0.0
        return min(1.0, len(measurements) / max(1, len(specs)) * 0.8)

    env = CircuitDesignEnv(adapter=None, reward_fn=reward_fn, max_steps=max_steps)
    obs = env.reset(task)

    # Build tool list for system prompt
    tool_names = ", ".join([t["function"]["name"] for t in TOOL_DEFINITIONS])
    system_content = (
        f"{SYSTEM_PROMPT}\n\nAvailable tools: {tool_names}\n\n"
        "To call a tool, use: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"
    )

    specs_str = json.dumps(task.get("specs", {}), indent=2)
    user_content = (
        f"Design: {task.get('description', task['id'])}\n"
        f"PDK: {task.get('pdk', 'sky130')}, Supply: {task.get('supply', 1.8)}V\n"
        f"Specifications:\n{specs_str}"
    )

    # Message history
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Trajectory recording
    trajectory_steps = [
        TrajectoryStep(step_index=0, role="user", content=user_content),
    ]
    step_idx = 1

    print(f"  Task: {task.get('description', task['id'])}")
    print(f"  Max steps: {max_steps}\n")

    total_gen_time = 0
    total_tokens = 0

    for step in range(max_steps):
        # Generate response
        t0 = time.time()
        response = generate_response(model, tokenizer, messages, max_tokens)
        gen_time = time.time() - t0
        total_gen_time += gen_time
        resp_tokens = len(tokenizer.encode(response))
        total_tokens += resp_tokens

        print(f"  Step {step+1}/{max_steps}: ({resp_tokens} tok, {gen_time:.1f}s)")
        print(f"    Model: {response[:120]}...")

        # Extract tool call
        tool_call = extract_tool_call(response)

        if tool_call:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            print(f"    Tool:  {tool_name}({json.dumps(tool_args)[:60]})")

            # Record assistant step
            trajectory_steps.append(TrajectoryStep(
                step_index=step_idx, role="assistant", content=response[:500],
                tool_call=ToolCall(name=tool_name, call_id=f"call_{step_idx:03d}", arguments=tool_args),
            ))
            step_idx += 1

            # Execute in environment
            result = env.step({"name": tool_name, "arguments": tool_args})
            print(f"    Result: reward={result.reward:+.3f} (total={result.info['total_reward']:.3f})")

            # Record tool result
            tool_result = result.observation[:500]
            trajectory_steps.append(TrajectoryStep(
                step_index=step_idx, role="tool", content=tool_result,
            ))
            step_idx += 1

            # Add to message history
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result: {tool_result[:300]}"})

            if result.done:
                print(f"    [DONE] Episode finished")
                break
        else:
            # No tool call — model is giving a final answer
            print(f"    [NO TOOL] Model gave text response (final answer)")
            trajectory_steps.append(TrajectoryStep(
                step_index=step_idx, role="assistant", content=response[:500],
            ))
            break

    # Final summary
    summary = env.get_episode_summary()
    avg_speed = total_tokens / total_gen_time if total_gen_time > 0 else 0

    print(f"\n{SEP}")
    print(f"   Agent Summary")
    print(f"{SEP}")
    print(f"  Steps:        {summary.get('steps', 0)}")
    print(f"  Total reward:  {summary.get('total_reward', 0):.3f}")
    print(f"  Success:       {summary.get('success', False)}")
    print(f"  Tokens:        {total_tokens}")
    print(f"  Avg speed:     {avg_speed:.1f} tok/s")
    print(f"  Total time:    {total_gen_time:.1f}s")

    # Record trajectory
    traj = Trajectory(
        id=f"agent_{uuid.uuid4().hex[:8]}",
        task_id=task.get("id", "unknown"),
        steps=trajectory_steps,
        success=summary.get("success", False),
        final_score=summary.get("total_reward", 0),
        duration_seconds=total_gen_time,
        metadata={"model": "local", "steps": summary.get("steps", 0)},
    )

    # Validate as SFT
    sft_msgs = format_trajectory_for_sft(traj)
    is_valid, errors = validate_sft_format(sft_msgs)
    print(f"  SFT valid:     {is_valid}")

    # Save trajectory
    out_path = Path("eval_results/agent_run.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": traj.id, "task_id": traj.task_id,
            "messages": sft_msgs, "score": traj.final_score,
            "success": traj.success, "steps": summary.get("steps", 0),
        }, ensure_ascii=False) + "\n")
    print(f"  Saved to:      {out_path}")
    print(f"{SEP}\n")

    return traj


def main():
    parser = argparse.ArgumentParser(description="Run ASIC-AI agent with real LLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", default="ota_2stage_001")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    task = load_task(args.task)
    run_agent(model, tokenizer, task, args.max_steps, args.max_tokens)


if __name__ == "__main__":
    main()
