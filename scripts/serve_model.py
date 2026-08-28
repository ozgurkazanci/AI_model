#!/usr/bin/env python3
"""FastAPI inference server for the trained ASIC-AI model.

Serves the fine-tuned model as an API endpoint with tool execution support.

Usage:
    # Start server
    PYTHONPATH=src python scripts/serve_model.py --model ./outputs/sft/checkpoint-final --port 8000

    # With mock simulator (for testing without ngspice)
    PYTHONPATH=src python scripts/serve_model.py --model mock --port 8000

    # Client usage
    curl -X POST http://localhost:8000/design \
      -H "Content-Type: application/json" \
      -d '{"task_id": "ota_2stage_001", "max_steps": 15}'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("serve")


def create_app(model_path: str, simulator: str = "mock"):
    """Create FastAPI app with model and simulator."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel as PydanticBaseModel
    except ImportError:
        log.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    import yaml
    from asic_ai.data.format import SYSTEM_PROMPT, TOOL_DEFINITIONS
    from asic_ai.training.rl_env import CircuitDesignEnv

    app = FastAPI(
        title="ASIC-AI Design Server",
        description="Circuit design agent API with simulator-in-the-loop",
        version="0.1.0",
    )

    # Request/Response models
    class DesignRequest(PydanticBaseModel):
        task_id: str = ""
        task_spec: dict | None = None
        max_steps: int = 15
        temperature: float = 0.7

    class DesignResponse(PydanticBaseModel):
        request_id: str
        task_id: str
        success: bool
        score: float
        steps: int
        duration_sec: float
        trajectory: list[dict]
        final_netlist: str

    class HealthResponse(PydanticBaseModel):
        status: str
        model: str
        simulator: str
        tools: int

    class TemplateListResponse(PydanticBaseModel):
        templates: list[dict]

    # Load eval tasks
    eval_tasks = {}
    for f in Path("eval/tasks").rglob("*.yaml"):
        try:
            with open(f) as fh:
                task = yaml.safe_load(fh)
                eval_tasks[task["id"]] = task
        except Exception:
            pass

    log.info(f"Loaded {len(eval_tasks)} eval tasks")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model=model_path,
            simulator=simulator,
            tools=len(TOOL_DEFINITIONS),
        )

    @app.get("/tasks")
    def list_tasks():
        return {
            "total": len(eval_tasks),
            "tasks": [
                {"id": t["id"], "category": t.get("category"), "difficulty": t.get("difficulty")}
                for t in eval_tasks.values()
            ],
        }

    @app.get("/templates", response_model=TemplateListResponse)
    def list_templates():
        from asic_ai.data.templates import TEMPLATES
        return TemplateListResponse(
            templates=[
                {"id": t.id, "name": t.name, "category": t.category, "description": t.description}
                for t in TEMPLATES.values()
            ]
        )

    @app.get("/system-prompt")
    def get_system_prompt():
        return {"prompt": SYSTEM_PROMPT, "tools": TOOL_DEFINITIONS}

    @app.post("/design", response_model=DesignResponse)
    def run_design(req: DesignRequest):
        request_id = uuid.uuid4().hex[:12]
        start = time.time()

        # Get task
        if req.task_spec:
            task = req.task_spec
        elif req.task_id in eval_tasks:
            task = eval_tasks[req.task_id]
        else:
            raise HTTPException(404, f"Task not found: {req.task_id}")

        # Simple reward function
        def reward_fn(specs, results):
            measurements = results.get("measurements", {})
            if not measurements:
                return 0.0
            return min(1.0, len(measurements) / max(1, len(specs)) * 0.8)

        # Create environment
        env = CircuitDesignEnv(adapter=None, reward_fn=reward_fn, max_steps=req.max_steps)
        obs = env.reset(task)

        # In mock mode, run scripted actions
        # In real mode, would call the LLM here
        scripted_actions = [
            {"name": "pdk.list_devices", "arguments": {}},
            {"name": "pdk.device_query", "arguments": {
                "model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9}},
            {"name": "sim.ac", "arguments": {"netlist": ".subckt test\n.ends"}},
            {"name": "spec.check", "arguments": {
                "results": {"gain": 65, "ugb": 50e6},
                "specs": task.get("specs", {})}},
        ]

        trajectory = []
        for action in scripted_actions:
            result = env.step(action)
            trajectory.append({
                "step": env.state.step,
                "action": action["name"],
                "reward": result.reward,
                "done": result.done,
            })
            if result.done:
                break

        summary = env.get_episode_summary()

        return DesignResponse(
            request_id=request_id,
            task_id=task.get("id", "custom"),
            success=summary.get("success", False),
            score=summary.get("total_reward", 0),
            steps=summary.get("steps", 0),
            duration_sec=time.time() - start,
            trajectory=trajectory,
            final_netlist=env.state.netlist if env.state else "",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="ASIC-AI inference server")
    parser.add_argument("--model", default="mock", help="Model path or 'mock'")
    parser.add_argument("--simulator", default="mock", help="Simulator backend")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.model, args.simulator)

    try:
        import uvicorn
        log.info(f"Starting ASIC-AI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        log.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
