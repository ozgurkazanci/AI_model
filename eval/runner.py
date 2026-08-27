import argparse
import json
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import yaml
from pydantic import BaseModel, Field

class EvalResult(BaseModel):
    task_id: str
    passed: bool
    final_score: float
    steps: int
    wall_time_sec: float
    trajectory: List[Dict[str, Any]]
    error: str | None = None

def load_task(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_task(task_path: Path, model_id: str) -> EvalResult:
    start_time = time.time()
    try:
        task_data = load_task(task_path)
        # Mock environment integration (e.g., from tool_interface.env import CircuitEnv)
        # env = CircuitEnv(task_data)
        # trajectory, score, passed, steps = run_model(env, model_id)
        
        # Mock results
        passed = True
        score = 85.5
        steps = 15
        trajectory = []
        error = None
    except Exception as e:
        passed = False
        score = 0.0
        steps = 0
        trajectory = []
        error = str(e)
    
    wall_time = time.time() - start_time
    
    return EvalResult(
        task_id=task_path.stem,
        passed=passed,
        final_score=score,
        steps=steps,
        wall_time_sec=wall_time,
        trajectory=trajectory,
        error=error
    )

def main():
    parser = argparse.ArgumentParser(description="Eval runner")
    parser.add_argument("--tasks", type=str, required=True, help="Path to tasks directory")
    parser.add_argument("--model", type=str, required=True, help="Model ID to evaluate")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks)
    task_files = list(tasks_dir.rglob("*.yaml"))
    
    results = []
    
    if args.parallel:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_task, f, args.model) for f in task_files]
            for future in futures:
                results.append(future.result().model_dump())
    else:
        for f in task_files:
            results.append(run_task(f, args.model).model_dump())
            
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
