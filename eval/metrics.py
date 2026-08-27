import json
import statistics
from typing import Dict, Any, List

def compute_metrics(results_path: str) -> Dict[str, Any]:
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data["results"]
    total = len(results)
    if total == 0:
        return {}
        
    passed = [r for r in results if r["passed"]]
    scores = [r["final_score"] for r in results]
    steps = [r["steps"] for r in passed] if passed else [0]
    
    metrics = {
        "pass_rate": len(passed) / total,
        "avg_score": sum(scores) / total,
        "avg_steps_to_solve": sum(steps) / len(passed) if passed else 0,
        "efficiency": sum(scores) / sum([r["steps"] for r in results if r["steps"] > 0]) if any(r["steps"] > 0 for r in results) else 0,
        "stats": {
            "score_mean": statistics.mean(scores),
            "score_median": statistics.median(scores),
            "score_std": statistics.stdev(scores) if total > 1 else 0.0,
            "score_min": min(scores),
            "score_max": max(scores)
        }
    }
    return metrics

def compare_runs(run1_path: str, run2_path: str) -> Dict[str, Any]:
    metrics1 = compute_metrics(run1_path)
    metrics2 = compute_metrics(run2_path)
    
    delta = {
        "pass_rate": metrics2.get("pass_rate", 0) - metrics1.get("pass_rate", 0),
        "avg_score": metrics2.get("avg_score", 0) - metrics1.get("avg_score", 0),
        "efficiency": metrics2.get("efficiency", 0) - metrics1.get("efficiency", 0)
    }
    
    return {
        "run1": metrics1,
        "run2": metrics2,
        "delta": delta
    }
