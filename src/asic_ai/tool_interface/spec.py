import math
from typing import Dict
from .schema import SpecDefinition, SpecCheckResult, SpecCheckDetail

def _compute_score(actual: float, min_val: float | None, max_val: float | None) -> tuple[bool, float]:
    """
    Compute logarithmic distance score for partial credit.
    If met, score is 1.0. If far, score decreases logarithmically down to -1.0.
    """
    score = 0.0
    met = True
    
    # Very basic safeguard against zero/negative log domains
    # In production, this would need careful handling based on spec context (e.g. negative gains).
    eps = 1e-12

    if min_val is not None:
        if actual >= min_val:
            s_min = 1.0
        else:
            met = False
            # Log ratio, clip to -1
            ratio = max(actual, eps) / max(min_val, eps) if min_val > 0 else max(min_val, eps) / max(actual, eps)
            s_min = max(-1.0, math.log10(ratio) / math.log10(2))
        score = s_min

    if max_val is not None:
        if actual <= max_val:
            s_max = 1.0
        else:
            met = False
            # Log ratio, clip to -1
            ratio = max(max_val, eps) / max(actual, eps) if actual > 0 else max(actual, eps) / max(max_val, eps)
            s_max = max(-1.0, math.log10(ratio) / math.log10(2))
            
        if min_val is not None:
            score = min(score, s_max) # Worst of both
        else:
            score = s_max
            
    return met, score


def check(results: Dict[str, float], specs: Dict[str, SpecDefinition]) -> SpecCheckResult:
    """
    Check simulation results against specifications and compute a reward score.
    
    Args:
        results: Dictionary mapping spec names to actual evaluated scalar values.
        specs: Dictionary of target specifications.
        
    Returns:
        SpecCheckResult containing overall score and per-spec breakdown.
    """
    breakdown = {}
    total_weight = 0.0
    weighted_score_sum = 0.0

    for spec_name, spec in specs.items():
        if spec_name not in results:
            # Missing result implies severe failure for this spec
            breakdown[spec_name] = SpecCheckDetail(
                target_value=spec.target,
                min_value=spec.min,
                max_value=spec.max,
                actual=0.0,
                met=False,
                score=-1.0
            )
            total_weight += spec.weight
            weighted_score_sum += -1.0 * spec.weight
            continue

        actual = results[spec_name]
        met, score = _compute_score(actual, spec.min, spec.max)
        
        breakdown[spec_name] = SpecCheckDetail(
            target_value=spec.target,
            min_value=spec.min,
            max_value=spec.max,
            actual=actual,
            met=met,
            score=score
        )
        
        total_weight += spec.weight
        weighted_score_sum += score * spec.weight

    overall_score = weighted_score_sum / total_weight if total_weight > 0 else 0.0

    return SpecCheckResult(
        score=overall_score,
        breakdown=breakdown
    )
