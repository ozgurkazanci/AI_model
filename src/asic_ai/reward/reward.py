"""Reward function for ASIC circuit design RL training.

This module implements the reward computation logic described in the design document
(Section 4, Stage 3). The reward signal comes from the simulator, not humans.

Key design principles:
- Partial credit: logarithmic distance, not binary pass/fail
- Corner-aware: PVT + Monte Carlo results included in reward
- Feasibility constraints: area, current, device size limits built in
- Non-convergence penalty: negative (not zero) to avoid conservatism
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RewardMode(Enum):
    """Reward computation mode."""
    NOMINAL_ONLY = "nominal_only"
    WORST_CORNER = "worst_corner"
    ALL_CORNERS_WEIGHTED = "all_corners_weighted"
    MONTE_CARLO = "monte_carlo"


@dataclass
class SpecTarget:
    """A single specification target."""
    name: str
    min_val: float | None = None
    max_val: float | None = None
    target_val: float | None = None
    weight: float = 1.0
    unit: str = ""
    # Feasibility bounds (hard limits, not optimization targets)
    feasibility_min: float | None = None
    feasibility_max: float | None = None

    def __post_init__(self) -> None:
        if self.min_val is None and self.max_val is None and self.target_val is None:
            raise ValueError(f"Spec '{self.name}' must have at least one of min, max, or target.")


@dataclass
class FeasibilityConstraint:
    """Hard constraint that must be satisfied for the design to be valid."""
    name: str
    parameter: str  # e.g., 'total_area', 'total_current', 'max_W', 'max_L'
    max_val: float | None = None
    min_val: float | None = None
    unit: str = ""


@dataclass
class SpecScore:
    """Score for a single specification."""
    name: str
    target_min: float | None
    target_max: float | None
    actual: float
    met: bool
    score: float  # -1.0 to 1.0, where 1.0 = fully met
    unit: str = ""
    margin_db: float | None = None  # margin in dB (for log-scale specs)


@dataclass
class FeasibilityResult:
    """Result of a feasibility check."""
    name: str
    parameter: str
    value: float
    limit: float
    passed: bool
    violation_ratio: float  # how much over/under the limit


@dataclass
class RewardResult:
    """Complete reward computation result."""
    total_reward: float
    spec_scores: list[SpecScore]
    feasibility_results: list[FeasibilityResult]
    feasibility_passed: bool
    all_specs_met: bool
    corner_scores: dict[str, float] | None = None  # corner_name -> score
    convergence_failed: bool = False
    penalty_applied: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardFunction:
    """Reward function for circuit design RL.

    The reward is the weighted sum of per-spec scores, with feasibility
    constraints and convergence penalties applied.

    Score computation uses logarithmic distance for partial credit:
    - For min specs: score = clip(log2(actual / target_min), -1, 1)
    - For max specs: score = clip(log2(target_max / actual), -1, 1)
    - For target specs: score = 1.0 - clip(|log2(actual / target)| / tolerance, 0, 2)

    This means:
    - Meeting the spec exactly: score = 0.0 (at boundary) to 1.0 (with margin)
    - 2x better than spec: score = 1.0
    - 2x worse than spec: score = -1.0
    - Between: logarithmic interpolation

    Feasibility violations multiply the total reward by a penalty factor.
    Non-convergence gets a fixed negative reward (not zero, to avoid conservatism).
    """

    # Constants
    NON_CONVERGENCE_REWARD: float = -0.5
    FEASIBILITY_PENALTY_FACTOR: float = 0.1  # multiply reward by this if infeasible
    LOG_BASE: float = 2.0  # log base for distance computation
    SCORE_CLIP_MIN: float = -1.0
    SCORE_CLIP_MAX: float = 1.0

    def __init__(
        self,
        specs: list[SpecTarget],
        feasibility_constraints: list[FeasibilityConstraint] | None = None,
        mode: RewardMode = RewardMode.WORST_CORNER,
        step_penalty: float = 0.0,
        max_steps: int = 20,
    ) -> None:
        """Initialize reward function.

        Args:
            specs: List of specification targets.
            feasibility_constraints: Optional hard constraints.
            mode: How to handle corners (nominal, worst, weighted, MC).
            step_penalty: Per-step penalty to encourage efficiency.
            max_steps: Maximum steps (for normalizing step penalty).
        """
        self.specs = {s.name: s for s in specs}
        self.feasibility_constraints = feasibility_constraints or []
        self.mode = mode
        self.step_penalty = step_penalty
        self.max_steps = max_steps

    def compute(
        self,
        results: dict[str, float],
        step: int = 0,
        corner_results: dict[str, dict[str, float]] | None = None,
        mc_results: list[dict[str, float]] | None = None,
        design_params: dict[str, float] | None = None,
        convergence_failed: bool = False,
    ) -> RewardResult:
        """Compute reward for a design.

        Args:
            results: Nominal simulation results, mapping spec_name -> actual_value.
            step: Current step number (for step penalty).
            corner_results: Per-corner results: corner_name -> {spec_name -> value}.
            mc_results: Monte Carlo results: list of {spec_name -> value}.
            design_params: Design parameters for feasibility checks.
            convergence_failed: Whether simulation failed to converge.

        Returns:
            RewardResult with total reward and breakdown.
        """
        # Handle non-convergence
        if convergence_failed:
            return RewardResult(
                total_reward=self.NON_CONVERGENCE_REWARD,
                spec_scores=[],
                feasibility_results=[],
                feasibility_passed=False,
                all_specs_met=False,
                convergence_failed=True,
                penalty_applied=0.0,
                metadata={"reason": "convergence_failure"},
            )

        # Compute per-spec scores for nominal
        nominal_scores = self._compute_spec_scores(results)

        # Handle corners
        corner_scores_map: dict[str, float] | None = None
        if self.mode == RewardMode.WORST_CORNER and corner_results:
            corner_scores_map = {}
            for corner_name, corner_vals in corner_results.items():
                c_scores = self._compute_spec_scores(corner_vals)
                corner_total = self._weighted_average(c_scores)
                corner_scores_map[corner_name] = corner_total

        elif self.mode == RewardMode.ALL_CORNERS_WEIGHTED and corner_results:
            corner_scores_map = {}
            for corner_name, corner_vals in corner_results.items():
                c_scores = self._compute_spec_scores(corner_vals)
                corner_total = self._weighted_average(c_scores)
                corner_scores_map[corner_name] = corner_total

        elif self.mode == RewardMode.MONTE_CARLO and mc_results:
            # Use worst-case across MC runs
            corner_scores_map = {}
            for i, mc_vals in enumerate(mc_results):
                mc_scores = self._compute_spec_scores(mc_vals)
                mc_total = self._weighted_average(mc_scores)
                corner_scores_map[f"mc_{i}"] = mc_total

        # Compute total spec score
        if self.mode == RewardMode.NOMINAL_ONLY or corner_scores_map is None:
            total_spec_score = self._weighted_average(nominal_scores)
        elif self.mode == RewardMode.WORST_CORNER:
            nominal_total = self._weighted_average(nominal_scores)
            worst_corner = min(corner_scores_map.values()) if corner_scores_map else nominal_total
            total_spec_score = min(nominal_total, worst_corner)
        elif self.mode == RewardMode.ALL_CORNERS_WEIGHTED:
            all_scores = [self._weighted_average(nominal_scores)]
            all_scores.extend(corner_scores_map.values())
            total_spec_score = sum(all_scores) / len(all_scores)
        elif self.mode == RewardMode.MONTE_CARLO:
            # Percentile-based: use 5th percentile (worst 5%)
            sorted_scores = sorted(corner_scores_map.values())
            idx = max(0, int(len(sorted_scores) * 0.05))
            total_spec_score = sorted_scores[idx] if sorted_scores else 0.0
        else:
            total_spec_score = self._weighted_average(nominal_scores)

        # Check feasibility
        feasibility_results = self._check_feasibility(design_params or {})
        feasibility_passed = all(fr.passed for fr in feasibility_results)

        # Apply feasibility penalty
        penalty = 0.0
        if not feasibility_passed:
            # Don't zero out, but severely penalize
            worst_violation = max(
                (fr.violation_ratio for fr in feasibility_results if not fr.passed),
                default=0.0,
            )
            penalty_factor = self.FEASIBILITY_PENALTY_FACTOR / (1.0 + worst_violation)
            penalty = total_spec_score * (1.0 - penalty_factor)
            total_spec_score *= penalty_factor

        # Apply step penalty (encourage efficiency)
        step_pen = self.step_penalty * (step / self.max_steps)
        total_reward = total_spec_score - step_pen
        penalty += step_pen

        # Check if all specs are met
        all_met = all(s.met for s in nominal_scores)
        if corner_results and self.mode != RewardMode.NOMINAL_ONLY:
            for corner_vals in corner_results.values():
                corner_spec_scores = self._compute_spec_scores(corner_vals)
                if not all(s.met for s in corner_spec_scores):
                    all_met = False
                    break

        return RewardResult(
            total_reward=total_reward,
            spec_scores=nominal_scores,
            feasibility_results=feasibility_results,
            feasibility_passed=feasibility_passed,
            all_specs_met=all_met,
            corner_scores=corner_scores_map,
            convergence_failed=False,
            penalty_applied=penalty,
            metadata={
                "mode": self.mode.value,
                "step": step,
                "nominal_score": self._weighted_average(nominal_scores),
            },
        )

    def _compute_spec_scores(self, results: dict[str, float]) -> list[SpecScore]:
        """Compute per-spec scores using logarithmic distance."""
        scores = []
        for name, spec in self.specs.items():
            actual = results.get(name)
            if actual is None:
                scores.append(SpecScore(
                    name=name,
                    target_min=spec.min_val,
                    target_max=spec.max_val,
                    actual=float("nan"),
                    met=False,
                    score=self.SCORE_CLIP_MIN,
                    unit=spec.unit,
                ))
                continue

            score = 0.0
            met = True
            margin_db = None

            if spec.min_val is not None and spec.max_val is not None:
                # Range spec: both min and max
                min_score = self._log_distance_min(actual, spec.min_val)
                max_score = self._log_distance_max(actual, spec.max_val)
                score = min(min_score, max_score)
                met = actual >= spec.min_val and actual <= spec.max_val

            elif spec.min_val is not None:
                # Minimum spec (e.g., gain >= 60 dB)
                score = self._log_distance_min(actual, spec.min_val)
                met = actual >= spec.min_val
                if spec.min_val > 0 and actual > 0:
                    margin_db = 20.0 * math.log10(actual / spec.min_val)

            elif spec.max_val is not None:
                # Maximum spec (e.g., current <= 200 µA)
                score = self._log_distance_max(actual, spec.max_val)
                met = actual <= spec.max_val
                if spec.max_val > 0 and actual > 0:
                    margin_db = 20.0 * math.log10(spec.max_val / actual)

            elif spec.target_val is not None:
                # Target spec (e.g., Vref = 1.2 V)
                score = self._log_distance_target(actual, spec.target_val)
                tolerance = 0.02  # 2% default tolerance
                met = abs(actual - spec.target_val) <= abs(spec.target_val * tolerance)

            scores.append(SpecScore(
                name=name,
                target_min=spec.min_val,
                target_max=spec.max_val,
                actual=actual,
                met=met,
                score=score,
                unit=spec.unit,
                margin_db=margin_db,
            ))

        return scores

    def _log_distance_min(self, actual: float, target_min: float) -> float:
        """Logarithmic distance for minimum spec.

        score = clip(log2(actual / target_min), -1, 1)
        - actual >= 2 * target_min: score = 1.0
        - actual == target_min: score = 0.0
        - actual == target_min / 2: score = -1.0
        """
        if target_min <= 0 or actual <= 0:
            return self.SCORE_CLIP_MIN if actual < target_min else self.SCORE_CLIP_MAX

        ratio = actual / target_min
        if ratio <= 0:
            return self.SCORE_CLIP_MIN

        score = math.log(ratio) / math.log(self.LOG_BASE)
        return max(self.SCORE_CLIP_MIN, min(self.SCORE_CLIP_MAX, score))

    def _log_distance_max(self, actual: float, target_max: float) -> float:
        """Logarithmic distance for maximum spec.

        score = clip(log2(target_max / actual), -1, 1)
        - actual <= target_max / 2: score = 1.0
        - actual == target_max: score = 0.0
        - actual == 2 * target_max: score = -1.0
        """
        if target_max <= 0 or actual <= 0:
            return self.SCORE_CLIP_MAX if actual <= 0 else self.SCORE_CLIP_MIN

        ratio = target_max / actual
        if ratio <= 0:
            return self.SCORE_CLIP_MIN

        score = math.log(ratio) / math.log(self.LOG_BASE)
        return max(self.SCORE_CLIP_MIN, min(self.SCORE_CLIP_MAX, score))

    def _log_distance_target(self, actual: float, target: float) -> float:
        """Logarithmic distance for target spec (exact value).

        score = 1.0 - clip(|log2(actual / target)| / tolerance_decades, 0, 2)
        """
        if target == 0:
            return self.SCORE_CLIP_MAX if actual == 0 else self.SCORE_CLIP_MIN
        if actual <= 0:
            return self.SCORE_CLIP_MIN

        log_ratio = abs(math.log(actual / target) / math.log(self.LOG_BASE))
        tolerance_decades = 0.1  # ~7% deviation for full score
        score = 1.0 - min(2.0, log_ratio / tolerance_decades)
        return max(self.SCORE_CLIP_MIN, min(self.SCORE_CLIP_MAX, score))

    def _weighted_average(self, scores: list[SpecScore]) -> float:
        """Compute weighted average of spec scores."""
        if not scores:
            return 0.0

        total_weight = sum(self.specs[s.name].weight for s in scores if s.name in self.specs)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            s.score * self.specs[s.name].weight
            for s in scores
            if s.name in self.specs
        )
        return weighted_sum / total_weight

    def _check_feasibility(
        self, design_params: dict[str, float]
    ) -> list[FeasibilityResult]:
        """Check hard feasibility constraints."""
        results = []
        for constraint in self.feasibility_constraints:
            value = design_params.get(constraint.parameter)
            if value is None:
                continue

            passed = True
            limit = 0.0
            violation_ratio = 0.0

            if constraint.max_val is not None:
                limit = constraint.max_val
                if value > constraint.max_val:
                    passed = False
                    violation_ratio = (value - constraint.max_val) / constraint.max_val

            if constraint.min_val is not None:
                limit = constraint.min_val
                if value < constraint.min_val:
                    passed = False
                    violation_ratio = (constraint.min_val - value) / constraint.min_val

            results.append(FeasibilityResult(
                name=constraint.name,
                parameter=constraint.parameter,
                value=value,
                limit=limit,
                passed=passed,
                violation_ratio=violation_ratio,
            ))

        return results

    @classmethod
    def from_eval_task(cls, task_dict: dict[str, Any]) -> RewardFunction:
        """Create RewardFunction from an eval task YAML dict.

        Args:
            task_dict: Parsed YAML eval task.

        Returns:
            Configured RewardFunction.
        """
        specs = []
        for spec_name, spec_def in task_dict.get("specs", {}).items():
            specs.append(SpecTarget(
                name=spec_name,
                min_val=spec_def.get("min"),
                max_val=spec_def.get("max"),
                target_val=spec_def.get("target"),
                weight=spec_def.get("weight", 1.0),
                unit=spec_def.get("unit", ""),
            ))

        constraints = []
        for fc in task_dict.get("feasibility_constraints", []):
            constraints.append(FeasibilityConstraint(
                name=fc["name"],
                parameter=fc["parameter"],
                max_val=fc.get("max"),
                min_val=fc.get("min"),
                unit=fc.get("unit", ""),
            ))

        # Determine mode from pass_criteria
        pass_criteria = task_dict.get("pass_criteria", "all_corners")
        if pass_criteria == "typical_only":
            mode = RewardMode.NOMINAL_ONLY
        elif pass_criteria == "all_corners":
            mode = RewardMode.WORST_CORNER
        else:
            mode = RewardMode.ALL_CORNERS_WEIGHTED

        return cls(specs=specs, feasibility_constraints=constraints, mode=mode)
