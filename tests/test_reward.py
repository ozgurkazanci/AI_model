"""Tests for the reward function module."""

import math
import pytest
from asic_ai.reward.reward import (
    RewardFunction,
    RewardResult,
    RewardMode,
    SpecTarget,
    SpecScore,
    FeasibilityConstraint,
    FeasibilityResult,
)


class TestLogDistance:
    """Test logarithmic distance scoring for partial credit."""

    def setup_method(self):
        self.rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0, unit="dB")],
        )

    def test_min_spec_met_exactly(self):
        """Spec met exactly at boundary: score should be ~0.0."""
        result = self.rf.compute(results={"gain_db": 60.0})
        score = result.spec_scores[0].score
        assert abs(score) < 1e-6, f"Expected ~0.0, got {score}"

    def test_min_spec_2x_margin(self):
        """Spec met with 2x margin: score should be ~1.0."""
        result = self.rf.compute(results={"gain_db": 120.0})
        score = result.spec_scores[0].score
        assert abs(score - 1.0) < 1e-6, f"Expected ~1.0, got {score}"

    def test_min_spec_missed_2x(self):
        """Spec missed by 2x: score should be ~-1.0."""
        result = self.rf.compute(results={"gain_db": 30.0})
        score = result.spec_scores[0].score
        assert abs(score - (-1.0)) < 1e-6, f"Expected ~-1.0, got {score}"

    def test_min_spec_partial(self):
        """Between: logarithmic interpolation."""
        result = self.rf.compute(results={"gain_db": 50.0})
        score = result.spec_scores[0].score
        assert -1.0 < score < 0.0, f"Expected between -1 and 0, got {score}"

    def test_min_spec_met_flag(self):
        """met flag should be True when spec is satisfied."""
        result = self.rf.compute(results={"gain_db": 65.0})
        assert result.spec_scores[0].met is True

    def test_min_spec_not_met_flag(self):
        result = self.rf.compute(results={"gain_db": 55.0})
        assert result.spec_scores[0].met is False


class TestMaxSpec:
    """Test maximum spec scoring (e.g., current <= 200 µA)."""

    def test_max_spec_met(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="current_a", max_val=200e-6, unit="A")],
        )
        result = rf.compute(results={"current_a": 100e-6})
        assert result.spec_scores[0].met is True
        assert result.spec_scores[0].score > 0.0

    def test_max_spec_exceeded(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="current_a", max_val=200e-6, unit="A")],
        )
        result = rf.compute(results={"current_a": 400e-6})
        assert result.spec_scores[0].met is False
        assert result.spec_scores[0].score < 0.0


class TestTargetSpec:
    """Test target spec scoring (e.g., Vref = 1.2 V)."""

    def test_target_exact(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="vref", target_val=1.2, unit="V")],
        )
        result = rf.compute(results={"vref": 1.2})
        assert result.spec_scores[0].score > 0.9


class TestCornerAwareReward:
    """Test corner-aware reward computation."""

    def test_worst_corner_mode(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
            mode=RewardMode.WORST_CORNER,
        )
        result = rf.compute(
            results={"gain_db": 70.0},
            corner_results={
                "tt": {"gain_db": 70.0},
                "ss": {"gain_db": 55.0},  # fails
                "ff": {"gain_db": 75.0},
            },
        )
        # Worst corner (ss) should dominate
        assert result.all_specs_met is False

    def test_nominal_only_ignores_corners(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
            mode=RewardMode.NOMINAL_ONLY,
        )
        result = rf.compute(
            results={"gain_db": 70.0},
            corner_results={
                "ss": {"gain_db": 55.0},  # fails but ignored
            },
        )
        assert result.spec_scores[0].met is True


class TestFeasibility:
    """Test feasibility constraint enforcement."""

    def test_feasibility_pass(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
            feasibility_constraints=[
                FeasibilityConstraint(
                    name="max_current",
                    parameter="total_current",
                    max_val=5e-3,
                )
            ],
        )
        result = rf.compute(
            results={"gain_db": 70.0},
            design_params={"total_current": 1e-3},
        )
        assert result.feasibility_passed is True

    def test_feasibility_violation_penalizes(self):
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
            feasibility_constraints=[
                FeasibilityConstraint(
                    name="max_current",
                    parameter="total_current",
                    max_val=5e-3,
                )
            ],
        )
        result_ok = rf.compute(
            results={"gain_db": 70.0},
            design_params={"total_current": 1e-3},
        )
        result_bad = rf.compute(
            results={"gain_db": 70.0},
            design_params={"total_current": 50e-3},
        )
        assert result_bad.total_reward < result_ok.total_reward
        assert result_bad.feasibility_passed is False


class TestNonConvergence:
    """Test non-convergence penalty."""

    def test_non_convergence_reward_is_negative(self):
        """Non-convergence should give -0.5, NOT zero."""
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
        )
        result = rf.compute(results={}, convergence_failed=True)
        assert result.total_reward == -0.5
        assert result.convergence_failed is True

    def test_non_convergence_is_not_zero(self):
        """Zero would make model too conservative (avoid risky topologies)."""
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
        )
        result = rf.compute(results={}, convergence_failed=True)
        assert result.total_reward != 0.0


class TestFromEvalTask:
    """Test factory method to create RewardFunction from eval task YAML."""

    def test_from_eval_task(self):
        task_dict = {
            "specs": {
                "dc_gain_db": {"min": 60, "weight": 1.0, "unit": "dB"},
                "current_a": {"max": 200e-6, "weight": 0.5, "unit": "A"},
            },
            "pass_criteria": "all_corners",
        }
        rf = RewardFunction.from_eval_task(task_dict)
        assert "dc_gain_db" in rf.specs
        assert "current_a" in rf.specs
        assert rf.mode == RewardMode.WORST_CORNER


class TestRewardResult:
    """Test RewardResult structure."""

    def test_result_fields(self):
        rf = RewardFunction(
            specs=[
                SpecTarget(name="gain_db", min_val=60.0),
                SpecTarget(name="ugb_hz", min_val=50e6),
            ],
        )
        result = rf.compute(results={"gain_db": 70.0, "ugb_hz": 60e6})
        assert isinstance(result, RewardResult)
        assert isinstance(result.total_reward, float)
        assert len(result.spec_scores) == 2
        assert isinstance(result.all_specs_met, bool)

    def test_missing_spec_result(self):
        """Missing measurement should get worst score."""
        rf = RewardFunction(
            specs=[SpecTarget(name="gain_db", min_val=60.0)],
        )
        result = rf.compute(results={})  # no gain_db measurement
        assert result.spec_scores[0].score == -1.0
        assert result.spec_scores[0].met is False
