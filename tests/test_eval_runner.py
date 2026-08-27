"""Tests for the eval runner and metrics modules."""

import json
import pytest
import yaml
from pathlib import Path
from eval.runner import EvalResult, load_task, run_task
from eval.metrics import compute_metrics


class TestTaskLoading:
    """Test YAML task file loading."""

    def test_load_yaml_task(self, tmp_path):
        task_data = {
            "id": "test_ota_001",
            "category": "analog",
            "pdk": "sky130",
            "supply": 1.8,
            "specs": {
                "dc_gain_db": {"min": 60},
                "current_a": {"max": 200e-6},
            },
            "corners": ["tt", "ss", "ff"],
            "pass_criteria": "all_corners",
        }
        task_file = tmp_path / "test_task.yaml"
        with open(task_file, "w") as f:
            yaml.dump(task_data, f)

        loaded = load_task(task_file)
        assert loaded["id"] == "test_ota_001"
        assert loaded["pdk"] == "sky130"
        assert "dc_gain_db" in loaded["specs"]

    def test_all_analog_tasks_load(self):
        """Verify all 6 analog eval tasks load without errors."""
        task_dir = Path("eval/tasks/analog")
        if not task_dir.exists():
            pytest.skip("eval tasks not found")
        for task_file in task_dir.glob("*.yaml"):
            task = load_task(task_file)
            assert "id" in task, f"{task_file.name}: missing 'id'"
            assert "specs" in task, f"{task_file.name}: missing 'specs'"
            assert "pass_criteria" in task, f"{task_file.name}: missing 'pass_criteria'"

    def test_all_digital_tasks_load(self):
        """Verify all 3 digital eval tasks load without errors."""
        task_dir = Path("eval/tasks/digital")
        if not task_dir.exists():
            pytest.skip("eval tasks not found")
        for task_file in task_dir.glob("*.yaml"):
            task = load_task(task_file)
            assert "id" in task
            assert "specs" in task

    def test_task_count(self):
        """Should have at least 9 eval tasks."""
        analog = list(Path("eval/tasks/analog").glob("*.yaml")) if Path("eval/tasks/analog").exists() else []
        digital = list(Path("eval/tasks/digital").glob("*.yaml")) if Path("eval/tasks/digital").exists() else []
        assert len(analog) + len(digital) >= 9, f"Expected >=9, got {len(analog) + len(digital)}"


class TestEvalResult:
    """Test EvalResult model."""

    def test_eval_result_creation(self):
        result = EvalResult(
            task_id="test_001",
            passed=True,
            final_score=0.85,
            steps=15,
            wall_time_sec=10.5,
            trajectory=[{"step": 1, "action": "simulate"}],
        )
        assert result.task_id == "test_001"
        assert result.passed is True
        assert result.final_score == 0.85

    def test_eval_result_with_error(self):
        result = EvalResult(
            task_id="test_002",
            passed=False,
            final_score=0.0,
            steps=0,
            wall_time_sec=0.1,
            trajectory=[],
            error="Convergence failure",
        )
        assert result.passed is False
        assert result.error is not None


class TestMetrics:
    """Test metrics computation."""

    def test_compute_metrics(self, tmp_path):
        results_data = {
            "model": "test-model",
            "results": [
                {"task_id": "t1", "passed": True, "final_score": 0.9, "steps": 10, "wall_time_sec": 5.0},
                {"task_id": "t2", "passed": True, "final_score": 0.8, "steps": 15, "wall_time_sec": 8.0},
                {"task_id": "t3", "passed": False, "final_score": 0.3, "steps": 20, "wall_time_sec": 12.0},
            ],
        }
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f)

        metrics = compute_metrics(str(results_file))
        assert "pass_rate" in metrics
        assert metrics["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)
        assert "avg_score" in metrics
        assert "stats" in metrics

    def test_empty_results(self, tmp_path):
        results_data = {"model": "test", "results": []}
        results_file = tmp_path / "empty.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f)

        metrics = compute_metrics(str(results_file))
        assert metrics == {}


class TestTaskSchema:
    """Test that eval task YAML files have correct schema."""

    REQUIRED_FIELDS = ["id", "category", "pdk", "specs", "pass_criteria"]

    def test_analog_tasks_have_required_fields(self):
        task_dir = Path("eval/tasks/analog")
        if not task_dir.exists():
            pytest.skip("eval tasks not found")
        for task_file in task_dir.glob("*.yaml"):
            task = load_task(task_file)
            for field in self.REQUIRED_FIELDS:
                assert field in task, f"{task_file.name}: missing required field '{field}'"

    def test_specs_have_minmax_or_target(self):
        """Every spec should have at least min, max, or target."""
        task_dir = Path("eval/tasks/analog")
        if not task_dir.exists():
            pytest.skip("eval tasks not found")
        for task_file in task_dir.glob("*.yaml"):
            task = load_task(task_file)
            for spec_name, spec_def in task.get("specs", {}).items():
                has_bound = "min" in spec_def or "max" in spec_def or "target" in spec_def
                assert has_bound, f"{task_file.name}/{spec_name}: missing min/max/target"
