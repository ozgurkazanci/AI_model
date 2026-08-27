"""Tests for the data validator module."""

import pytest
from asic_ai.data.validator import (
    validate_trajectory,
    validate_dataset,
    validate_tool_call_format,
    ValidationResult,
    DatasetValidationReport,
)
from asic_ai.data.trajectory import Trajectory, TrajectoryStep, ToolCall, TrajectoryDataset


def _make_trajectory(steps=None, success=True, task_id="test_001"):
    """Helper to build a valid trajectory."""
    if steps is None:
        steps = [
            TrajectoryStep(step_index=0, role="user", content="Design an OTA with 60dB gain"),
            TrajectoryStep(
                step_index=1,
                role="assistant",
                content="I'll start with a two-stage OTA topology.",
                tool_call=ToolCall(
                    name="sim.ac",
                    arguments={"netlist": ".subckt test...", "params": {}},
                    call_id="call_001",
                ),
            ),
            TrajectoryStep(
                step_index=2,
                role="tool",
                content='{"gain_db": 55.0, "ugb_hz": 40000000.0}',
                tool_result={"gain_db": 55.0, "ugb_hz": 40e6},
            ),
        ]
    return Trajectory(
        id="traj_001",
        task_id=task_id,
        steps=steps,
        success=success,
        final_score=0.85,
        metadata={"model": "test"},
        duration_seconds=10.0,
    )


class TestValidateToolCallFormat:
    """Test individual tool call format validation."""

    def test_valid_tool_call(self):
        tc = {"name": "sim.ac", "arguments": {"netlist": "...", "params": {}}}
        assert validate_tool_call_format(tc) is True

    def test_missing_name(self):
        tc = {"arguments": {"netlist": "..."}}
        assert validate_tool_call_format(tc) is False

    def test_missing_arguments(self):
        tc = {"name": "sim.ac"}
        assert validate_tool_call_format(tc) is False

    def test_wrong_type(self):
        assert validate_tool_call_format("not a dict") is False
        assert validate_tool_call_format(None) is False

    def test_name_not_string(self):
        tc = {"name": 123, "arguments": {}}
        assert validate_tool_call_format(tc) is False

    def test_arguments_not_dict(self):
        tc = {"name": "sim.ac", "arguments": "not a dict"}
        assert validate_tool_call_format(tc) is False


class TestValidateTrajectory:
    """Test trajectory-level validation."""

    def test_valid_trajectory_passes(self):
        traj = _make_trajectory()
        result = validate_trajectory(traj)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_trajectory_with_success(self):
        traj = _make_trajectory(success=True)
        result = validate_trajectory(traj)
        assert result.is_valid is True


class TestValidateDataset:
    """Test dataset-level validation."""

    def test_valid_dataset(self):
        dataset = TrajectoryDataset(trajectories=[_make_trajectory()])
        report = validate_dataset(dataset)
        assert isinstance(report, DatasetValidationReport)
        assert report.total_trajectories == 1

    def test_empty_dataset(self):
        dataset = TrajectoryDataset(trajectories=[])
        report = validate_dataset(dataset)
        assert report.total_trajectories == 0
