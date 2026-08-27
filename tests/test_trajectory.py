"""Tests for the trajectory data format module."""

import json
import pytest
from asic_ai.data.trajectory import (
    ToolCall,
    TrajectoryStep,
    Trajectory,
    TrajectoryDataset,
)


def _step(idx, role, content=None, tool_call=None, tool_result=None):
    """Helper to create a TrajectoryStep."""
    return TrajectoryStep(
        step_index=idx,
        role=role,
        content=content,
        tool_call=tool_call,
        tool_result=tool_result,
    )


def _make_trajectory(tid="traj_001", success=True, score=0.85):
    """Helper to create a valid Trajectory."""
    return Trajectory(
        id=tid,
        task_id="task_001",
        steps=[
            _step(0, "user", content="Design a two-stage OTA"),
            _step(1, "assistant", content="I'll select a two-stage topology.",
                  tool_call=ToolCall(name="sim.ac", arguments={"netlist": "..."}, call_id="c1")),
            _step(2, "tool", tool_result={"gain_db": 55.0},
                  tool_call=ToolCall(name="sim.ac", arguments={}, call_id="c1")),
        ],
        success=success,
        final_score=score,
        metadata={"model": "test"},
        duration_seconds=10.0,
    )


class TestToolCall:
    def test_tool_call_creation(self):
        tc = ToolCall(name="sim.ac", arguments={"netlist": "test.sp"}, call_id="call_001")
        assert tc.name == "sim.ac"
        assert tc.call_id == "call_001"

    def test_tool_call_requires_call_id(self):
        with pytest.raises(Exception):
            ToolCall(name="sim.ac", arguments={})


class TestTrajectoryStep:
    def test_step_creation(self):
        step = _step(0, "user", content="Design an OTA")
        assert step.role == "user"
        assert step.step_index == 0

    def test_step_with_tool_call(self):
        tc = ToolCall(name="sim.ac", arguments={"params": {}}, call_id="c1")
        step = _step(1, "assistant", content="Simulating...", tool_call=tc)
        assert step.tool_call is not None
        assert step.tool_call.name == "sim.ac"

    def test_step_requires_index(self):
        with pytest.raises(Exception):
            TrajectoryStep(role="user", content="hello")


class TestTrajectory:
    def test_creation(self):
        traj = _make_trajectory()
        assert traj.id == "traj_001"
        assert len(traj.steps) == 3
        assert traj.success is True

    def test_to_jsonl_and_back(self):
        traj = _make_trajectory()
        jsonl = traj.to_jsonl()
        restored = Trajectory.from_jsonl(jsonl)
        assert restored.id == traj.id
        assert len(restored.steps) == len(traj.steps)
        assert restored.success == traj.success

    def test_to_chat_format(self):
        traj = _make_trajectory()
        chat = traj.to_chat_format()
        assert isinstance(chat, list)
        assert len(chat) >= 2
        # First message should be user
        assert chat[0]["role"] == "user"

    def test_validate_returns_list(self):
        traj = _make_trajectory()
        errors = traj.validate()
        assert isinstance(errors, list)

    def test_get_tool_calls(self):
        traj = _make_trajectory()
        tool_calls = traj.get_tool_calls()
        assert isinstance(tool_calls, list)


class TestTrajectoryDataset:
    def test_dataset_creation(self):
        dataset = TrajectoryDataset(trajectories=[_make_trajectory()])
        assert len(dataset.trajectories) == 1

    def test_statistics(self):
        dataset = TrajectoryDataset(
            trajectories=[
                _make_trajectory("t1", success=True, score=0.9),
                _make_trajectory("t2", success=False, score=0.3),
            ]
        )
        stats = dataset.statistics()
        assert "count" in stats or "total" in stats or len(stats) > 0

    def test_filter_successful(self):
        t1 = _make_trajectory("t1", success=True, score=0.9)
        t2 = _make_trajectory("t2", success=False, score=0.3)
        dataset = TrajectoryDataset(trajectories=[t1, t2])
        filtered = dataset.filter_successful()
        assert all(t.success for t in filtered.trajectories)

    def test_split(self):
        trajs = [_make_trajectory(f"t{i}") for i in range(10)]
        dataset = TrajectoryDataset(trajectories=trajs)
        train, val = dataset.split(train_ratio=0.8, seed=42)
        assert len(train.trajectories) + len(val.trajectories) == 10
