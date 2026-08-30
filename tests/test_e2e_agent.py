import pytest
import os
import tempfile
from pathlib import Path
from asic_ai.adapters.mock import MockSimulatorAdapter
from asic_ai.adapters.base import AdapterConfig
from asic_ai.tool_interface.schema import SimParams, PVTCorner
from asic_ai.reward.reward import RewardFunction, SpecTarget, RewardMode
from asic_ai.agent.loop import AgentLoop, AgentConfig, EvalTask, Trajectory

@pytest.fixture
def mock_adapter(tmp_path):
    config = AdapterConfig(
        binary_path="dummy",
        work_dir=str(tmp_path),
        seed=42
    )
    return MockSimulatorAdapter(config)

@pytest.fixture
def sample_netlist(tmp_path):
    netlist = tmp_path / "circuit.sp"
    netlist.write_text("m1 d g s b nmos w=10u l=1u\nm2 d g s b pmos w=20u l=1u\nr1 d g 1k")
    return str(netlist)

def test_mock_adapter_dc(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="dc")
    result = mock_adapter.dc(sample_netlist, params)
    assert result.op_points is not None
    assert "v(out)" in result.op_points

def test_mock_adapter_ac(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="ac")
    result = mock_adapter.ac(sample_netlist, params)
    assert result.frequencies
    assert "v(out)" in result.signals
    
    # Gain should be deterministic due to seed 42
    gain = result.signals["v(out)"].y_values[0]
    assert gain > 0

def test_mock_adapter_corners(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="ac")
    corners = [
        PVTCorner(process="tt", voltage=1.2, temperature=25),
        PVTCorner(process="ss", voltage=1.08, temperature=125)
    ]
    results = mock_adapter.corners(sample_netlist, corners, params)
    assert len(results) == 2
    
    tt_gain = results[0].ac.signals["v(out)"].y_values[0]
    ss_gain = results[1].ac.signals["v(out)"].y_values[0]
    
    # SS should have lower gain than TT due to 0.8 derating
    assert ss_gain < tt_gain

def test_mock_adapter_netlist_sensitivity(mock_adapter, tmp_path):
    params = SimParams(analysis_type="ac")
    
    # Small device
    small_netlist = tmp_path / "small.sp"
    small_netlist.write_text("m1 d g s b nmos w=1u l=1u")
    res_small = mock_adapter.ac(str(small_netlist), params)
    gain_small = res_small.signals["v(out)"].y_values[0]
    
    # Large device
    large_netlist = tmp_path / "large.sp"
    large_netlist.write_text("m1 d g s b nmos w=100u l=1u")
    res_large = mock_adapter.ac(str(large_netlist), params)
    gain_large = res_large.signals["v(out)"].y_values[0]
    
    assert gain_large > gain_small

def test_reward_with_mock_results(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="ac")
    result = mock_adapter.ac(sample_netlist, params)
    
    gain = result.signals["v(out)"].y_values[0]
    
    specs = [
        SpecTarget(name="gain", min_val=40.0, weight=1.0)
    ]
    reward_fn = RewardFunction(specs=specs, mode=RewardMode.NOMINAL_ONLY)
    
    # Pass mock result as dictionary to reward function
    mock_metrics = {"gain": gain}
    reward_result = reward_fn.compute(results=mock_metrics)
    
    assert reward_result.total_reward is not None

class DummyModel:
    pass

class DummyOptimizer:
    pass

import asyncio

def test_a_model_that_cannot_generate_is_an_error_not_a_completed_episode(mock_adapter):
    """DummyModel has no generate(). The old loop ran zero steps and reported
    status="max_steps_reached", i.e. it claimed to have exhausted a 2-step
    budget without ever calling anything."""
    loop = AgentLoop(model=DummyModel(), simulator=mock_adapter,
                     optimizer=DummyOptimizer(), config=AgentConfig(max_steps=2))
    trajectory = asyncio.run(loop.run(EvalTask(spec={"gain": {"min": 60}})))

    assert trajectory.status == "error"
    assert "generate" in trajectory.final_result["error"]
    assert trajectory.steps == []


def test_a_real_engine_produces_a_real_trajectory(mock_adapter):
    """A scripted engine drives the shared agent loop end to end."""
    from asic_ai.inference.engine import GenerationResult

    class ScriptedEngine:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, messages, **kwargs):
            text = self.replies.pop(0) if self.replies else "Done."
            return GenerationResult(text=text, prompt_tokens=1, completion_tokens=1)

    engine = ScriptedEngine([
        '<tool_call>{"name": "pdk.list_devices", "arguments": {}}</tool_call>',
        "That is enough for now.",
    ])
    loop = AgentLoop(model=engine, simulator=mock_adapter,
                     optimizer=DummyOptimizer(), config=AgentConfig(max_steps=4))
    trajectory = asyncio.run(loop.run(EvalTask(spec={"gain": {"min": 60}})))

    assert trajectory.status in ("stopped", "success", "max_steps_reached")
    assert trajectory.steps, "a real engine must produce a real trajectory"
    assert trajectory.steps[0]["tool_calls"] == ["pdk.list_devices"]


def test_stuck_detection_needs_repetition_not_just_length(mock_adapter):
    """The old check fired on trajectory LENGTH, so a productive episode was
    declared stuck."""
    loop = AgentLoop(model=DummyModel(), simulator=mock_adapter,
                     optimizer=DummyOptimizer(),
                     config=AgentConfig(max_retries_same_error=3))
    varied = Trajectory(steps=[{"tool_calls": [n]} for n in
                               ("sim.dc", "sim.ac", "spec.check", "sim.tran")])
    repeated = Trajectory(steps=[{"tool_calls": ["sim.ac"]} for _ in range(4)])
    assert loop._check_stuck(varied) is False
    assert loop._check_stuck(repeated) is True

def test_mock_consistency(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="ac")
    res1 = mock_adapter.ac(sample_netlist, params)
    
    # Re-initialize mock_adapter with same seed to ensure exact same sequence of random numbers
    mock_adapter2 = MockSimulatorAdapter(mock_adapter.config)
    res2 = mock_adapter2.ac(sample_netlist, params)
    
    gain1 = res1.signals["v(out)"].y_values[0]
    gain2 = res2.signals["v(out)"].y_values[0]
    
    assert gain1 == gain2
