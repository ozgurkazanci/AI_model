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

def test_full_trajectory_creation(mock_adapter):
    agent_config = AgentConfig(max_steps=2)
    loop = AgentLoop(model=DummyModel(), simulator=mock_adapter, optimizer=DummyOptimizer(), config=agent_config)
    
    task = EvalTask(spec={"gain": {"min": 60}})
    trajectory = asyncio.run(loop.run(task))
    
    # Loop should complete
    assert trajectory.status == "max_steps_reached"

def test_mock_consistency(mock_adapter, sample_netlist):
    params = SimParams(analysis_type="ac")
    res1 = mock_adapter.ac(sample_netlist, params)
    
    # Re-initialize mock_adapter with same seed to ensure exact same sequence of random numbers
    mock_adapter2 = MockSimulatorAdapter(mock_adapter.config)
    res2 = mock_adapter2.ac(sample_netlist, params)
    
    gain1 = res1.signals["v(out)"].y_values[0]
    gain2 = res2.signals["v(out)"].y_values[0]
    
    assert gain1 == gain2
