import pytest
from asic_ai.data.format import SYSTEM_PROMPT, TOOL_DEFINITIONS, format_trajectory_for_sft, validate_sft_format

def test_system_prompt_content():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert 'circuit' in SYSTEM_PROMPT
    assert 'ASIC' in SYSTEM_PROMPT
    assert 'tool' in SYSTEM_PROMPT.lower()
    assert 'never memorize PDK parameters' in SYSTEM_PROMPT.lower() or 'never memorize' in SYSTEM_PROMPT.lower()
    assert 'step by step' in SYSTEM_PROMPT.lower()

def test_tool_definitions_length():
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) == 15

def test_tool_definitions_fields():
    for tool in TOOL_DEFINITIONS:
        assert tool.get('type') == 'function'
        func = tool.get('function', {})
        assert 'name' in func
        assert 'description' in func
        assert 'parameters' in func

def test_tool_names_match_contract():
    expected_names = {
        'sim.dc', 'sim.ac', 'sim.tran', 'sim.noise', 'sim.stb', 
        'sim.corners', 'sim.mc', 'meas.eval', 'spec.check', 
        'pdk.device_query', 'pdk.list_devices', 'pdk.get_corners', 
        'netlist.patch', 'lint.check', 'opt.suggest'
    }
    actual_names = {tool['function']['name'] for tool in TOOL_DEFINITIONS}
    assert actual_names == expected_names

class MockTrajectory:
    def __init__(self, messages):
        self.messages = messages

def test_format_trajectory_for_sft():
    mock_traj = MockTrajectory(messages=[
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": "Design LDO"},
        {"role": "assistant", "content": "I will design it."}
    ])
    formatted = format_trajectory_for_sft(mock_traj)
    
    assert len(formatted) == 3
    assert formatted[0]['role'] == 'system'
    assert 'Available Tools' in formatted[0]['content']
    assert formatted[1]['role'] == 'user'
    assert formatted[2]['role'] == 'assistant'

def test_validate_sft_format_valid():
    valid_data = [
        {"role": "system", "content": "You are an expert circuit designer."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    is_valid, errors = validate_sft_format(valid_data)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_sft_format_invalid():
    invalid_data_empty = []
    is_valid, errors = validate_sft_format(invalid_data_empty)
    assert is_valid is False
    assert "empty" in errors[0].lower()

    invalid_data_role = [
        {"role": "user", "content": "No system prompt first"}
    ]
    is_valid, errors = validate_sft_format(invalid_data_role)
    assert is_valid is False
    assert "system" in errors[0].lower()

    invalid_data_bad_role = [
        {"role": "system", "content": "Sys prompt"},
        {"role": "invalid_role", "content": "bad"}
    ]
    is_valid, errors = validate_sft_format(invalid_data_bad_role)
    assert is_valid is False
    assert any("invalid role" in err.lower() for err in errors)
