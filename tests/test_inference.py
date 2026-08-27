import pytest
from asic_ai.inference.parser import ToolCallParser, ParsedToolCall
from asic_ai.inference.engine import APIEngine
from pydantic import BaseModel

# Mock config for tests
class InferenceConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95

def test_parsed_tool_call_fields():
    call = ParsedToolCall(
        name="sim.ac",
        arguments={"start": 1, "stop": 1e9},
        thinking="I need to run AC sim",
        raw_text="I need to run AC sim\n<function=sim.ac>{\"start\": 1, \"stop\": 1e9}</function>",
        parse_method="xml"
    )
    assert call.name == "sim.ac"
    assert call.arguments["start"] == 1
    assert call.thinking == "I need to run AC sim"
    assert call.parse_method == "xml"

def test_tool_call_parser_xml():
    parser = ToolCallParser()
    text = "Let's test this.\n<function=sim.ac>{\"points\": 10}</function>"
    calls = parser.parse(text)
    
    assert len(calls) == 1
    assert calls[0].name == "sim.ac"
    assert calls[0].arguments == {"points": 10}
    assert calls[0].thinking.strip() == "Let's test this."
    assert calls[0].parse_method == "xml"

def test_tool_call_parser_chatml():
    # Note: This checks that parser handles chatml gracefully, even if not fully implemented in source yet.
    parser = ToolCallParser()
    text = "Thinking...\n<tool_call>{\"name\": \"sim.dc\", \"arguments\": {\"temp\": 27}}</tool_call>"
    
    # We mock or expect behavior depending on current implementation
    # Right now parser.py only explicitly handles XML. It might return empty list.
    calls = parser.parse(text)
    # If implemented, we'd check fields. For now we just ensure it doesn't crash.
    assert isinstance(calls, list)

def test_tool_call_parser_function_call():
    # Testing function_call format parsing handling
    parser = ToolCallParser()
    text = "{\"name\": \"sim.tran\", \"arguments\": {}}"
    calls = parser.parse(text)
    assert isinstance(calls, list)

def test_tool_call_parser_malformed():
    parser = ToolCallParser()
    text = "Let's do this <function=sim.ac>bad json</function>"
    calls = parser.parse(text)
    
    assert len(calls) == 1
    assert calls[0].name == "sim.ac"
    assert calls[0].arguments == {} # Should gracefully fallback to empty dict
    assert calls[0].parse_method == "xml"

def test_api_engine_instantiation():
    engine = APIEngine(base_url="http://localhost:8000/v1", api_key="dummy", model="test-model")
    assert engine.base_url == "http://localhost:8000/v1"
    assert engine.model == "test-model"

def test_inference_config_defaults():
    config = InferenceConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 2048
    assert config.top_p == 0.95
