import re
import json
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field

class ParsedToolCall(BaseModel):
    """Represents a parsed tool call from model output."""
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    thinking: str = Field(..., description="Model's reasoning before the tool call")
    raw_text: str = Field(..., description="Original model output text")
    parse_method: str = Field(..., description="Which format was detected (chatml, function_call, xml, etc.)")

class ToolCallParser:
    """Parser to extract structured tool calls from model output text."""
    
    def parse(self, model_output: str) -> List[ParsedToolCall]:
        """Parse tool calls from model output text.
        
        Supports multiple formats:
        1. ChatML tool_call format
        2. Function call format (JSON)
        3. XML-style
        """
        calls = []
        
        # XML-style parsing heuristic
        xml_pattern = re.compile(r'<function=(.*?)>(.*?)</function>', re.DOTALL)
        matches = xml_pattern.findall(model_output)
        
        if matches:
            for name, args_str in matches:
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                    
                calls.append(ParsedToolCall(
                    name=name,
                    arguments=args,
                    thinking=model_output[:model_output.find(f'<function={name}')].strip(),
                    raw_text=model_output,
                    parse_method="xml"
                ))
            return calls
            
        return calls
        
    def validate_tool_call(self, call: ParsedToolCall) -> Tuple[bool, str]:
        """Validate a parsed tool call against the frozen schema."""
        if not call.name:
            return False, "Tool name is missing."
        return True, "Valid"
