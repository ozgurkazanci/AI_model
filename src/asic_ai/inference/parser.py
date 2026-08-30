"""Extract tool calls from model output.

The format this must read is fixed by asic_ai.data.format, which is what every
training example uses and what the fine-tuned model emits:

    <tool_call>{"name": "sim.ac", "arguments": {"netlist": "..."}}</tool_call>

This module previously implemented only a `<function=name>...</function>`
pattern. That string occurs ZERO times across data/sft/*.jsonl, which contains
4322 `<tool_call>` blocks, so the parser returned no calls for real model output
and no calls for the project's own training data. The model could emit a
perfect, in-contract tool call and the inference loop would see nothing.

Parsing is deliberately tolerant of what a small model actually produces --
whitespace and newlines inside the tags, single quotes, a trailing comma,
several calls in one turn -- but never tolerant about the RESULT: a call whose
JSON cannot be recovered is reported as a parse failure rather than silently
becoming an empty-argument call, because "sim.ac with no arguments" is a
plausible-looking request that would run and return nonsense.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

__all__ = ["ParsedToolCall", "ToolCallParser"]

# The canonical format. DOTALL so netlists with newlines survive.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Legacy/alternate shapes, kept so a differently-templated base model still works.
_FUNCTION_TAG_RE = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)


class ParsedToolCall(BaseModel):
    """Represents a parsed tool call from model output."""
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    thinking: str = Field(..., description="Model's reasoning before the tool call")
    raw_text: str = Field(..., description="Original model output text")
    parse_method: str = Field(..., description="Which format was detected (chatml, function_call, xml, etc.)")


def _loads_lenient(raw: str) -> Any:
    """Recover JSON a small model got slightly wrong. Raises if it cannot."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Trailing commas before a closing brace/bracket.
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Single-quoted keys/strings, but only when there are no double quotes to
    # damage. Anything more aggressive risks corrupting a netlist payload.
    if '"' not in cleaned and "'" in cleaned:
        try:
            return json.loads(cleaned.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    raise ValueError("could not recover JSON from tool_call body")


class ToolCallParser:
    """Parser to extract structured tool calls from model output text.

    Takes no constructor arguments (several call sites depend on that).
    """

    def parse(self, model_output: str) -> List[ParsedToolCall]:
        """Parse tool calls from model output text.

        Recognised, in order:
          1. <tool_call>{"name": ..., "arguments": {...}}</tool_call>  -- canonical
          2. <function=name>{...}</function>                           -- legacy

        Malformed calls are skipped rather than returned with empty arguments.
        Use `parse_errors` alongside this when the caller needs to tell "the
        model wrote nothing" apart from "the model wrote something broken".
        """
        calls, _ = self._parse_with_errors(model_output or "")
        return calls

    def parse_errors(self, model_output: str) -> List[str]:
        """Human-readable reasons any tool_call block failed to parse."""
        _, errors = self._parse_with_errors(model_output or "")
        return errors

    def _parse_with_errors(self, text: str) -> Tuple[List[ParsedToolCall], List[str]]:
        calls: List[ParsedToolCall] = []
        errors: List[str] = []

        for match in _TOOL_CALL_RE.finditer(text):
            body = match.group(1)
            try:
                data = _loads_lenient(body)
            except ValueError as exc:
                errors.append(f"{exc}: {body[:80]!r}")
                continue
            if not isinstance(data, dict):
                errors.append(f"tool_call body is {type(data).__name__}, not an object")
                continue

            name = data.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"tool_call missing a 'name': {body[:80]!r}")
                continue

            args = data.get("arguments", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                errors.append(f"{name}: 'arguments' is {type(args).__name__}, not an object")
                continue

            calls.append(ParsedToolCall(
                name=name,
                arguments=args,
                thinking=text[:match.start()].strip(),
                raw_text=text,
                parse_method="chatml",
            ))

        if calls or errors:
            return calls, errors

        for match in _FUNCTION_TAG_RE.finditer(text):
            name, args_str = match.group(1), match.group(2)
            try:
                args = _loads_lenient(args_str)
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ParsedToolCall(
                name=name.strip(),
                arguments=args,
                thinking=text[:match.start()].strip(),
                raw_text=text,
                parse_method="xml",
            ))

        return calls, errors

    def validate_tool_call(self, call: ParsedToolCall) -> Tuple[bool, str]:
        """Check a parsed call against the frozen tool contract.

        Verifies the tool exists and that every required parameter is present.
        A hallucinated tool or a missing required argument is caught here rather
        than at the adapter, where it would surface as an opaque failure.
        """
        if not call.name:
            return False, "Tool name is missing."

        from asic_ai.data.format import TOOL_DEFINITIONS

        spec = None
        for tool in TOOL_DEFINITIONS:
            if tool["function"]["name"] == call.name:
                spec = tool["function"]
                break
        if spec is None:
            known = sorted(t["function"]["name"] for t in TOOL_DEFINITIONS)
            return False, f"Unknown tool {call.name!r}. Available: {known}"

        params = spec.get("parameters", {}) or {}
        required = params.get("required", []) or []
        missing = [p for p in required if p not in call.arguments]
        if missing:
            return False, f"{call.name} is missing required argument(s): {missing}"

        return True, "Valid"
