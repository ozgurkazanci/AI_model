"""GBNF grammar that makes illegal tool calls unsamplable.

The 945ex eval found two failure classes that no amount of prompt or feedback
fixed: 30 calls to tools that do not exist (int.patch, sim.dc_ac,
simulate_simulate, ...) and 20 completed-but-malformed JSON bodies (unquoted
SPICE units, `new FloatValue(1.2)`, brace miscounts). Both are DECODING
failures: the model composes them token by token, and the error observation
fed back afterwards does not stop it emitting the byte-identical call again --
fifo_sync_001 repeated `int.patch` seven times against seven "Unknown tool"
replies.

llama.cpp's grammar-based sampling removes the failure at the only place the
0.5B model cannot ignore it: a token that would leave the grammar is masked
before sampling, so a tool name outside TOOL_DEFINITIONS or a bare `10u` as a
JSON value cannot be generated at all. This constrains SYNTAX only -- names
and JSON shape. Whether the arguments are semantically right (a real netlist,
required keys present) stays with the parser and the simulator, which give
honest feedback either way.

The grammar is derived from TOOL_DEFINITIONS at call time, never written by
hand, for the same reason build_system_message() exists: two sources of truth
for the contract is how the last one silently rotted (a parser that matched a
format occurring zero times in the corpus).

Shape allowed:  prose  (<tool_call>{constrained JSON}</tool_call>  prose)*
The one deliberate distortion: prose cannot contain '<', so the only way to
open an angle bracket is a well-formed tool call. Training prose does not use
'<' outside the tags; the placeholder netlist string "<netlist>" that 258 sim
calls passed becomes unsamplable in prose too, which is a feature.
"""
from __future__ import annotations

from typing import Iterable

from asic_ai.data.format import TOOL_DEFINITIONS


def contract_tool_names() -> list[str]:
    """The only names a generated call may use, from the frozen contract."""
    return sorted(t["function"]["name"] for t in TOOL_DEFINITIONS)


def _grammar_for(names: Iterable[str]) -> str:
    """GBNF for prose interleaved with contract-only tool calls.

    Split out from tool_call_grammar() so tests can prove the derivation:
    a name absent from `names` must be absent from the grammar.
    """
    names = list(names)
    if not names:
        raise ValueError("refusing to build a grammar with zero tool names; "
                         "it would forbid every tool call")
    name_alt = " | ".join(f'"\\"{n}\\""' for n in names)
    # JSON value rules follow llama.cpp's own json.gbnf: strings exclude raw
    # control characters (a raw newline inside a string is a syntax error in
    # JSON, and allowing it re-admits half the malformed bodies the eval saw).
    return f"""# Auto-generated from TOOL_DEFINITIONS -- do not edit by hand.
root ::= prose (toolcall prose)*
prose ::= [^<]*
toolcall ::= "<tool_call>" sp callobj sp "</tool_call>"
callobj ::= "{{" sp "\\"name\\"" sp ":" sp toolname sp "," sp "\\"arguments\\"" sp ":" sp jobject sp "}}"
toolname ::= {name_alt}
jobject ::= "{{" sp ( jmember ( sp "," sp jmember )* )? sp "}}"
jmember ::= jstring sp ":" sp jvalue
jvalue ::= jstring | jnumber | jobject | jarray | "true" | "false" | "null"
jarray ::= "[" sp ( jvalue ( sp "," sp jvalue )* )? sp "]"
jstring ::= "\\"" jchar* "\\""
jchar ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
jnumber ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
sp ::= [ \\t\\n\\r]*
"""


def tool_call_grammar() -> str:
    """The GBNF to send with every constrained generation request."""
    return _grammar_for(contract_tool_names())
