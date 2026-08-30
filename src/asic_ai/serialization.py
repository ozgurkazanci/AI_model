"""The one place a simulation result becomes text.

WHY THIS EXISTS
---------------
The measurement layer is deliberately allowed to carry non-finite floats:
measure.db20() returns -inf for a magnitude that is genuinely zero, and
measure.transfer_function() returns NaN for a phase that genuinely does not
exist. Those are ordered, arithmetic-safe values, every scan in measure.py
skips them, and clamping them to a finite floor (-6000 dB) is worse -- it
survives a mean(), it survives a reward, and it reads as "very small gain"
instead of "no signal at all".

What they are NOT is JSON. `json.dumps(float("-inf"))` writes `-Infinity`, and
`json.dumps(float("nan"))` writes `NaN`. Neither is in the JSON grammar:

  - json.loads accepts them only because CPython opts in by default; the same
    document fails the moment a caller passes parse_constant to reject them,
  - jq, JavaScript's JSON.parse, Rust's serde_json and Go's encoding/json all
    reject the document outright,
  - the HuggingFace datasets loader rejects the line, which means an SFT file
    with one such record fails to load AT TRAINING TIME, not here.

training/rl_env.py hands the serialised tool result straight to the model, and
the same payload flows on into data/trajectory.py and data/sft_generator.py and
from there into a .jsonl. One ordinary AC run -- an AC 1 input and a DC-only
supply rail -- put 183 `-Infinity` and 183 `NaN` tokens into that text.

WHAT THIS DOES
--------------
`json_safe()` replaces every non-finite float with None (JSON `null`) and
records where it did so, and `dumps()` is json.dumps with allow_nan=False so
that anything that slips past raises here, loudly, instead of being written.
None is the right replacement: it is exactly what the rest of this codebase
uses for "the data does not define this", and spec_extract already drops
non-finite values for the same reason.

This is a SERIALIZATION guard. It does not, and must not, change what the
metric layer computes.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["json_safe", "dumps", "count_non_finite"]


def _is_bad_float(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def json_safe(obj: Any, _path: str = "", _found: list[str] | None = None) -> Any:
    """A copy of `obj` in which no float is inf, -inf or NaN.

    Mappings, lists, tuples and sets are walked. A non-finite float becomes
    None. Everything else is returned unchanged; objects json.dumps cannot
    handle are left for its own `default=` to deal with.
    """
    found = [] if _found is None else _found
    out = _walk(obj, _path, found)
    if found and _found is None:
        log.warning(
            "json_safe: %d non-finite float(s) replaced with null before "
            "serialisation (%s%s). A non-finite float is not valid JSON and "
            "would reach the model, a trajectory and an SFT file as -Infinity "
            "or NaN.",
            len(found), ", ".join(found[:5]),
            ", ..." if len(found) > 5 else "",
        )
    return out


def _walk(obj: Any, path: str, found: list[str]) -> Any:
    if _is_bad_float(obj):
        found.append(path or "<root>")
        return None
    if isinstance(obj, dict):
        return {k: _walk(v, f"{path}.{k}" if path else str(k), found)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, f"{path}[{i}]", found) for i, v in enumerate(obj)]
    if isinstance(obj, set):
        return [_walk(v, f"{path}{{}}", found) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return _walk(dump(), path, found)
    return obj


def count_non_finite(obj: Any) -> int:
    """How many non-finite floats `obj` contains. For tests and diagnostics."""
    found: list[str] = []
    _walk(obj, "", found)
    return len(found)


def dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps that CANNOT emit -Infinity, Infinity or NaN.

    The object is sanitised first, then serialised with allow_nan=False so a
    non-finite value that arrives some other way raises ValueError here rather
    than being written into a file that will not load.
    """
    kwargs.setdefault("default", str)
    kwargs["allow_nan"] = False
    return json.dumps(json_safe(obj), **kwargs)
