"""Guard the single most fragile invariant in the project.

Every SFT training example and every inference call must carry the byte-identical
system message returned by build_system_message(). The corpus once held two
variants (657 examples without the tool list, 393 with it) and the scripts held
five different hand-rolled ways of assembling one. A model trained on one system
prompt and served with another silently stops emitting tool calls.

These tests fail loudly if that drift comes back.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from asic_ai.data.format import (
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    build_system_message,
    format_trajectory_for_sft,
)

REPO_ROOT = Path(__file__).parent.parent
SFT_DIR = REPO_ROOT / "data" / "sft"

KNOWN_TOOLS = {t["function"]["name"] for t in TOOL_DEFINITIONS}
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _sft_files() -> list[Path]:
    return sorted(SFT_DIR.glob("*.jsonl"))


def _examples(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def _messages(example: dict) -> list[dict]:
    return example.get("messages") or example.get("conversations") or []


# --------------------------------------------------------------- builder ---

def test_builder_matches_format_trajectory_output():
    """The builder and the trajectory formatter must not diverge."""
    class _Empty:
        steps: list = []

    from_formatter = format_trajectory_for_sft(_Empty())[0]["content"]
    assert from_formatter == build_system_message()


def test_builder_includes_prompt_and_every_tool():
    system = build_system_message()
    assert SYSTEM_PROMPT.strip() in system
    assert "## Available Tools" in system
    for name in KNOWN_TOOLS:
        assert f"### {name}" in system, f"{name} missing from system message"


def test_builder_is_deterministic():
    assert build_system_message() == build_system_message()


# ------------------------------------------------------------------ data ---

@pytest.mark.parametrize("path", _sft_files(), ids=lambda p: p.name)
def test_sft_file_uses_canonical_system_prompt(path: Path):
    """Exactly one system message variant across the whole corpus."""
    canonical = build_system_message()
    offenders = []

    for line_no, example in _examples(path):
        messages = _messages(example)
        if not messages:
            continue
        first = messages[0]
        if first.get("role") != "system":
            offenders.append(f"line {line_no}: first message is {first.get('role')!r}, not 'system'")
        elif first.get("content") != canonical:
            offenders.append(
                f"line {line_no}: system message is {len(first.get('content') or '')} chars, "
                f"expected {len(canonical)}"
            )

    assert not offenders, (
        f"{path.name} deviates from the canonical system prompt "
        f"({len(offenders)} example(s)). First few: {offenders[:3]}. "
        "Fix with: PYTHONPATH=src python scripts/normalize_sft_system_prompt.py --write"
    )


@pytest.mark.parametrize("path", _sft_files(), ids=lambda p: p.name)
def test_sft_file_only_calls_contract_tools(path: Path):
    """No training example may teach a tool that does not exist."""
    offenders = []

    for line_no, example in _examples(path):
        for message in _messages(example):
            for raw in TOOL_CALL_RE.findall(message.get("content") or ""):
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError as exc:
                    offenders.append(f"line {line_no}: unparseable tool_call ({exc})")
                    continue
                name = call.get("name")
                if name not in KNOWN_TOOLS:
                    offenders.append(f"line {line_no}: unknown tool {name!r}")

    assert not offenders, f"{path.name}: {offenders[:5]}"


def test_corpus_has_exactly_one_system_prompt_variant():
    """Belt-and-braces check across every file at once."""
    variants: set[str] = set()
    total = 0
    for path in _sft_files():
        for _, example in _examples(path):
            messages = _messages(example)
            if messages and messages[0].get("role") == "system":
                variants.add(messages[0]["content"])
                total += 1

    assert total > 0, "no SFT examples found"
    assert len(variants) == 1, f"{len(variants)} system prompt variants across {total} examples"
    assert variants.pop() == build_system_message()


# ------------------------------------------------------------------ code ---

def test_no_module_builds_a_system_message_by_hand():
    """Only format.py may reference SYSTEM_PROMPT; everyone else calls the builder."""
    allowed = {
        REPO_ROOT / "src" / "asic_ai" / "data" / "format.py",
        REPO_ROOT / "scripts" / "normalize_sft_system_prompt.py",  # mentions it in its docstring
        Path(__file__),
    }

    offenders = []
    for path in list((REPO_ROOT / "scripts").rglob("*.py")) + list((REPO_ROOT / "src").rglob("*.py")):
        if path in allowed:
            continue
        if "SYSTEM_PROMPT" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "These modules reference SYSTEM_PROMPT directly instead of calling "
        f"build_system_message(): {offenders}"
    )
