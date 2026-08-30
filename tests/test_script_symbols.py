"""Guard against scripts calling an asic_ai symbol they never imported.

This is the test that should have existed before Phase 68. That commit
mechanically rewrote 54 call sites across 24 scripts to call
build_system_message(), but the helper that was supposed to add the import
guarded with

    if "build_system_message" in src: return src

and ran AFTER the call sites were rewritten, so the string was always already
present and the import was never added. A later pass then removed the
now-unused SYSTEM_PROMPT imports, leaving 22 scripts calling an unbound name.

Nothing caught it:
  - py_compile only checks SYNTAX; an unbound name is a runtime NameError.
  - the test suite never imports anything under scripts/.
So 242 tests stayed green while 22 CLI entry points were broken.

Scope note: this deliberately checks only names that asic_ai actually exports,
resolved against real module-level and function-level bindings. A full
undefined-name analysis (a mini-pyflakes) would catch more but would also
produce false positives on conditional imports, star imports and late globals,
and a flaky guard is one that gets deleted. This narrow check has no false
positives and covers the failure mode that actually occurred: a refactor that
removes or forgets an import while leaving the call behind.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src" / "asic_ai"
SCRIPTS = REPO_ROOT / "scripts"

BUILTINS = set(dir(builtins))


def _asic_ai_exports() -> set[str]:
    """Public top-level names defined anywhere under src/asic_ai."""
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:  # pragma: no cover - would fail elsewhere first
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                    names.add(node.target.id)
    return names


ASIC_AI_EXPORTS = _asic_ai_exports() - BUILTINS


def _bound_at_module_level(tree: ast.Module) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        # Walk everything: `try: import x except ImportError:` and
        # `if TYPE_CHECKING:` blocks bind at module level too.
        if isinstance(node, ast.ImportFrom):
            bound |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _bound_in_scope(fn: ast.AST) -> set[str]:
    """Names bound inside one function: params, assignments, local imports."""
    bound: set[str] = set()
    args = getattr(fn, "args", None)
    if args is not None:
        for a in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            bound.add(a.arg)
        if args.vararg:
            bound.add(args.vararg.arg)
        if args.kwarg:
            bound.add(args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            bound |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    bound.add(n.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    bound.add(n.id)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _unbound_asic_ai_names(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, str(path))
    module_bound = _bound_at_module_level(tree)

    findings: list[str] = []
    fn_line_ranges: list[tuple[int, int, ast.AST]] = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_line_ranges.append((fn.lineno, fn.end_lineno or fn.lineno, fn))

    for fn_start, fn_end, fn in fn_line_ranges:
        local = _bound_in_scope(fn)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            name = node.id
            if name not in ASIC_AI_EXPORTS:
                continue
            if name in local or name in module_bound:
                continue
            findings.append(f"{path.name}:{node.lineno} {name} (in {fn.name}())")

    covered = set()
    for s, e, _ in fn_line_ranges:
        covered.update(range(s, e + 1))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if node.lineno in covered:
            continue
        if node.id in ASIC_AI_EXPORTS and node.id not in module_bound:
            findings.append(f"{path.name}:{node.lineno} {node.id} (module level)")

    return findings


SCRIPT_FILES = sorted(SCRIPTS.glob("*.py"))


def test_asic_ai_export_set_is_populated():
    """Sanity: if this collapses to nothing the whole guard is vacuous."""
    assert len(ASIC_AI_EXPORTS) > 50
    assert "build_system_message" in ASIC_AI_EXPORTS


@pytest.mark.parametrize("path", SCRIPT_FILES, ids=lambda p: p.name)
def test_script_imports_every_asic_ai_symbol_it_uses(path: Path):
    findings = _unbound_asic_ai_names(path)
    assert not findings, (
        "these asic_ai names are used but never imported in scope "
        f"(NameError at runtime): {findings}"
    )


def test_build_system_message_is_bound_wherever_it_is_called():
    """The exact Phase 68 regression, called out by name."""
    offenders = []
    for path in SCRIPT_FILES:
        if "build_system_message" not in path.read_text(encoding="utf-8"):
            continue
        if any("build_system_message" in f for f in _unbound_asic_ai_names(path)):
            offenders.append(path.name)
    assert not offenders, (
        f"{len(offenders)} script(s) call build_system_message() without "
        f"importing it: {offenders}"
    )
