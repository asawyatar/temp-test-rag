"""Non-negotiable: the tenant is data, never a branch.

Scans the source for comparisons between something tenant-like and a string
literal, e.g. `if tenant == "atar"` or `tenant_id in ("retal", "diriyah")`.
Per-tenant differences belong in config or data. Comparing with a named
constant (SHARED_SCOPE) is allowed: it is one rule for everyone.
"""

import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"


def _mentions_tenant(node: ast.AST) -> bool:
    return any(
        (isinstance(n, ast.Name) and "tenant" in n.id.lower())
        or (isinstance(n, ast.Attribute) and "tenant" in n.attr.lower())
        for n in ast.walk(node)
    )


def _has_string_literal(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in ast.walk(node))


def _tenant_literal_comparisons(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            tenant_side = any(_mentions_tenant(s) for s in sides)
            literal_side = any(_has_string_literal(s) and not _mentions_tenant(s) for s in sides)
            if tenant_side and literal_side:
                lines.append(node.lineno)
        elif isinstance(node, ast.match_case) and isinstance(node.pattern, ast.MatchValue):
            lines.append(node.pattern.lineno)
    return lines


def test_detector_catches_a_tenant_branch() -> None:
    sample = 'if ctx.tenant_id == "atar":\n    pass\nif tenant in ("retal", "diriyah"):\n    pass\n'

    assert _tenant_literal_comparisons(ast.parse(sample)) == [1, 3]


def test_no_source_file_branches_on_a_tenant_name() -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{line}"
        for path in sorted(SRC.rglob("*.py"))
        for line in _tenant_literal_comparisons(ast.parse(path.read_text(), filename=str(path)))
    ]

    assert offenders == [], f"tenant compared with a string literal at: {offenders}"
