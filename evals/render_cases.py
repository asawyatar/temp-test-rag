"""Render evals/retrieval_cases.jsonl as a Markdown table for review.

    uv run python evals/render_cases.py

The JSONL file is the source of truth; this page is regenerated from it.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIND_LABELS = {
    "same_language": "Same language",
    "cross_language": "Cross-language",
    "shared": "Shared app help",
    "unanswerable": "Not in any document",
    "cross_tenant_probe": "Only the other tenant knows",
}


def main() -> None:
    rows = [
        json.loads(line)
        for line in (HERE / "retrieval_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lines = [
        "# Retrieval cases",
        "",
        "Generated from `retrieval_cases.jsonl` by `render_cases.py`; edit the JSONL.",
        "",
        "Tenant A is Qamar Bay, tenant B is Rimal Terrace (see `fixtures/README.md`).",
        "",
    ]
    for kind, label in KIND_LABELS.items():
        group = [r for r in rows if r["kind"] == kind]
        lines += [
            f"## {label} ({len(group)})",
            "",
            "| ID | Tenant | Asked in | Question | Answer should come from |",
            "|---|---|---|---|---|",
        ]
        for r in group:
            expected = ", ".join(f"`{s}`" for s in r["expected_sources"]) or "nothing"
            tenant = "A" if r["tenant"] == "tenant_a" else "B"
            lines.append(f"| {r['id']} | {tenant} | {r['lang']} | {r['question']} | {expected} |")
            if r.get("note"):
                lines.append(f"| | | | ↳ {r['note']} | |")
        lines.append("")
    (HERE / "CASES.md").write_text("\n".join(lines))
    print(f"wrote evals/CASES.md ({len(rows)} cases)")


if __name__ == "__main__":
    main()
