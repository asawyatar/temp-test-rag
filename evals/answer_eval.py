"""Answer evaluation: ask every case through the real pipeline and grade it.

    make eval                                  # the configured model
    make eval MODELS="gpt-6.1-terra gpt-6.1-sol"
    uv run python evals/answer_eval.py --regrade   # re-score saved answers, no model calls

Ingests the fixtures into its own database (`atar_ai_eval`) with the real
embedding model, asks the 32 questions in evals/answer_cases.jsonl through
the same `answer_events` as POST /v1/answer, and grades each answer with
the deterministic checks in evals/grading.py. Questions run one at a time so
the latency numbers are clean. Writes evals/results/answers-<model>.json and
a combined evals/results/answer_eval.md.

Calls the configured answer model (paid): 32 short requests per model.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from embedding_bakeoff import ensure_database
from grading import grade

from atar_ai_agent.answer import Citation, Delta, Done, Failure, Sources, answer_events
from atar_ai_agent.ask import operator_context
from atar_ai_agent.config import Settings
from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import Pool, create_pool, migrate
from atar_ai_agent.ingest.job import sync_folder
from atar_ai_agent.llm import build_llm
from atar_ai_agent.llm.client import LLMClient
from atar_ai_agent.rag.embeddings import build_embedder
from atar_ai_agent.rag.protocols import Embedder
from atar_ai_agent.rag.store import PgVectorStore

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "answer_cases.jsonl"
RESULTS = ROOT / "evals" / "results"
DEFAULT_DATABASE_URL = "postgresql://atar:atar_local_dev@localhost:5433/atar_ai_eval"
FIXTURE_SCOPES = {
    "tenant_a": TenantId("tenant_a"),
    "tenant_b": TenantId("tenant_b"),
    "shared": SHARED_SCOPE,
}


@dataclass
class CaseRun:
    id: str
    kind: str
    pair: str
    question: str
    answer: str
    cited: list[str]
    retrieved: list[str]
    retrieval_rank: int | None
    grounded: bool
    model: str | None
    failure: str | None
    first_text_s: float | None
    total_s: float
    passed: bool
    checks: dict[str, bool]
    problems: list[str]


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def pair(case: dict[str, Any]) -> str:
    if case["expect"] == "not_found":
        return f"{case['lang']}->none"
    if case["kind"] == "shared":
        return f"{case['lang']}->shared"
    return f"{case['lang']}->{Path(case['sources'][0]).suffixes[-2].lstrip('.')}"


async def ingest_fixtures(pool: Pool, embedder: Embedder) -> None:
    for folder, scope in FIXTURE_SCOPES.items():
        await sync_folder(
            scope, ROOT / "fixtures" / folder, store=PgVectorStore(pool), embedder=embedder
        )


async def run_case(
    case: dict[str, Any],
    *,
    pool: Pool,
    embedder: Embedder,
    llm: LLMClient,
    top_k: int,
) -> CaseRun:
    context = operator_context(TenantId(case["tenant"]), case["lang"])
    started = time.perf_counter()
    first_text: float | None = None
    text: list[str] = []
    cited: list[str] = []
    retrieved: list[str] = []
    grounded, model, failure = False, None, None
    async for event in answer_events(
        case["question"],
        context,
        embedder=embedder,
        store=PgVectorStore(pool),
        llm=llm,
        top_k=top_k,
    ):
        match event:
            case Sources(sources):
                retrieved = [s.source_key for s in sources]
            case Delta(piece):
                if first_text is None and piece.strip():
                    first_text = time.perf_counter() - started
                text.append(piece)
            case Citation(_, source_key, _, _):
                cited.append(source_key)
            case Failure(code, _):
                failure = code
            case Done(done_grounded, _, _, done_model):
                grounded, model = done_grounded, done_model
    total = time.perf_counter() - started
    answer = "".join(text).strip()
    result = grade(case, answer, cited, grounded)
    if failure:
        result.passed = False
        result.problems.append(f"model error: {failure}")
    rank = next((i for i, s in enumerate(retrieved, 1) if s in case["sources"]), None)
    return CaseRun(
        id=case["id"],
        kind=case["kind"],
        pair=pair(case),
        question=case["question"],
        answer=answer,
        cited=sorted(set(cited)),
        retrieved=retrieved,
        retrieval_rank=rank,
        grounded=grounded,
        model=model,
        failure=failure,
        first_text_s=round(first_text, 2) if first_text is not None else None,
        total_s=round(total, 2),
        passed=result.passed,
        checks=result.checks,
        problems=result.problems,
    )


def _rate(runs: list[CaseRun]) -> str:
    return f"{sum(r.passed for r in runs)}/{len(runs)}"


def _p(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def summarise(runs: list[CaseRun]) -> dict[str, Any]:
    answerable = [r for r in runs if r.kind not in ("unanswerable", "cross_tenant_probe")]
    check_names = sorted({name for r in runs for name in r.checks})
    first = [r.first_text_s for r in runs if r.first_text_s is not None]
    total = [r.total_s for r in runs]
    return {
        "passed": _rate(runs),
        "pass_rate": round(sum(r.passed for r in runs) / len(runs), 3),
        "by_kind": {
            k: _rate([r for r in runs if r.kind == k]) for k in sorted({r.kind for r in runs})
        },
        "by_pair": {
            p: _rate([r for r in runs if r.pair == p]) for p in sorted({r.pair for r in runs})
        },
        "checks": {
            name: f"{sum(r.checks[name] for r in runs if name in r.checks)}/"
            f"{sum(name in r.checks for r in runs)}"
            for name in check_names
        },
        "retrieval_hit@1": f"{sum(r.retrieval_rank == 1 for r in answerable)}/{len(answerable)}",
        "retrieval_in_top_k": f"{sum(r.retrieval_rank is not None for r in answerable)}"
        f"/{len(answerable)}",
        "first_text_s": {"median": _p(first, 0.5), "p90": _p(first, 0.9)} if first else None,
        "total_s": {"median": _p(total, 0.5), "p90": _p(total, 0.9)},
    }


def write_markdown() -> Path:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("answers-*.json"))]
    lines = [
        "# Answer evaluation",
        "",
        "Generated by `evals/answer_eval.py` (`make eval`); do not edit by hand.",
        "Cases: `evals/answer_cases.jsonl`; checks: `evals/grading.py`.",
        "With 32 cases, one case is about 3 points: treat differences under about",
        "10 points as noise.",
        "",
        "| Model | Run | Passed | Same language | Cross-language | Not found / probes "
        "| Injection | First text (median / p90) | Full answer (median / p90) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        s = run["summary"]
        kinds = s["by_kind"]
        silent = f"{kinds.get('unanswerable', '–')} + {kinds.get('cross_tenant_probe', '–')}"
        ft = s["first_text_s"] or {"median": "–", "p90": "–"}
        lines.append(
            f"| `{run['model']}` | {run['run_at'][:16]} | **{s['passed']}** | "
            f"{kinds.get('same_language', '–')} | {kinds.get('cross_language', '–')} | {silent} | "
            f"{kinds.get('injection', '–')} | {ft['median']} s / {ft['p90']} s | "
            f"{s['total_s']['median']} s / {s['total_s']['p90']} s |"
        )
    lines += [
        "",
        "## Checks",
        "",
        "| Model | " + " | ".join(runs[0]["summary"]["checks"]) + " | Retrieval hit@1 | In top k |",
    ]
    lines.append("|---|" + "---|" * (len(runs[0]["summary"]["checks"]) + 2))
    for run in runs:
        s = run["summary"]
        lines.append(
            f"| `{run['model']}` | "
            + " | ".join(s["checks"].values())
            + f" | {s['retrieval_hit@1']} | {s['retrieval_in_top_k']} |"
        )
    for run in runs:
        failed = [c for c in run["cases"] if not c["passed"]]
        lines += ["", f"## Failures: `{run['model']}` ({len(failed)})", ""]
        if not failed:
            lines.append("None.")
        for c in failed:
            answer = " ".join(c["answer"].split())
            lines.append(f"- **{c['id']}** ({c['pair']}) {c['question']}")
            lines.append(f"  - problems: {'; '.join(c['problems'])}")
            lines.append(f"  - answer: {answer[:300]}{'…' if len(answer) > 300 else ''}")
    path = RESULTS / "answer_eval.md"
    path.write_text("\n".join(lines) + "\n")
    return path


async def main_async(models: list[str] | None) -> None:
    settings = Settings()
    embedder = build_embedder(settings)
    if embedder is None:
        raise SystemExit("OPENAI_API_KEY is not set (the embedding model needs it)")
    url = os.environ.get("EVAL_DATABASE_URL", DEFAULT_DATABASE_URL)
    ensure_database(url)
    await migrate(url)
    pool = create_pool(
        settings.model_copy(update={"database_url": type(settings.database_url)(url)})
    )
    await pool.open()
    try:
        await ingest_fixtures(pool, embedder)
        model_field = "openai_model" if settings.answer_provider == "openai" else "anthropic_model"
        for model in models or [str(getattr(settings, model_field))]:
            llm = build_llm(settings.model_copy(update={model_field: model}))
            if llm is None:
                raise SystemExit(f"no {settings.answer_provider} API key configured")
            try:
                cases = load_cases()
                runs = []
                for case in cases:
                    run = await run_case(
                        case, pool=pool, embedder=embedder, llm=llm, top_k=settings.search_top_k
                    )
                    runs.append(run)
                    verdict = "pass" if run.passed else "FAIL"
                    print(f"  {verdict}  {run.id}  {run.total_s:>5}s  {'; '.join(run.problems)}")
            finally:
                await llm.close()
            summary = summarise(runs)
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / f"answers-{model}.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "provider": settings.answer_provider,
                        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "summary": summary,
                        "cases": [asdict(r) for r in runs],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(f"{model}: passed {summary['passed']} | {json.dumps(summary['by_kind'])}")
    finally:
        await pool.close()
        await embedder.aclose()
    print(f"wrote {write_markdown().relative_to(ROOT)}")


def regrade() -> None:
    """Re-score saved answers with the current cases and checks.

    For when a check was wrong (a correct answer phrased in a way the case
    did not allow): fixing the case should not cost another paid run.
    """
    cases = {case["id"]: case for case in load_cases()}
    for path in sorted(RESULTS.glob("answers-*.json")):
        saved = json.loads(path.read_text())
        runs = []
        for row in saved["cases"]:
            run = CaseRun(**row)
            result = grade(cases[run.id], run.answer, run.cited, run.grounded)
            if run.failure:
                result.passed = False
                result.problems.append(f"model error: {run.failure}")
            run.passed, run.checks, run.problems = result.passed, result.checks, result.problems
            runs.append(run)
        saved["summary"] = summarise(runs)
        saved["cases"] = [asdict(r) for r in runs]
        saved["regraded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
        print(f"{saved['model']}: passed {saved['summary']['passed']}")
    print(f"wrote {write_markdown().relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models", nargs="*", help="answer models to compare (default: configured)"
    )
    parser.add_argument(
        "--regrade", action="store_true", help="re-score saved answers without calling any model"
    )
    args = parser.parse_args()
    if args.regrade:
        regrade()
    else:
        asyncio.run(main_async(args.models))


if __name__ == "__main__":
    main()
