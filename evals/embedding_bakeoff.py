"""Embedding bake-off: which model retrieves best over Arabic and English?

    uv run python evals/embedding_bakeoff.py --model openai-3-large
    uv run --group bakeoff python evals/embedding_bakeoff.py --model bge-m3

For each candidate model it ingests every fixture folder into its own database
(`atar_ai_bakeoff`), runs every case in evals/retrieval_cases.jsonl through the
real `search()`, and writes per-case rows plus a summary to evals/results/.

Metrics, per kind of question and per language pair (question -> document):
- hit@k: an expected source appears among the top k chunks.
- MRR: mean of 1/rank of the first chunk from an expected source.
- For questions the content cannot answer: the top similarity score. The gap
  between answerable and unanswerable top scores says whether a score
  threshold can refuse before the answer model is even called.

The candidates live here, not in src/: only the winner moves into the service.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import openai
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from atar_ai_agent.config import Settings
from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import Pool, create_pool, migrate
from atar_ai_agent.ingest.loaders import loader_for
from atar_ai_agent.rag.chunking import CHUNKER_VERSION, chunk_document, embedding_input
from atar_ai_agent.rag.embeddings import OpenAIEmbedder
from atar_ai_agent.rag.protocols import Embedder
from atar_ai_agent.rag.search import search
from atar_ai_agent.rag.store import PgVectorStore
from atar_ai_agent.rag.types import SourceRecord

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "retrieval_cases.jsonl"
RESULTS = ROOT / "evals" / "results"
FIXTURE_SCOPES = {"tenant_a": TenantId("tenant_a"), "tenant_b": TenantId("tenant_b")}
FIXTURE_SCOPES["shared"] = SHARED_SCOPE
DEFAULT_DATABASE_URL = "postgresql://atar:atar_local_dev@localhost:5433/atar_ai_bakeoff"
TOP_K = 10


# --- Candidates -------------------------------------------------------------


class _BakeoffSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    voyage_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None


class VoyageEmbedder:
    """Voyage AI hosted embeddings over HTTPS (docs checked 2026-11-17).

    `input_type` tells the model whether it sees a question or a document.
    The docs do not promise response order, so results are sorted by `index`.
    """

    URL = "https://api.voyageai.com/v1/embeddings"
    BATCH = 128  # far below the 1,000-item and token limits per request

    def __init__(self, model: str, client: httpx2.AsyncClient | None = None) -> None:
        key = _BakeoffSettings().voyage_api_key
        if client is None and key is None:
            raise SystemExit("VOYAGE_API_KEY is not set (add it to .env).")
        self.model_name = model
        self._client = client or httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {key.get_secret_value() if key else ''}"},
            timeout=60.0,
        )

    async def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            body = {
                "input": list(texts[start : start + self.BATCH]),
                "model": self.model_name,
                "input_type": input_type,
                "output_dimension": 1024,
                "truncation": False,  # an over-long chunk is a bug to see, not hide
            }
            for attempt in range(4):
                response = await self._client.post(self.URL, json=body)
                if response.status_code != 429 or attempt == 3:
                    break
                await asyncio.sleep(2**attempt)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend([float(v) for v in item["embedding"]] for item in data)
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "document")

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._embed([text], "query")
        return vector


class SentenceTransformerEmbedder:
    """BAAI/bge-m3 run in-process with sentence-transformers (the service's
    embedder until 2026-11-22, see docs/DECISIONS.md). Kept so the bake-off can
    be re-run on real documents. Needs the `bakeoff` dependency group (torch).
    """

    MODEL = "BAAI/bge-m3"
    # Pinned commit: the safetensors weights, verified against BAAI's own.
    REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"

    def __init__(self, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = f"{self.MODEL}@{self.REVISION[:12]}"
        self._model = SentenceTransformer(
            self.MODEL,
            revision=self.REVISION,
            device=device,
            trust_remote_code=False,
            # safetensors only: pickled weights can run code on load.
            model_kwargs={"use_safetensors": True},
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(
            self._model.encode,
            list(texts),
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(value) for value in row] for row in vectors]

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self.embed_documents([text])
        return vector


def _openai(model: str) -> Embedder:
    key = _BakeoffSettings().openai_api_key
    if key is None:
        raise SystemExit("OPENAI_API_KEY is not set (add it to .env).")
    return OpenAIEmbedder(openai.AsyncOpenAI(api_key=key.get_secret_value()), model)


CANDIDATES: dict[str, Callable[[], Embedder]] = {
    # BAKEOFF_DEVICE=cpu measures what a GPU-less server would see.
    "bge-m3": lambda: SentenceTransformerEmbedder(os.environ.get("BAKEOFF_DEVICE")),
    # The service's own embedder (src/), so the bake-off measures what runs.
    "openai-3-large": lambda: _openai("text-embedding-3-large"),
    "openai-3-small": lambda: _openai("text-embedding-3-small"),
    "voyage-4": lambda: VoyageEmbedder("voyage-4"),
    "voyage-4-large": lambda: VoyageEmbedder("voyage-4-large"),
}


# --- Cases and results -------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    tenant: str
    lang: str
    kind: str
    question: str
    expected_sources: list[str]


@dataclass(frozen=True)
class CaseResult:
    id: str
    kind: str
    pair: str
    rank: int | None
    top_score: float
    top_source: str
    retrieved: list[str]
    # Best chunk from an expected source minus best chunk from anywhere else.
    # Positive = right answer ranked first; larger = more robust. Answerable only.
    margin: float | None


def load_cases() -> list[Case]:
    rows = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    return [Case(**{k: row[k] for k in Case.__dataclass_fields__}) for row in rows]


def language_pair(case: Case) -> str:
    if case.kind in ("unanswerable", "cross_tenant_probe"):
        return f"{case.lang}->none"
    if case.kind == "shared":
        return f"{case.lang}->shared"
    document_lang = Path(case.expected_sources[0]).suffixes[-2].lstrip(".")
    return f"{case.lang}->{document_lang}"


# --- Database ------------------------------------------------------------------


def ensure_database(url: str) -> None:
    name = conninfo_to_dict(url)["dbname"]
    assert isinstance(name, str)
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def read_fixtures() -> dict[TenantId, list[tuple[str, bytes]]]:
    return {
        scope: [
            (path.name, path.read_bytes())
            for path in sorted((ROOT / "fixtures" / folder).iterdir())
            if loader_for(path.name)
        ]
        for folder, scope in FIXTURE_SCOPES.items()
    }


async def ingest(pool: Pool, embedder: Embedder) -> int:
    """Store every fixture under its scope with this embedder. Returns chunk count."""
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE sources CASCADE")
    store = PgVectorStore(pool)
    total = 0
    for scope, files in (await asyncio.to_thread(read_fixtures)).items():
        for name, data in files:
            loader = loader_for(name)
            assert loader is not None
            document = loader.load(data, name)
            chunks = chunk_document(document)
            vectors = await embedder.embed_documents([embedding_input(c) for c in chunks])
            record = SourceRecord(source_key=name, title=document.title, fingerprint="bakeoff")
            await store.replace_source(scope, record, chunks, vectors, embedder.model_name)
            total += len(chunks)
    return total


async def run_case(pool: Pool, embedder: Embedder, case: Case) -> CaseResult:
    tenant = TenantId(case.tenant)
    chunks = await search(
        case.question, tenant, embedder=embedder, store=PgVectorStore(pool), top_k=TOP_K
    )
    # Isolation must hold here too; a failure here would be a bug, not a score.
    assert all(c.tenant_id in (tenant, SHARED_SCOPE) for c in chunks), case.id
    sources = [c.source_key for c in chunks]
    rank = next((i for i, s in enumerate(sources, 1) if s in case.expected_sources), None)
    right = [c.score for c in chunks if c.source_key in case.expected_sources]
    wrong = [c.score for c in chunks if c.source_key not in case.expected_sources]
    margin = round(max(right) - max(wrong), 4) if right and wrong else None
    return CaseResult(
        id=case.id,
        kind=case.kind,
        pair=language_pair(case),
        rank=rank,
        top_score=round(chunks[0].score, 4) if chunks else 0.0,
        top_source=sources[0] if sources else "",
        retrieved=sources,
        margin=margin,
    )


# --- Summary -------------------------------------------------------------------


def _metrics(rows: list[CaseResult]) -> dict[str, float]:
    margins = [r.margin for r in rows if r.margin is not None]

    def hit(k: int) -> float:
        return sum(1 for r in rows if r.rank is not None and r.rank <= k) / len(rows)

    return {
        "n": len(rows),
        "hit@1": round(hit(1), 3),
        "hit@3": round(hit(3), 3),
        "hit@5": round(hit(5), 3),
        "mrr": round(sum(1 / r.rank for r in rows if r.rank) / len(rows), 3),
        "margin_mean": round(statistics.mean(margins), 3) if margins else 0.0,
        "margin_min": round(min(margins), 3) if margins else 0.0,
    }


def summarise(rows: list[CaseResult]) -> dict[str, Any]:
    answerable = [r for r in rows if r.kind not in ("unanswerable", "cross_tenant_probe")]
    unanswerable = [r for r in rows if r not in answerable]
    by_pair = {
        pair: _metrics([r for r in answerable if r.pair == pair])
        for pair in sorted({r.pair for r in answerable})
    }
    answerable_scores = [r.top_score for r in answerable]
    unanswerable_scores = [r.top_score for r in unanswerable]
    return {
        "overall": _metrics(answerable),
        "by_language_pair": by_pair,
        "by_kind": {
            kind: _metrics([r for r in answerable if r.kind == kind])
            for kind in sorted({r.kind for r in answerable})
        },
        "top_score": {
            "answerable_min": min(answerable_scores),
            "answerable_median": round(statistics.median(answerable_scores), 4),
            "unanswerable_max": max(unanswerable_scores) if unanswerable_scores else None,
            "unanswerable": {r.id: r.top_score for r in unanswerable},
            "separable": bool(unanswerable_scores)
            and max(unanswerable_scores) < min(answerable_scores),
        },
        "weakest": [
            {"id": r.id, "pair": r.pair, "margin": r.margin}
            for r in sorted(
                (r for r in answerable if r.margin is not None),
                key=lambda r: r.margin or 0.0,
            )[:3]
        ],
        "misses": [
            {"id": r.id, "pair": r.pair, "rank": r.rank, "top_source": r.top_source}
            for r in answerable
            if r.rank != 1
        ],
    }


def write_markdown(results_dir: Path) -> Path:
    """One table across every model that has a results file."""
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("bakeoff-*.json"))]
    pairs = sorted({pair for run in runs for pair in run["summary"]["by_language_pair"]})
    lines = [
        "# Embedding bake-off",
        "",
        "Generated by `evals/embedding_bakeoff.py`; do not edit by hand.",
        f"Cases: `evals/retrieval_cases.jsonl` (top {TOP_K} chunks per question).",
        "",
        "## Retrieval quality on answerable questions",
        "",
        "| Model | Run | Chunks | hit@1 | hit@3 | hit@5 | MRR | "
        + " | ".join(f"MRR {p}" for p in pairs)
        + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(pairs),
    ]
    for run in runs:
        s = run["summary"]
        overall = s["overall"]
        per_pair = [str(s["by_language_pair"].get(p, {}).get("mrr", "–")) for p in pairs]
        lines.append(
            f"| `{run['key']}` | {run['run_at'][:10]} | {run['chunks']} "
            f"| {overall['hit@1']} | {overall['hit@3']} | {overall['hit@5']} "
            f"| {overall['mrr']} | " + " | ".join(per_pair) + " |"
        )
    lines += [
        "",
        "## How clearly does the right chunk win?",
        "",
        "Margin = best chunk from the right document minus the best chunk from any other",
        "document. hit@1 can hide a near tie; on messier real content a small margin flips.",
        "Each model's scores sit on their own scale, so compare margins *within* a model",
        "(e.g. across language pairs) and compare models by rank. A negative margin is a miss.",
        "",
        "| Model | Mean margin | Smallest margin | "
        + " | ".join(f"Mean {p}" for p in pairs)
        + " | Weakest cases |",
        "|---|---|---|" + "---|" * len(pairs) + "---|",
    ]
    for run in runs:
        s = run["summary"]
        per_pair = [str(s["by_language_pair"].get(p, {}).get("margin_mean", "–")) for p in pairs]
        weakest = ", ".join(f"{w['id']} ({w['margin']})" for w in s["weakest"])
        lines.append(
            f"| `{run['key']}` | {s['overall']['margin_mean']} | {s['overall']['margin_min']} | "
            + " | ".join(per_pair)
            + f" | {weakest} |"
        )
    lines += [
        "",
        "## Can a score threshold refuse unanswerable questions?",
        "",
        "| Model | Lowest answerable top score | Highest unanswerable top score | Separable |",
        "|---|---|---|---|",
    ]
    for run in runs:
        t = run["summary"]["top_score"]
        lines.append(
            f"| `{run['key']}` | {t['answerable_min']} | {t['unanswerable_max']} "
            f"| {'yes' if t['separable'] else 'no'} |"
        )
    lines += ["", "## Questions not answered by the top chunk", ""]
    for run in runs:
        misses = run["summary"]["misses"]
        lines.append(
            f"- `{run['key']}`: "
            + (
                ", ".join(
                    f"{m['id']} ({m['pair']}, rank {m['rank'] or '>' + str(TOP_K)}, "
                    f"top: {m['top_source']})"
                    for m in misses
                )
                or "none"
            )
        )
    lines += [
        "",
        "## Timing",
        "",
        "| Model | Device | Ingest seconds | Mean query ms |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{run['key']}` | {run['device']} | {run['ingest_seconds']} | {run['mean_query_ms']} |"
        for run in runs
    ]
    path = results_dir / "embedding_bakeoff.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- Main ------------------------------------------------------------------------


async def run(model_key: str, database_url: str) -> Path:
    ensure_database(database_url)
    await migrate(database_url)
    embedder = CANDIDATES[model_key]()
    pool = create_pool(Settings(database_url=SecretStr(database_url)))
    await pool.open()
    try:
        started = time.perf_counter()
        chunk_count = await ingest(pool, embedder)
        ingest_seconds = round(time.perf_counter() - started, 1)

        cases = load_cases()
        started = time.perf_counter()
        rows = [await run_case(pool, embedder, case) for case in cases]
        mean_query_ms = round((time.perf_counter() - started) * 1000 / len(cases), 1)
    finally:
        await pool.close()

    model_attr = getattr(embedder, "_model", None)
    device = str(getattr(model_attr, "device", "api"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    forced = os.environ.get("BAKEOFF_DEVICE")
    key = f"{model_key}-{forced}" if forced else model_key
    out = RESULTS / f"bakeoff-{key}.json"
    out.write_text(
        json.dumps(
            {
                "key": key,
                "model": embedder.model_name,
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "chunker_version": CHUNKER_VERSION,
                "chunks": chunk_count,
                "device": device,
                "ingest_seconds": ingest_seconds,
                "mean_query_ms": mean_query_ms,
                "summary": summarise(rows),
                "cases": [asdict(r) for r in rows],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return write_markdown(RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(CANDIDATES), required=True)
    args = parser.parse_args()
    url = os.environ.get("BAKEOFF_DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"wrote {asyncio.run(run(args.model, url)).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
