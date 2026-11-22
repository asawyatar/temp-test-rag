"""`atar-ask`: ask a tenant's knowledge base a question from the terminal.

    atar-ask --tenant tenant_a "متى يفتح المسبح؟"
    atar-ask --tenant tenant_b            (no question: ask several, one per line)

An operator tool for trying the answer pipeline without the mobile app or the
backend: the operator names the tenant, exactly as with `atar-ingest`. It runs
the same `answer_events` as POST /v1/answer, so what you see here is what a
resident of that tenant would get.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from typing import TextIO

from atar_ai_agent.answer import Citation, Delta, Done, Failure, Sources, answer_events
from atar_ai_agent.config import Settings
from atar_ai_agent.context import Locale, RequestContext, TenantId
from atar_ai_agent.db import create_pool
from atar_ai_agent.ingest.cli import aclose_if_supported, tenant_argument
from atar_ai_agent.llm import build_llm
from atar_ai_agent.llm.client import LLMClient
from atar_ai_agent.llm.prompts import HistoryTurn
from atar_ai_agent.rag.embeddings import build_embedder
from atar_ai_agent.rag.protocols import Embedder
from atar_ai_agent.rag.store import PgVectorStore

EXIT_OK, EXIT_NOT_CONFIGURED = 0, 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atar-ask", description=__doc__.split("\n\n")[0])
    parser.add_argument("--tenant", type=tenant_argument, required=True)
    parser.add_argument(
        "--locale",
        choices=["ar", "en"],
        default="ar",
        help="the app's language; only used when the question has no letters",
    )
    parser.add_argument("question", nargs="?", help="omit to ask several, one per line")
    return parser


def operator_context(tenant: TenantId, locale: Locale) -> RequestContext:
    # No resident token: this is an operator asking on a tenant's behalf.
    # Fine for answering from documents; phase 2 tools that call the backend
    # as the resident will need a real RequestContext from a verified request.
    return RequestContext(
        tenant_id=tenant, tenant_subdomain="", user_id="operator", locale=locale, token=""
    )


async def ask_one(
    question: str,
    context: RequestContext,
    *,
    embedder: Embedder,
    store: PgVectorStore,
    llm: LLMClient,
    top_k: int,
    out: TextIO,
    history: Sequence[HistoryTurn] = (),
) -> str:
    """Print the answer and its sources; return the answer text."""
    started = time.perf_counter()
    sources: list[str] = []
    cited: dict[int, str] = {}
    answer: list[str] = []
    async for event in answer_events(
        question, context, embedder=embedder, store=store, llm=llm, top_k=top_k, history=history
    ):
        match event:
            case Sources(found, query):
                sources = [s.title for s in found]
                if query and query != question:
                    out.write(f"  (searched as: {query})\n")
            case Delta(text):
                answer.append(text)
                out.write(text)
                out.flush()
            case Citation(index, source_key, title, cited_text):
                cited.setdefault(index, f"{source_key} > {title.split(' > ')[-1]}: {cited_text}")
            case Failure(code, message):
                out.write(f"\n[{code}] {message}")
            case Done(grounded, language, _, model):
                seconds = time.perf_counter() - started
                out.write("\n\n")
                for index, line in sorted(cited.items()):
                    out.write(f"  [{index + 1}] {_shorten(line)}\n")
                out.write(
                    f"  grounded: {'yes' if grounded else 'no'} | language: {language} | "
                    f"retrieved: {len(sources)} | model: {model or 'none'} | {seconds:.1f}s\n"
                )
    return "".join(answer).strip()


def _shorten(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


async def run(
    argv: Sequence[str],
    *,
    settings: Settings,
    make_embedder: Callable[[Settings], Embedder | None],
    make_llm: Callable[[Settings], LLMClient | None],
    stdin: TextIO,
    out: TextIO,
) -> int:
    args = build_parser().parse_args(argv)
    llm = make_llm(settings)
    if llm is None:
        key = f"{settings.answer_provider.upper()}_API_KEY"
        print(f"error: no answer model configured; set {key} in .env", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    embedder = make_embedder(settings)
    if embedder is None:
        print("error: no embedding model configured; set OPENAI_API_KEY in .env", file=sys.stderr)
        await llm.close()
        return EXIT_NOT_CONFIGURED
    context = operator_context(args.tenant, args.locale)
    pool = create_pool(settings)
    await pool.open()
    try:
        store = PgVectorStore(pool)
        questions = [args.question] if args.question else None
        if questions is None:
            out.write(
                f"Asking as tenant {args.tenant}. One question per line; empty line to quit.\n"
                "Follow-up questions remember this session's earlier turns.\n"
            )
        # Interactive sessions keep their history in memory only (nothing is
        # saved); like the app, only the last few exchanges are sent.
        history: list[HistoryTurn] = []
        while True:
            if questions is not None:
                if not questions:
                    break
                question = questions.pop(0)
            else:
                out.write("\n> ")
                out.flush()
                question = stdin.readline().strip()
                if not question:
                    break
            answer = await ask_one(
                question,
                context,
                embedder=embedder,
                store=store,
                llm=llm,
                top_k=settings.search_top_k,
                out=out,
                history=history[-2 * settings.chat_history_exchanges :],
            )
            history += [HistoryTurn("user", question), HistoryTurn("assistant", answer)]
    finally:
        await pool.close()
        await llm.close()
        await aclose_if_supported(embedder)
    return EXIT_OK


def main() -> None:
    settings = Settings()
    code = asyncio.run(
        run(
            sys.argv[1:],
            settings=settings,
            make_embedder=build_embedder,
            make_llm=build_llm,
            stdin=sys.stdin,
            out=sys.stdout,
        )
    )
    raise SystemExit(code)
