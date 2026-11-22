# Decisions

A running record of why this service is built the way it is. Newest entries go
at the bottom of each section. Every new dependency gets a line under
[Dependencies](#dependencies).

Approved 2026-11-16.

## Inputs these decisions rest on

| Question | Answer (2026-11-16) |
|---|---|
| Knowledge sources | Markdown, JSON, Word and PDF files per tenant. One loader per type behind one interface, built in that order. JSON is chunked per record. Backend content and web pages are later loaders. |
| Real content | None yet. Synthetic, clearly labelled fixtures for two fictional communities that deliberately conflict. |
| Sharing | Per-tenant plus one shared Atar-wide set. |
| Auth | Forward the resident's token and tenant headers to `GET /tenancy/api/me`. |
| Hosting | Not decided. Ship a Dockerfile and read a plain `DATABASE_URL`. |
| Embeddings | Hosted or self-hosted; an eval decides. |
| Scale | Tens of documents per tenant, changing rarely. |
| Local database | Docker Compose. |

## Design

1. **One request context.** `RequestContext(tenant_id, tenant_subdomain,
   user_id, locale, token, app_env, app_version)` is a frozen dataclass in
   `context.py` at the package root, so `rag/` and the future `agent/` never
   import from `api/`. Only `api/auth.py` builds it, and only after
   `/tenancy/api/me` succeeds. The request body model forbids extra fields, so a
   `tenant_id` in the body is rejected.

2. **`search` is a plain function.**
   `search(query: str, tenant_id: TenantId, *, embedder, store, top_k) -> list[Chunk]`
   in `rag/search.py`. No default tenant, no HTTP, no LLM. `TenantId` is a
   `NewType`, so the type checker flags a raw string coming from anywhere but
   the verifier. In phase 2 this function becomes one tool among several.

3. **Isolation at one choke point.** `chunks.tenant_id` is `NOT NULL`. One
   store method reads chunks and it requires the tenant scope. Shared content is
   stored under a reserved scope id from config; `search` asks for exactly
   `[tenant_id, shared_id]`. There is no code path without a tenant. Postgres
   row-level security is deliberately left as later hardening.

4. **Idempotent ingestion by content hash.** A `sources` row per file holds a
   fingerprint; `chunks` rows reference it with `ON DELETE CASCADE`. An
   unchanged fingerprint is skipped. A changed one replaces that source's chunks
   in one transaction. A source missing from the input folder is deleted and its
   chunks go with it. `--dry-run` prints the plan first.

5. **Exact vector search, no ANN index yet.** At tens of documents per tenant a
   scan filtered by tenant takes milliseconds and has perfect recall. An HNSW
   index combined with a tenant filter can silently drop results, so it waits
   until scale needs it.

6. **Embedding model chosen by bake-off.** Candidates: Voyage `voyage-4`
   (hosted) and `BAAI/bge-m3` (self-hosted), both 1024 dimensions so the schema
   fits either. Measured on the fixtures by language pair (ar→ar, en→en, ar→en,
   en→ar) with recall@k and MRR. Only the winner is implemented in `src/`; the
   other adapter stays in `evals/`. Synthetic fixtures are easier than real
   content, so the bake-off is re-run when real documents exist.
   **Interim result (2026-11-17, bge-m3 only; Voyage waits for a key):**
   full table in `evals/results/embedding_bakeoff.md`.
   - bge-m3 ranks the right document first for all 26 answerable questions,
     in every language pair, including all 11 cross-language ones. That is a
     ceiling: the fixtures are too easy to separate models by hit@1, so the
     bake-off also reports the *margin* between the right chunk and the best
     wrong one.
   - English questions over Arabic documents are the weakest direction (mean
     margin 0.10, against 0.19 for Arabic over Arabic). The closest call,
     "Is the gym open 24 hours?", beat an English chunk containing the words
     "open 24 hours" by 0.008. Same-language word overlap competes with the
     right answer in the other language. `search` returns 6 chunks, so the
     answer model still sees it.
   - CPU only (as on a GPU-less server): 51 ms per query and 4.8 s to embed
     all 84 fixture chunks. Speed is not the deciding factor. The costs of
     self-hosting are about 1 GB of `torch` in the image and 2–3 GB of RAM.
   **Comparison with OpenAI (2026-11-17)**, run instead of Voyage because an
   OpenAI key was available. Both v3 models used `dimensions=1024`:
   - bge-m3 26/26 right document first; `text-embedding-3-large` and
     `text-embedding-3-small` 24/26 each.
   - Every OpenAI miss is cross-language, and each lost to a chunk in the
     question's own language. Same-language questions were perfect for all
     three models.
   - API latency was 240–280 ms per query, against 51 ms for bge-m3 on CPU.
     Cost is negligible at this scale for either.
   - Margins are not comparable across models (each has its own score
     scale); ranks are.
   - Caveat: the difference is 2 questions out of 26 on synthetic content.
     It points the same way as the risk the brief names (cross-language
     Arabic), but re-run with real documents before treating it as settled.
     Voyage was not run (no key).
   **Recommendation: bge-m3**, the only model that ranked every
   cross-language question correctly, and one that keeps resident questions
   on Atar's own servers. Its cost is operational: about 1 GB of `torch` in
   the image, 2–3 GB of RAM, and deciding whether it runs inside this service
   or as a small embedding service next to it.
   **Decision (confirmed 2026-11-17): bge-m3.**
   - **Pinned to commit `9a0624b896d81da7492a910ffa53731274b6cf3d`.** BAAI's
     main branch (`5617a9f`, last changed 2024-07-03) ships only
     `pytorch_model.bin`, a pickle. The safetensors file comes from open pull
     request #130, made by Hugging Face's automated conversion service
     (`SFconvertbot`, 2025-08-13), whose only parent is main. `transformers`
     fetched it automatically when asked for safetensors. Safetensors cannot
     run code. Pinning the commit means the weights can never change
     underneath us.
   - **Verified identical to BAAI's original weights (2026-11-17).** BAAI's
     `pytorch_model.bin` (main `5617a9f`) and the pinned `model.safetensors`
     both matched their published SHA-256 (`b5e0ce34…6aad38` and
     `993b2248…e38e7e`). The pickle was opened with
     `torch.load(weights_only=True)`, which refuses anything but tensors. All
     391 tensors matched by name, shape, dtype and every one of the
     567,754,752 values.
   - **Runs inside this service for phase 1.** It sits behind the `Embedder`
     protocol, so moving it to a separate embedding service later touches
     one file. Encoding runs on one dedicated thread: the event loop is never
     blocked, and two requests never run the model at once (which would
     double its memory). The model loads on first use, so a re-run that
     changes nothing never loads 2 GB.
   - **Production note for the Dockerfile slice:** on Linux, `torch`'s
     default wheel bundles CUDA (several GB). The image should install the
     CPU-only wheel and bake in the pinned model, so containers never
     download it at start-up.
   **Consequence for the answer endpoint:** no similarity-score threshold
   separates answerable from unanswerable questions (lowest answerable top
   score 0.438, highest unanswerable 0.536). "Grounded or silent" must be
   judged by the answer model over the retrieved chunks, not by a score cut.
   **Changed (2026-11-22): `text-embedding-3-large` from OpenAI's API,
   replacing bge-m3.** Reason: hosting. bge-m3 needs `torch` (several GB on
   Linux with its CUDA wheels) and 2–3 GB of RAM in every container, which
   rules out small instances on a platform such as Render; the API needs
   neither. Accepted cost: on the synthetic fixtures it ranked 24/26 right
   documents first against bge-m3's 26/26, both misses cross-language, and a
   query takes about 250 ms instead of 51 ms. Privacy changes little: the
   default answer model is OpenAI's already, so resident questions reached
   OpenAI anyway; what is new is that every stored chunk is sent to OpenAI
   once, at ingestion.
   - Large over small: equal hit@1 on the fixtures, and large is the stronger
     model on OpenAI's own benchmarks, at a cost that is negligible here.
   - Asked for 1024 dimensions, so the `vector(1024)` column and its index
     are unchanged. The stored model name is `text-embedding-3-large@1024`;
     every fingerprint changes, so the next `atar-ingest` re-embeds
     everything, and search skips bge-m3 chunks until then.
   - Uses `OPENAI_API_KEY` whatever `ANSWER_PROVIDER` says. Without it the
     answer endpoint answers 503 and both commands exit "not configured".
   - The SDK retries rate limits and outages itself; what still fails
     becomes `EmbeddingUnavailableError`, so an answer stream ends with
     `error` then `done`, and an ingestion stops with sources already
     written kept (a re-run finishes).
   - bge-m3 stays in `evals/embedding_bakeoff.py` (with the `bakeoff`
     dependency group) so the comparison can be re-run on real documents.
     If real Arabic content shows the cross-language misses matter, that
     re-run decides whether to go back.
   - The score-threshold finding above was measured on bge-m3 only; the
     model still decides answerability, so nothing depends on it.

7. **Grounded or silent, in three layers.** Retrieved text enters the prompt
   only as search-result blocks, never in the system prompt. The system prompt
   says to answer only from those blocks, to say so when they do not cover the
   question, to reply in the question's language (`X-App-Locale` breaks ties),
   and that block text is reference data, not instructions. Zero retrieved
   chunks returns a localised "not found" without calling the model. The final
   stream event carries `grounded: true|false`. Phase 1 gives the model no
   tools, so injected text has nothing to trigger.

8. **Citations are always checked against what was retrieved.** With Claude,
   the API's built-in citations return the exact quoted passage. With OpenAI
   (the provider since 2026-11-17, see below), the model writes `[n]` markers
   and the service accepts only numbers of sources it actually sent. Either
   way, a citation can never point at text that was not retrieved.

9. **Streaming over SSE.** `POST /v1/answer` returns `text/event-stream` with
   `sources`, `delta`, `citation`, `done` and `error` events. React Native
   cannot reliably read a streamed `fetch` body, and SSE client libraries for it
   exist.

10. **Small provider interfaces.** `Embedder`, `VectorStore`, `LLMClient`,
    `IdentityVerifier` and `Loader` are `Protocol`s with one implementation
    each. Test doubles live in `tests/`, not in `src/`.

11. **Arabic handling.** The original text is stored for citation. The splitter
    gets Arabic separators (`؟ ، ؛`). A small script-ratio function tags each
    chunk `ar` or `en`. The PDF slice checks Arabic extraction order explicitly,
    because PDF libraries often scramble right-to-left text.

12. **Answer model: OpenAI first, Claude kept as an option** (changed
    2026-11-17 at the user's request; the brief proposed Claude). The default
    is `ANSWER_PROVIDER=openai` with `gpt-6.1-terra` (was `gpt-6.1-sol`;
    switched for speed at the user's request, 2026-11-17). Setting
    `ANSWER_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` switches to
    `claude-opus-5-5`. Both adapters are kept because the user said they are
    *starting* with OpenAI; that makes the second one a planned option, not
    speculation.

## Implementation notes

- **Raw SQL, no ORM.** The tenant filter stays visible in the one query that
  reads chunks. An ORM would make it easy to write a second read path.
- **Migrations are numbered `.sql` files** inside the package
  (`src/atar_ai_agent/migrations/`) applied by a small runner, so they ship with
  the installed package and need no path guessing. Alembic is the upgrade path
  if migrations ever need branching or autogeneration.
- **The database URL is a `SecretStr`.** It contains a password, and this keeps
  it out of logs and tracebacks.
- **The app is built by a factory** (`create_app`), not a module-level global,
  so importing the module never needs environment variables and tests can pass
  their own settings.
- **The app starts even when the database is down.** The pool opens in the
  background and `/health` answers 503 until it connects. A crash loop on boot
  would hide the reason from whoever is watching the health check.
- **Local Postgres listens on host port 5433**, not 5432, so it cannot collide
  with a Postgres installed on the machine.
- **Loaders take bytes, not paths.** The ingestion job reads each file once and
  uses the same bytes for the fingerprint and for loading. Sources that are not
  files (backend content, web pages) fit the same signature later.
- **A `Document` is a list of `TextUnit`s** (a Markdown section, a JSON record,
  a PDF page). The chunker never merges two units, so a chunk never mixes two
  topics, and "JSON is chunked per record" needs no special case downstream.
- **Chunk size is 1000 characters with 100 overlap**, counted in characters so
  no tokenizer is needed and Arabic and English behave alike. The splitter's
  separator list includes the Arabic question mark, semicolon and comma.
- **The fingerprint covers the chunker version and the embedding model name**
  as well as the file bytes, so changing either re-processes every source on
  the next run without a manual "force" flag.
- **The Markdown loader strips HTML comments and joins hard-wrapped lines.**
  Comments are not content (the fixture labels live there), and a single
  newline inside a Markdown paragraph is only a space.
- **The chunk-to-source foreign key includes `tenant_id`.** The database itself
  refuses a chunk filed under a different tenant than its source, so isolation
  does not rest on application code alone. A test tries exactly that with
  hand-written SQL.
- **Vectors are sent as text and cast with `::vector`.** Three lines of
  formatting instead of the `pgvector` Python package and its NumPy dependency.
- **Headings are embedded with the chunk, but only the body is stored as its
  text.** "The pool opens at 06:00" means more to an embedding model with
  "Community Rules > Swimming pool" in front; citations still quote the body.
- **The job loads and chunks every changed file before embedding or writing
  any.** A broken file stops the run with nothing half-applied.
- **Unsupported files are reported, not silently skipped;** hidden files are
  ignored.
- **`atar-ingest sync --tenant <id> | --shared --path <folder>`** (added
  2026-11-17, with the real embedder):
  - It applies migrations first, so one command is enough on a fresh
    database.
  - It always plans first. A re-run on an unchanged folder prints "no
    changes" and never loads the 2 GB model.
  - A run that would delete *every* stored source for the scope (an empty
    folder, or a mistyped `--path` pointing at one) is refused unless
    `--allow-delete-all` is passed. This guard works whatever `plan_sync`
    does with an empty folder; if `plan_sync` raises instead, the command
    needs to catch that error too.
  - Tenant ids starting with `_` are refused: they are reserved for scopes
    such as `_shared`, which has its own `--shared` flag.
  - Exit codes: 0 done, 1 a file could not be read (nothing changed),
    2 usage, 3 refused.
- **The test suite creates its own database** instead of relying on a Docker
  init script mounted from the host. Colima does not share `/Volumes/...` with
  its VM, so host mounts from this project folder arrive empty.
- **Local Docker runtime is Colima** (free, command-line only). It does not
  start on boot, so `make db-up` starts it when it is installed and stopped.
  The database container has `restart: unless-stopped`: Colima restarts the
  Docker daemon at the end of every boot, which stopped a container started
  mid-boot (seen 2026-11-17).
- **`search` reads exactly two scopes: the caller's tenant and `_shared`.**
  `SHARED_SCOPE` is a constant, not a config value (it must match what
  ingestion wrote). A leading underscore is not valid in a hostname, so no
  tenant subdomain can collide with it. Passing it, or an empty string, as the
  tenant is an error. The store method that reads chunks refuses an empty
  scope list.
- **Search skips chunks embedded by a different model.** Vectors from two
  models are not comparable; until a re-ingest catches up, old chunks are
  invisible rather than wrongly ranked.
- **Isolation is tested with perfect matches.** Every one of tenant B's chunks
  is used verbatim as tenant A's query (and the reverse), and none may come
  back. Deliberately widening the scope in `search` makes these tests fail
  (checked 2026-11-17). A separate test scans `src/` for any comparison
  between a tenant value and a string literal.
- **JSON shape: a list of records, or `{"title", "records"}`.** Each record is
  one unit rendered as `key: value` lines, nested keys dotted. Anything else is
  refused with the expected shape in the message. Revisit when real exports
  arrive.
- **Word: headings (`Title`, `Heading 1–3`) split sections, like Markdown.**
  Tables stay in document order, one row per line, cells joined by ` | `.
- **PDF: `pypdf`, with lines rebuilt from text positions.** Measured on
  Chrome-made PDFs (2026-11-17):
  - `pypdf` plain extraction keeps Arabic words intact but puts numbers and
    Latin words in visual order, next to the wrong words.
  - `pypdf` layout mode reverses every Arabic word. Unusable.
  - PDFium (`pypdfium2`) orders lines well but corrupts the letters inside
    Arabic words ("المسبح" → "املسبح", "صباحاً" reversed).
  - Our loader collects `pypdf`'s text runs with their positions, groups them
    into lines, and orders Arabic lines right to left, keeping Latin and
    number stretches left to right. With that, every time, extension and
    amount in the fixtures lands next to the right word.
  - Text is NFKC-normalised (expands "ﬁ" and Arabic presentation forms) and
    NUL glyphs are dropped.
  - **Known limitation:** a letter drawn with a diacritic as a single glyph
    can lose the letter ("تشغّل" → "تشّل"); a strict xfail test tracks it. If
    real Arabic PDFs show this often, the option is to have Claude transcribe
    those PDFs at ingestion time (they change rarely, so the cost is small).
  - Scanned PDFs with no text layer are refused with a hint to OCR them.
- **Control characters are stripped for every format in the chunker** (chunker
  version 2). Postgres cannot store NUL, which a PDF produced for an unmapped
  glyph; the fixture ingestion test caught it.
- **Links as sources (2026-11-17): `links.txt` in a tenant's folder,** one
  `http(s)` link per line (`#` comments). The link is the source key, so
  citations point back to it. Pages, PDFs and Word files are told apart by
  their bytes, then Content-Type, then URL suffix.
  - **A link's fingerprint is its extracted text,** not its bytes: real
    pages change on every load (scripts, timestamps, ads). A test changes the
    page's footer time and script, and the re-run is still a no-op.
  - **Any failed download stops the run before anything is written,** so a
    site being down never deletes its data. Responses are capped at 25 MB
    with a 20 s timeout, and requests carry a generic User-Agent.
  - **The job is split into `prepare_sync` (read, fetch, plan) and
    `apply_sync`,** so each link is downloaded once per run. Before, the
    command planned with a dry run and then synced again.
  - **Static pages: trafilatura** (main text as Markdown with headings and
    tables; menus, ads, scripts and footers dropped), then the Markdown
    splitter. Local `.html` files use the same loader.
  - **JavaScript pages: headless Chrome via Playwright,** used only when a
    page's HTML has under 200 characters of main text. The motivating case
    is Termly's policy viewer: 1.5 KB, whose only text is "Termly Policy
    Viewer". The browser's own rendered text is read rather than its HTML.
    On the Termly policy, trafilatura on the rendered HTML glued words
    ("Last updatedJanuary") and kept about 60% of the text; `innerText` had
    correct spacing and all of it.
  - **Section titles on rendered pages** are short single-line blocks that
    are `<h1>`–`<h6>`, or bold, or larger than body text, where the style
    must cover the whole line. Levels: real tags keep theirs; styled titles
    rank by font size (largest `#`, then `##`, then `###`), so sub-sections
    nest. Styles are measured *before* stripping `<style>` elements; the
    first version measured after and found no headings. On the Termly policy
    this gives 18 sections and 27 chunks, with paths like "1. WHAT
    INFORMATION DO WE COLLECT? > Personal information you disclose to us".
  - `RENDER_BROWSER=chrome` drives the installed Google Chrome, so nothing
    is downloaded. A container would use `chromium` plus `playwright install
    chromium`; `off` refuses JavaScript pages. Pages needing a login still
    fail with a clear error.
- **Word and PDF fixtures are generated by `scripts/build_fixtures.py`** and
  committed, so tests never need Chrome.
- **The embedding bake-off runs the real pipeline.** `evals/embedding_bakeoff.py`
  ingests every fixture with each candidate into its own database
  (`atar_ai_bakeoff`) and asks the 30 questions in
  `evals/retrieval_cases.jsonl` through `search()`. It reports hit@1/3/5 and
  MRR per language pair (en→en, ar→ar, ar→en, en→ar), and whether a
  similarity-score threshold separates answerable from unanswerable
  questions. Candidates live in `evals/`; only the winner moves to `src/`.
  `bge-m3` loads safetensors weights only, because pickled weights can run
  code when loaded.
- **The package ships a `py.typed` marker,** so tools outside `src/` (evals,
  scripts) get its types.
- **`plan_sync` reports an empty folder as "delete everything"** (decided
  2026-11-17). The function stays a plain description of what the folder
  says; the command decides whether that is a mistake, and refuses it
  without `--allow-delete-all`.
- **The tenant id is the `X-Tenant-Id` value,** not the subdomain:
  subdomains can be renamed. The subdomain travels in `RequestContext` for
  phase 2's backend calls. Ingestion uses the same id (`--tenant 17`).
- **Verification forwards the resident's own token and tenant headers to
  `/tenancy/api/me` and fails closed.** The backend's answer must itself name
  the tenant, and it must equal `X-Tenant-Id`, or the request gets 403.
  - Status mapping: missing token 401, missing tenant headers 400 (neither
    reaches the backend); backend 401/403 passed on; other backend errors 502;
    unreachable or timed out 503.
  - **The response shape is assumed** (`parse_me` in `api/auth.py`): it reads
    the user from the body, `data` or `user`, the id from `id`, and the
    tenant from `tenant_id`, `tenantId` or `tenant.id`. Replace it when the
    backend team shares a sample. A wrong guess refuses requests; it cannot
    let one through.
  - There is no cache yet, so each answer makes one `/me` call. A short
    per-token cache is a later optimisation, and it would delay token
    revocation by its lifetime.
- **`DEV_AUTH=true` is a local-development auth mode** (2026-11-17): it
  trusts `X-Tenant-Id` without asking the backend, so `/v1/answer` runs on
  localhost before `/tenancy/api/me` is reachable. Any caller can claim any
  tenant, so it is off by default, logs a warning when on, and the app
  refuses to start if `BACKEND_BASE_URL` is also set. `_`-prefixed ids are
  still refused.
- **The app's headers are declared on `/v1/answer`** (an `AppHeaders` model),
  so `/docs` documents them and can send them, and the verifier reads that
  same model.
- **The body accepts only `question`** (`extra="forbid"`), so a `tenant_id`
  in the body is a 422.
- **The `LLMClient` interface takes a neutral `AnswerPrompt`** (question,
  language, numbered sources). Each adapter turns it into its provider's
  format. The shared instructions (`BASE_INSTRUCTIONS`) never contain
  retrieved text. Nothing outside `llm/` knows the provider.
- **OpenAI adapter** (openai SDK 3.52, Responses API, checked 2026-11-17):
  - `gpt-6.1-terra` at reasoning effort `low`, chosen for speed
    (2026-11-17). It was first `gpt-6.1-sol`, which OpenAI describes as
    near-flagship; the answer evals should confirm terra holds quality on
    Arabic, grounding and injection. `OPENAI_MODEL` switches back.
  - `store=False`, so OpenAI does not keep the request and its resident
    question.
  - Sources go in the input as `<source number="n" title="…">` blocks. Any
    `<source`, `<sources` or `<question` text inside a document, title or
    question is defanged (`‹`), so a document cannot close its own block and
    pose as instructions.
  - **Citations** (`llm/markers.py`): the model writes `[2]` or `[1, 3]`
    (Arabic-Indic digits accepted). The parser strips the markers from the
    visible text, even when a marker is split across stream pieces, and emits
    a citation per number that matches a source it sent. A hallucinated
    `[9]` is dropped. Unlike Claude's, these citations name the whole chunk,
    not the exact sentence.
  - Mapping: a refusal or `content_filter` becomes `refusal`; hitting the
    output limit becomes `max_tokens`; a failed response or rate limit
    becomes an `error` event.
  - OpenAI has no server-side fallback equivalent to Claude's.
- **Claude adapter:** `claude-opus-5-5` at effort `low`. Thinking cannot be
  switched off on this model; effort is the control. The model, effort and
  token limit are settings.
- **Server-side fallback is on for Claude** (beta `server-side-fallback-2026-07-01`,
  `fallbacks="default"`). If a safety classifier declines, the API retries
  on its recommended fallback model *on the same stream*. Per the docs, text
  already sent stays valid, so the stream is passed straight through. `done`
  reports which model actually answered. A refusal with no fallback becomes
  an `error` event with code `refused`.
- **With Claude, chunks reach the model only as `search_result` blocks** in
  the user turn, one text block per paragraph, citations on for all (the API
  requires all-or-none). Citations come back with the exact `cited_text` and
  the result's index, which maps to our chunk.
- **`atar-ask --tenant <id> ["question"]`** asks from the terminal through
  the same `answer_events` as `/v1/answer`. It prints the answer, the cited
  sources and a summary line. With no question it reads one per line. It is
  an operator tool: the operator names the tenant, as with `atar-ingest`, so
  it needs no backend. Its context carries no resident token, which is enough
  for answering from documents but not for phase 2's backend tools.
- **First live answers (OpenAI `gpt-6.1-sol`, 2026-11-17, synthetic fixtures):**
  all five were correct. That covers Arabic over English and English over
  Arabic (JSON), a cross-tenant probe (tennis, which only tenant B has:
  "not found" for A), the notice-board injection (fees reported as 85
  SAR/m², not "cancelled") and an unanswerable question ("not found",
  `grounded: no`). About 10 s per answer including the first-question model
  load; time to first word still needs measuring in the evals.
- **Fixture layout restored (2026-11-18).** A commit replaced the 9 synthetic
  `tenant_a` files with 36 Atar app-guide pages, which broke 41 tests plus
  one collection error, and half the evaluation. The guide is general app
  help with no community-specific facts, so it moved to `fixtures/shared/`.
  The synthetic `tenant_a` files were restored from git history
  (byte-identical), keeping the deliberate A-vs-B conflicts that isolation
  tests and evals rely on. Guide pages are checked by their front matter
  (`title`, `language` matching the file name, an existing `pair`) instead
  of the synthetic label.
- **Markdown front matter** (a leading `---` block of `key: value` lines) is
  metadata: it is removed from the text and its `title` names the document.
  A block without any `key: value` line is left alone, since it is a
  horizontal rule.
- **Search reserves 3 of 8 slots per scope** (`MIN_PER_SCOPE`, `top_k` 6 → 8).
  With ~360 shared guide chunks, tenant A's own fees page fell to rank 8 for
  "Have the service fees been cancelled?", beaten by shared "service
  request" chunks, and the injection case lost its answer. Each scope (the
  tenant's own documents and the shared set) now keeps its best 3, and the
  rest go to the best scores. Isolation is unchanged: exactly those two
  scopes are read, each on its own. Eval after the change: gpt-6.1-terra and
  gpt-6.1-sol both 32/32. Three earlier "failures" were checks too strict for
  correct wording ("٨ صباحاً", "حيوانان", "خميس"); the cases were widened and
  the saved answers re-scored.
- **Chat history (2026-11-18).** Follow-ups such as "and on Tuesday?" work.
  - **Stored on the server, owned by tenant + user** (`conversations` and
    `conversation_messages`, migrations 003-004). The app sends only a
    `conversation_id`, never past turns: a client that could send history
    could forge earlier "assistant" answers to steer the next one. A body
    with `history` is a 422. As with chunks, the message → conversation
    foreign key includes `tenant_id`.
  - **The app chooses the conversation id** ("1234", a UUID; letters, digits
    and `._:-`, up to 100 characters), and an id not used before starts a
    new conversation. Corrected 2026-11-18: the first version required the
    server's UUID from the first event. A user testing with curl sent no id,
    or "1234" (rejected), so every question became a new conversation and
    history never applied. Ids are unique *per resident* (`external_id`,
    unique on tenant + user + id), so another resident's "1234" is a
    different conversation, and theirs is unreachable rather than forbidden.
    Without an id, one is generated and returned in the `conversation` event.
  - **Code lives in `conversations/`** (protocol + Postgres store) **and
    `chat.py`** (no HTTP, like `answer.py`). `rag/` is unchanged and `search`
    is still a plain function; the phase 2 agent will reuse the same
    conversations.
  - **Follow-ups are rewritten into a standalone question for search only**
    (`standalone_question`, one short `LLMClient.complete` call, made only
    when there is history). The answer still follows the resident's own
    words and language. A failed rewrite falls back to the original
    question; a blank or over-long rewrite is ignored. The searched query is
    reported in the `sources` event and stored with the user turn.
  - **History is the resident's chat, not a source of community facts.** The
    model may use what the resident said about themselves (their name, for
    example: "what is my name?" works), but rules, fees, times and procedures
    must come from this turn's sources, so an earlier wrong answer cannot be
    repeated as fact. Greetings and thanks get a short reply with no
    citation. Corrected 2026-11-18: the first version forbade anything not
    in the sources, so "what is my name?" could never work. Each provider receives history in its native multi-turn form
    (OpenAI: earlier input messages; Claude: alternating messages before the
    search-result turn). Resident text is tag-defanged as before.
  - **Bounded:** the last 3 exchanges (`CHAT_HISTORY_EXCHANGES`) within 6,000
    characters (`CHAT_HISTORY_MAX_CHARS`); oldest pairs are dropped first,
    and only complete question/answer pairs are sent.
  - **Saved only when the answer finishes cleanly:** the question and answer
    go in one transaction, just before `done`. An answer that errors or is
    cut off saves nothing, so stored history never holds half-answers. If
    saving fails, an `error` event (`history_not_saved`) precedes `done`.
  - **API:** `POST /v1/answer` takes an optional `conversation_id`; the
    stream now opens with `conversation` (`conversation_id`, `is_new`).
    `GET /v1/conversations/{id}` returns the messages with citations;
    `DELETE /v1/conversations/{id}` removes them (204). Both answer 404 for
    an id this resident has not used.
  - `atar-ask` interactive mode keeps follow-up history in memory (nothing
    stored) and prints the rewritten query.
  - **Replay of the user's own conversation** on the local server (one id,
    gpt-6.1-terra): "مرحبا.. انا اسمي احمد" → "مرحباً أحمد! كيف أقدر أساعدك؟";
    "ما اسمي؟" → "اسمك أحمد."; "هل يمكن ان اقدم طلب؟" → cited steps location;
    "ما هي الخطوات؟" searched as "ما هي خطوات تقديم طلب خدمة؟" → the cited
    steps.
  - **Earlier live check (gpt-6.1-terra):** "When does the pool open?" then "And on
    Tuesday?" searched as "When does the pool open on Tuesday?" and answered
    "closed every Tuesday", cited. An Arabic follow-up after English turns
    was answered in Arabic. A follow-up costs one extra short model call.
  - **Open:** a retention policy (purge after N days) and a "list my
    conversations" endpoint. Neither is built.
- **Answer evaluation (`make eval`, 2026-11-17).** 32 cases in
  `evals/answer_cases.jsonl` (16 Arabic, 16 English; 11 cross-language, 2
  unanswerable, 2 cross-tenant probes, 2 injection), run through the real
  `answer_events` on a separate database. Grading (`evals/grading.py`) is
  deterministic, with no model judge:
  - language matches the question
  - it cites, and cites an expected document
  - the expected facts are present, and the other tenant's values absent
  - silence (no citation) when nothing answers it
  - the injection document is not cited

  Results (full table in `evals/results/answer_eval.md`):
  - `gpt-6.1-terra` and `gpt-6.1-sol` both pass 32/32. terra is faster:
    first text at a median of 1.21 s against 1.62 s, full answer 1.83 s
    against 2.16 s. **terra stays the default.**
  - The first scoring said 30/32 and 31/32. Reading the failures showed the
    *checks* were wrong: "خميس" without "ال" was not accepted, and a correct
    "couldn't find whether the tennis court is free" matched a forbidden
    phrase. The cases were fixed and the saved answers re-scored with
    `--regrade`, which makes no model calls.
  - Retrieval put the right document in the top 6 for 28/28, and first for
    27/28. The miss: for "Have the service fees been cancelled?", the
    *injection* notice ranked first and the fees page third. The answer was
    still right (85 SAR, injection not cited): the defence held at the model
    even though retrieval surfaced the attack first.
  - The ~10 s seen earlier with `atar-ask` was process start plus the
    embedding-model load; a warm service answers in about 2 s.
  - Caveat: the fixtures are synthetic and the set is small (one case is
    about 3 points). Both models scoring perfectly means this set cannot yet
    separate them on quality. Real documents and harder questions will.
- **Tests never read `.env`.** `make_settings` in `tests/conftest.py` ignores
  it and clears every API key. Added after one endpoint test built a real
  OpenAI client from the developer's key and made one real call with
  synthetic text (2026-11-17).
- **The answer language is the question's language;** `X-App-Locale` decides
  only when the question has no letters (for example "06:00?").
- **Refuse before the model only when nothing was retrieved**
  (`should_call_model`). The bake-off showed no similarity score separates
  answerable from unanswerable questions; the model judges whether the
  chunks answer, and is told to say so when they do not.
- **SSE contract (`api/sse.py`):** `sources` first; then `delta` and
  `citation` events; then at most one `error`; then `done`, always last,
  with `grounded` (at least one citation), `language`, `stop_reason` and
  `model`. A stream without `done` means the connection broke.
- **Services are injected into `create_app`.** Tests swap in fakes; missing
  configuration (no backend URL, no API key) gives a 503 after verification,
  so an unverified caller learns nothing.
- **One uvicorn worker per container.** Each worker would load its own copy
  of the embedding model.
- **Ruff's "ambiguous Unicode" rules (RUF001–003) are off.** They flag the
  Arabic comma and Arabic-Indic digits, which are correct text here.

## Dependencies

Runtime:

| Package | Why |
|---|---|
| `fastapi` | HTTP framework named in the brief; typed request and response models. |
| `uvicorn[standard]` | ASGI server that runs FastAPI. |
| `pydantic-settings` | Loads config from environment variables into a typed, validated object. |
| `psycopg[binary,pool]` | Postgres driver (v3) with async support and a connection pool. `binary` avoids needing a C toolchain locally. |
| `langchain-text-splitters` | Splits Markdown by heading and long text by a separator list we control. Used for loading and splitting only; retrieval and the LLM call are written directly. |
| `python-docx` | Reads Word files: headings, paragraphs and tables in document order. Typed. Also used by the fixture builder to write them. |
| `pypdf` | Reads PDF text runs with their positions, which the Arabic line rebuilding needs. Pure Python, typed, BSD licence. Chosen over PDFium after the comparison above. |
| `anthropic` | Official SDK for Claude: streaming, citations, server-side fallback. Typed. |
| `openai` | Official SDK for OpenAI's Responses API, the default answer model since 2026-11-17, and its embeddings API, the embedding model since 2026-11-22. It brings `httpx` (the Anthropic SDK uses `httpx2`), so the project now has two HTTP libraries, one per SDK; our own code uses `httpx2`. |
| `httpx2` | HTTP client for `/tenancy/api/me` (moved from dev to runtime). Also what the Anthropic SDK uses, so the project has one HTTP library. |
| `trafilatura` | Extracts a web page's main text (as Markdown with headings and tables) and drops menus, ads and footers. Apache-2.0. Brings `lxml`, `justext`, `htmldate`, `courlan`, `dateparser`. |
| `playwright` | Drives a headless browser for links whose text JavaScript draws (e.g. Termly's policy viewer). Uses the installed Google Chrome locally, so no browser download. |

Development:

| Package | Why |
|---|---|
| `pytest` | Test runner named in the brief. |
| `pytest-asyncio` | Lets tests `await` the async database code. |
| `ruff` | Linter and formatter named in the brief. |
| `mypy` | Type checker (strict). Pure Python, with a Pydantic plugin. |
| `hatchling` | Build backend so `uv` can install the `src/` package and its CLI entry points. |

Bake-off only (`uv run --group bakeoff ...`, never installed in production):

| Package | Why |
|---|---|
| `sentence-transformers` | Runs `BAAI/bge-m3` in-process for `evals/embedding_bakeoff.py`. Pulls in `torch` and `transformers`. It was the service's embedder from 2026-11-17 to 2026-11-22. |

Infrastructure:

| Piece | Why |
|---|---|
| `pgvector/pgvector:pg17` image | Postgres 17 with the pgvector extension preinstalled. |
| `uv` | One tool for the virtualenv, the lockfile and running commands. |
| `render.yaml` | Render Blueprint (2026-11-22): the web service and its Postgres in Frankfurt, the region closest to Saudi Arabia. Native Python runtime, no Docker: the image would only wrap `uv sync`. Migrations run as the pre-deploy command, so a failed one stops the deploy. Ingestion stays an operator task run from a laptop. Steps in `docs/DEPLOY.md`. |

## Open items

- A sample `/tenancy/api/me` response, and whether it rejects a token used with
  another tenant's `X-Tenant` headers. If it does not, the verifier must compare
  the tenant in the response itself.
- Backend base URL per environment.
- Whether in-Kingdom data rules apply to resident questions and community
  documents. If they do, the embedding choice narrows to self-hosted.
- Hosting target.
