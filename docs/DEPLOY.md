# Deploying to Render

`render.yaml` describes everything: one web service and one Postgres
database, both in Frankfurt. No Docker: Render's native Python runtime builds
the service with `uv`.

## First deploy

1. Push the branch Render should deploy (`branch:` in `render.yaml`, `main`
   by default) to GitHub.
2. Render dashboard -> **New** -> **Blueprint** -> pick this repository.
   Render reads `render.yaml` and lists the service and the database.
3. Render asks for the values marked `sync: false`:
   - `OPENAI_API_KEY`: the embedding model's key, and the answer model's.
   - `BACKEND_BASE_URL`: the Atar backend, without a trailing path
     (residents are verified against `{BACKEND_BASE_URL}/tenancy/api/me`).
4. **Apply**. Render creates the database, builds the service, runs
   `atar-migrate` (pre-deploy), then starts it.
5. Check `https://<service>.onrender.com/health` answers `{"status": "ok", "database": "ok"}`.

From then on, every push to that branch deploys again. Migrations run on
every deploy and are skipped when there is nothing new.

## Loading documents

Ingestion runs from a laptop, not on the server (it may need a browser, and
it is an operator task). Point it at the Render database for one command:

1. Render dashboard -> the database -> **Connect** -> copy the
   **External Database URL**.
2. Run, with that URL (an environment variable wins over `.env`):

   ```bash
   DATABASE_URL="<external url>" uv run atar-ingest sync --tenant 17 --path data/17
   ```

   The tenant id is the backend's `tenant_id` for that community. Documents
   every community shares go in with `--shared` instead of `--tenant`. Use
   `--dry-run` first to see what would change.
3. Try it without the app:

   ```bash
   DATABASE_URL="<external url>" uv run atar-ask --tenant 17 "متى يفتح المسبح؟"
   ```

The external URL contains the database password: paste it into the command
only, never into a file in the repository.

## What runs where

| Setting | Value | Why |
|---|---|---|
| Region | Frankfurt | Closest Render region to Saudi Arabia. |
| Web plan | starter | Free instances sleep when idle (the first question then waits for a cold start) and cannot run a pre-deploy command. |
| Database plan | basic-256mb | Free databases expire after 30 days. Tens of documents per tenant fit easily. |
| Workers | 1 | Concurrency comes from async; see `api/app.py`. |
| `RENDER_BROWSER` | off | Only ingestion uses it, and ingestion does not run here. |
| `DEV_AUTH` | not set | It trusts any `X-Tenant-Id` header; the app refuses to start with it and `BACKEND_BASE_URL` together. |

Plans can be changed in the dashboard at any time without redeploying from
scratch.

## When something fails

- **Build fails on `uv sync --frozen`**: `uv.lock` is out of date. Run
  `uv lock` locally and push it.
- **Deploy stops at pre-deploy**: a migration failed; its error is in the
  deploy log. Nothing new went live.
- **`/health` answers 503**: the service cannot reach the database. Check
  that `DATABASE_URL` comes from the database (`fromDatabase` in
  `render.yaml`) and that both are in the same region.
- **`/v1/answer` answers 503 "not configured"**: `OPENAI_API_KEY` or
  `BACKEND_BASE_URL` is missing in the service's Environment tab.
- **Answers say nothing was found**: no documents for that tenant in the
  Render database yet; see "Loading documents".
