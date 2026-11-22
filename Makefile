.PHONY: check lint format typecheck test db-up db-down migrate ingest-fixtures eval dev

# Everything that must pass before a slice is done.
check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

# Starts Colima first if it is installed and stopped (it does not survive a
# reboot), then the database. Other Docker runtimes are left alone.
db-up:
	@if command -v colima >/dev/null 2>&1 && ! colima status >/dev/null 2>&1; then colima start; fi
	docker compose up -d --wait

db-down:
	docker compose down

migrate:
	uv run atar-migrate

# The synthetic fixtures, into the dev database (needs OPENAI_API_KEY). Safe to
# re-run: unchanged folders report "no changes" without calling the embedding model.
ingest-fixtures:
	uv run atar-ingest sync --tenant tenant_a --path fixtures/tenant_a
	uv run atar-ingest sync --tenant tenant_b --path fixtures/tenant_b
	uv run atar-ingest sync --shared --path fixtures/shared

# Ask every eval question through the real pipeline and grade the answers
# (calls the configured answer model). Compare models with:
#   make eval MODELS="gpt-6.1-terra gpt-6.1-sol"
eval:
	uv run python evals/answer_eval.py $(if $(MODELS),--models $(MODELS))

dev:
	uv run uvicorn atar_ai_agent.api.app:create_app --factory --reload --port 8000
