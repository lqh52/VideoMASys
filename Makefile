.PHONY: install db-up db-down init-db api worker test lint

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

init-db:
	.venv/bin/videomasys-init-db

api:
	.venv/bin/videomasys-api

worker:
	.venv/bin/videomasys-worker

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

