# Task runner. Windows has no make by default, so make.ps1 mirrors every target here.
# If you add a target, add it there too -- tests/unit/test_task_runner.py enforces it.

UV ?= uv
RUN := $(UV) run

.DEFAULT_GOAL := help
.PHONY: help setup lint format format-check typecheck test test-cov test-live check audit config docker-build docker-up docker-down docker-logs clean

help: ## List available targets
	@grep -hE '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Install dependencies and git hooks
	$(UV) sync --all-groups
	$(RUN) pre-commit install

lint: ## Check style and common defects
	$(RUN) ruff check src tests

format: ## Rewrite files to the project format
	$(RUN) ruff format src tests

format-check: ## Fail if any file is unformatted
	$(RUN) ruff format --check src tests

typecheck: ## Type check src in strict mode
	$(RUN) mypy

test: ## Run the offline suite (no API key required)
	$(RUN) pytest

test-cov: ## Run the offline suite with a coverage report
	$(RUN) pytest --cov --cov-report=term-missing

test-live: ## Run the suite against real providers. Costs money.
	$(RUN) pytest -m live

check: lint format-check typecheck test ## Everything CI runs

audit: ## Check installed dependencies for known vulnerabilities
	$(RUN) pip-audit

config: ## Print the resolved configuration, secrets redacted
	$(RUN) python -m agentgate

docker-build: ## Build the container image
	docker compose build

docker-up: ## Start the stack
	docker compose up -d

docker-down: ## Stop the stack and remove volumes
	docker compose down -v

docker-logs: ## Follow stack logs
	docker compose logs -f

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
