# Task runner. Windows has no make by default, so make.ps1 mirrors every target here.
# If you add a target, add it there too -- tests/unit/test_task_runner.py enforces it.

UV ?= uv
RUN := $(UV) run

# pip-audit is pointed at the exported lockfile rather than the installed environment.
# Auditing the environment means auditing agentgate itself, which is not on PyPI, and
# --strict treats an unauditable distribution as a failure.
AUDIT_REQUIREMENTS := .audit-requirements.txt

.DEFAULT_GOAL := help
.PHONY: help setup lint format format-check typecheck test test-cov test-live check audit models measure config docker-build docker-up docker-down docker-logs clean

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

test-live: ## Run the suite against real providers. Estimates, confirms, then enforces.
	$(RUN) python scripts/run_live.py

check: lint format-check typecheck test ## Everything CI runs

audit: ## Check locked dependencies for known vulnerabilities
	$(UV) export --all-groups --no-emit-project --no-hashes --format requirements-txt \
		-o $(AUDIT_REQUIREMENTS) --quiet
	$(RUN) pip-audit --strict -r $(AUDIT_REQUIREMENTS)

models: ## List model identifiers this key can reach, with a price-table skeleton
	$(RUN) python -m agentgate.models.catalogue

measure: ## Measure what one full run consumes, to derive the token ceiling
	$(RUN) python scripts/measure_run.py

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
