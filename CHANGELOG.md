# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `src/` layout, packaged with hatchling, exposing a typed `agentgate` distribution.
- Pinned dependency set: LangGraph 1.2.10 and LangChain 1.3.14 for orchestration, both SQLite
  and Postgres checkpointers, FastAPI and Typer surfaces, and the structlog / OpenTelemetry /
  Prometheus observability stack.
- MIT license and a README carrying the project thesis.
- Lint, format, and type configuration: ruff with a bugbear/bandit/pathlib rule set, mypy in
  strict mode over `src/`, and pytest with a `live` marker deselected by default.
- Pre-commit hooks mirroring the CI gate, including `detect-secrets` against a committed
  baseline.
- `agentgate.config`: the single source of every tunable, validated at import so a broken
  environment fails at startup with a message naming the variable at fault. Defaults are
  offline and free -- the `fake` lane with an in-memory checkpointer -- so reaching a real
  provider is opt-in. Unrecognised `AGENTGATE_*` variables are rejected with a spelling
  suggestion rather than silently ignored.
- `python -m agentgate` prints the resolved configuration as JSON with every secret masked,
  exiting non-zero when the environment does not describe a runnable system.
- `Makefile` with a `make.ps1` shim exposing the same targets on Windows, so the documented
  commands work on every machine the project is developed on.

[Unreleased]: https://github.com/HamzaAhmedWajeeh/agentgate/commits/main/
