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

[Unreleased]: https://github.com/HamzaAhmedWajeeh/agentgate/commits/main/
