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
- Multi-stage `Dockerfile` producing a 404 MB image that runs as uid 10001, and a Compose
  stack with Postgres. No healthcheck is declared yet; there is no endpoint to call.
- GitHub Actions CI: ruff, ruff-format, mypy, pytest on Python 3.12 and 3.13, `pip-audit`,
  and a Docker build that asserts the image runs non-root. Fake lane throughout, no secrets.

- Deterministic fake lane (`agentgate.models.fake`) with scriptable replies, scriptable
  failures, and honest `usage_metadata`. The whole suite and all of CI run on it.
- Cost controls in config: per-call-class output ceilings, per-run and per-session spend
  ceilings, and a per-model price table. A networked lane with an unpriced model refuses to
  start rather than treating unknown cost as zero.

- Lane registry and capability matrix. Every entry records how it was learned (live probe,
  stub, in-process, or operator declaration) and when; the suite fails if any entry on a
  networked lane rests on assumption. An unmeasured capability reads as unsupported.
- Structured output with a validate-and-repair fallback for lanes lacking native support,
  proven against a committed OpenAI-compatible stub that returns prose-wrapped JSON.
- `make models` lists the identifiers a key can reach and emits a zeroed, paste-ready price
  table. It states plainly that the API exposes no pricing, and infers nothing from a name.

- Tracing configuration: `AGENTGATE_TRACING_BACKEND` selects `none` (default), `langsmith`,
  or `otlp`. OpenTelemetry is the instrumentation in every case; the backend is only the
  exporter behind it. Off by default, and a backend selected without its destination is a
  startup error. Design recorded in ADR 0008; implementation lands in Phase 8.

- Spend ledger with per-run and per-session ceilings, accounting from `usage_metadata`. A
  reply without usage is an error rather than a free call.
- `make test-live` estimates the cost, asks for confirmation, and aborts if actual spend
  exceeds the estimate by more than a configurable factor. Five live cases, deselected by
  default and never run in CI.
- `docs/concept-map.md`, maintained as the build proceeds, and ADR 0004 carrying the leak
  inventory: what is known to differ between lanes and how each difference was established.

- Typed graph state with reducers on the channels that fan out, and deliberately without one
  on the channels a single node writes.
- The core graph: classify, a policy gate as a conditional edge returning a `Literal`, lane
  binding, a supervisor returning `Command`, a budget guard, and finalisation. Checkpointer
  chosen by configuration across in-memory, SQLite, and Postgres.
- Append-only audit events recording what decided, what it decided, and on what input hash --
  never on the input itself.
- `make measure` derives the token ceiling from an instrumented run; the spend ceilings follow
  from that budget priced at the configured table. Both bases are recorded in `.env.example`.

- The live suite has its own token and spend ceilings, on their own basis: the gatekeeper's
  estimate times the tolerance, which is the bound it already applies to dollars, applied to
  tokens as well. A suite is not a run, and charging it to a per-run budget aborts it for
  being a suite. `make test-live` now prints estimated against actual tokens alongside
  dollars, so the figure the ceiling rests on stays observed.
### Fixed

- `make test-live` could not start. The gatekeeper set two `AGENTGATE_*` variables on the
  pytest subprocess that were not declared settings, and the unknown-variable guard rejected
  them, so every live case failed at configuration before reaching a provider. Both are
  declared settings now. Recorded as item 7 of the leak inventory in ADR 0004.
- `AGENTGATE_LIVE_SPEND_ABORT_USD` was computed, printed, and read by nothing. It now bounds
  the suite while it runs, rather than only being compared against the total afterwards.

### Changed

- `SpendLedger` requires the ceilings it enforces rather than reading the run ceilings off
  configuration. A ledger that inferred its own scope is how the live suite came to be
  measured against a per-run budget. See item 8 of the leak inventory in ADR 0004.

- Configuration tolerates unrelated keys in a shared `.env` rather than rejecting them.
  Typo protection for the `AGENTGATE_` namespace is unchanged and remains stricter than
  `extra="forbid"` ever was. Every unprefixed environment name the application reads is now
  declared explicitly and pinned by a test that asserts no field consumes an undeclared one.
  See ADR 0009.
- Configuration is validated on an explicit `get_settings()` call at each entry point
  rather than as a side effect of importing `agentgate.config`. The startup guarantee is
  unchanged; importing the module now has no side effects and cannot raise. See ADR 0007.

[Unreleased]: https://github.com/HamzaAhmedWajeeh/agentgate/commits/main/
