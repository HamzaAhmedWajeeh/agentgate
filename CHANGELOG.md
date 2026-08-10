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
- The cloud lane's first capability-matrix row, from a live probe against a real key on
  2026-08-10: native structured output is supported. The live suite enforces the row.

- A committed corpus of four synthetic documents describing an organisation that does not
  exist, chunked on Markdown headings because a heading is the author's own statement about
  where one idea ends. `make seed` indexes it and prints what sample queries retrieve.
- Dense in-process retrieval: embeddings chosen by lane like models, an exhaustive cosine
  search written rather than imported, and Qdrant declared in configuration but raising rather
  than silently falling back. Recorded in ADR 0010, including the two things about the offline
  embedder that were found by running it — a 70% hash-collision rate at the first dimension
  chosen, and `hash()` being salted per process.

- Research fan-out: the supervisor dispatches sub-questions to a compiled retrieval subgraph
  with one `Send` per question, and each branch hands its finding back with
  `Command(graph=Command.PARENT)`. Fan-in is the parent's `operator.add` reducer, so it is a
  property of the state schema rather than of any collecting code.
- `AGENTGATE_MAX_FAN_OUT` caps how many branches one dispatch may open, enforced where `Send`
  objects are constructed. This is the only budget decided before the spending rather than
  counted after it: the list being fanned out over is model output, so without it the model
  chooses how many calls get paid for.
- A branch that fails is caught, recorded, and does not take its siblings down with it, and
  the run that results is marked `answer_complete: false` rather than presenting a partial
  answer in the shape of a whole one. `dispatched` is compared against the outcomes so a
  branch that reports nothing at all is still counted as missing.

- A drafter worker built with `create_agent` — the one prebuilt agent in the system, so the
  repository shows the fast path as well as the explicit one, and shows what it costs: the
  model-tool loop is not visible in `build.py`, which is exactly why the allowlist is
  middleware rather than a list of bound tools.
- Per-agent tool allowlists enforced in `wrap_tool_call`, between the model's request and the
  executor. The drafter cannot reach an irreversible tool — not "does not": a model scripted
  to demand `issue_refund` is refused before the handler runs, and the refusal is an audit
  event. No part of the enforcement is in a prompt, and a test asserts the prompt stays out of
  it. Tool failures are summarised back to the model rather than raised.
- Ceilings re-derived now that fan-out exists: `AGENTGATE_MAX_TOTAL_TOKENS` moves from 2,370 to
  19,200, from a measured heaviest run of 1,920 tokens at the fan-out limit. Spend ceilings
  follow. The Phase 3 note predicted the old ceiling would reject every Phase 4 run; it would
  not have, and what it recorded instead is in `.env.example`.
- Recorded, not fixed: the spend guard accounts chat-model usage and cannot see embedding
  spend, which is free on the fake lane and real on the cloud lane. ADR 0004, item 9.

- The human gate: `interrupt()` called from inside the node, resumed with `Command(resume=...)`.
  Nothing above the pause has a side effect, because resume re-executes the node from its top —
  proven by observation rather than quoted, with a counter watched going up on every resume.
- Reject-with-feedback returns the draft to the drafter and comes back to the gate. That loop
  is the first thing in this system that can fail to stop on its own, so the iteration cap is
  now exercised end to end against a reviewer who never approves: the run terminates on the
  budget and records `budget_exceeded` as the reason.
- `execute` is reachable only past the approved branch, and checks the decision on state as
  well. The topology is true until someone draws another edge; the node's own check is not.

- State channels hold JSON-serialisable data only. A checkpoint is a persistence format, not
  an in-process value: it outlives the process, the deploy, and under Postgres the container,
  so anything crossing that boundary is a wire format with a schema. `Finding`,
  `Classification` and `ResearchOutcome` stay as parse-and-serialise helpers at node
  boundaries. Recorded in ADR 0011, including why registering the types was the wrong half of
  the problem to solve and why the pytest configuration this was meant to use does not exist.

- Recorded, not fixed: `LANGGRAPH_STRICT_MSGPACK=true` is a control that makes things worse
  when enabled. It does not raise on an unregistered type in a checkpoint — it drops the value
  and lets the run continue, turning a visible notice into silent data loss. ADR 0004, item 10.

- An output guardrail on citation provenance: every source the draft cites must be one
  research actually returned, and a fabricated citation is an audit event. Exact rather than
  heuristic on purpose — a guardrail that is right most of the time converts "we do not check
  this" into "we check this", and the second is false in the cases that matter. It cannot see
  an uncited fabrication, and says so.
- `docs/concept-map.md` is now held to the repository by a test, in both directions, after it
  spent a phase claiming five concepts were simultaneously built and not built.

### Fixed

- `make test-live` could not start. The gatekeeper set two `AGENTGATE_*` variables on the
  pytest subprocess that were not declared settings, and the unknown-variable guard rejected
  them, so every live case failed at configuration before reaching a provider. Both are
  declared settings now. Recorded as item 7 of the leak inventory in ADR 0004.
- The retrieval corpus was not copied into the container image. The runtime stage ships the
  virtualenv, which covers code and not data, so every research branch would have failed
  inside the container while every offline test passed — the suite runs from a checkout where
  `corpus/` is simply there. Copied now, and pinned by a static check of the runtime stage.
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
