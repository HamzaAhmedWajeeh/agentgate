# agentgate

[![CI](https://github.com/HamzaAhmedWajeeh/agentgate/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/HamzaAhmedWajeeh/agentgate/actions/workflows/ci.yml)
[![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Every agent action passes a gate. A policy gate decides which model tier may see the data.
> Budget gates cap iterations, tokens, and spend. A human gate approves anything irreversible.
> All three write to an append-only audit trail.

An agent runtime for environments where an autonomous action has consequences: the model tier a
request reaches is a policy decision, the number of steps it may take is a budget decision, and
anything irreversible is a human decision. `agentgate` makes all three explicit, enforces them in
the graph rather than in a prompt, and records every one of them.

Built on LangGraph and LangChain. The demo domain is a compliance-aware research and drafting
workflow.

## Status

**The build is paused at Phase 5 of nine.** Phases 6 to 9 — long-term memory and time travel,
the API and CLI surfaces, production hardening, and the eval suite — are planned and unstarted.

This README documents only what is covered by a test or by a check that has actually been run.
Nothing here is aspirational, and nothing below is a promise about what will exist.

**Working today**

- Configuration: every tunable in one module, validated at import, failing at startup with a
  message that names the variable at fault rather than a provider stack trace mid-run.
- A safe-by-default posture: an unconfigured process comes up on a deterministic offline lane
  with an in-memory checkpointer, so the test suite needs no API key and CI needs no secrets.
- `python -m agentgate` prints the resolved configuration with every secret masked.
- Toolchain: ruff, mypy in strict mode, pytest, pre-commit, and CI running all of it.

- Three model lanes behind one interface, with a capability matrix in which every entry records
  how it was learned and when. The suite fails if any entry about a networked lane rests on
  assumption rather than measurement.
- Structured output with a validate-and-repair fallback, proven against a committed
  OpenAI-compatible stub server that returns prose-wrapped JSON the way self-hosted endpoints
  actually do.
- `make models` lists the model identifiers a key can reach. It cannot tell you what they cost,
  and says so: that API exposes no pricing, and no price is ever inferred from a name.

- The graph: classification, a policy gate that routes restricted content to the sovereign lane,
  research fanned out over a compiled retrieval subgraph, a drafting worker, and a human
  approval gate that pauses before anything irreversible. Rejection returns the draft for
  revision, and a reviewer who never approves is stopped by the iteration budget rather than by
  giving up.
- Dense in-process retrieval over a committed corpus of synthetic documents. No service is
  needed to run it and no key is needed to test it.
- Per-agent tool allowlists enforced between the model's request and the executor, not by the
  prompt. A model scripted to demand an irreversible tool is refused before the handler runs.
- Budget gates on iterations, tokens, and fan-out width. The width cap is the only one decided
  before the spending rather than counted after it, because the list being fanned out over is
  model output.
- An append-only audit trail in graph state, recording what decided, what it decided, and on
  what input hash — never on the input itself.

**Not built.** Long-term memory and time travel, the FastAPI and CLI surfaces, streaming,
structlog/OpenTelemetry/Prometheus instrumentation, and the eval suite. `docs/concept-map.md`
lists every concept and marks each one built or not built; a row there describes the repository
as it is today.

**Built but not yet wired.** The spend ledger enforces run and session ceilings and is exercised
by the live suite, but nothing in the graph accounts against it yet. Until it is, the token and
spend ceilings bound what `make measure` derives rather than what a run actually consumes.

**Not accounted at all.** Embedding calls. The ledger reads chat-model usage metadata, and
embeddings do not produce it, so on the cloud lane indexing and querying the corpus costs money
no ceiling here can see. Recorded as item 9 of the leak inventory in
[ADR 0004](docs/adr/0004-provider-abstraction-and-lanes.md); the decision is that the run budget
means all spend, and the code has not caught up with it.

**Deferred verification.** The sovereign lane is exercised against the committed stub server
over real HTTP. Ollama and vLLM are the intended targets and neither has been run against yet,
so neither is claimed. The cloud lane has one live-probed capability row; tool calling and
embeddings have never been watched against a real provider. The same applies to observability:
LangSmith and OTLP are designed for and unverified, and until each has actually run, neither is
claimed either.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Windows users need neither `make`
nor WSL — `./make.ps1 <target>` exposes the same targets.

```bash
git clone https://github.com/HamzaAhmedWajeeh/agentgate.git
cd agentgate
cp .env.example .env
make setup
```

Confirm the toolchain, offline and with no API key:

```bash
make check      # lint, format check, type check, tests
make config     # the configuration this process would run under
```

`.env.example` is shipped set to the fake lane, so a verbatim copy runs with no credentials and
no cost. Switching to a real provider is an explicit edit, described in the file itself.

### Task reference

| Target | What it does |
| --- | --- |
| `make setup` | Install the locked dependency set and git hooks |
| `make check` | Everything CI runs: lint, format check, type check, tests |
| `make test` | The offline suite |
| `make test-cov` | The offline suite with a coverage report |
| `make test-live` | Run against real providers. Costs money. |
| `make config` | Print the resolved configuration, secrets redacted |
| `make audit` | Check dependencies for known vulnerabilities |
| `make docker-build` | Build the container image |
| `make docker-up` | Start the local stack |

## Configuration

Everything is driven by environment variables read in `src/agentgate/config.py`, and every one
of them is documented in [`.env.example`](.env.example). There are no feature flags anywhere
else and no model identifiers in application code.

Two behaviours are worth knowing about:

**Defaults cannot cost you anything.** The default lane is `fake`, the default checkpointer is
in-memory. Reaching a real provider is something you opt into.

**A misspelled variable is a startup error.** `AGENTGATE_MAX_ITERATION` does not silently do
nothing; it stops the process and suggests `AGENTGATE_MAX_ITERATIONS`. The failure mode this
prevents is an operator believing a budget is in force while the process runs on defaults.

**No model identifier ships with a default.** Which models a given key can actually reach is not
something this repository can know, and a hardcoded name that has since been retired fails at
request time instead of at startup.

## What this deliberately does not do

**It does not solve prompt injection.** Retrieved documents and tool output are treated as data,
never as instruction, and the drafting agent holds no tool that can perform an irreversible
action. That is containment through privilege separation and a human gate — not a defence
against a sufficiently clever input, and it is not presented as one.

**It is not a framework.** It is one worked example of a governed agent runtime, small enough to
read in an afternoon. Lift the patterns; do not depend on the package.

**It does not make the model trustworthy.** Every gate here constrains what an untrustworthy
model is *permitted to do*. None of them make its output correct.

**It does not pick an observability backend for you.** OpenTelemetry is the instrumentation and
the trace backend is deliberately a deployment decision, because a system that keeps a
restricted document away from a cloud model and then ships it to a managed trace backend has
only moved the leak somewhere less visible. Tracing is off by default. See
[ADR 0008](docs/adr/0008-tracing-backend-is-a-deployment-decision.md).

**It does not deploy to Kubernetes.** Docker Compose is the deployment story here, on purpose.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
