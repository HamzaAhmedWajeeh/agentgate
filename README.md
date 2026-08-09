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

Under construction, built in phases. This README documents only what is covered by a test or by
a check that has actually been run — nothing here is aspirational, and the architecture diagram
arrives with the graph it describes rather than before it.

**Working today**

- Configuration: every tunable in one module, validated at import, failing at startup with a
  message that names the variable at fault rather than a provider stack trace mid-run.
- A safe-by-default posture: an unconfigured process comes up on a deterministic offline lane
  with an in-memory checkpointer, so the test suite needs no API key and CI needs no secrets.
- `python -m agentgate` prints the resolved configuration with every secret masked.
- Toolchain: ruff, mypy in strict mode, pytest, pre-commit, and CI running all of it.

**Not built yet:** the graph, the model lanes, retrieval, the gates, and every surface.

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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
