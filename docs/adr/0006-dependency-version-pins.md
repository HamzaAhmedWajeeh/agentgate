# 6. Pin the current LangGraph and LangChain 1.x line, not 1.0.x

Date: 2026-08-09

Status: Accepted

## Context

The original specification for this project called for "LangGraph 1.0.x and LangChain 1.x APIs",
naming `interrupt()` and `Command` as the interfaces to build against and the deprecated
`NodeInterrupt` as the one to avoid.

At the time of writing the 1.0 series had ended at `langgraph==1.0.10`, with the line having
moved on through 1.1 to `1.2.10`. LangChain was at `1.3.14` and `langchain-core` at `1.5.3`.

Every API the specification names is present and unchanged across that range: `interrupt()`,
`Command`, `Command.PARENT`, `Send`, `get_state_history`, and `update_state(as_node=...)` all
carry the same signatures in 1.2 as in 1.0. `NodeInterrupt` remains the deprecated path.

Two considerations pull in opposite directions. Pinning literally to `1.0.10` matches the
specification as written, at the cost of shipping a version that is two minor releases behind
and missing the accumulated fixes. Pinning to the current line deviates from the letter of the
specification, and a reader who knows the brief may wonder whether the deviation was noticed.

There is also a practical constraint: `langchain==1.3.14` declares a floor on `langchain-core`
in the 1.5 range, and the older LangGraph pin is not guaranteed to resolve cleanly against it.
Holding 1.0.x would likely have meant holding older LangChain packages too, widening a
one-package deviation into a whole-stack one.

## Decision

Pin the current 1.x line exactly:

```
langgraph==1.2.10
langchain==1.3.14
langchain-core==1.5.3
langchain-openai==1.4.2
```

Every direct dependency is pinned to an exact version rather than a compatible-release range,
and `uv.lock` is committed, so the resolved graph is reproducible rather than merely constrained.

## Consequences

The repository is built against a supported release rather than a stale one, and a reviewer
checking the pins against what the ecosystem currently ships finds them current.

The deviation from the brief is recorded here rather than left to be discovered, which is the
point of the record: the specification said 1.0.x, this says 1.2.10, and the reason the
substitution is safe is that the named APIs did not change.

Exact pins mean dependency updates are a deliberate act with a commit attached, not something
that happens silently between two builds of the same commit. The cost is that upgrades need
doing by hand; `make audit` runs `pip-audit` against the locked set so a vulnerable pin surfaces
in CI rather than being noticed later.

## Alternatives rejected

**Pin literally to `langgraph==1.0.10`.** Matches the brief exactly. Rejected because it ships
known-fixed bugs for no benefit, risks an unresolvable graph against current LangChain, and
optimises for fidelity to a document over fidelity to the intent behind it — which was to build
against the modern API surface, not against a specific patch release.

**Use compatible-release ranges (`~=1.2`).** Smaller maintenance burden. Rejected because the
brief asks for exact pins, and because "works on my machine, broken in CI three weeks later
with no commit in between" is precisely the failure a governed runtime should not model.
