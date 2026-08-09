# 8. OpenTelemetry is the instrumentation; the trace backend is a deployment decision

Date: 2026-08-09

Status: Accepted (design settled; implementation lands in Phase 8)

## Context

An agent runtime is close to unobservable without traces. A single request fans out across a
classifier, a router, a supervisor, several parallel researchers, a retrieval subgraph, and a
drafter, and the interesting failures are the ones that live between those steps rather than
inside any one of them. Something has to record the shape of a run.

The obvious answer in a LangChain codebase is LangSmith. It is purpose-built for this, the
integration is a single environment variable, and the traces it produces are better than
anything generic tooling gives you for agent workloads.

It is also, by design, a service that receives the full text of every prompt, every retrieved
document, and every model response.

That is the whole problem. This runtime exists for environments where a policy gate decides
which *model tier* may see a piece of data — where routing a restricted request to a
self-hosted endpoint instead of a cloud API is the entire point of the sovereign lane. A
system that carefully keeps a restricted document away from a cloud model, and then ships that
same document to a managed trace backend as a span attribute, has not protected anything. It
has moved the leak somewhere less visible.

So the question is not "LangSmith or not". It is: what layer does instrumentation belong at,
such that the destination can change without the instrumentation changing?

There is a second, sharper problem. Both LangChain and the OpenTelemetry SDK auto-configure
from the environment. A developer with `LANGSMITH_TRACING=true` exported for another project,
or a CI runner with an inherited `OTEL_*` block, turns an otherwise offline test suite into
one that uploads prompts — silently, with nothing failing to announce it. Tracing that can
switch itself on is not a deployment decision at all.

## Decision

**OpenTelemetry is the instrumentation. Always, and in every configuration.** Spans are created
around each node and each model call regardless of where — or whether — they are exported.

**The backend is one selection behind that instrumentation**, chosen by
`AGENTGATE_TRACING_BACKEND`:

| Value | Behaviour |
| --- | --- |
| `none` (default) | No exporter. The process still logs structurally. Nothing leaves it. |
| `langsmith` | Managed backend. Correct when the data is allowed to leave the boundary. |
| `otlp` | Any OTLP collector, including one inside your own network. The self-hostable path. |

This is deliberately the same shape as the model lanes: one abstraction, several destinations,
and which destination you get is a policy decision made in configuration rather than a
property of the code. The sovereign lane and the OTLP collector exist for the same reason.

Four rules follow, and each is enforced rather than documented:

1. **Default off.** An unconfigured process traces nowhere. Turning it on is an explicit act.
2. **A backend without a destination is a startup error.** Selecting `otlp` with no endpoint,
   or `langsmith` with no key, fails validation. Spans dropped on the floor while the operator
   believes tracing is on is worse than tracing being off, because it buys false confidence.
3. **The suite and CI never trace.** `isolated_env` strips `LANGSMITH_*`, `LANGCHAIN_TRACING*`,
   `LANGCHAIN_ENDPOINT`, and `OTEL_*` from the environment of every test, and
   `tests/unit/test_offline_isolation.py` asserts it for each known leak vector by name.
4. **No secret is ever a span attribute.** Redaction happens on the way out, the same way the
   configuration dump masks `SecretStr`. A trace backend credential is a secret like any other.

## Consequences

Switching observability backends is a change to `.env`, not a change to code, and not a change
to any node. The same spans reach LangSmith in a permissive environment and a collector inside
the boundary in a restrictive one.

The cost is that the LangSmith integration is not free the way `LANGSMITH_TRACING=true` is
free. Routing it through OpenTelemetry means accepting whatever fidelity that path gives us
rather than the native tracer's. That is a real loss of convenience, and it is the price of the
destination being swappable. For this runtime, in these environments, it is worth paying.

There is a maintenance cost too: the strip-list in `isolated_env` has to keep up with whatever
variables LangChain and the OTel SDK read next. A new auto-configuration variable that nobody
adds to that list is a hole. The parametrised test names each vector explicitly so the list is
at least visible and reviewable, but it cannot catch a variable nobody knows about.

Nothing here is verified yet. The implementation lands in Phase 8, and until each backend has
actually been run, neither is claimed as working — LangSmith against a free-tier account, OTLP
against a collector in the Compose stack.

## Alternatives rejected

**Use LangSmith directly as the instrumentation.** Far less work, better agent-specific traces,
and the integration is one variable. Rejected because it makes the trace destination a property
of the code. In an environment where the data cannot leave, the only remedy would be turning
tracing off entirely — so the deployments that most need to understand what their agents did
would be the ones flying blind.

**OTLP only, no managed option.** Simplest possible story, and never a data-egress question.
Rejected because it is needlessly puritan: when the data does permit it, LangSmith is the right
tool and refusing to support it helps nobody.

**Trace by default, with an opt-out.** Better ergonomics for the common case, and observability
that is off by default is often observability nobody ever switches on. Rejected because the
failure mode is asymmetric. Forgetting to enable tracing costs you a debugging session;
forgetting to disable it ships regulated content to a third party, and you find out later.

**Kubernetes-native tracing conventions and a sidecar collector.** Rejected along with
Kubernetes generally. Docker Compose is the deployment story for this repository, and an OTLP
collector as a Compose service covers the self-hosted path without importing an orchestrator.
