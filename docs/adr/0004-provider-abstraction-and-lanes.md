# 4. Provider abstraction, three lanes, and the leak inventory

Date: 2026-08-09

Status: Accepted

## Context

This runtime has to reach language models in three quite different situations, and the
difference between them is a matter of policy rather than performance:

- A request whose content is unremarkable should reach a capable commercial model.
- A request classified as restricted must not leave infrastructure the operator controls, no
  matter how much better the commercial model would be.
- Every test, and all of CI, must run with no network and no cost.

The naive response is three code paths, and the failure that follows is predictable: the
classifier node grows a branch on which provider it is talking to, then the researcher grows
one, and eventually the policy decision — the thing this system exists to enforce — is smeared
across a dozen call sites where nobody can audit it.

The opposite failure is a leak-free abstraction that pretends the three are interchangeable.
They are not. A self-hosted endpoint speaking the OpenAI dialect does not behave like OpenAI,
and code written as though it does breaks in ways that surface far from the cause.

## Decision

**One interface, three lanes, selected by configuration.**

| Lane | What it is | Used for |
| --- | --- | --- |
| `cloud` | OpenAI, two tiers | the default path |
| `sovereign` | any OpenAI-compatible endpoint via `base_url` | restricted-sensitivity requests |
| `fake` | a deterministic in-process model | the entire suite and CI |

Callers ask `build_model(settings, tier, call_class)` for a model and never learn which lane
answered. The lane is chosen upstream, by the policy router, which is the single auditable place
that decision lives.

Three supporting decisions matter as much as the shape:

**The safe default is the free one.** An unconfigured process comes up on `fake`. Reaching a
real provider is opt-in, which is the only reason `make test` is a thing you can run without
thinking about it.

**Both networked lanes are one integration pointed at different endpoints.** `cloud` and
`sovereign` construct the same client with a different `base_url`. Supporting the sovereign lane
therefore costs almost nothing — it is not a second provider integration, and
`test_both_networked_lanes_are_the_same_integration_pointed_elsewhere` pins that.

**The abstraction is allowed to leak, but only in writing.** Where the lanes genuinely differ,
the difference is recorded as an observation with a provenance and a date, and callers branch on
that record rather than on a guess. See the inventory below.

## The leak inventory

This is the part worth reading. An abstraction over three providers *will* leak; the only
question is whether the leaks are written down or discovered at three in the morning.

Two kinds of entry appear here, and they turned out to be the same shape. Some are differences
between providers. Others are places where **a tool reported success without having checked** --
mypy accepting a node it should have rejected, a configuration key accepted and discarded. Both
are gaps between what something claims and what it does, and both are only findable by running
the thing rather than reading about it.

Every row was established by running something. None is inferred. Each is pinned by a test, so
a row that stops being true fails the build instead of quietly rotting.

### 1. Native structured output — the sovereign lane does not have it

| | |
| --- | --- |
| **Difference** | The cloud lane can be asked for a schema-valid object. A self-hosted OpenAI-compatible endpoint typically ignores `response_format` and returns the right JSON wrapped in prose and a code fence. |
| **How established** | Measured against `tests/doubles/openai_compatible.py` over real HTTP, through the real client library. |
| **Evidence** | `test_native_structured_output_fails_against_the_sovereign_lane` — the native path raises `ValidationError: Invalid JSON`. `test_repair_loop_rescues_prose_wrapped_json_from_the_sovereign_lane` — the same request yields a validated object via validate-and-repair. |
| **Consequence** | `invoke_structured` dispatches on the capability matrix. A lane recorded as lacking native support goes straight to the repair loop; a lane wrongly recorded as having it degrades to repair rather than surfacing a provider error to a node that has no idea what a lane is. |
| **Recorded** | `(SOVEREIGN, NATIVE_STRUCTURED_OUTPUT) = supported: False, provenance: STUB, 2026-08-09` |

### 2. The output ceiling is not called `max_tokens` on the wire

| | |
| --- | --- |
| **Difference** | The client is configured with `max_tokens`. `langchain-openai` **1.4.2** emits `max_completion_tokens`, following the current OpenAI API. |
| **How established** | Observed in the stub server's request log. Found because an assertion written against the client attribute passed while a wire assertion failed. |
| **Evidence** | `test_the_output_ceiling_travels_with_the_call_class`, and `test_a_renamed_wire_field_fails_loudly`, which pins the trap itself. |
| **Consequence** | A general rule for this codebase: **every assertion about what reaches a provider reads the observed request body, never the constructed client.** A client attribute and the wire are different things, and the gap between them is invisible until something depends on it. The registry tests carry a note saying so, and the wire helper raises on a missing field rather than returning `None`, so a rename goes red instead of vacuous. |
| **Recorded** | Here, and in the docstrings of `tests/integration/test_resilience.py`. Version-specific: re-check on any `langchain-openai` upgrade. |

### 3. `functools.partial` hides a wrongly-shaped node from mypy

| | |
| --- | --- |
| **Difference** | `partial` types as `partial[T]`, whose parameter list is effectively `...`. Wrapping a graph node in it silences the signature check completely — including for a node that takes a second required argument nothing will ever supply. mypy strict reports success; the runtime raises `TypeError`. |
| **How established** | A real mypy run over two snippets differing only in whether the node is wrapped. The unwrapped one is rejected; the wrapped one passes. |
| **Evidence** | `tests/integration/test_toolchain_blind_spots.py::test_partial_hides_a_wrongly_shaped_node_from_mypy`, with `::test_without_partial_mypy_catches_the_same_node` as the control and `::test_the_wrongly_shaped_node_really_does_fail_at_runtime` closing the loop. |
| **Consequence** | Lane nodes in `build.py` are bound with an explicit closure rather than `partial`, and the closure's return type is a `GraphNode` protocol so the signature is still checked. Note the trap in the control itself: the first attempt at it annotated the graph as `Any`, which erased `add_node` and made *both* snippets pass — a test proving nothing while looking green. |
| **Recorded** | Pinned. If mypy ever closes this, the test fails and the workaround can go. |

### 4. `interrupt_before` is compile-time only, and ignored silently otherwise

| | |
| --- | --- |
| **Difference** | `interrupt_before` passed in the invoke config is discarded without warning. Only the `compile()` argument has any effect. |
| **How established** | Observed. A resume test asked the graph to pause before `finalise` via config and got a fully completed run back, with the finalisation audit event present. |
| **Evidence** | `tests/integration/test_toolchain_blind_spots.py::test_interrupt_before_in_the_invoke_config_is_silently_ignored`, paired with `::test_interrupt_before_at_compile_time_actually_pauses`. |
| **Consequence** | This is worse than an error, because the failure is invisible: a gate that does not gate looks exactly like a gate that does. `build_graph` takes `interrupt_before` as a compile-time parameter and says so. Phase 5's approval gate uses `interrupt()` from inside the node instead, which pauses from the node body rather than from the graph definition and therefore cannot be silently dropped by being passed in the wrong place. |
| **Recorded** | Pinned before the behaviour becomes load-bearing, which is the only useful time to record it. |

### 5. `extra="forbid"` does not police environment variables

| | |
| --- | --- |
| **Difference** | pydantic-settings builds its environment source per declared field, so an unknown `AGENTGATE_*` variable is silently dropped rather than rejected — `extra="forbid"` never sees it. |
| **How established** | Reproduced directly: `AGENTGATE_MAX_ITERATION=3` was accepted and ignored. |
| **Evidence** | `test_tolerance_does_not_extend_to_the_agentgate_namespace`. |
| **Consequence** | A bespoke guard scans the environment and `.env` for prefixed names matching no field and suggests the closest match. Without it, an operator can believe a budget is in force while the process runs on defaults. |
| **Recorded** | ADR 0009. |

### 6. A shared `.env` is read by fields that never declared the name

| | |
| --- | --- |
| **Difference** | Unprefixed keys in `.env` were matched to fields with similar names. An unprefixed `LANGSMITH_API_KEY` populated `langsmith_api_key` despite no alias declaring it. |
| **How established** | Found on a real developer `.env`, not constructed. |
| **Evidence** | `tests/unit/test_env_namespace.py` parametrises over every field and asserts none consumes the bare form of its own name unless declared. |
| **Consequence** | Every unprefixed read is now an explicit `AliasChoices`, and the permitted set is asserted against the model so widening it is deliberate. |
| **Recorded** | ADR 0009. |

### 7. The gatekeeper's own environment stopped the suite it was guarding

| | |
| --- | --- |
| **Difference** | `scripts/run_live.py` sets `AGENTGATE_LIVE_SPEND_LEDGER` and `AGENTGATE_LIVE_SPEND_ABORT_USD` on the pytest subprocess it launches. Neither was a declared setting, and the unknown-variable guard from item 5 rejects any unrecognised `AGENTGATE_*` name. Every live case died at `get_settings()`, before spending anything and before proving anything. |
| **How established** | Run, on 2026-08-10: constructing `Settings` under exactly the environment the gatekeeper builds returned `ConfigurationError: AGENTGATE_LIVE_SPEND_ABORT_USD ... is not a known setting, did you mean AGENTGATE_MAX_SPEND_USD?` |
| **Evidence** | `tests/unit/test_live_suite_ceilings.py::test_every_variable_the_gatekeeper_injects_is_a_declared_setting`, which reads the injected names out of the script rather than restating them, guarded by `::test_the_gatekeeper_injects_something_at_all` so an empty match cannot pass vacuously. `::test_the_suite_starts_under_the_environment_the_gatekeeper_builds` constructs the settings rather than comparing name sets. |
| **Consequence** | Both are declared settings now, which makes them legal, typed, and documented in `.env.example`. The wider lesson is that two correct mechanisms can be jointly wrong: the guard was right to reject unknown names and the gatekeeper was right to pass configuration by environment, and nothing owned the interaction. It survived because the live suite is deselected by default — the only code here that CI cannot exercise, and therefore the only code where "it has never been run" and "it passes" look identical. |
| **Recorded** | Here. `AGENTGATE_LIVE_SPEND_ABORT_USD` is also now read rather than merely set — see item 8. |

### 8. A test suite accounted against a per-run ceiling

| | |
| --- | --- |
| **Difference** | The live suite's ledger enforced `max_total_tokens` and `max_spend_usd`, which bound one request through the graph. The suite is six independent cases sharing one book so the total is visible to the gatekeeper. Six cases at roughly 250 tokens each against a 2,370-token run ceiling is most of the budget spent on being a suite. |
| **How established** | Read off the arithmetic once the run ceiling was derived from measurement in Phase 3, and confirmed by construction: `SpendLedger` took its ceilings off `Settings` inside `check()`, so every ledger was a run ledger whether or not it was accounting a run. |
| **Evidence** | `tests/unit/test_spend.py::test_the_live_suite_is_not_accounted_against_the_run_ceiling` and `::test_the_live_suite_trips_its_own_token_ceiling` — the separation must not disarm the guard, only re-scope it. `tests/unit/test_live_suite_ceilings.py::test_the_suite_token_ceiling_is_still_the_estimate_times_the_tolerance` recomputes the ceiling from the estimate it was derived from. |
| **Consequence** | `Ceilings` is a required argument to `SpendLedger`, so a ledger has to say what it is accounting. The suite's ceiling has its own basis — the gatekeeper's estimate times the tolerance, the same bound as the dollar abort in the other unit — and `make test-live` now prints estimated against actual tokens as well as dollars, so the figure the ceiling rests on is observed rather than assumed. `AGENTGATE_LIVE_SPEND_ABORT_USD` tightens it, which is the first time that value has been enforced during a suite rather than reported after one. |
| **Recorded** | `.env.example`, next to the run ceilings it is deliberately not part of. The failure mode worth naming: charged against the wrong ceiling, the suite aborts for being a suite, and the obvious remedy is to raise the run ceiling — weakening the guard that was working. |

### 9. The spend guard cannot see embedding spend

| | |
| --- | --- |
| **Difference** | `SpendLedger` accounts from `usage_metadata` on chat-model replies. Embeddings do not produce one. On the cloud lane, indexing the corpus and embedding every research query costs real money that no ceiling in this system can observe — the run ceiling, the session ceiling, and the live-suite ceiling are all blind to it. |
| **How established** | Found while re-deriving the ceilings at the end of Phase 4. `make measure` reports two model calls for a run that saturates the fan-out; the five research branches make none, because on the fake lane retrieval is embedding-only and embeddings are free. The zero is real on the fake lane and false on the cloud lane, which is the worst combination: the offline suite will never show it. |
| **Evidence** | The measurement itself: `a request at the fan-out limit — 2 calls, 1,920 tokens`. Five branches, zero accounted calls. Nothing yet asserts the gap, which is why this row says "cannot see" rather than naming a test. |
| **Consequence** | Stated in `.env.example` next to the token ceiling rather than left for someone to discover from a bill. Closing it means either accounting embedding usage into the ledger or declaring the corpus index a build-time cost outside the run budget — a real decision, not a patch, and it belongs with the guardrails work in Phase 5 rather than being improvised here. Until then no claim is made that the ceilings bound total spend; they bound *chat* spend. |
| **Recorded** | Here, and in `.env.example`. Open. |

### 10. A checkpoint notice that no configuration can turn into an error, and a flag that makes it worse

| | |
| --- | --- |
| **Difference** | Resuming a run logs `Deserializing unregistered type ... This will be blocked in a future version` for every custom type in state. Three plausible ways to promote it to a failure, and none of them does: it is **not a Python warning**, so `filterwarnings` cannot see it; it is **not printed**, so `capsys` cannot either — it is a `logging` record from `langgraph.checkpoint.serde.jsonplus`; and `LANGGRAPH_STRICT_MSGPACK=true` does **not raise**, it blocks the value and continues. |
| **How established** | Run, on 2026-08-10. `warnings.catch_warnings(record=True)` around a full run and resume returned `warnings caught: 0` while the line still appeared. Under `LANGGRAPH_STRICT_MSGPACK=true` the run completed normally, printing `Blocked deserialization of agentgate.graph.state.Finding`. |
| **Evidence** | `tests/integration/test_checkpoint_serialisation.py::test_a_real_run_never_logs_the_deserialisation_notice`, guarded by `::test_that_check_would_actually_notice`, which puts a custom type in a channel on purpose and asserts the notice appears. The guard is not decoration: it is what caught the `capsys` version passing against an empty capture. |
| **Consequence** | Two separate lessons. First, a `filterwarnings` entry was written, verified to enforce nothing, and **removed** — the absence is now a comment in `pyproject.toml` explaining why, because a rule that reads as enforcement and enforces nothing is worse than no rule. Second, and worth stating on its own: **`LANGGRAPH_STRICT_MSGPACK=true` is a control that makes things worse when enabled.** It converts a visible notice into a silent data loss — the value is dropped and the run carries on — so someone who turns it on believing it hardens the system has made a resume fail quietly instead of loudly. Do not set it. The real fix is ADR 0011: keep custom types out of channels, and enforce that with a walk over a real run's channels that depends on no framework behaviour at all. |
| **Recorded** | Here and in ADR 0011. Version-specific: re-check on any LangGraph upgrade, including whether the flag has learned to raise. |

### 11. Not measured yet, and therefore not claimed

These are absent from the capability matrix on purpose. An absent row means "nobody asked",
which `supports()` reads as unsupported — the pessimistic direction, where being wrong costs
some tokens on a fallback rather than a provider exception in a node that cannot interpret it.

| Gap | What would close it |
| --- | --- |
| Tool calling, on any lane | Tools exist as of Phase 4 and the drafter binds them, but no live case has watched a real provider emit a tool call. A live case would change the suite's cost estimate and therefore its ceiling, so it lands with the next measured live run. |
| Embeddings, on the cloud lane | `OpenAIEmbeddings` is wired and has never been called. The offline lane embeds in-process, so nothing in CI touches this path — see item 9 for the ceiling consequence. |
| Streaming, on any lane | Phase 7, when the SSE surface exists. |
| Ollama and vLLM behaviour | Neither has been run against. The stub stands in for the shape, not for a specific server. |

`(CLOUD, NATIVE_STRUCTURED_OUTPUT)` left this table on 2026-08-10. `scripts/probe_capabilities.py`
was run against a real key and observed the native path returning a valid object with no
post-processing, so the row is recorded as `supported: True, provenance: LIVE_PROBE` and the
live suite now enforces it. The gap closed the way every gap here is meant to: by asking.

**This inventory is incomplete, and it grows by measurement.** Every entry above exists because
something was run and produced a surprising answer, which means the ones not yet found are the
ones nothing has exercised. The correct response to a suspected difference is to write a test
that provokes it, not to add a defensive branch.

## Consequences

The policy decision lives in one place and can be audited there. Adding a fourth lane is a
registry change and a matrix row, not a sweep through the nodes.

The cost is indirection: reading `build_model` does not tell you what will actually be
constructed without also reading configuration. That is the price of the destination being a
deployment decision, and it is the same trade made for the tracing backend in ADR 0008.

The matrix will go stale. Providers change, and an observation dated today is evidence about
today. Dates are recorded so an old row reads as a reason to re-probe rather than as a fact.

## Alternatives rejected

**One provider, no abstraction.** Much simpler, and honest about what a portfolio project needs.
Rejected because the sovereign lane is the thesis: a policy gate that can only route to one
place is not a gate.

**LiteLLM or a similar universal proxy.** Would give many more providers for less code.
Rejected because it moves the leak rather than documenting it — the differences in this
inventory would still exist, just one layer further away and harder to observe. It is also a
dependency in the path of every model call, which is a lot of surface for a repository meant to
be read in an afternoon.

**Assume all OpenAI-compatible endpoints behave like OpenAI.** What the phrase invites you to
believe. Rejected on the first measurement: item 1 in the inventory is exactly this assumption
failing.
