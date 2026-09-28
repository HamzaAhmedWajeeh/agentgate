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

Rows 13 to 16 share a shape worth naming before you read them, because it is the one this
repository is least able to see on its own: **a component that is correct in isolation, tested in
isolation, and connected to nothing.** The policy router chose lanes correctly for four phases.
`build_resilient_model` retries correctly today. Neither was reachable from the system that
claimed the behaviour. A unit test cannot find this, and an integration test only finds it if it
asserts on something outside the process -- which is why the guard for item 13 reads two servers'
request logs rather than any value the graph produced.

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

### 9. Embedding spend was invisible to every ceiling — CLOSED, via item 19

| | |
| --- | --- |
| **Difference** | `SpendLedger` accounts from `usage_metadata` on chat-model replies. Embeddings do not produce one. On the cloud lane, indexing the corpus and embedding every research query costs real money that no ceiling in this system can observe — the run ceiling, the session ceiling, and the live-suite ceiling are all blind to it. |
| **How established** | Found while re-deriving the ceilings at the end of Phase 4. `make measure` reports two model calls for a run that saturates the fan-out; the five research branches make none, because on the fake lane retrieval is embedding-only and embeddings are free. The zero is real on the fake lane and false on the cloud lane, which is the worst combination: the offline suite will never show it. |
| **Evidence** | The measurement itself: `a request at the fan-out limit — 2 calls, 1,920 tokens`. Five branches, zero accounted calls. Nothing yet asserts the gap, which is why this row says "cannot see" rather than naming a test. |
| **Consequence** | Stated in `.env.example` next to the token ceiling rather than left for someone to discover from a bill. Closing it means either accounting embedding usage into the ledger or declaring the corpus index a build-time cost outside the run budget — a real decision, not a patch, and it belongs with the guardrails work in Phase 5 rather than being improvised here. Until then no claim is made that the ceilings bound total spend; they bound *chat* spend. |
| **Recorded** | Here and in `.env.example`. **Closed 2026-08-10.** |
| **Decided** | **The run budget means all spend, not chat spend.** A gate that claims to cap spend and means "some spend" is misdescribed, and the specific reason it mattered here is that embedding cost scales with fan-out width — the one quantity a model chooses rather than the system. The unaccounted path was exactly the path with model-controlled multiplication in it. The alternative on the table was to declare the corpus index a build-time cost outside the run budget; rejected because querying is not indexing, every research branch embeds its sub-question at run time, and a budget with a carve-out is a budget someone has to remember. |
| **Reopened 2026-09-27** | **This row was wrong.** `AccountedEmbeddings` books nothing, because nothing constructs it -- `build_embeddings` returns a plain `OpenAIEmbeddings`, and embedding spend is still invisible to every ceiling. The decision below stands; the implementation is not on any path. See item 19, which is where the correction is recorded rather than hidden by an edit. The one part that was true: the price guard does include the embedding model, so an unpriced embedder still refuses to start. |
| **Closed 2026-09-28, for real** | `build_embeddings` now constructs `AccountedEmbeddings` on the cloud lane, billed to the run that is embedding and checked against its ceiling after every batch. Item 19 records how. |
| **Claimed by** | `retrieval/accounting.py:AccountedEmbeddings` books every embedding call into the ledger on the same three rules as a chat call: a response with no usage is an error rather than a zero, an unpriced embedding model refuses to start (`config.py:_every_reachable_model_has_a_price` now includes it), and spend is recorded per model so the summary names `text-embedding-3-small` rather than "embeddings". `check()` runs after every batch, so a runaway index trips the ceiling while it is running rather than reporting the bill afterwards. Pinned by `tests/unit/test_embedding_accounting.py`. |

### 10. A checkpoint notice that no configuration can turn into an error, and a flag that makes it worse

| | |
| --- | --- |
| **Difference** | Resuming a run logs `Deserializing unregistered type ... This will be blocked in a future version` for every custom type in state. Three plausible ways to promote it to a failure, and none of them does: it is **not a Python warning**, so `filterwarnings` cannot see it; it is **not printed**, so `capsys` cannot either — it is a `logging` record from `langgraph.checkpoint.serde.jsonplus`; and `LANGGRAPH_STRICT_MSGPACK=true` does **not raise**, it blocks the value and continues. |
| **How established** | Run, on 2026-08-10. `warnings.catch_warnings(record=True)` around a full run and resume returned `warnings caught: 0` while the line still appeared. Under `LANGGRAPH_STRICT_MSGPACK=true` the run completed normally, printing `Blocked deserialization of agentgate.graph.state.Finding`. |
| **Evidence** | `tests/integration/test_checkpoint_serialisation.py::test_a_real_run_never_logs_the_deserialisation_notice`, guarded by `::test_that_check_would_actually_notice`, which puts a custom type in a channel on purpose and asserts the notice appears. The guard is not decoration: it is what caught the `capsys` version passing against an empty capture. |
| **Consequence** | Two separate lessons. First, a `filterwarnings` entry was written, verified to enforce nothing, and **removed** — the absence is now a comment in `pyproject.toml` explaining why, because a rule that reads as enforcement and enforces nothing is worse than no rule. Second, and worth stating on its own: **`LANGGRAPH_STRICT_MSGPACK=true` is a control that makes things worse when enabled.** It converts a visible notice into a silent data loss — the value is dropped and the run carries on — so someone who turns it on believing it hardens the system has made a resume fail quietly instead of loudly. Do not set it. The real fix is ADR 0011: keep custom types out of channels, and enforce that with a walk over a real run's channels that depends on no framework behaviour at all. |
| **Recorded** | Here and in ADR 0011. Version-specific: re-check on any LangGraph upgrade, including whether the flag has learned to raise. |

### 11. `OpenAIEmbeddings` discards the usage the budget depends on

| | |
| --- | --- |
| **Difference** | The OpenAI embeddings API returns a `usage` block with the token count it charged for. `langchain_openai.OpenAIEmbeddings.embed_documents` returns the vectors and drops it. |
| **How established** | Found while closing item 9. There is no accessor for it on the LangChain object; the field exists on the response the client underneath returns. |
| **Evidence** | `retrieval/accounting.py:OpenAIEmbeddingsWithUsage` exists only because of this, and its docstring says so. The offline half is pinned by `tests/unit/test_embedding_accounting.py::test_an_embedder_that_reports_no_usage_is_an_error_not_a_free_call`; nothing has yet watched a real provider report it, which is recorded in the gaps table below. |
| **Consequence** | The embedding call goes through the provider client directly and reads the reported count. The two alternatives were both worse: estimating tokens from the text is a guess presented as a measurement, and accounting zero is the bug being fixed. Note the shape — it is the same one as item 2, where the output cap was renamed on the wire: **a client is a convenience over a protocol, and what it chooses not to surface is invisible until something depends on it.** |
| **Recorded** | Here. Version-specific: re-check on any `langchain-openai` upgrade, in case the usage is surfaced and the direct client call can go. |

### 12. Not measured yet, and therefore not claimed

These are absent from the capability matrix on purpose. An absent row means "nobody asked",
which `supports()` reads as unsupported — the pessimistic direction, where being wrong costs
some tokens on a fallback rather than a provider exception in a node that cannot interpret it.

| Gap | What would close it |
| --- | --- |
| Tool calling, on any lane | Tools exist as of Phase 4 and the drafter binds them, but no live case has watched a real provider emit a tool call. A live case would change the suite's cost estimate and therefore its ceiling, so it lands with the next measured live run. |
| Embeddings, on the cloud lane | Reached only by a cloud-only deployment since item 17 closed. Wired and accounted since item 19 closed, and never called against a real provider. The usage field `AccountedEmbeddings` would read (item 11) has only been exercised against a double. This cell said *accounted* until item 20. The offline lane embeds in-process, so nothing in CI touches this path — see item 9 for the ceiling consequence. |
| Streaming, on any lane | Phase 7, when the SSE surface exists. |
| Ollama and vLLM behaviour | Neither has been run against. The stub stands in for the shape, not for a specific server. |

`(CLOUD, NATIVE_STRUCTURED_OUTPUT)` left this table on 2026-08-10. `scripts/probe_capabilities.py`
was run against a real key and observed the native path returning a valid object with no
post-processing, so the row is recorded as `supported: True, provenance: LIVE_PROBE` and the
live suite now enforces it. The gap closed the way every gap here is meant to: by asking.

### 13. The routed lane never reached model construction

| | |
| --- | --- |
| **Difference** | `route_by_policy` chose a lane, the lane node wrote it to `state["lane"]`, and the audit trail recorded it. Nothing applied it. Neither `classify` nor `draft` passed `lane=` to the model factory, so `build_model` resolved `lane or settings.lane` to the configured default on every call. A request the router sent to the sovereign lane was drafted by whatever the deployment defaulted to. **Second half:** `model_for` resolved the identifier against `settings.lane` too, so fixing only the endpoint would have sent a request for `gpt-4.1-nano` to the operator's own server and recorded that name as the model that answered. |
| **How established** | Two stub servers on separate ports with separate request logs, one standing in for each lane, running the real graph. A restricted request containing invented PII produced **three requests to the cloud endpoint and zero to the sovereign one**, while the drafter's own audit event read `lane='sovereign' model='cloud-capable-stub'`. Originally found by reading the audit trail of a real run against OpenAI on 2026-08-16, where `AGENTGATE_SOVEREIGN_BASE_URL` was not even set -- had the drafter honoured the routing, `build_model` would have raised `LaneUnavailableError` rather than succeeding. |
| **Evidence** | `tests/integration/test_routed_lane_enforcement.py::test_restricted_content_never_reaches_the_cloud_endpoint` is the primary guard, and it asserts an **absence**: canaries that appear only in the findings, which nothing but the drafter is shown, must appear in no request body the cloud endpoint received. `::test_the_draft_is_asked_of_the_sovereign_model_not_merely_the_sovereign_endpoint` pins the identifier half. `::test_a_sovereign_default_never_reaches_the_cloud_endpoint_for_public_content` pins the failure mode of the obvious repair. `::test_the_audit_trail_names_the_endpoint_that_actually_answered` checks the trail against the request log rather than against itself. |
| **Consequence** | The route **narrows** and never widens. `CONTAINMENT` orders the lanes by how far data travels and `narrower_of` takes the minimum, because passing the routed lane straight through introduces a worse leak than the one it fixes: `route_by_policy` sends public content to the cloud lane by design, so on a sovereign-default deployment the obvious version starts calling a third party on behalf of an operator who configured their own endpoint as the default. The same rule is what keeps the offline suite offline -- a fake deployment stays fake whatever the router says. Two startup guards follow from what the fix makes reachable: a sovereign base URL with no model is now rejected, and the price guard's "reachable" set is computed from `routable_lanes` rather than the configured lane, which had excluded a hybrid deployment's sovereign model entirely. A single-lane deployment now **refuses** a request policy sends somewhere stricter, rather than serving it from the lane policy just ruled out. |
| **Recorded** | Here, and in `models/registry.py:build_model`, `config.py:narrower_of` and `graph/state.py:AgentState.lane`, whose docstring now says it records what policy permits rather than where a call went. |
| **Closed by** | `fix(policy): send restricted content to the lane the router chose`. Mutation-checked: the drafter dropping `lane=`, the identifier resolving on the configured lane, `narrower_of` degrading to an override or inverting, `routable_lanes` forgetting or over-reporting, and both audit events reverting -- all red. |

**The old test is the point of this row.** `test_restricted_content_reaches_the_sovereign_lane_in_a_real_run` existed throughout, and passed throughout. It asserted `result["lane"] == "sovereign"`, which was correct: the state field was always right. The name claimed an endpoint and the body checked a label, and no amount of running it could have told the difference, because the scripted factory it used ignores the lane argument entirely. That is this inventory's whole thesis in one test: **a check that reports success without checking.** It is renamed to `test_restricted_content_records_the_sovereign_binding_and_its_reason`, the same treatment the `filterwarnings` rule got in ADR 0011, and the enforcement it appeared to provide now lives in a file that cannot pass without a request reaching the right server.

### 14. The classifier sees content before the policy decision exists

| | |
| --- | --- |
| **Difference** | `classify` runs before `route_by_policy`, so there is no routed lane to honour and the request has not been judged yet. It called the model factory with no lane, so on a cloud-default deployment the raw request -- names, account numbers, anything else in it -- went to the third party **in order to decide whether it was allowed to go there**. |
| **How established** | The same two-endpoint run as item 13. All three request canaries appeared in the cloud endpoint's request log before any lane had been selected. Recorded first as a *passing* assertion that the egress happened, so that closing it would be a visible change to a named test rather than an improvement nobody could date. |
| **Evidence** | `tests/integration/test_routed_lane_enforcement.py::test_the_classifier_never_shows_the_raw_request_to_the_cloud_endpoint` is that assertion, inverted. `::test_a_cloud_only_deployment_classifies_on_the_cloud_lane_as_documented_egress` holds the part that stays open. `::test_classification_does_not_take_the_native_path_on_a_lane_recorded_as_non_native` guards the capability lookup, with the cloud-only test as its control -- the cloud lane *is* recorded as native, so `response_format` must appear there, or the absence assertion is watching a field item 2 has already renamed once. |
| **Consequence** | Classification runs on `Settings.most_contained_lane`: the most contained lane in `routable_lanes`, folded with `narrower_of`. Fake stays fake, hybrid classifies on sovereign, cloud-only classifies on cloud. No new setting -- a dedicated classification-lane variable would be a third lane selector whose only safe values are "sovereign" or "the default", and an operator who set it to `cloud` would have re-created this leak while believing they had configured a control. The `supports()` lookup moved with it, because asking a non-native endpoint for native structured output does not fail, it falls through to the repair loop after wasting a call: **measured against the stub, 2 calls and ~733 prompt tokens where 1 and ~537 would do**, on every classification, silently. |
| **Not closed** | A cloud-only deployment has nowhere else to send a request to be judged. That is a property of having one lane rather than a defect in code, so it remains, recorded, and pinned by a passing test. The honest reading of ADR 0004's own line -- *a policy gate that can only route to one place is not a gate* -- is that single-lane deployments do not get this guarantee. |
| **Unmeasured** | **Classification quality on a self-hosted lane is not claimed anywhere.** Ollama and vLLM have never been run against, so what a weaker classifier does to routing is unknown. The direction is safe: a classifier that cannot produce a verdict fails closed to `restricted`, so the cost of a bad one is **the cloud lane going unused** rather than restricted content escaping. Nobody has measured how often that happens, and until someone has, "hybrid deployments classify on their own endpoint" is a statement about where the call goes and not about how good the answer is. |
| **Recorded** | Here, in `graph/nodes/classify.py`'s module docstring, and in `config.py:most_contained_lane`. |
| **Closed by** | `fix(policy): classify on the most contained lane the deployment can reach` -- narrowed, not closed. Mutation-checked: the lane lookup, the capability lookup, `most_contained_lane` reverting to the default, the audit event, and the single-lane refusal -- all red. |

**A test fixture was wrong in a way only the trail assertion could catch.** With classification moved to the sovereign lane, the stub on that lane was still replying with a draft, so the classifier failed to parse it and fell closed to `restricted` -- the same destination the test was checking for, reached without a verdict ever being parsed. The test now asserts `classification_failed is None`, so the routing under test has to come from a classification rather than from the fail-closed branch. Worth recording because it is the third time in this repository a guard has passed for a reason unrelated to what it claimed.

### 15. `with_retry` and `with_fallbacks` are tested and unreachable

| | |
| --- | --- |
| **Difference** | `build_resilient_model` composes retry and fallback, is covered by `tests/integration/test_resilience.py` against a server returning real HTTP errors, and **is called from nowhere in `src/`**. Both nodes that construct models call `build_model` directly. `_init_openai_compatible` passes `max_retries=0` under a comment reading *"retries are applied by build_resilient_model, in one place"* -- true about the design, false about the running system since Phase 2. **This system performs no retries and no fallbacks on any lane.** |
| **How established** | `grep -rn build_resilient_model src/` returns one hit: the comment asserting where retries happen. Found while tracing which code paths a routed lane had to reach, not by a failing test -- nothing was looking, because the tests that cover it construct it themselves. |
| **Evidence** | `docs/concept-map.md` now carries a third status, **built, not wired**, and `tests/unit/test_concept_map.py::test_every_built_not_wired_row_exists_and_is_called_from_nowhere` enforces it in both directions: the symbol must exist, and no file in `src/` other than the one defining it may mention it. Adding an import of `build_resilient_model` to any node turns that test red. |
| **Consequence** | Same shape as item 13 and worth stating as a general rule: **a function tested in isolation is evidence about the function, not about the system.** The concept map said *done* for four phases on the strength of a green test file. The row now says what is true, and the status is enforced rather than asserted, so the map cannot rot in the comfortable direction again. |
| **Deliberately not wired here** | Wiring it is a behaviour change to every model call and belongs in its own change. **The constraint for whoever does it: a fallback must never cross to a less contained lane.** `build_resilient_model` already threads `lane` into both tiers, and `narrower_of` makes the widening case impossible at construction -- but a fallback chain assembled per lane is a new place for the same mistake, and the guard it needs is the one from item 13: two endpoints, and an assertion that the fallback never appears in the wrong request log. |
| **Recorded** | Here, in the corrected comment in `models/registry.py:_init_openai_compatible`, and in the concept map's status legend. |
| **Closed by** | Nothing. The row is the record; the fix is not in this change. |

### 16. The routed tier never reached model construction either

| | |
| --- | --- |
| **Difference** | `bind_lane` binds a tier as well as a lane -- `cloud_capable` binds `Tier.CAPABLE`, `cloud_cheap` binds `Tier.CHEAP` -- and records it in the audit trail. No channel carries it, and the drafter always asks for `Tier.CAPABLE`. So `route_by_policy` distinguishing simple from involved requests changes the audit trail and nothing else. |
| **How established** | Observed on the wire in the same two-endpoint harness. A public, simple request routes to `cloud_cheap`, the lane event records `tier: cheap`, and the cloud endpoint is asked for the **capable** model. |
| **Evidence** | `tests/integration/test_routed_lane_enforcement.py::test_a_hybrid_deployment_still_uses_the_cloud_lane_for_public_content` asserts both halves as the current truth, so wiring the tier has to come through that line. `docs/concept-map.md` carries the row as *not built*. |
| **Consequence** | Invisible in the reference configuration, because both cloud tiers name the same model -- which is itself deliberate and documented, the split being a policy boundary rather than a cost claim. That is exactly why it survived: the one deployment shape that would reveal it is the one nobody runs. |
| **Deliberately not fixed here** | It needs a `tier` channel in `AgentState`, which is a schema change and a checkpoint-compatibility question, and it is a cost decision rather than a containment one. Item 13 was neither. |
| **Recorded** | Here and in the concept map. |
| **Closed by** | Nothing yet. |

### 17. Retrieval embeds on the configured lane, ignoring the route — CLOSED on hybrid deployments

| | |
| --- | --- |
| **Difference** | `build_embeddings` dispatches on `settings.lane` and never sees the routed lane. So on a hybrid deployment, a request the policy gate sent to the sovereign lane has its research queries -- and the whole corpus -- embedded by the third party anyway. Item 13 fixed the chat calls in `classify` and `draft`; this is the identical defect in the function next door, and item 13's guards could not see it because they seed findings and never research. |
| **How established** | A real research branch, on a hybrid cloud-default deployment, with the account number in the sub-question -- which is where it lives in reality, because a question about a refund window is not answerable without naming the account. The account number and the name both arrive at the cloud embedding endpoint. Two facts had to be fixed before it could be established at all, and both are the finding: `OpenAIEmbeddings` was constructed **with no endpoint to override**, so every request went to `api.openai.com` and no double could ever see one; and it **tokenises client-side**, so the body carries `[[2, 68538, ...]]` and a canary assertion that greps it matches nothing. The first version of the test did exactly that and passed against a leak in progress. |
| **Evidence, when it was open** | `test_routed_lane_enforcement.py::test_a_restricted_research_query_is_embedded_by_the_cloud_provider` asserted the leak *happened* -- the same treatment as item 14's documented egress -- so that closing it would be a visible inversion of a named test rather than an improvement nobody could date. Beside it: the corpus shown to be indexed through the same egress, which is what made the fix a design question, and a control on the instrument that checks the decode against a line known to be in the corpus on disk, because a decoder using the wrong encoding returns plausible rubbish and every canary assertion reading it goes quiet. |
| **Consequence, first** | Only observability. `openai_api_base` is passed now, and `decode_embedding_input` exists, so the path can be watched -- **a leak nothing can observe is a leak nothing can pin.** Note which field name: `base_url` is the pydantic alias and mypy rejects it, exactly as item 2 predicted one layer up. The canary helper in that test file now reads the decoded embedding texts alongside the chat log, because an absence assertion that reads only the chat log is blind to this entire egress. |
| **Decided 2026-09-28** | **Build and query the index on the most contained lane the deployment can reach** -- `Settings.most_contained_lane`, the rule classification already uses, rather than a second answer to "which lane is most contained" that could drift from the first. `build_embeddings` dispatches on it instead of `settings.lane`. No new setting. One index, no routing inside retrieval: a hybrid deployment indexes the corpus and embeds every research query in process, public requests included, and nothing from either reaches the third party. The options it was chosen from are kept below. |
| **Why this one** | It is the only option that makes the guarantee unconditional on a hybrid deployment rather than a property of the route, and it needs nothing that has not been run. **Per-lane indexes were ruled out:** with no sovereign embedding endpoint the sovereign index would be built by the hashing embedder, so retrieval quality would correlate exactly with sensitivity -- restricted requests getting the weaker search -- which is worse than the leak. A sovereign embedding endpoint composes with this decision later, as a better contained embedder behind the same rule. |
| **What it costs** | **Retrieval quality on every request, not just restricted ones.** A public request drafted by the cloud lane now searches a bag-of-words index. Measured below: **90% top-4 against a 20% chance floor**, 4 of 5 synonym-only questions, and one miss that shares no content word with its answer. That figure belongs to *this* 20-chunk corpus at **301 distinct terms**. ADR 0010's collision curve is why it does not travel: a hashing index degrades as vocabulary grows, so a corpus an order of magnitude larger needs re-measuring before the same conclusion can be drawn. |
| **Re-measured 2026-09-28** | After the run ledger changed the cloud embedder to `AccountedEmbeddings` on the raw client. **The hashing figure is unchanged: 90% top-4, 4 of 5 synonym-only.** What moved is the control and the timings -- the stub baseline went from 10% to 30% because the stub hashes what it is sent and the client now sends strings rather than token ids, and the build times now overlap in both directions (0.006-0.017 s against 0.010-0.017 s). Neither bears on the decision, which rested on the hit rate. The measurement table below has both dates. |
| **Item 19** | Sidestepped on every deployment this closes: a hybrid deployment has no cloud embedding spend left to account. **Not sidestepped on a cloud-only one**, which still embeds on the cloud -- and until the run ledger (2026-09-28) accounted that spend to nothing. It is now billed to the run that made it; item 19 is closed. |
| **Not closed** | **A cloud-only deployment.** Its most contained lane is the cloud, so the corpus and every public sub-question are embedded there -- the same shape as item 14, a property of having one lane rather than a defect in code, and pinned the same way, by a passing assertion. Restricted content does not reach it: a cloud-only deployment refuses a restricted request before research, which is item 13's refusal. |
| **Evidence** | `test_routed_lane_enforcement.py::test_a_restricted_research_query_never_reaches_the_cloud_embedding_endpoint` is the inversion: no canary in any embedding request the cloud endpoint received, decoded through `decode_embedding_input`, paired with a presence assertion -- the run's findings must come from corpus files -- so it cannot pass against a run that never embedded. `::test_the_corpus_is_not_indexed_through_the_cloud_endpoint_either` holds the indexing half. `::test_a_hybrid_deployment_embeds_public_research_in_process_too` pins the cost: a public request routed to the cloud still searches the contained index. `::test_a_cloud_only_deployment_embeds_on_the_cloud_lane_as_documented_egress` holds what stays open. `tests/unit/test_embeddings.py` pairs the dispatch with its control, so returning the hashing embedder unconditionally fails too. |
| **Recorded** | Here, in `retrieval/embeddings.py`'s module docstring, in `config.py:most_contained_lane`, in the concept map, and in the README's claim about egress. |
| **Closed by** | `fix(retrieval): embed on the most contained lane the deployment can reach` -- closed for hybrid deployments, open for cloud-only ones. Mutation-checked with a script holding the original bytes: dispatching on `settings.lane` again turns the three hybrid absence tests and the unit test red; returning the hashing embedder unconditionally turns the cloud-only tests and the unit control red; narrowing the canary helper to the chat log turns its own test red. |

**The guard this row could not have while it was open.** The canary helper was widened to read decoded embedding texts, which makes every absence assertion in that file cover retrieval egress. While the leak was live, the leak itself exercised the widening. Once closed, no hybrid run embeds on the cloud, so narrowing the helper back to the chat log would pass every absence assertion -- and would keep passing if the leak came back. `::test_the_canary_helper_reads_the_embedding_log` now holds it directly: one embedding request and no chat call, so the helper can only see the marker through the embedding log.

### 18. A streamed call reports no usage, and the CLI is the only thing that streams — CLOSED

| | |
| --- | --- |
| **Difference** | A streamed OpenAI response carries no `usage` block unless the caller sets `stream_options.include_usage`. `langchain-openai` does not set it. The CLI runs the graph with `stream_mode=["updates", "messages"]`, which makes every model call it issues a streamed one -- so **every model call made through the command line reports no token usage at all.** Not an error; an absent number. |
| **How established** | Observed in the stub's request log: every streamed body carries `"stream": True` and no `stream_options` key. Found while writing the first CLI test to run against a networked lane, which is a combination nothing had ever exercised -- every other command test runs on the fake lane, where no provider exists to have defaults. |
| **Evidence** | `tests/integration/test_cli.py::test_streamed_calls_ask_for_no_usage_block_which_is_recorded_not_accepted`, asserting the current truth. Its control is `::test_the_endpoint_does_return_usage_when_a_stream_asks_for_it`, which asks the endpoint directly over HTTP with the flag set and requires the usage chunk to arrive -- because "no usage came back" has two possible causes and only one of them is a fact about this system. |
| **Consequence** | Latent, and it is worth being precise about why. Chat calls are not wired to the spend ledger (README, and item 19 below), so nothing currently reads the number. It stops being latent the moment they are: `usage_of` refuses to treat an unmeasured call as free, and on this path there would be nothing to refuse, because the field is simply absent. Same shape as item 11 -- **a client is a convenience over a protocol, and what it declines to send is invisible until something depends on it.** Third instance now. |
| **Also** | The first version of that test asserted the property of both stubs, and `all()` over an empty list is `True`: on a restricted request the cloud endpoint receives no streamed calls at all, so half the assertion was passing by looking at nothing. Caught by the other half failing. |
| **Recorded** | Here and in `tests/doubles/openai_compatible.py`. |
| **Closed 2026-09-28** | With the run ledger, exactly as predicted: once chat spend was accounted, an unfixed item 18 would have stopped every networked CLI run at its first model call, because the ledger refuses a call that reports nothing. `stream_usage=True` on every OpenAI-compatible client. The pinning test is inverted -- `test_cli.py::test_every_streamed_call_asks_for_its_usage_block` requires `include_usage` on every streamed request and the run to complete -- and removing the setting turns the CLI tests red. |
| **Unverified** | The sovereign lane receives the same `stream_options`. Whether Ollama and vLLM honour it is unknown, like everything else about them; one that ignores it will be refused by the ledger rather than billed at zero. |
| **Closed by** | `feat(spend): wire one run ledger through every model call in the graph`. |

### 19. `AccountedEmbeddings` is not wired, so item 9 is not closed — CLOSED

| | |
| --- | --- |
| **Difference** | Item 9 above says **Closed 2026-08-10**, and names `retrieval/accounting.py:AccountedEmbeddings` as what closed it. Nothing constructs it. `build_embeddings` returns a plain `langchain_openai.OpenAIEmbeddings`, which -- as `accounting.py`'s own module docstring says in the course of explaining why `OpenAIEmbeddingsWithUsage` exists -- discards the `usage` block the budget depends on. So on the cloud lane, embedding spend is invisible to the run ceiling, the session ceiling and the live-suite ceiling: **exactly the state item 9 describes as fixed.** |
| **How established** | `grep -rn "AccountedEmbeddings\|OpenAIEmbeddingsWithUsage" src/ tests/ scripts/` returns the class definitions and `tests/unit/test_embedding_accounting.py`, and nothing else. Found while tracing the embedding path for item 17. |
| **Evidence** | `tests/integration/test_routed_lane_enforcement.py::test_no_embedding_call_is_accounted_for_anywhere` asserts that the embedder the index builds is `OpenAIEmbeddings`, so wiring the accounted one turns it red. On a cloud-only deployment since item 17 closed, because that is the only one left with cloud embedding spend. `docs/concept-map.md` carries the row as **built, not wired**, which is enforced in both directions. |
| **Consequence** | The half of item 9 that *is* true is the price guard: `_every_reachable_model_has_a_price` does include the embedding model, so an unpriced embedder still refuses to start. Everything about accounting is not. Item 9's **Closed by** row is corrected rather than deleted, because the correction is the interesting part -- and the general rule it produces is the same one as item 15: **a class with thorough unit tests and no caller is evidence about the class.** Second instance in this inventory, found six weeks apart, both by asking "what actually calls this". |
| **Recorded** | Here, in the corrected item 9, and in the concept map. |
| **Closed 2026-09-28** | Wired in the same change that gave the graph a run ledger, because a ledger that saw chat spend and not embeddings would be item 20 restated -- a ceiling that looks like it is working. `build_embeddings` constructs `AccountedEmbeddings` on the cloud lane. The corpus index outlives any one run, so the embedder cannot hold a run's ledger; it resolves one per call from the charge the research branch sets (`guardrails/run_ledger.py:charging`), and the index build is billed to the run that caused it. As predicted, an index build can now fail partway through on a ceiling. The research branch catches its own failures, and would have swallowed that one into a failed outcome the drafter then wrote up -- so budget errors now pass through it. |
| **Evidence, now** | `test_routed_lane_enforcement.py::test_every_embedding_call_is_accounted_in_the_run_ledger` -- the inversion -- reads the run's spend record against the tokens the stub reported for the texts it was sent. `test_run_ledger.py::test_the_ceiling_trips_inside_retrieval_and_is_not_swallowed` holds the pass-through. Mutation-checked: removing the charge, or the pass-through, turns them red. |
| **Closed by** | `feat(spend): wire one run ledger through every model call in the graph`. |

Two of the seven rows above were closed when they were found and five were recorded and left open,
which is the honest ratio for a session spent asking one question. Item 17 has since closed for
hybrid deployments. Items 15 to 19 all came out of item 13's measurement:
once you are asking "does the thing the trail claims actually happen", the same question has
obvious next targets, and it keeps finding the same answer. Three of them -- 15, 19 and the tier
in 16 -- are the identical shape: **code that is correct, tested, and called from nowhere.**

**This inventory is incomplete, and it grows by measurement.** Every entry above exists because
something was run and produced a surprising answer, which means the ones not yet found are the
ones nothing has exercised. The correct response to a suspected difference is to write a test
that provokes it, not to add a defensive branch.

### 20. The README said embedding spend was accounted, and chat spend was the half left to wire

| | |
| --- | --- |
| **Difference** | Item 19 inverted, in prose. The README's list of what exists said *"Embedding spend goes through the same ledger as chat spend, on the same rules"*, and its limitations section said *"the embedding path accounts against it, but the chat calls in the graph do not yet ... the run and session ceilings bound embedding spend"*. Both halves are false: `AccountedEmbeddings` is constructed by nothing (item 19) and no chat call reaches the ledger (items 18 and 19 both say so). **Neither kind of spend is accounted.** The same README, three paragraphs further down, carried item 19's correction -- which was added next to the claim it contradicts without removing it. |
| **Also said it** | Item 12's gaps table (*"Wired, accounted"*), `retrieval/accounting.py`'s module docstring (*"accounted the same way chat spend is"*), and the changelog entry that recorded item 9 as closed. |
| **How established** | A status summary repeated the README's version back, and was checked against items 9 and 19. Then `grep -rn -i "ledger\|accounted"` over `README.md`, `docs/`, `CHANGELOG.md` and `src/`, reading every hit against what calls it. |
| **Consequence** | The rule item 19 produced -- **a class with no caller is evidence about the class** -- applies to sentences as well. A correction recorded beside a claim rather than in place of it leaves both standing, and a reader who stops at the first one leaves with the wrong answer. Every hit is now corrected in place. |
| **Not guarded** | Nothing enforces prose. The enforced record is `docs/concept-map.md`, where `AccountedEmbeddings` is **built, not wired** and a test holds it there; a sentence elsewhere that contradicts that row is found only by reading. A phrase-matching test over the README was considered and not written: it would pass the moment the wording changed, which is the vacuous guard this repository keeps finding. |
| **Recorded** | Here, and in the corrected README, item 12, `accounting.py` and the changelog. |
| **Closed by** | `docs: correct the claim that embedding spend is accounted` -- the correction, not the wiring. |
| **Since** | The wiring, 2026-09-28: chat and embedding spend both reach one run ledger now, and the README says which spend the ceilings bound and which they do not. |

### 22. A node's `config` typed as an optional union is silently not injected

| | |
| --- | --- |
| **Difference** | LangGraph passes a node its `RunnableConfig` only if it recognises the parameter's annotation. Under `from __future__ import annotations` -- every module here -- `config: RunnableConfig \| None = None` is the string `"RunnableConfig \| None"`, which it does not recognise. So it passes **nothing**, and says so only in a `UserWarning` whose text recommends that exact spelling. `RunnableConfig` and `Optional[RunnableConfig]` both work. |
| **How established** | The run ledger travels in the config, and the nodes that read it fail closed. The first run after wiring stopped at classification with "this run has no spend ledger" while `run_config` had plainly supplied one. A probe of three spellings under string annotations found the one that is dropped. |
| **Evidence** | `tests/integration/test_toolchain_blind_spots.py::test_a_config_typed_as_an_optional_union_is_silently_not_injected`, asserting the current truth, with `::test_a_required_config_is_injected` as its control. |
| **Consequence** | Every node that reads its config takes a required `config: RunnableConfig`, which also means a node exercised directly in a test has to be handed a run's config -- the ledger rule, applied to unit tests. The general point is item 3's: **a toolchain can drop something without failing, and a component that fails closed is what finds it.** A node that read an optional setting from its config would have run on without it. |
| **Recorded** | Here and in the test. Version-specific: re-check on a `langgraph` upgrade. |
| **Closed by** | Nothing to close in this code. Pinned so an upgrade that fixes or changes it is a visible inversion. |

### 23. Native structured output threw away the usage of a billed call whose reply did not parse

| | |
| --- | --- |
| **Difference** | `with_structured_output` hands the OpenAI client a pydantic class, and the client parses **inside the model call**. A reply that does not validate raises before the call returns, so a request that reached the provider, and was billed, arrives at the ledger's callback as an error with no usage. The classifier takes the native path on the cloud lane (it is recorded as native) and falls back to repair when it fails, so a misbehaving reply cost a call nothing counted. |
| **How established** | The run ledger's first wire test counted five requests at the stub and four in the book. The stub answers native requests in prose, as a misbehaving model would; the native attempt was the missing one. |
| **Evidence** | `tests/integration/test_run_ledger.py::test_a_native_structured_call_that_fails_to_parse_is_still_accounted`, and the request-count test that found it. Mutation-checked: restoring `with_structured_output` turns both red. |
| **Consequence** | The native path binds the schema as a plain JSON-schema dict and parses the reply itself, strictly -- JSON and nothing else, so a lane that does not really do native is still caught and still falls back. `response_format` is still on the wire, which item 14's control depends on. Fourth instance of item 11's shape: **a client is a convenience over a protocol, and what it declines to hand back is invisible until something depends on it.** |
| **Recorded** | Here and in `models/structured.py:invoke_structured`. |
| **Closed by** | `feat(spend): wire one run ledger through every model call in the graph`. |

### 24. The human gate approves a draft, and no action exists for it to approve — CLOSED

| | |
| --- | --- |
| **Difference** | The thesis is that a human gate approves anything irreversible. A run that passes the gate reaches `execute`, which records `irreversible_effects: []` -- because **no irreversible effect has ever existed.** Nothing proposes one: the executor's allowlist (`issue_refund`, `send_customer_email`) is declared in `tools/registry.py` and held by no agent, there is no state channel a proposal could travel in, and `execute` reads nothing that could name one. What the gate actually approves is a draft being released. The README and the concept map described a gate that pauses before anything irreversible. |
| **How established** | While designing the decider's assess node, which was to send Jev structured facts including the proposed actions. There are none: the list would be empty on every run, and the decider's verdict would have depended on nothing that varies. Then shown on a real run: a fake-lane request that asks for a refund in so many words, approved at the gate, reaches `execute` once and records no effect. |
| **Evidence** | `tests/integration/test_gate_guards_a_draft.py`, asserting the current truth so that closing this inverts named tests. `::test_an_approved_run_reaches_execute_and_nothing_irreversible_happens` is the run, with its presence half -- the run did pass the gate and reach `execute`. `::test_there_is_no_state_channel_a_proposal_could_travel_in` reads the schema, `::test_execute_ignores_a_proposal_even_when_one_is_handed_to_it` hands `execute` a fabricated approved proposal and watches it do nothing, and `::test_the_executor_allowlist_is_declared_and_held_by_no_agent` checks that nothing in `src/` takes the executor's tools. Mutation-checked: adding a proposals channel, having `execute` record one, or referencing the executor's allowlist from a node each turns its test red. |
| **Consequence** | Same shape as items 15 and 19 -- **correct in isolation, tested in isolation, connected to nothing** -- and the most load-bearing instance yet, because it is the headline claim. Every piece of the gate is real and pinned: `interrupt()` pauses, resume re-executes from the top, nothing has a side effect before the pause, `execute` is reachable only past an approval and refuses otherwise. What it guards is a release, not an effect. The tool allowlists already half-implement the privilege separation the thesis needs -- the drafter provably cannot call an irreversible tool -- and the other half, something that proposes an action for the gate to approve, was never built. |
| **Corrected** | The README now says the gate guards a draft today and names this row; the concept map carries *irreversible action executed only past the gate* and *proposals carried to the gate* as **not built**. The thesis blockquote is left verbatim as the design statement it is, with the scope stated directly beneath it. |
| **Recorded** | Here, in the README, in the concept map, and in `graph/nodes/execute.py`'s own comment, which has said so since Phase 5. |
| **Closed 2026-09-28** | The drafter's final message is a JSON object -- the draft and **proposed actions**, each a tool name and arguments -- parsed by this code, strictly on a native lane and by extraction on one that is not, and failing closed: a reply that is not the object asked for proposes nothing, and the drop is audited. Proposals are screened before the gate against the `EXECUTOR` allowlist and each tool's own schema; the rest are dropped and audited, so the human is never shown an action that cannot run. The gate shows the actions and a hash of exactly them, and an approval must carry that hash back: one that does not match what is in state -- a proposal changed after the pause -- is refused and recorded as a refusal, not run. `execute` is the executor: it checks the hash again, screens each proposal again, requires the run's ledger and checks its ceiling, and performs each approved proposal through the effect sink. |
| **What makes it safe to close** | **The only effect sink is an append-only outbox**, and `AGENTGATE_EFFECT_SINK` refuses any other value at startup -- so "this performs nothing real" is enforced by configuration rather than documented. **Every effect is keyed by run, position and argument hash, and the outbox never writes a key twice**, which is what makes a re-run of `execute` harmless: pinned by an effect written, a crash before the checkpoint, and a resume from a fresh process that must not write it again. The tool handlers still raise: an effect never goes through them. |
| **What it cost** | Two state channels (`proposed_actions`, `approved_digest`), read tolerantly so a checkpoint from before them approves a draft and performs nothing. A longer drafter prompt, carrying the JSON contract on every draft. An approval now has to name what it approves: the CLI's `approve` carries back the hash stored with the pause, and a resume that omits it is refused whenever there is anything to act on. The prompt guard in `test_tool_allowlist.py` was narrowed rather than kept -- the prompt now names the executor's tools, as a menu of what may be proposed, and still names none the drafter holds and carries no prohibition. And item 25, found on the way: the obvious mechanism for structured output turns non-compliance into a paid retry loop, so it is not used. |
| **Known limits** | The outbox is scanned in full on every record, which is fine at demo scale and linear in effects. A crash *during* a write could leave a torn last line, and the next read would fail on it -- closed, not open: nothing is performed twice, but nothing further is performed until the file is repaired. How often a real model returns the JSON asked for, rather than prose, is unmeasured; on a lane where it does not, actions simply never reach the gate. |
| **Evidence, now** | `tests/integration/test_gate_guards_a_draft.py` -- the four tests that pinned this row open, each inverted. `tests/integration/test_actions_past_the_gate.py`: performed once and read off the outbox; a non-JSON reply proposing nothing, audited; an invalid proposal dropped before the gate; approval without the hash refused; a proposal changed after the pause refused; `execute` refusing a mismatched hash on its own; the crash-and-resume case; `execute` refusing without the ledger and over the ceiling; an old checkpoint. `tests/unit/test_effects.py`: screening, the hash, the key, the sink's idempotency across instances, and the configuration refusal. `test_cli.py::test_the_packet_shows_the_action_and_approve_performs_exactly_it`, across two processes. |
| **Closed by** | `feat(gate): propose actions, approve exactly what is shown, perform each once`. |

### 25. `create_agent`'s structured output turns a model that does not comply into a paid retry loop

| | |
| --- | --- |
| **Difference** | `create_agent(response_format=...)`, as installed, gets structured output by having the model call a synthetic response tool. A model that answers with the right JSON *as text* never calls it -- and nothing treats that as an error. The agent re-prompts, each turn a request the provider bills, until LangGraph's recursion limit stops it. Non-compliance is not a refusal; it is a loop with a meter running. |
| **How established** | Probing the mechanism proposed for closing item 24, before building on it. The fake model scripted with the correct JSON, and the OpenAI-compatible stub answering correctly with native structured output on and off: **8 requests at a recursion limit of 8**, every time, and no error until the limit. |
| **Evidence** | `tests/integration/test_toolchain_blind_spots.py::test_structured_output_the_fake_cannot_satisfy_loops_to_the_recursion_limit` and `::test_every_turn_of_that_loop_is_a_billed_request` (native on and off), asserting the current truth. `::test_the_run_ledger_stops_that_loop_at_the_spend_ceiling_first`: with the model charged to a run ledger the loop ends at the ceiling, as a refusal naming what was spent, before the recursion limit. |
| **Consequence** | Not used. Item 24's proposals come from the drafter's final message as JSON, parsed by this code on the same lane-aware path classification uses, and a parse failure fails closed -- no proposals, the drop audited -- rather than being retried at the provider's expense. It also avoids depending on tool calling, which item 12 lists as never measured on any lane. And it is the run ledger earning its place a second time: under it, this loop is a visible refusal at the spend ceiling rather than a silent runaway that only a recursion limit -- a backstop, not a budget -- would end. Same family as items 15 and 22: a toolchain behaviour that is not an error, found by running it. |
| **Recorded** | Here and in the test. Version-specific: re-check on a `langchain` upgrade. |
| **Closed by** | Nothing to close in this code; the mechanism is avoided, and the pin makes a change in it visible. |
### 21. An aliased field cannot be set by its own name, and the keyword is dropped without a word

| | |
| --- | --- |
| **Difference** | A field with a validation alias cannot be set in-process by its field name. Under `case_sensitive=False` a constructor keyword is matched against the field's aliases, never the name, and `extra="ignore"` discards one that matches nothing. So `Settings(jev_api_key="x")` constructs cleanly and the key is `None`. `openai_api_key=` works only because its unprefixed alias `OPENAI_API_KEY` happens to spell the field name, and so do both LangSmith fields -- which is why the trap stayed hidden: every existing aliased field was a coincidence. |
| **How established** | Two of the B2 decider-configuration tests passed against a settings model that did not have the fields yet. The keyword was accepted, dropped, and the assertions -- one about a valid shape, one about a key's absence from `repr` -- held for a model that knew nothing about them. Caught because the other 32 went red and those two did not. |
| **Evidence** | `tests/unit/test_env_namespace.py::test_an_aliased_field_cannot_be_set_by_its_own_name`, parametrised over every aliased field whose aliases do not spell its name, so a field added later is covered without anyone adding it. Its control, `::test_a_field_whose_alias_spells_its_name_works_only_by_coincidence`, pins the three that do. Mutation-checked: `populate_by_name=True` turns the `jev_api_key` case red. |
| **Consequence** | The same shape as item 5, which is this failure for the environment -- an unknown variable dropped rather than rejected -- and not for in-process construction, which nothing guarded. **A landmine for every aliased field from here, B4's included:** a test that sets one by name tests nothing. Tests pass the key under its declared name (`test_decider_config.py:KEY` says why). Not fixed with `populate_by_name`: that would make the field name an input too, and a construction path that accepts names the environment does not is a second answer to "what does this field read". |
| **Recorded** | Here, and in `tests/unit/test_decider_config.py`. |
| **Closed by** | Nothing. Pinned as the current truth, so any change to it is a visible inversion. |

## Closing item 17: the index is the hard part

**Decided 2026-09-28: index on the contained lane.** Item 17 above records the decision, what it
costs and what it leaves open. The options are kept as they were laid out, because the reasons the
others lost are part of the decision.

The constraint that makes this awkward: **a vector index belongs to the model that built it.** A
query embedded by one model cannot meaningfully search an index built by another, so "embed
restricted queries on the contained lane" implies an answer to "what indexed the corpus".

| Option | What it costs |
| --- | --- |
| **Per-lane indexes.** One index per lane, each built by that lane's embedder; a routed request searches the index matching its effective lane. | Honest and complete, and the most expensive. Index build time and storage multiply by the number of lanes, and the offline suite gains a second index build per retrieval test. Startup or first-query latency rises for whichever index is built lazily. The sovereign lane needs a real embedding endpoint (see below) or its index is built by `HashingEmbeddings`, which would mean restricted requests search a bag-of-words index while public ones search a semantic one -- **a quality difference that correlates exactly with sensitivity**, which is worse than it sounds and needs saying out loud. |
| **Index both ways, always.** Build every index with every configured embedder up front; route the query to the matching one. | Same storage and build cost as per-lane indexes with none of the laziness, so it is strictly worse on cost and strictly better on first-query latency. Simpler to reason about: no lane can arrive and find no index. Offline suite unaffected if the fake lane counts as one embedder. |
| **A sovereign embedding endpoint.** `AGENTGATE_SOVEREIGN_EMBEDDING_MODEL` plus the existing base URL; restricted queries embed there. | Smallest conceptual change and it composes with either option above -- but it does not stand alone, because it still leaves the corpus indexed by whichever lane built it. Adds a setting, a capability-matrix row, and a price entry, and it is another networked path with no live verification: Ollama and vLLM serve embeddings, and neither has ever been run against. |
| **Index on the contained lane only.** Build one index with the most contained embedder available and let every lane search it. | Cheapest, one index, no routing logic in retrieval at all, and it makes the guarantee unconditional -- no corpus content reaches a third party either. The cost is retrieval quality for every request including public ones, and on a fake-lane deployment it means the hashing embedder indexes production, which is not a thing to ship quietly. |
| **Do not retrieve for restricted requests.** Route restricted requests down a path with no research. | No embedding egress at all and nothing to index twice. It changes what the product does rather than how it does it: restricted requests get the weakest answers, which inverts the usual expectation and should be a stated product decision rather than a consequence of an infrastructure constraint. |
| **Keep it open and scope the claim.** What was in force until 2026-09-28: the leak is pinned by a passing test, the README says retrieval is not covered, and no claim is made that restricted content never reaches the cloud. | Costs nothing and fixes nothing. Defensible only for as long as the claim stays scoped, which is why the scoping is enforced by the tests above rather than by remembering. |

### Item 19 comes first, for any option that keeps embedding on the cloud lane

**Per-lane indexes, indexing both ways, and keeping it open all leave embedding spend on the cloud
lane, and that spend is currently accounted by nothing.** `AccountedEmbeddings` is not wired -- item
19 -- so every one of those options ships an egress whose cost no ceiling can see. Wiring it is a
prerequisite rather than a follow-up.

And wiring it **changes when ceilings trip**. `AccountedEmbeddings.check()` runs after every batch,
which is deliberate: a runaway index trips the ceiling while it is running rather than reporting the
bill afterwards. The consequence for these options is that an index build becomes a thing that can
*fail partway through* on a spend limit, where today it cannot fail at all. Per-lane indexes make
that worse in proportion to the number of lanes, and a first-query lazy build makes it worse again
by moving the failure into a request rather than into startup.

Indexing on the contained lane is the only option that sidesteps this, because there is no cloud
embedding spend left to account -- **on a deployment with a sovereign lane.** A cloud-only
deployment's contained lane is the cloud, so it still embeds there and item 19 still applies to it.

*Item 19 closed on 2026-09-28, and the prediction above held: an index build can now fail partway
through on a ceiling, and did in the test written to make it. Kept as written, as the reasoning.*

### What has now been measured

`scripts/measure_retrieval.py`, **re-run 2026-09-28 on `main` at `38522da`**, against the
committed 20-chunk corpus, with the questions and expected chunks committed in
`scripts/retrieval_questions.json`. Seven runs; build time is the median of three builds within each
run, and the range across runs is given because the range is the finding.

| | `HashingEmbeddings` | `AccountedEmbeddings` -> stub |
| --- | --- | --- |
| index build, median of 3, range over 7 runs | **0.010 - 0.017 s** | **0.006 - 0.017 s** |
| embedding calls for 20 chunks | 0 | **1** (one batch) |
| top-4 hit rate, 10 questions | **90%** (9/10), every run | 30% -- see below |
| shared-vocabulary questions | 5/5 | 3/5 |
| synonym-only questions | 4/5 | 0/5 |

**Why the second column moved, and why that is not a different comparison.** The first version of
this table, run 2026-09-27, measured `OpenAIEmbeddings` against the stub: 0.020 s to build, 10% hit
rate, 0/5 and 1/5. The run ledger then made the cloud embedder `AccountedEmbeddings` over the raw
OpenAI client, which is what production runs, so that is what this column now times. Two things
changed with it. The stub hashes whatever representation of the text it is sent, and the old client
sent token ids where the new one sends strings -- so the meaningless embedder became a *different*
meaningless embedder, and 10% and 30% are two draws from the same kind of control, both consistent
with a 20% floor on ten questions. And the old client tokenised client-side before sending, which
the new one does not. The comparison is the same one: hashing, against a deliberately meaningless
embedder, against chance. The hashing column did not change at all.

**Build cost still does not distinguish the options -- more plainly than before.** One batched call
and single-digit milliseconds either way against a loopback stub, and the two ranges overlap: the
provider path was faster in most runs and not in all of them, so the ordering is not stable at this
size. The first version read 0.013 s against 0.020 s and the provider path slower; both orderings
are noise at this scale. A real endpoint adds a round trip, so treat the provider figure as a floor
-- but at this corpus size, indexing either way costs nothing worth deciding on. **The decision on
item 17 never rested on this line**: it rested on the hit rate, which reproduced exactly.

**The hit rate does.** 90% against a **20% chance floor** (top-4 of 20), and 4 of the 5
synonym-only questions were found -- including "can a customer still get their money back after six
weeks", which shares one content word with its answer. The stub column is a deliberately
meaningless embedder and scored 30% (10% on 2026-09-27, as a different meaningless embedder), which
is what makes the 90% readable: if a hash of the input had scored well, the questions would be
answerable by anything.

**The one miss is the honest limit.** *"What is the deadline for the write-up after something goes
wrong?"* has zero content words in common with the section it should find, and term frequency over
a fixed vocabulary has nothing to work with. It retrieved the refund eligibility window instead.
That is what a bag of words cannot do, stated by measurement rather than by argument.

**What this does not license.** The number belongs to *this corpus at this size*. Top-4 of 20 is a
generous test, and ADR 0010 already records the collision curve that makes a hashing index degrade
as vocabulary grows -- 7% of terms sharing a bucket at 4096 dimensions against 301 distinct terms
today. A corpus an order of magnitude larger would need re-measuring before the same conclusion
could be drawn, and this table is only as good as its date.

**Still not measured**, and neither can be from here: retrieval quality of a real cloud embedder,
which needs a key and a live probe -- the script refuses to print a figure it cannot observe -- and
whether a sovereign embedding endpoint reports usage at all, which items 11 and 18 both suggest
treating as unknown until seen on a wire.

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
