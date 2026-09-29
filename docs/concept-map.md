# Concept map

Where each LangGraph and LangChain concept actually lives in this repository.

**Maintained as the build proceeds, not written at the end.** A map assembled retrospectively
records where things ended up; this one is updated in the same commit as the code it points at,
so a row marked *done* has a file behind it today.

**The build is paused at Phase 5. Phases 6 to 9 are planned and unstarted, and a row here is a
claim about what exists rather than a promise about what will.** Rows marked *not built* have
nothing behind them today; some name the file they would live in, which is a plan and not a
commitment.

**Three statuses, and the third one exists because two rows were wrong.** *done* means it is in
the system. *not built* means it is absent. **built, not wired** means the code exists and is
tested and **nothing in `src/` calls it** -- so it is not behaviour this system has, however
green its tests are. `with_retry` and `with_fallbacks` sat at *done* for four phases on the
strength of `build_resilient_model`, which no node called; leak inventory item 15, closed now
that the factory is the one every model-calling node is given. The status is enforced the same
way the other two are: a *built, not wired* row whose symbol gains a caller anywhere in `src/`
fails the build until someone moves the row -- which is what happened here, and the row moved
only once a test read the retry off an endpoint's request log.

---

## LangGraph

| Concept | Status | Location |
| --- | --- | --- |
| Typed state with `add_messages` reducer | **done** | `graph/state.py:AgentState.messages` |
| `operator.add` reducer on append-only findings | **done** | `graph/state.py:AgentState.findings`, `.audit_trail` |
| Last-write-wins fields + `InvalidUpdateError` note | **done** | `graph/state.py` docstring; demonstrated in `tests/integration/test_parallel_writes.py` |
| Conditional edge returning a `Literal` | **done** | `graph/routing.py:route_by_policy` |
| `Command` combining update and `goto` | **done** | `graph/nodes/supervisor.py:supervise` |
| `Command(graph=Command.PARENT)` handoff | **done** | `graph/subgraphs/retrieval.py:deliver` |
| `Send` fan-out, reducer fan-in | **done** | `graph/nodes/researcher.py:dispatch`; fan-in via `graph/state.py:findings` |
| Fan-out width capped before dispatch | **done** | `graph/nodes/researcher.py:cap_fan_out`; `tests/integration/test_fan_out.py` |
| Fan-out survives a failing branch, visibly | **done** | `graph/subgraphs/retrieval.py`; `graph/nodes/finalise.py:research_gaps` |
| Compiled subgraph used as a node | **done** | `graph/subgraphs/retrieval.py:build_retrieval_subgraph` |
| `InMemorySaver` | **done** | `graph/build.py:checkpointer_for` |
| `SqliteSaver` | **done** | `graph/build.py:checkpointer_for` |
| `PostgresSaver` | **done** (wired; server-backed run is Phase 8) | `graph/build.py:checkpointer_for` |
| Checkpointer chosen by config, never by editing code | **done** | `graph/build.py`; asserted in `tests/integration/test_graph.py` |
| `interrupt()` approval gate | **done** | `graph/nodes/approval.py:approval_gate` |
| Resume with `Command(resume=...)` | **done** | `tests/integration/test_approval_gate.py` |
| Test proving node re-execution from top on resume | **done** | `test_the_interrupted_node_re_executes_from_its_top_on_resume` |
| No side effect before the interrupt | **done** | `graph/nodes/approval.py`; `test_nothing_above_the_pause_writes_an_audit_event` |
| Reject-with-feedback → revise → back to the gate | **done** | `graph/nodes/approval.py`; `graph/nodes/supervisor.py` |
| `execute` reachable only past the gate | **done** | `graph/nodes/execute.py` |
| Irreversible action executed only past the gate | **done** | `graph/nodes/execute.py:execute`, holding the `EXECUTOR` allowlist; `tests/integration/test_gate_guards_a_draft.py`. Leak inventory item 24, closed |
| Proposed actions carried to the gate as structured data | **done** | `graph/state.py:AgentState.proposed_actions`; `effects/proposals.py:screen_proposals` drops what the executor could not run, before the gate |
| Approval authorises exactly what was shown | **done** | `effects/proposals.py:digest_of`; `graph/nodes/approval.py:approval_gate` refuses a hash that no longer matches; `tests/integration/test_actions_past_the_gate.py` |
| Effects performed exactly once, across a crash | **done** | `effects/proposals.py:effect_key`; `effects/sink.py:OutboxSink`; the crash-and-resume case in `tests/integration/test_actions_past_the_gate.py` |
| One effect sink, and configuration refuses any other | **done** | `effects/sink.py:build_effect_sink`; `config.py:EffectSinkBackend`; `tests/unit/test_effects.py` |
| Iteration cap exercised end to end | **done** | `test_a_reviewer_who_never_approves_is_stopped_by_the_iteration_cap` |
| `Store` long-term memory, namespaced per user | not built | `config.py` `StoreBackend` exists |
| Store vs. checkpoint distinction documented | **done** | `config.py` `StoreBackend` docstring; `.env.example` |
| `get_state_history` | not built | `cli.py` |
| Fork from a past checkpoint | not built | `cli.py` |
| `update_state(as_node=...)` | not built | `cli.py` |
| Time travel exposed via CLI | not built | Phase 6 primitives absent; `cli.py --help` says so |
| `stream_mode=["updates", "messages"]` | **done** | `cli.py:_stream`; SSE surface still to come |
| Streaming surfaced as SSE | not built | `api/` |
| CLI: run, resume, approve, reject | **done** | `cli.py`; `tests/integration/test_cli.py` |
| Resume across processes, not just sessions | **done** | `tests/integration/test_cli.py::test_approving_from_a_fresh_process_resumes_the_same_run` |
| `recursion_limit` | **done** | passed at invoke; ordering vs the budget guard asserted in `tests/integration/test_graph.py` |
| Iteration counter + budget check in a conditional edge | **done** | `graph/routing.py:route_by_budget`; counter in `state.iterations` |

## LangChain

| Concept | Status | Location |
| --- | --- | --- |
| `init_chat_model` as the provider abstraction | **done** | `models/registry.py:_init_openai_compatible` |
| `base_url` pattern for the sovereign lane | **done** | `models/registry.py:build_model`; `config.py:sovereign_base_url` |
| LCEL pipe chain | not built | `graph/nodes/researcher.py` |
| `RunnableParallel` | not built | `graph/nodes/researcher.py` |
| `with_structured_output` — classifier | **done** | `graph/nodes/classify.py` via `models/structured.py:invoke_structured` |
| `with_structured_output` — supervisor router | not built | `graph/routing.py`, once the supervisor chooses between workers |
| Validate-and-repair structured output fallback | **done** | `models/structured.py:invoke_with_repair` |
| ↳ its test, against a lane that really lacks native support | **done** | `tests/integration/test_sovereign_lane_structured_output.py::test_repair_loop_rescues_prose_wrapped_json_from_the_sovereign_lane` |
| ↳ the leak it exists for, demonstrated | **done** | same file, `::test_native_structured_output_fails_against_the_sovereign_lane` |
| `with_retry` | **done** | `models/resilient.py:ResilientChatModel`, built by `models/registry.py:build_resilient_model` — the factory every model-calling node is given |
| `with_fallbacks` | **done** | same, one tier down and never a lane out |
| ↳ tested against induced HTTP failures | **done** | `tests/integration/test_resilience.py` |
| ↳ reached by a node, read off an endpoint | **done** | `tests/integration/test_resilience_wiring.py` |
| `@tool` with Pydantic arg schemas | **done** | `tools/registry.py` |
| Tools bound to the model | **done** | `graph/nodes/drafter.py` via `tools_for(Agent.DRAFTER)` |
| `ToolNode` execution | not built | `create_agent` runs the tool loop internally; no explicit `ToolNode` in this repository |
| Tool failures summarised into state, not raised | **done** | `tools/allowlist.py:wrap_tool_call` |
| Per-agent tool allowlist, enforced not documented | **done** | `tools/allowlist.py:AllowlistMiddleware`; `tests/integration/test_tool_allowlist.py` |
| `trim_messages` | not built | `graph/nodes/summarise.py` |
| Summarisation node | not built | `graph/nodes/summarise.py` |
| Retrieval over the committed corpus | **done** | `retrieval/`; `corpus/` |
| Hybrid vs. dense decision documented | **done** | `docs/adr/0010-dense-in-process-retrieval-by-default.md` |
| `create_agent` in exactly one worker | **done** | `graph/nodes/drafter.py:draft` |
| ADR on why explicit graphs elsewhere | not built | `docs/adr/` |

## Beyond the brief

Things this build surfaced that were not on the original list, kept because each is load-bearing.

| Concept | Status | Location |
| --- | --- | --- |
| Capability matrix with recorded provenance | **done** | `models/registry.py:CAPABILITY_MATRIX` |
| Suite fails on an unmeasured networked-lane entry | **done** | `tests/unit/test_registry.py::test_no_networked_lane_entry_rests_on_assumption` |
| Deterministic fake lane with honest usage metadata | **done** | `models/fake.py` |
| Committed OpenAI-compatible stub server | **done** | `tests/doubles/openai_compatible.py` |
| Run ledger reaching every model call in the graph, ceiling checked after each | **done** | `graph/build.py:run_config`; `guardrails/run_ledger.py:ledger_of`, `guardrails/run_ledger.py:accounted`; `tests/integration/test_run_ledger.py` |
| Run spend survives a pause and resume in another process | **done** | `graph/build.py:resume_config`; `graph/state.py:AgentState.spend`, written by `graph/nodes/supervisor.py:supervise` |
| Session ceiling across runs in one process | not built | `guardrails/spend.py` has the parent-ledger mechanism; the CLI runs one run per process, and the multi-run surface it bounds is Phase 7's API |
| Streamed calls report their token usage | **done** | `models/registry.py:_init_openai_compatible` sets `stream_usage`. Leak inventory item 18, closed |
| Unmeasured usage is an error, never a free call | **done** | `guardrails/spend.py:usage_of` |
| Embedding spend accounted in the same ledger | **done** | `retrieval/embeddings.py:build_embeddings` constructs `retrieval/accounting.py:AccountedEmbeddings`, billed via `guardrails/run_ledger.py:charging`. Leak inventory items 9 and 19, closed |
| ↳ why that row changed twice | **done** (pinned) | `tests/integration/test_routed_lane_enforcement.py::test_every_embedding_call_is_accounted_in_the_run_ledger`; it said *done* while nothing constructed the class (item 19), then *built, not wired*, and is now wired |
| Decider in front of the approval gate (Jev, LLM, fake), fail-closed to a human | **done** | `graph/nodes/assess.py:assess` calls `decider/build.py:build_decider` once per draft, before the gate, on cloud-routed requests only; `tests/integration/test_decider_gate.py` |
| Decider sent structured facts, never the draft | **done** | `graph/nodes/assess.py:decision_facts`; `tests/integration/test_decider_gate.py::test_the_decider_is_sent_structured_facts_and_never_the_draft` |
| Gate checks deterministic preconditions in code before reading a verdict | **done** | `graph/nodes/approval.py:auto_approval`; `decider/assessment.py:DETERMINISTIC_PRECONDITIONS` |
| Shadow mode: verdict recorded beside the human's decision, never acted on | **done** | `graph/nodes/approval.py`, the `approved_by` and `assessment` fields of the approval event |
| Decider capability rows with provenance, STUB until a live probe | **built, not wired** | `decider/capabilities.py:DECIDER_CAPABILITY_MATRIX`. A record the probe replaces; no code consults it |
| Output check: citation provenance | **done** | `guardrails/output.py:check_provenance`; `tests/unit/test_output_guardrail.py` |
| Durable append-only audit trail | **done** | `audit/writer.py:write_events`; `graph/nodes/finalise.py` |
| Trail readable with no imports from agentgate | **done** | `tests/integration/test_audit_trail.py::test_the_trail_reads_with_the_standard_library_alone` |
| Gate coverage enforced by discovery | **done** | `tests/integration/test_audit_trail.py::test_every_discovered_gate_wrote_an_event` |
| Per-call-class output ceilings | **done** | `config.py:max_tokens_for`; asserted on the wire in `tests/integration/test_resilience.py` |
| Wire-level assertions, never client attributes | **done** | `tests/integration/test_resilience.py:on_the_wire` |
| Unknown `AGENTGATE_*` variable rejected with a suggestion | **done** | `config.py:_reject_unknown_variables` |
| No undeclared unprefixed environment reads | **done** | `tests/unit/test_env_namespace.py` |
| Tracing off by default, backend as deployment choice | **done** (config) | `config.py:TracingBackend`; instrumentation in Phase 8 |
| Test suite cannot phone home | **done** | `tests/conftest.py:TRACING_PREFIXES`; `tests/unit/test_offline_isolation.py` |
| Live suite bounded by estimate, confirmation, and hard abort | **done** | `scripts/run_live.py` |
| Capability discovery split from capability enforcement | **done** | `scripts/probe_capabilities.py` produces; `tests/live/` enforces |
| Ceilings derived from a measured run, not chosen | **done** | `scripts/measure_run.py`; basis recorded in `.env.example` |
| Policy gate fails closed on an unparseable verdict | **done** | `graph/nodes/classify.py`; `tests/unit/test_nodes.py` |
| Routed lane applied at model construction, asserted on the wire | **done** | `models/registry.py:build_model`; `tests/integration/test_routed_lane_enforcement.py` |
| Route narrows the configured lane and cannot widen it | **done** | `config.py:narrower_of`; `config.py:CONTAINMENT` |
| Classification runs on the most contained lane available | **done** | `config.py:most_contained_lane`; `graph/nodes/classify.py` |
| Retrieval embeds on the most contained lane available, one index for every route | **done** | `retrieval/embeddings.py:build_embeddings` reads `config.py:most_contained_lane`. Closes ADR 0004 item 17 on hybrid deployments; a cloud-only deployment still embeds on the cloud |
| Embedding requests can be observed on the wire | **done** | `retrieval/embeddings.py`; `tests/doubles/openai_compatible.py:decode_embedding_input` |
| The stub server speaks SSE, so a networked lane can be driven from the CLI | **done** | `tests/doubles/openai_compatible.py`; `tests/integration/test_cli.py` |
| An agentgate error from the CLI is reported, not traced | **done** | `cli.py:main`; `tests/integration/test_cli.py` |
| Routed **tier** applied at model construction | not built | would need a `tier` channel in `graph/state.py`; `bind_lane` binds one and nothing carries it. Leak inventory item 16 |
| Crash mid-run resumes from the last checkpoint | **done** | `tests/integration/test_crash_and_resume.py` |
| `partial` erases node signatures from mypy | **done** (pinned) | `tests/integration/test_toolchain_blind_spots.py` |
| `interrupt_before` in invoke config is silently ignored | **done** (pinned) | `tests/integration/test_toolchain_blind_spots.py` |

## Decision records

| ADR | Subject |
| --- | --- |
| [0004](adr/0004-provider-abstraction-and-lanes.md) | Provider abstraction, three lanes, **and the leak inventory** |
| [0006](adr/0006-dependency-version-pins.md) | Pinning the current 1.x line rather than 1.0.x |
| [0007](adr/0007-configuration-validated-at-startup.md) | Validation at startup, not at import |
| [0008](adr/0008-tracing-backend-is-a-deployment-decision.md) | OpenTelemetry as instrumentation; backend as deployment choice |
| [0009](adr/0009-env-is-a-shared-namespace.md) | `.env` is read, not owned |
| [0010](adr/0010-dense-in-process-retrieval-by-default.md) | Dense in-process retrieval by default |
| [0011](adr/0011-checkpoints-are-a-durable-format.md) | A checkpoint is a durable format, so state channels hold data |
| [0012](adr/0012-the-decider-is-a-declared-egress.md) | The decider is a declared egress that can only ask for a human |

Pending: 0001 (LangGraph over a custom loop), 0002 (supervisor over swarm),
0003 (checkpointer selection), 0005 (interrupt placement and idempotency).
