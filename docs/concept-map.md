# Concept map

Where each LangGraph and LangChain concept actually lives in this repository.

**Maintained as the build proceeds, not written at the end.** A map assembled retrospectively
records where things ended up; this one is updated in the same commit as the code it points at,
so a row marked *done* has a file behind it today. Rows marked *pending* name the phase that
will land them, and are not claims about anything that exists.

Status: **Phase 2 complete.** Phases 3 to 9 pending.

---

## LangGraph

| Concept | Status | Location |
| --- | --- | --- |
| Typed state with `add_messages` reducer | pending — Phase 3 | `graph/state.py` |
| `operator.add` reducer on append-only findings | pending — Phase 3 | `graph/state.py` |
| Last-write-wins fields + `InvalidUpdateError` note | pending — Phase 3 | `graph/state.py` |
| Conditional edge returning a `Literal` | pending — Phase 3 | `graph/routing.py` |
| `Command` combining update and `goto` | pending — Phase 3 | `graph/nodes/supervisor.py` |
| `Command(graph=Command.PARENT)` handoff | pending — Phase 4 | `graph/subgraphs/retrieval.py` |
| `Send` fan-out, reducer fan-in | pending — Phase 4 | `graph/nodes/researcher.py` |
| Compiled subgraph used as a node | pending — Phase 4 | `graph/subgraphs/retrieval.py` |
| `InMemorySaver` | pending — Phase 3 | `graph/build.py` |
| `SqliteSaver` | pending — Phase 3 | `graph/build.py` |
| `PostgresSaver` | pending — Phase 3 | `graph/build.py` |
| Checkpointer chosen by config, never by editing code | **done** | `config.py` `CheckpointerBackend`; selection in Phase 3 |
| `interrupt()` approval gate | pending — Phase 5 | `graph/nodes/approval.py` |
| Resume with `Command(resume=...)` | pending — Phase 5 | `graph/nodes/approval.py` |
| Test proving node re-execution from top on resume | pending — Phase 5 | `tests/integration/` |
| No side effect before the interrupt | pending — Phase 5 | `graph/nodes/approval.py` |
| `Store` long-term memory, namespaced per user | pending — Phase 6 | `config.py` `StoreBackend` exists |
| Store vs. checkpoint distinction documented | **done** | `config.py` `StoreBackend` docstring; `.env.example` |
| `get_state_history` | pending — Phase 6 | `cli.py` |
| Fork from a past checkpoint | pending — Phase 6 | `cli.py` |
| `update_state(as_node=...)` | pending — Phase 6 | `cli.py` |
| Time travel exposed via CLI | pending — Phase 6 | `cli.py` |
| `stream_mode=["updates", "messages"]` | pending — Phase 7 | `api/` |
| Streaming surfaced as SSE | pending — Phase 7 | `api/` |
| `recursion_limit` | **done** (configured) | `config.py:recursion_limit`; wired in Phase 3 |
| Iteration counter + budget check in a conditional edge | pending — Phase 3 | `graph/routing.py` |

## LangChain

| Concept | Status | Location |
| --- | --- | --- |
| `init_chat_model` as the provider abstraction | **done** | `models/registry.py:_init_openai_compatible` |
| `base_url` pattern for the sovereign lane | **done** | `models/registry.py:build_model`; `config.py:sovereign_base_url` |
| LCEL pipe chain | pending — Phase 3 | `graph/nodes/classify.py` |
| `RunnableParallel` | pending — Phase 4 | `graph/nodes/researcher.py` |
| `with_structured_output` — classifier | pending — Phase 3 | `graph/nodes/classify.py` (mechanism in `models/structured.py`) |
| `with_structured_output` — supervisor router | pending — Phase 3 | `graph/routing.py` |
| Validate-and-repair structured output fallback | **done** | `models/structured.py:invoke_with_repair` |
| ↳ its test, against a lane that really lacks native support | **done** | `tests/integration/test_sovereign_lane_structured_output.py::test_repair_loop_rescues_prose_wrapped_json_from_the_sovereign_lane` |
| ↳ the leak it exists for, demonstrated | **done** | same file, `::test_native_structured_output_fails_against_the_sovereign_lane` |
| `with_retry` | **done** | `models/registry.py:build_resilient_model` |
| `with_fallbacks` | **done** | `models/registry.py:build_resilient_model` |
| ↳ tested against induced HTTP failures | **done** | `tests/integration/test_resilience.py` |
| `@tool` with Pydantic arg schemas | pending — Phase 4 | `tools/` |
| Tools bound to the model | pending — Phase 4 | `tools/` |
| `ToolNode` execution | pending — Phase 4 | `graph/build.py` |
| Tool failures summarised into state, not raised | pending — Phase 4 | `tools/` |
| Per-agent tool allowlist | pending — Phase 4 | `tools/` |
| `trim_messages` | pending — Phase 6 | `graph/nodes/summarise.py` |
| Summarisation node | pending — Phase 6 | `graph/nodes/summarise.py` |
| Retrieval over the committed corpus | pending — Phase 4 | `retrieval/` |
| Hybrid vs. dense decision documented | pending — Phase 4 | ADR |
| `create_agent` in exactly one worker | pending — Phase 4 | `graph/nodes/drafter.py` |
| ADR on why explicit graphs elsewhere | pending — Phase 4 | `docs/adr/` |

## Beyond the brief

Things this build surfaced that were not on the original list, kept because each is load-bearing.

| Concept | Status | Location |
| --- | --- | --- |
| Capability matrix with recorded provenance | **done** | `models/registry.py:CAPABILITY_MATRIX` |
| Suite fails on an unmeasured networked-lane entry | **done** | `tests/unit/test_registry.py::test_no_networked_lane_entry_rests_on_assumption` |
| Deterministic fake lane with honest usage metadata | **done** | `models/fake.py` |
| Committed OpenAI-compatible stub server | **done** | `tests/doubles/openai_compatible.py` |
| Spend ledger with run and session ceilings | **done** | `guardrails/spend.py` (wired into the graph in Phase 5) |
| Unmeasured usage is an error, never a free call | **done** | `guardrails/spend.py:usage_of` |
| Per-call-class output ceilings | **done** | `config.py:max_tokens_for`; asserted on the wire in `tests/integration/test_resilience.py` |
| Wire-level assertions, never client attributes | **done** | `tests/integration/test_resilience.py:on_the_wire` |
| Unknown `AGENTGATE_*` variable rejected with a suggestion | **done** | `config.py:_reject_unknown_variables` |
| No undeclared unprefixed environment reads | **done** | `tests/unit/test_env_namespace.py` |
| Tracing off by default, backend as deployment choice | **done** (config) | `config.py:TracingBackend`; instrumentation in Phase 8 |
| Test suite cannot phone home | **done** | `tests/conftest.py:TRACING_PREFIXES`; `tests/unit/test_offline_isolation.py` |
| Live suite bounded by estimate, confirmation, and hard abort | **done** | `scripts/run_live.py` |

## Decision records

| ADR | Subject |
| --- | --- |
| [0004](adr/0004-provider-abstraction-and-lanes.md) | Provider abstraction, three lanes, **and the leak inventory** |
| [0006](adr/0006-dependency-version-pins.md) | Pinning the current 1.x line rather than 1.0.x |
| [0007](adr/0007-configuration-validated-at-startup.md) | Validation at startup, not at import |
| [0008](adr/0008-tracing-backend-is-a-deployment-decision.md) | OpenTelemetry as instrumentation; backend as deployment choice |
| [0009](adr/0009-env-is-a-shared-namespace.md) | `.env` is read, not owned |

Pending: 0001 (LangGraph over a custom loop), 0002 (supervisor over swarm),
0003 (checkpointer selection), 0005 (interrupt placement and idempotency).
