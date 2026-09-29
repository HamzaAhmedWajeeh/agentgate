# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Retries and fallbacks, reaching an endpoint at last** (leak inventory item 15, closed).
  `build_resilient_model` is now the model factory every model-calling node is given, so a
  transient provider failure is retried on the tier that failed and a durable one falls back to
  the cheap tier. It returns a `ResilientChatModel` rather than a composed `Runnable`, because
  `create_agent` takes a `BaseChatModel` and calls `bind_tools` on it -- a composed chain is
  accepted at construction and fails at the first invocation, which put the drafter out of reach.
  **A fallback narrows the lane and never widens it:** both leaves are built from one routed lane
  through `build_model`, so there is no second answer to which lane is more contained, and a
  sovereign endpoint that goes down is answered by the sovereign endpoint or by nothing.
  `tests/integration/test_resilience_wiring.py` reads all of it off two endpoints' request logs
  rather than constructing a chain of its own.
- A rejection predicate on the OpenAI-compatible stub, so a test can put one tier out of service
  and leave the other healthy. `fail_first_n` can only express "the first few", which is the wrong
  shape for the outage a fallback exists for.

### Fixed

- **A retry no longer overrules a guard.** `with_retry` retries on every exception by default, and
  the run ledger raises its ceiling error from inside the callback of the call that crossed it --
  so wiring the resilient path turned a budget ceiling into four more billed calls after the run
  was supposed to have stopped. Nothing deriving from `AgentgateError` is retried now. The two
  ceiling tests in `tests/integration/test_run_ledger.py` are what caught it.
- **The model that answered is the model that is billed.** `accounted` pushes the ledger callback
  down into a resilient model's leaves instead of wrapping the composite. A callback on the
  composite reads the primary's identifier from its own metadata, so a reply the cheap fallback
  produced during an outage would have been priced at capable-tier rates, silently. Same shape as
  item 11.

- `env/sovereign.env` and `env/hybrid.env`: deployment examples with every setting stated -- no profile
  variable -- including the effect sink and outbox path. Secrets are left blank and model identifiers
  are placeholders; a test holds each file to the settings model and loads it once its secrets are
  supplied. The hybrid example runs Jev in shadow mode with no threshold set.
- ADR 0012, the decider: a declared egress on the cloud lane only, that can approve or ask a human but
  never reject, shadow by default with thresholds that must come from measured agreement (none
  exists), preconditions checked in code before any verdict, the assess node separate from the gate,
  no SDK, and the injection limitation stated plainly -- structured facts narrow the surface and do
  not close it, because a proposal's arguments are model-authored.
- **The decider in front of the approval gate** (Part B, B5). An `assess` node, between the
  supervisor and the gate, asks the decider once per draft -- only on a cloud-routed request, and
  charged to the run ledger -- and stores its verdict as JSON. It is sent structured facts: the
  routed lane, the finding count, the denied tools, the provenance result and the proposed actions;
  never the draft. The gate checks the deterministic preconditions (provenance passed, no denied
  tool) in code before it reads the verdict at all, and approves in a human's place only in enforce
  mode, when the assessed proposals still hash to what is in state and every threshold holds.
  Everything else -- shadow mode, a restricted route, any decider failure, any unmet condition -- is
  a human, and the approval event records who approved alongside the whole stored verdict, so shadow
  agreement can be measured from the trail. A skipped assessment overwrites any earlier verdict.
- ADR 0004 item 26: `update_state` on a paused interrupt drops the pause silently. Written as the
  paused node, the resume value is discarded and a static successor runs without the human's
  decision; with no node named, this graph's run ends and the next resume is a no-op. This graph
  survives because the approval gate leaves only by `Command` -- pinned, and mutation-checked by
  adding a static edge out of the gate, which lets an approval written as the gate reach `execute`.
- **Actions past the gate** (ADR 0004 item 24, closed). The drafter's final message is a JSON
  object -- the draft and proposed actions, each a tool name and arguments -- parsed strictly by
  this code and failing closed. Proposals the executor could not run are dropped before the gate and
  audited. The gate shows the actions and a hash of exactly them; an approval must carry that hash,
  and one that no longer matches is refused rather than run. `execute` is the executor: it holds the
  `EXECUTOR` allowlist, checks the hash and the run ledger's ceiling, and performs each approved
  action once, keyed by run, position and argument hash, through the effect sink. The only sink is
  an append-only outbox, and `AGENTGATE_EFFECT_SINK` refuses any other value at startup. A crash
  after an effect is written and before the checkpoint does not perform it twice. The CLI shows the
  actions on the packet, and `approve` carries back the hash of what was shown.
- ADR 0004 item 25: `create_agent`'s structured output turns a model that does not comply into a
  paid retry loop -- 8 billed requests at a recursion limit of 8, from the fake and from the stub with
  native structured output on and off, and no error until the limit. Pinned in
  `test_toolchain_blind_spots.py`, with the run ledger shown stopping it at the spend ceiling first.
  Not used: item 24's proposals are parsed from the drafter's final message instead.
- **One run ledger reaching every model call in the graph.** `run_config` creates it per run and
  carries it in the config; classification, the drafter's whole `create_agent` loop and the
  embeddings behind retrieval are charged to it, and the token and dollar ceilings are checked after
  every call, so a crossing stops the run at that call. Model-calling nodes refuse to run without a
  ledger. The spend so far is written to a `spend` state channel at every supervisor turn, and
  `resume_config` starts a resumed run's ledger from it -- so an approval in another process does not
  reset the ceiling. The CLI reports what the run spent, and a crossed ceiling exits 2 with the
  message and no traceback. What the ceilings do not bound is in the README: the session ceiling,
  calls outside the graph, and spend lost to a crash between a call and the next turn.
- ADR 0004 items 22 and 23, both found by the ledger. 22: under `from __future__ import annotations`
  LangGraph silently does not inject a node's `config` typed `RunnableConfig | None`, and warns
  recommending that spelling; pinned in `test_toolchain_blind_spots.py`. 23: native structured
  output let the client parse inside the call, so a billed reply that failed to validate reported no
  usage -- five requests at the stub, four in the book.
- The decider, `decider/`, with three backends behind `build_decider`: `jev` (TypeSafe's
  `POST /v1/systemone` over plain `httpx`, no SDK), `llm` (the cloud chat lane asked the route
  question) and `fake` (deterministic, and unscripted it asks a human). Every outcome is an
  `Assessment`, and every failure -- a timeout, 401, 422, a second 429 or 5xx, a non-JSON body, a
  missing field, a missing usage block, or a response from a model other than the pinned one --
  is an assessment with no route, which asks a human. At most one retry, on 429 or 5xx, honouring
  `retry-after` up to five seconds. Usage is recorded against the pinned model before the answer
  is judged, and a crossed spend ceiling aborts rather than becoming a human review. Confidence is
  the reported field, never recomputed. The `llm` backend reports a route and nothing numeric, so
  it can never meet the enforce-mode thresholds. **Built, not wired:** nothing calls
  `build_decider` until the assess node exists, so no request is assessed today.
- `usage_of` reads a raw usage block as well as a chat reply, and refuses a partial one: on an API
  that charges for input only, a block without `input_tokens` is an unmeasured call, not a free one.
- A TypeSafe stub in `tests/doubles/`, shaped from the published API reference, with a request log
  of path, headers and body, and every documented failure status: 422, 429 and 529, plus 500.
- `DECIDER_CAPABILITY_MATRIX`, every Jev row `STUB` until a live probe runs, and
  `scripts/probe_capabilities.py jev`, which makes one call and emits `LIVE_PROBE` rows -- and
  refuses to run against anything but the official endpoint, so a stub cannot be recorded as the
  real thing.
- The offline suite strips every `TYPESAFE_*` variable, which covers the four TypeSafe's SDKs
  read on their own -- key, base URL, default model, log level -- and any a later release adds.
  `TYPESAFE_API_KEY` is also a declared alias of the decider key, so a developer with TypeSafe
  configured would otherwise have handed every test a live decider key. Named case in
  `test_offline_isolation.py`: the variables are exported before isolation runs, and none
  survives.
- ADR 0004 item 21: a settings field with a validation alias cannot be set by its own name. The
  keyword is matched against the aliases and silently dropped, so `Settings(jev_api_key="x")`
  gives `None`; `openai_api_key=` works only because its alias spells the field name. Pinned as
  the current truth over every aliased field, so fields added later are covered automatically.
- `.env.example` records the Jev price beside its pinned version as a dated comment: $0.042 per
  million input tokens, output free, read 2026-09-28.
- Decider configuration, validated at startup: `AGENTGATE_DECIDER_BACKEND` (`none` | `jev` | `llm`,
  default `none`), `AGENTGATE_DECIDER_MODE` (`shadow` | `enforce`, default `shadow`), the Jev base
  URL, key and model, and three auto-approve thresholds -- route probability, route confidence and
  irreversibility -- each reading a field the TypeSafe API returns. **Configuration only: nothing
  reads these settings yet**, so no decider runs and nothing here is a claim about behaviour.
  Refused at startup, each with a message naming the variable: a Jev backend with no key; any
  decider on a deployment with no routable cloud lane; a Jev model that is not an exact version;
  an unpriced Jev model; enforce mode with no backend, a missing threshold, or a threshold no
  answer can fail. `TYPESAFE_API_KEY` is a declared alias, added to the permitted unprefixed reads.
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
- `tests/integration/test_routed_lane_enforcement.py`: the policy gate asserted on the wire,
  against two stub endpoints with separate request logs. The primary assertion is an absence --
  canaries visible only to the drafter appearing in no request body the cloud endpoint received --
  and it is paired with a presence assertion so it cannot pass against a run that drafted nothing.
- `docs/concept-map.md` gains a third status, **built, not wired**, enforced in both directions by
  `test_every_built_not_wired_row_exists_and_is_called_from_nowhere`: the symbol must exist and
  nothing in `src/` outside its own file may mention it. Added because `with_retry` and
  `with_fallbacks` sat at *done* for four phases on the strength of `build_resilient_model`, which
  no node has ever called -- so this system performs no retries at all. Item 15, recorded and
  deliberately not wired here; the constraint for whoever does it is that a fallback must never
  cross to a less contained lane.
- Leak inventory items 13 to 19 in ADR 0004. They share a shape worth naming: a component correct
  in isolation, tested in isolation, and connected to nothing. Item 19 is a correction to item 9,
  which said *closed* and named the class that closed it -- and nothing constructs that class.
  The row is corrected in place rather than deleted.
- The stub server speaks SSE, serves `/v1/embeddings` on its own request log, and can decode the
  token ids an embedding request actually carries. All three exist because something could not
  otherwise be observed: the CLI streams, so a networked lane had never been driven through the
  command line at all; and `OpenAIEmbeddings` tokenises client-side, so a canary assertion over an
  embedding body matches nothing and passes while content leaves.
- `tiktoken` declared explicitly, for the same reason `openai` is: the suite calls it directly.
  Without decoding, no absence assertion about the embedding path can ever be non-vacuous.
- A section in ADR 0004 laying out the options for closing item 17 -- per-lane indexes, indexing
  both ways, a sovereign embedding endpoint, indexing on the contained lane, not retrieving for
  restricted requests -- with what each costs the offline suite and startup. Not decided: the
  measurements that would settle it have not been taken, and it says so.
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
- `AccountedEmbeddings`, which books embedding calls into the ledger on the same three rules as a
  chat call: a response with no usage is an error rather than a zero, an unpriced embedding model
  refuses to start, and spend is recorded per model. Checked after every batch, so a runaway
  index would trip the ceiling while it runs rather than reporting the bill afterwards. **The run
  budget means all spend, not chat spend** — the decision is recorded in ADR 0004 item 9.
  **Nothing constructs it**, so embedding spend is not accounted: this entry said it was until
  ADR 0004 item 20, and item 9 is reopened as item 19.
- Recorded: `langchain_openai.OpenAIEmbeddings` discards the `usage` block the API returns,
  which is the one field the budget depends on, so the embedding call goes through the provider
  client directly. ADR 0004, item 11.

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

- A durable audit trail: append-only JSON lines, self-describing field names, timezone-aware
  timestamps, and the request recorded as a hash rather than content. Chosen for a reader who
  does not have this repository — the same argument that decided the checkpoint boundary in ADR
  0011, one level out.
- Gate coverage enforced by discovery rather than by a list. The gates are enumerated from the
  code and each must have written an event, so a gate added later without one fails the build.
  Mutation-checked across eight ways a gate could go silent.
- `openai` declared as a direct dependency. It was already installed as a transitive one, and
  `retrieval/accounting.py` imports it directly because `OpenAIEmbeddings` drops the usage block
  the budget depends on. An undeclared transitive import is a coupling nobody can see.

- A Typer command line -- `run`, `resume`, `approve`, `reject` -- treated as an interface
  rather than a test harness. The review packet is formatted for a person, progress streams
  node by node through `stream_mode=["updates", "messages"]`, and the thread id is on screen
  with the exact commands that use it. Installed as `agentgate`.
- Resume works across processes, not just across sessions: `run` pauses and exits, `approve`
  is a separate invocation that picks the run up from the checkpoint. Proven with subprocesses,
  and guarded by a test that the same flow fails on the in-memory checkpointer.
- `history` and `fork` are absent rather than stubbed, and `--help` says why: the Phase 6 time
  travel primitives are not built.

### Fixed

- **The README described a human gate that approves anything irreversible, and no irreversible
  action exists.** A run that passes the gate reaches `execute`, which records
  `irreversible_effects: []`: nothing proposes an effect, no state channel could carry one, and the
  executor's allowlist is held by no agent. The gate approves a draft being released. ADR 0004
  item 24, pinned as the current truth by `test_gate_guards_a_draft.py`; the README and concept map
  now say what the gate guards today. Not fixed here -- the fix is an architecture change.
- **The retrieval measurement described an embedder that was no longer on `main`.** ADR 0004's table
  and `measure_retrieval.py`'s label still named `OpenAIEmbeddings` after the run ledger made the
  cloud embedder `AccountedEmbeddings` on the raw client -- on the row that justified closing item 17.
  Re-measured 2026-09-28: hashing unchanged at 90% top-4; the meaningless stub baseline moved from 10%
  to 30%, because it hashes what it is sent and the client now sends strings rather than token ids;
  build times now overlap in both directions. Row 17 says what moved and why. The decision rested on
  the hit rate, which reproduced exactly.
- **Embedding spend is accounted** (ADR 0004 items 9 and 19, closed): `build_embeddings` constructs
  `AccountedEmbeddings`, billed per call to the run that is embedding, because the corpus index
  outlives any one run. A research branch no longer swallows a crossed ceiling into a failed outcome.
- **Streamed calls report their usage** (item 18, closed): `stream_usage=True` on every
  OpenAI-compatible client. Without it, accounting would have refused every networked CLI call.
- **A native structured-output reply that fails to parse is still accounted** (item 23): the schema
  is bound as a plain dict and the reply parsed strictly by this code, not inside the client.
- `make seed` said an embed was billed by the configured lane; it now asks the lane embeddings
  actually use, which on a hybrid deployment is in process and free.
- **Retrieval embeds on the most contained lane the deployment can reach**, so on a hybrid
  deployment neither a restricted request's research queries nor the corpus reach the third party.
  `build_embeddings` reads `most_contained_lane` -- the rule classification already uses -- instead
  of the configured default. One index, no routing inside retrieval, no new setting. The cost is
  retrieval quality on every request, public ones included: 90% top-4 against a 20% chance floor,
  measured on this 20-chunk corpus at 301 distinct terms and not beyond it. A cloud-only deployment
  still embeds on the cloud, and is pinned as documented egress. ADR 0004 item 17, closed for hybrid
  deployments; the test that asserted the leak happened now asserts its absence.
- **The README said embedding spend was accounted and chat spend was the half left to wire.**
  Neither is accounted: `AccountedEmbeddings` is constructed by nothing and no chat call reaches
  the ledger. The same claim was in ADR 0004's gaps table, `accounting.py`'s docstring and the
  item 9 entry above; each is corrected in place, and item 19's correction had been sitting three
  paragraphs below the claim it contradicts. ADR 0004 item 20.
- **The CLI printed a readable message and then eighty-eight lines of traceback**, exiting 1 rather
  than the 2 its handler claimed. `typer.Exit` raised outside the click invocation is an ordinary
  exception, and `main()`'s docstring said "reported, not traced" while both happened. Nothing
  caught it because no CLI test could make the CLI raise: every command test runs on the fake lane,
  where no lane is unavailable and no provider can fail. Reachable now that a single-lane deployment
  refuses a restricted request.
- The CLI stated the configured lane as the lane in use -- `lane  cloud` -- which on a hybrid
  deployment is what an operator reads while a restricted request is classified and drafted on
  their own endpoint. It is labelled `default` now, and the lanes actually used are reported from
  the audit trail at the end of a run.
- `build_embeddings` and `OpenAIEmbeddingsWithUsage` take the endpoint, so the embedding path can
  be pointed at a double. Until now it was the only egress in the system that **could not be
  observed at all**. The field is `openai_api_base`, not the `base_url` alias, because mypy knows
  the field names -- the same trap as leak inventory item 2, one layer up.
- **The policy gate's routing decision was recorded but never applied.** `route_by_policy` chose
  a lane, the lane node wrote it to state, the audit trail reported it, and no node passed `lane=`
  to the model factory -- so a request routed to the sovereign lane was drafted by whatever the
  deployment defaulted to. Measured against two stub endpoints: three requests to the cloud
  endpoint and zero to the sovereign one, while the drafted event read
  `lane='sovereign' model='cloud-capable-stub'`. The model *identifier* had the same defect, so
  fixing the endpoint alone would have sent a cloud model's name to the operator's own server.
  Item 13 of the leak inventory in ADR 0004.
- The route now **narrows** the configured lane and cannot widen it. `CONTAINMENT` orders the
  lanes by how far data travels and `narrower_of` takes the minimum, because passing the routed
  lane through unconditionally makes a sovereign-default deployment call a third party for public
  content -- a worse leak than the one being fixed. It is also what keeps the offline suite
  offline.
- **Classification no longer runs on the configured lane.** It runs before the router, so on a
  cloud default the raw request went to the third party in order to decide whether it was allowed
  to go there. It now runs on the most contained lane in `routable_lanes`, with no new setting:
  fake stays fake, hybrid classifies on its own endpoint, cloud-only classifies on cloud and that
  remaining egress is recorded as item 14 and pinned by a passing test. The native-structured-
  output lookup moved with it, because asking a non-native endpoint for it wastes a call rather
  than failing -- measured as 2 calls and ~733 prompt tokens where 1 and ~537 would do.
- A sovereign base URL with no model, and a hybrid deployment whose sovereign model has no price,
  are both startup failures now. Neither was reachable before the routed lane was wired through;
  the price guard computed "reachable" from the configured lane, which excluded the very model the
  policy gate exists to route restricted content to.
- A deployment with one lane now **refuses** a request policy sends somewhere stricter, rather
  than serving it from the lane policy just ruled out.
- The audit trail was wrong in both directions and is now checked against the request log rather
  than against itself. The lane event carries the policy route and the effective lane; the drafted
  and classified events name where the call actually went and what it asked for.
- `test_restricted_content_reaches_the_sovereign_lane_in_a_real_run` is renamed to
  `test_restricted_content_records_the_sovereign_binding_and_its_reason`. It passed throughout the
  two phases in which restricted content did not reach that lane, because it asserted a state
  field and the field was always correct. Same treatment as the `filterwarnings` rule in ADR 0011.
- `make test-live` could not start. The gatekeeper set two `AGENTGATE_*` variables on the
  pytest subprocess that were not declared settings, and the unknown-variable guard rejected
  them, so every live case failed at configuration before reaching a provider. Both are
  declared settings now. Recorded as item 7 of the leak inventory in ADR 0004.
- The lane nodes recorded themselves in the audit trail as `bind_cloud_capable` while the graph
  knew them as `cloud_capable`, so a reader correlating the trail against the topology found no
  such node. Found by the gate-discovery test on its first run.
- The retrieval corpus was not copied into the container image. The runtime stage ships the
  virtualenv, which covers code and not data, so every research branch would have failed
  inside the container while every offline test passed — the suite runs from a checkout where
  `corpus/` is simply there. Copied now, and pinned by a static check of the runtime stage.
- `AGENTGATE_LIVE_SPEND_ABORT_USD` was computed, printed, and read by nothing. It now bounds
  the suite while it runs, rather than only being compared against the total afterwards.

### Changed

- Nodes that call a model or record spend take a required `config: RunnableConfig`, and a test
  that calls one directly passes a run's config (item 22 is why it is required rather than optional).
- The fake model records the tools bound to it in place, so a shallow copy -- which is how a model is
  charged to a ledger -- shares the record.
- `Settings.classification_lane` is renamed `most_contained_lane`. Classification runs on it and
  retrieval now embeds on it, so a name for one caller was already wrong. No behaviour change.
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
