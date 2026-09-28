# Working rules for this repository

What a fresh session needs and cannot get by reading the code. Everything else is already
written down somewhere better:

| Looking for | Read |
| --- | --- |
| Where something claimed one thing and did another | `docs/adr/0004-provider-abstraction-and-lanes.md` — the leak inventory, 19 rows |
| Whether a concept is built | `docs/concept-map.md` — three statuses, all enforced by `tests/unit/test_concept_map.py` |
| What changed and why | `CHANGELOG.md` |
| Why a design is the way it is | `docs/adr/` |
| Git, authorship and commit discipline | `.dev/CLAUDE.md` — **read it before touching git.** Gitignored, so it is not in a clone |
| Current phase and tick state | `.dev/PLAN.md` — also gitignored |

Do not summarise those files here. A summary that drifts is worse than a pointer.

## House rules

Non-negotiable, and the reason each exists is in the inventory somewhere.

- **Tests first, and shown red.** A guard nobody watched fail is a guard nobody has tested.
- **Assert on the wire.** Read the observed request body — a stub's request log — never a client
  attribute (item 2) and never a state field, which is a label the graph wrote about itself (13).
- **Assert absence, not just presence**, and pair it with a presence assertion so it cannot pass
  against a run that did nothing.
- **Mutation-check every guard.** Remove the fix, confirm red, restore. Do it with a script that
  holds the original bytes, not `git checkout --` — that discards uncommitted work.
- **The offline suite stays offline and free.** No new network path in CI, no key required.
- **State channels hold JSON only** (ADR 0011). Parse on the way in, plain data on the way out.
- **No new dependency without justification, and exact pins** (ADR 0006). The bar is the one
  `openai` and `tiktoken` meet: the code calls it directly, and an undeclared transitive import is
  a coupling nobody can see.
- **Stop and ask rather than guess at a design decision.** Small commits, conventional messages,
  one branch per unit of work, cut from `main`.

## The failure mode this repo keeps finding

**A component correct in isolation, tested in isolation, and connected to nothing.** Items 13, 15,
16, 17 and 19 are all that shape. The policy router chose lanes correctly for four phases while
nothing applied the choice. `build_resilient_model` retries correctly today and no node calls it.
Item 19 was a row that said *closed*, named the class that closed it, and that class was on no
path until the run ledger wired it.

A unit test cannot find this. An integration test only finds it if it asserts on something outside
the process. **When something looks done, check what calls it.**

## Item 17: decided and closed for hybrid deployments

Retrieval embedded on the configured lane and never saw the routed lane, so on a hybrid deployment
a restricted request had its research queries — and the whole corpus — embedded by the third party.

**Decided 2026-09-28: index on the most contained lane.** `build_embeddings` dispatches on
`Settings.most_contained_lane`. One index, no routing inside retrieval, no new setting. On a hybrid
deployment the corpus and every research query are embedded in process, public requests included,
and nothing from either reaches the third party. The test that asserted the leak happened now
asserts its absence; ADR 0004 item 17 has the full record.

- **Open for cloud-only deployments.** Their most contained lane *is* the cloud, so the corpus and
  public sub-questions are still embedded there, pinned as documented egress — the same shape as
  item 14. That embedding spend is accounted: every call is billed to the run's ledger (item 19,
  closed).
- **The cost is retrieval quality on every request.** Measured via `make measure-retrieval`: **90%
  top-4 against a 20% chance floor**. That figure belongs to this **20-chunk corpus at 301 distinct
  terms**. ADR 0010's collision curve degrades a hashing index as vocabulary grows, so a corpus an
  order of magnitude larger needs re-measuring before the figure means anything.
- **Per-lane indexes were ruled out:** without a sovereign embedding endpoint the sovereign index
  would be built by the hashing embedder, so retrieval quality would correlate exactly with
  sensitivity. A sovereign embedding endpoint composes with this later, behind the same rule.

**`most_contained_lane` is the single rule for "which lane is most contained".** Classification
and retrieval both read it, and Part B will be its third caller. Do not write a second answer to
that question — two homes for one policy eventually give two answers.

## Every run has a ledger

**Every run starts through `graph/build.py:run_config`** (or `resume_config`), which carries the
run's spend ledger. Model-calling nodes refuse to run without it, and any new model call -- the
decider's included -- is charged to it. A node reading its config takes a required
`config: RunnableConfig`: the optional-union spelling is silently not injected (item 22).

## Part B: the next step is B1

A decider in front of the approval gate. Jev is TypeSafe AI's decision model — typed questions in,
typed answers with probabilities out. **Not a chat model**, so do not give it a chat interface.

- `build_decider()`, its own abstraction. Not a lane behind `build_model`.
- **A cloud egress**, so it runs only when the routed lane is cloud. A sovereign-routed request
  never calls it and always goes to a human.
- Its own `assess` node **before** the gate, never inside it — the gate re-executes from its top on
  resume, so a call there would run twice and could disagree with itself.
- It can only **auto-approve or ask a human**, never reject. Every error path — timeout, HTTP error,
  parse failure, missing verdict, low confidence, restricted lane — resolves to a human.
- **Shadow mode by default.** The threshold comes from measured shadow-mode agreement, not chosen.
- Budgets, caps and spend stay deterministic code. The decider decides nothing numeric.
- Startup validation rejects the contradictory combinations (backend on with no key; a decider on a
  deployment with no cloud lane; enforce mode with an implausible threshold).
- No SDK and no third-party free-key mirrors.
- **B1 is next**: verify the request and response schema against docs.typesafe.ai before
  writing any code. Do not build against a guessed schema.

## Two local facts

- **`INTERVIEW-PREP.md` is excluded via `.git/info/exclude` and must never be committed.** Not
  `.gitignore`, deliberately: that ships with the repo and would tell every reader it exists.
- **`Makefile` and `make.ps1` help strings must match exactly.** `tests/unit/test_task_runner.py`
  enforces it, and it has caught a one-word difference.
