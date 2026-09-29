# Working rules for this repository

What a fresh session needs and cannot get by reading the code. Everything else is already
written down somewhere better:

| Looking for | Read |
| --- | --- |
| Where something claimed one thing and did another | `docs/adr/0004-provider-abstraction-and-lanes.md` — the leak inventory, 27 rows |
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
- **Wait for every check before merging.** The repo has no branch protection, so `gh pr merge`
  succeeds while checks are still running. "Merge #N" never means "before CI is green".
- **When a branch, a PR or a premise is not what was described, say so and ask** — do not act on
  your own judgement of whether it matters. A branch described as merged that carries an unmerged
  commit is reported, not deleted.

## The failure mode this repo keeps finding

**A component correct in isolation, tested in isolation, and connected to nothing.** Items 13, 15,
16, 17, 19 and 24 are all that shape — 24 was the headline claim: the human gate approved a draft
while no action existed for it to approve. The policy router chose lanes correctly for four phases
while nothing applied the choice. Item 19 was a row that said *closed*, named the class that closed
it, and that class was on no path until the run ledger wired it.

A unit test cannot find this. An integration test only finds it if it asserts on something outside
the process. **When something looks done, check what calls it.**

### The chain that closed 15, 16 and 27 — the argument for wiring things

All three are closed now, and *how* they closed is the point rather than the fact:

1. **Item 15** was `build_resilient_model`: retries and fallbacks, correct, tested against real HTTP
   errors, called from nowhere for four phases.
2. **Wiring it produced item 27** in the first full test run. `with_retry` retries on every
   exception, and the ledger raises a crossed ceiling from inside the callback of the call that
   crossed it — so a budget guard read as a 500 and the answer to it was four more billed calls.
   That defect had been in the code since the day it was written. It could not fail, because nothing
   called it. **An unwired component's defects are not dormant, they are unobservable.**
3. **Deciding item 16** produced the next one. Removing the cheap tier's fallback — because a
   fallback to the same model through the same connection pool made `max_retries` mean something
   other than what it says — exposed that a sovereign deployment then has no working fallback on
   any request. Three decisions compose into that, and the third is that the sovereign lane names
   one model for both tiers, so even the capable tier's fallback degrades to itself.

So the chain runs: a green test file hid a defect → wiring exposed it → fixing it exposed a
property of the recommended deployment. **None of those three steps was reachable from the one
before it by reading.** That is the case for wiring a thing rather than trusting its tests, and it
is why `docs/concept-map.md` carries an enforced *built, not wired* status at all.

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
and retrieval both read it. Do not write a second answer to that question — two homes for one
policy eventually give two answers. (The decider does not read it: it follows the *routed* lane,
because it runs only on requests the router sent to the cloud.)

## Every run has a ledger

**Every run starts through `graph/build.py:run_config`** (or `resume_config`), which carries the
run's spend ledger. Model-calling nodes refuse to run without it, and any new model call -- the
decider's included -- is charged to it. A node reading its config takes a required
`config: RunnableConfig`: the optional-union spelling is silently not injected (item 22).

## Part B: complete through B8, merged in PR #18

The Jev decider in front of the approval gate: configuration and startup validation, offline
isolation, the decider itself, the `assess` node before the gate, the deployment examples in `env/`,
and ADR 0012. ADR 0012 is the record of every design decision; read it rather than a summary here.

### Not done, and each needs Hamza's say-so

- **No live Jev probe has run.** Every Jev capability row is `STUB`. `scripts/probe_capabilities.py
  jev` makes one billed call and refuses to run against anything but the official endpoint.
- **No enforce-mode threshold has been measured.** Thresholds must come from measured shadow-mode
  agreement, and none exists. Enforce mode is refused at startup without all three, so the
  decider can only run in shadow mode today.
- **Calibration is not claimed**, and cannot be from a single probe.

### One commit is not green in a clean checkout — decided, left as is

**Commit `a795e1f` is not green in a clean checkout.** The `.gitignore` rule `env/`, there for
virtualenvs, also excluded the new `env/sovereign.env` and `env/hybrid.env`, so that commit carries
their test without the files; the next commit tracks them and narrows the rule to `!/env/`. The tip
and CI are green. **Decided 2026-09-29: leave it.** Rewriting pushed history costs more than the
inconsistency, and the commit is inside a merge commit now. Do not rewrite it.

## One undecided question

**Does an air-gapped operator want a degradation step at all?** A sovereign deployment makes
`max_retries + 1` attempts and then the run fails, with no fallback on any request — stated in the
README's limitations and pinned by
`tests/unit/test_registry.py::test_a_sovereign_deployment_has_no_working_fallback_on_either_tier`.
Whether such an operator would rather a run fail cleanly than come back with a weaker answer from
a smaller local model is not known, because nobody has asked one, and this repository does not
invent an answer it has not measured.

**The trigger to reopen it is a second sovereign model identifier, not a preference.** While the
sovereign lane resolves both tiers to one `sovereign_model`, there is nothing to degrade *to* and
the question is academic. The day `sovereign_capable_model` and `sovereign_cheap_model` exist, the
capable tier's fallback becomes real on that lane and the cheap tier's own question reopens on its
own terms. ADR 0004 item 16 has the record.

## What comes next, in order

**Phases 6 → 7 → 9 → 8:** store-backed memory and time travel; the FastAPI/SSE surface (which is
also where the session ceiling gets a caller); evals; observability.

**Nothing in the leak inventory is open and closable by a code change.** Items 15, 16 and 27 are
closed. What remains open needs something other than code, and it is worth knowing which is which
before anyone goes looking for work there:

- **Item 12** is a list of capabilities nobody has measured. Each needs a billed call against a
  real provider, or a real Ollama or vLLM, and streaming belongs to Phase 7 anyway.
- **Items 14 and 17** are narrowed to a remainder that is structural rather than unfixed: a
  cloud-only deployment has nowhere more contained to classify or embed on. No code closes that;
  a second lane does.
- **Part B's Jev rows** are all `STUB`, and enforce mode has no measured threshold. Both need a
  billed call Hamza authorises.

Nothing on this list starts without Hamza's go-ahead.

## Two local facts

- **`INTERVIEW-PREP.md` is excluded via `.git/info/exclude` and must never be committed.** Not
  `.gitignore`, deliberately: that ships with the repo and would tell every reader it exists.
- **`Makefile` and `make.ps1` help strings must match exactly.** `tests/unit/test_task_runner.py`
  enforces it, and it has caught a one-word difference.
