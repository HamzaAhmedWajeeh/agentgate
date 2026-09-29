# 12. The decider is a declared egress that can only ask for a human, never overrule one

Date: 2026-09-28

Status: Accepted

## Context

The approval gate stops every run for a human. That is the right default and an expensive one: most
drafts in a routine workflow would be approved, and a person reading each one is the bottleneck the
rest of the system was built to avoid. A *decider* -- something that assesses a draft before the gate
and may approve in a human's place when it is confident -- is the obvious relief.

The obvious relief is also a new way for content to leave the boundary and a new authority to act,
both in front of the one gate whose job is to hold. TypeSafe's Jev is the candidate: a decision model,
not a chat model -- typed questions in (Choice, Score, Noul), typed answers out with probabilities,
and a `confidence` on Choice and Score answers. Its request and response shapes were verified against
docs.typesafe.ai on 2026-09-28 before any code was written (B1).

## Decision

**An `assess` node, before the gate, asks the decider once per draft; the gate reads what it
stored.** The decider can produce "auto-approve" or "ask a human" and nothing else, and every path
that is not a clean, current, confident approval -- a failure, a restricted route, a missing field,
an unmet threshold, shadow mode -- resolves to a human.

### It is a declared egress, and rides only the cloud lane

The decider is a call to a third party. Configuration decides that it is *available*; the policy gate
decides, per request, whether it is *allowed* -- and a request routed anywhere more contained than the
cloud never reaches it. The same rule is enforced at startup from the other side: a decider on a
deployment with no routable cloud lane is refused, because it could never run and because a
deployment built not to send data to OpenAI should not send it to TypeSafe either. There is no second
egress decision; there is the router's.

### It can never reject

A rejection sends the draft back for revision, and a model that rejects on its own authority can
drive the revision loop with nobody watching. More simply: the cost of a wrong "ask a human" is a
person's time, and the cost of a wrong rejection is invisible work. A decider that can only ever say
"yes, confidently" or "a human should look" cannot make the gate *stricter* behind a person's back,
only faster -- and the "yes" path is the one every safeguard below is aimed at.

### Shadow is the default, and the threshold must be measured, not chosen

In shadow mode the verdict is recorded beside the human's decision and never acted on. That is not
caution for its own sake: it is **the only way to get the number enforce mode needs**. Whether an
`auto_approve` at probability 0.9 and confidence 0.8 agrees with what a person would have decided is
an empirical question about this workflow's drafts, and a threshold picked because it looks
conservative is a guess presented as a control. The approval event carries the full stored verdict
and who decided, so agreement can be read off the audit trail without the decider in the loop.

**No such measurement exists yet.** No threshold ships. Configuration refuses enforce mode without all
three -- minimum route probability, minimum route confidence, maximum irreversibility -- and refuses a
value that no answer can fail (a probability floor at or below 0.5, which a two-option Choice's winner
always meets; a confidence floor of 0; an irreversibility ceiling at or above 0.5). Those bounds rule
out the vacuous; they are not a recommendation.

### The deterministic preconditions are code, checked first

Whether the provenance check passed and whether a tool was denied are facts already in state. The
gate checks them in code **before it reads the verdict at all**, and no question to the decider names
them (`DETERMINISTIC_PRECONDITIONS`). Asking a model to re-derive a boolean only adds a way to get it
wrong, and a verdict cannot outvote a fact.

### The assess node is separate from the gate

The gate re-executes from its top on every resume -- `graph/nodes/approval.py`'s module docstring,
proved by `test_the_interrupted_node_re_executes_from_its_top_on_resume`. A decider called inside it would be
called again each time -- a second billed request that could disagree with the first, about the same
draft, with no record of which one the approval rested on. Called once before the gate and stored,
the verdict is a fact the gate reads. A resume makes no further decider call, and a test counts the
requests to prove it. A skipped assessment overwrites the stored one, so a verdict about an earlier,
rejected draft can never be read as one about the current draft.

### No SDK

One documented endpoint, a JSON body in and out, and `httpx` already a declared dependency. The SDK
would add a dependency to save a few dozen lines, and it configures itself from four `TYPESAFE_*`
environment variables on its own -- including a default model of `jev-latest`, the alias this system
refuses because a threshold tuned against one version says nothing about the next. Reading
configuration nobody declared is the problem ADR 0009 exists to prevent. The model is pinned to an
exact version, and a response naming a different one is a failure that asks a human.

### Every call is accounted

Each assessment is charged to the run ledger against the pinned model, before the answer is judged:
an answer that is then discarded was still paid for. The usage block is documented as required and has
not been seen on a real wire; if it is absent, the call is refused rather than recorded as free. A
crossed ceiling aborts the run like any other call rather than becoming "ask a human".

## The limitation that is not closed: injection through the proposals

Jev's documentation says plainly that it does not treat state as hostile: content written to steer the
model can move the answer. So the decider is **not** shown the draft -- model output built from
retrieved content, and the most obvious channel through which the corpus could address it. It is shown
structured facts instead: the routed lane, the finding count, the denied tools, the provenance result,
and the proposed actions.

**That narrows the surface. It does not close it.** The proposed actions are tool names and arguments
*the drafter authored*. A drafter talked into proposing something can still describe it to the decider
in the arguments it chose -- a customer email's body is free text, and so is anything else a tool
accepts as a string. The screen before the gate ensures the tool is one the executor could run and the
arguments satisfy its schema; it cannot ensure the arguments are not persuasive.

What caps the damage is not the structured state; it is three properties that hold whatever the
decider is told:

- **It can never reject**, so the worst it can do is approve what a human would otherwise have seen.
- **Shadow is the default**, so out of the box it approves nothing at all.
- **The deterministic preconditions are checked in code before any verdict is read**, so a draft whose
  provenance failed, or whose drafter was refused a tool, goes to a human no matter how confident the
  decider is.

And whatever is approved performs nothing real: the only effect sink is an append-only outbox, and
configuration refuses any other (ADR 0004 item 24). In enforce mode, then, the residual risk is an
action the drafter proposed, the decider approved, and a person never saw -- recorded to the outbox.
That is the risk a measured threshold is meant to bound, and it is the reason enforce mode waits for
one.

## What is not claimed

- **Calibration.** Whether Jev's probabilities are *calibrated* -- whether its 0.9s are right nine times
  in ten on these drafts -- is a claim about many answers, and no single probe can observe it, live or
  stubbed. It is what shadow mode measures, and it is claimed nowhere until that measurement exists.
  The capability matrix records what a response *carries*, not how good it is.
- **Live behaviour.** Every Jev capability row is `STUB` until `scripts/probe_capabilities.py jev` has
  run against the official endpoint, which it refuses to do against anything else. No live probe has
  run.
- **The `llm` backend in enforce mode.** It reports a route and nothing numeric -- a chat model's
  self-reported number is generated text, not a measurement -- so it can never meet the thresholds. It
  is a shadow-mode comparison and nothing more.

## Alternatives rejected

**Jev on the classifier.** Classification decides where a request may go, and it runs before any
routing decision exists -- on the most contained lane, precisely so the raw request is not shown to a
third party in order to decide whether it may be (ADR 0004 item 14). A cloud decider there reintroduces
that egress at the one point it was removed from.

**A global privacy toggle.** A single "send nothing to third parties" switch would be a second egress
decision beside the router's, and two answers to one question eventually disagree. The router already
decides per request; the decider follows it.

**A profile variable** (`AGENTGATE_PROFILE=hybrid` setting a bundle of others). Settings that change
because another setting implied them are invisible reads -- ADR 0009. The deployment examples in `env/`
state every setting explicitly, and startup validation catches the combinations that contradict
themselves.

**Third-party key mirrors.** Services that resell or proxy access to a model on shared keys put a party
nobody vetted between this system and the provider, on the path that carries the data. The decider
talks to the official endpoint or to a stub under test, and the probe refuses to record anything else
as live.

**`create_agent`'s structured output, for the `llm` backend and for proposals.** It asks the model to
call a synthetic response tool, and a model that answers with the right JSON as text instead is
re-prompted until the recursion limit -- a paid retry loop, not a refusal (ADR 0004 item 25). The `llm`
backend parses with this code's own validate-and-repair, and the drafter's proposals are parsed from
its final message, both failing closed.

## Consequences

The gate can be made faster without being made weaker, and whether it *should* be is a measurement
this repository does not yet have. Until shadow mode has produced one, the decider's only effect is a
line in the audit trail beside each human decision -- which is the evidence the next step needs.
