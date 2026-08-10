"""The graph's typed state, and the reducers that make concurrency safe.

LangGraph executes a graph in super-steps. Every node scheduled in the same super-step runs
concurrently and returns a partial state update, and those updates are merged when the step
completes. How they merge is decided per channel, by the reducer annotated on that field.

**A field with no reducer is last-write-wins, and last-write-wins cannot resolve a tie.** If two
nodes in the same super-step both write ``draft``, LangGraph has no rule for which one should
survive, so it refuses: ``InvalidUpdateError: At key 'draft': Can receive only one value per
step``. That error is a feature. Silently keeping one of two answers would be a data-loss bug
that surfaces as a mysteriously incomplete result three nodes later.

The fix is not to serialise the nodes. It is to say what merging *means* for that channel:

- ``messages`` merges with ``add_messages``, which appends and reconciles by message id.
- ``findings`` merges with ``operator.add``, which concatenates. Research fans out over
  sub-questions and every branch contributes; there is no tie to break because none of them
  are competing to be the answer.
- Everything else is last-write-wins, which is correct precisely because only one node ever
  writes it. ``sensitivity`` is written by the classifier and nothing else.

So the reducer is not a stylistic choice or a way to silence an error. It is the declaration
that a channel accumulates rather than replaces, and choosing wrong in either direction is a
correctness bug: a reducer on a single-writer field hides a double-write, and its absence on a
fan-out field turns a legitimate merge into a crash.

``tests/integration/test_parallel_writes.py`` demonstrates both halves against a real graph.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Sensitivity(StrEnum):
    """How restricted the request's content is.

    Drives the policy gate. This is the decision that determines which lane a request may
    reach, so it is a closed set rather than free text.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Complexity(StrEnum):
    """How much work the request plausibly needs. Drives tier selection, not lane selection."""

    SIMPLE = "simple"
    INVOLVED = "involved"


class Decision(StrEnum):
    """What the human gate returned."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Classification(BaseModel):
    """The classifier's structured output.

    A pydantic model rather than loose fields because it is produced by a model and must be
    validated before anything routes on it. A malformed sensitivity would otherwise become a
    policy decision.
    """

    sensitivity: Sensitivity
    complexity: Complexity
    contains_pii: bool = Field(description="Whether the request appears to contain personal data.")
    reason: str = Field(default="", description="One line, for the audit trail.")


class Finding(BaseModel):
    """One piece of researched evidence.

    Findings accumulate across parallel branches, which is why the channel holding them needs
    a reducer.
    """

    question: str
    content: str
    source: str = ""


class ResearchOutcome(BaseModel):
    """What one research branch did, whether or not it produced a finding.

    Separate from ``Finding`` because a branch that failed has no finding to record and is
    exactly the branch worth knowing about. Counting findings alone cannot distinguish "the
    question had no answer in the corpus" from "that branch raised and nobody noticed"; both
    show up as one fewer item in a list.
    """

    question: str
    ok: bool
    detail: str = ""


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    ``total=False`` because nodes return partial updates; requiring every key on every return
    would make each node responsible for state it has no business touching.
    """

    # --- accumulating channels -------------------------------------------------------
    #
    # These have reducers because more than one node can legitimately write them in the same
    # super-step, and the merge is well defined.

    messages: Annotated[list[AnyMessage], add_messages]
    """Conversation so far. ``add_messages`` appends and reconciles by id, so a node that
    revises a message it already emitted updates it rather than duplicating it."""

    findings: Annotated[list[Finding], operator.add]
    """Append-only research output. The researcher fans out over sub-questions with ``Send``
    and every branch contributes; ``operator.add`` concatenates the branches on fan-in."""

    audit_trail: Annotated[list[dict[str, Any]], operator.add]
    """Append-only record of what each node decided. Concatenating is the only merge that
    preserves an audit trail -- last-write-wins would silently discard events."""

    research_outcomes: Annotated[list[ResearchOutcome], operator.add]
    """One entry per research branch that reported back, successful or not.

    Findings alone cannot tell you whether a fan-out completed. Three findings from a fan-out
    of three and three findings from a fan-out of five are the same list, and the second is a
    partial answer. This channel is what makes the difference visible, and it accumulates for
    the same reason ``findings`` does: every branch contributes and none of them compete."""

    # --- single-writer channels ------------------------------------------------------
    #
    # No reducer, deliberately. Each is written by exactly one node, so last-write-wins is
    # correct and a concurrent write is a bug that should surface as InvalidUpdateError rather
    # than as one of two answers arbitrarily winning.

    request: str
    """The original user request. Written once, at the start."""

    correlation_id: str
    """Ties every audit event and log line for this run together."""

    classification: Classification
    """Written by the classifier node alone."""

    lane: str
    """Resolved by the policy router alone. A string rather than the Lane enum because state
    is serialised into checkpoints and must survive a round trip through JSON."""

    sub_questions: list[str]
    """Written by the supervisor when it dispatches research."""

    dispatched: int
    """How many branches the last fan-out opened.

    Compared against ``len(research_outcomes)`` to catch a branch that reported nothing at all.
    A branch that fails and says so appears in the outcomes; a branch that vanishes does not,
    and without this number the two are indistinguishable from the fan-in side -- which is the
    failure mode a fan-out has that a sequential loop does not."""

    draft: str
    """Written by the drafter alone."""

    answer_complete: bool
    """Whether every dispatched branch reported success. Written by ``finalise`` alone.

    Recorded rather than derived at the point of display, so that a caller reading the final
    state cannot present a partial answer as a whole one by forgetting to check."""

    decision: Decision
    """Written by the approval gate alone."""

    feedback: str
    """Reviewer's reason for rejection, fed back into the revision loop."""

    iterations: int
    """Supervisor hand-offs so far. Compared against the budget in a conditional edge.

    Single-writer on purpose: an incrementing counter with ``operator.add`` would double-count
    the moment two nodes ran in the same super-step, which is exactly when a budget matters.
    """

    finalised: bool
    """Set when the budget guard or the supervisor decides the run is over."""


def initial_state(request: str, correlation_id: str) -> AgentState:
    """The state a run starts from.

    Accumulating channels start empty rather than absent, so a node can append without first
    checking whether the key exists.
    """
    return AgentState(
        request=request,
        correlation_id=correlation_id,
        messages=[],
        findings=[],
        audit_trail=[],
        iterations=0,
        finalised=False,
        decision=Decision.PENDING,
    )
