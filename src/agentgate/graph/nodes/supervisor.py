"""The supervisor: decides what happens next, and says so in one object.

Returns a ``Command``, which carries a state update and a ``goto`` together. The alternative --
writing a "next" field to state and having a conditional edge read it back out -- splits one
decision across two places and lets them disagree. With ``Command`` the decision and its
consequence are the same return value, so there is no window in which state says one thing and
control flow does another.

The supervisor is re-entered after research: every branch of the fan-out hands control back
here with ``Command(graph=Command.PARENT)``. So its decision has to be a function of what has
already happened rather than of where it was called from -- ``dispatched`` and the outcomes
tell it whether research is behind it, and it must reach the same conclusion whether it is
being entered for the first time or by the last of five returning branches.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.routing import budget_exhausted
from agentgate.graph.state import AgentState, Decision

NODE = "supervisor"

Destination = Literal["researcher", "drafter", "approval_gate", "budget_guard"]


def supervise(state: AgentState, settings: Settings) -> Command[Destination]:
    """Advance the run by one turn.

    Increments the iteration counter as part of the same update that moves control, so a
    replayed super-step cannot advance control without also advancing the count. Splitting
    those would let a crash-and-resume cycle take a free turn.
    """
    iterations = state.get("iterations", 0) + 1
    request = state.get("request", "")

    outstanding = [question for question in state.get("sub_questions", []) if question.strip()]
    dispatched = state.get("dispatched", 0)

    # Research happens once per run. `dispatched` is the record that it did, and it is checked
    # rather than inferred from the findings: a fan-out in which every branch failed produces
    # no findings at all, and re-dispatching it would loop against a corpus that is not going
    # to start working.
    # The budget is checked before the work, not after it, because the revision loop can
    # genuinely fail to stop: a reviewer who keeps rejecting drives drafter -> gate -> drafter
    # indefinitely, and that loop never passes through budget_guard on its own. Routing there
    # on exhaustion is what makes the cap reachable rather than decorative.
    #
    # `finalised` is deliberately NOT set on this branch. The supervisor's job is to notice
    # that the budget is spent; deciding what that means is `route_by_budget`'s, and a
    # supervisor that pre-decided would make the guard a rubber stamp on its own conclusion.
    exhausted = budget_exhausted(state, settings)

    if exhausted:
        goto: Destination = "budget_guard"
    elif outstanding and dispatched == 0:
        goto = "researcher"
    elif dispatched and not state.get("draft"):
        # Research is behind us and there is no deliverable yet. Note that this is reached
        # even when every branch failed: drafting from nothing produces a document that says
        # it found nothing, which is a better answer than silence and is the only path on
        # which the incompleteness gets written down where a reader will see it.
        goto = "drafter"
    elif state.get("draft") and state.get("decision") is not Decision.APPROVED:
        # A draft exists and no human has approved it. Rejection clears the draft, so a
        # rejected run falls through to the drafter above on its next turn -- that fall-through
        # is the revision loop, and it is a loop precisely because neither branch is terminal.
        goto = "approval_gate"
    else:
        goto = "budget_guard"

    done = goto == "budget_guard" and not exhausted

    return Command(
        update={
            "iterations": iterations,
            "finalised": done,
            "audit_trail": [
                audit_event(
                    node=NODE,
                    decided=Decided.DISPATCHED,
                    correlation_id=state.get("correlation_id", ""),
                    input_digest=digest(request),
                    lane=state.get("lane"),
                    detail={
                        "iteration": iterations,
                        "of_budget": settings.max_iterations,
                        "outstanding": len(outstanding),
                        "dispatched_already": dispatched,
                        "revisions": state.get("revisions", 0),
                        "budget_exhausted": exhausted,
                        "goto": goto,
                        "finishing": done,
                    },
                )
            ],
        },
        goto=goto,
    )
