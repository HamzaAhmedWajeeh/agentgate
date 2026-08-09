"""The supervisor: decides what happens next, and says so in one object.

Returns a ``Command``, which carries a state update and a ``goto`` together. The alternative --
writing a "next" field to state and having a conditional edge read it back out -- splits one
decision across two places and lets them disagree. With ``Command`` the decision and its
consequence are the same return value, so there is no window in which state says one thing and
control flow does another.

In Phase 3 the supervisor can only dispatch or finish, because the researcher and drafter do
not exist yet. The dispatch branch is wired but currently goes straight to finalisation; Phase
4 replaces that target without changing this node's shape.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.state import AgentState

NODE = "supervisor"

Destination = Literal["budget_guard"]


def supervise(state: AgentState, settings: Settings) -> Command[Destination]:
    """Advance the run by one turn.

    Increments the iteration counter as part of the same update that moves control, so a
    replayed super-step cannot advance control without also advancing the count. Splitting
    those would let a crash-and-resume cycle take a free turn.
    """
    iterations = state.get("iterations", 0) + 1
    request = state.get("request", "")

    # Phase 3 has no workers, so there is never anything to dispatch and the supervisor
    # finishes on its first turn. Phase 4 replaces this condition with real work; the shape of
    # the decision does not change. Seeding `sub_questions` is how a test drives the loop and
    # exercises the budget guard.
    outstanding = [q for q in state.get("sub_questions", []) if q]
    done = not outstanding

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
                        "finishing": done,
                    },
                )
            ],
        },
        goto="budget_guard",
    )
