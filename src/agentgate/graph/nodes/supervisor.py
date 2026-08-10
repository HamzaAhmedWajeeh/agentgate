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
from agentgate.graph.state import AgentState

NODE = "supervisor"

Destination = Literal["researcher", "budget_guard"]


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
    if outstanding and dispatched == 0:
        goto: Destination = "researcher"
    else:
        goto = "budget_guard"

    done = goto == "budget_guard"

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
                        "goto": goto,
                        "finishing": done,
                    },
                )
            ],
        },
        goto=goto,
    )
