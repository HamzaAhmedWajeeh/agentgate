"""Close the run, recording why it ended and whether the answer is whole.

"Finished the work" and "ran out of budget" produce the same absence of further activity and
are not the same event. A trail that cannot distinguish them cannot answer the only question
anyone asks about a truncated result.

The same argument applies one level down, to the fan-out. Three findings from three branches
and three findings from five branches are the same list. A run that lost two branches and says
nothing about it has not failed -- it has done something worse, which is to present a partial
answer in the shape of a complete one. So completeness is computed here and written to state
rather than left for a caller to work out, because a caller who forgets is the whole failure
mode.

This is also where the run's trail is appended to the durable log, because it is the one node
every completed run passes through, and one append means a reader never sees half a run.
**A run that never reaches here has its trail in the checkpoint and not in the file** -- one
still paused at the approval gate, or one that died mid-flight. That is recoverable rather than
lost, since the checkpoint holds the same events, but it is a real limitation and it is written
here rather than left to be discovered from an empty log after an incident.
"""

from __future__ import annotations

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.audit.writer import write_trail
from agentgate.config import Settings
from agentgate.graph.completeness import research_gaps
from agentgate.graph.routing import budget_exhausted
from agentgate.graph.state import AgentState

NODE = "finalise"


def finalise(state: AgentState, settings: Settings) -> AgentState:
    """Mark the run complete and record the reason, and whether anything is missing."""
    exhausted = budget_exhausted(state, settings)
    gaps = research_gaps(state)
    complete = gaps.complete

    if not complete:
        decided = Decided.FINALISED_INCOMPLETE
    elif exhausted:
        decided = Decided.BUDGET_EXCEEDED
    else:
        decided = Decided.FINALISED

    event = audit_event(
        node=NODE,
        decided=decided,
        correlation_id=state.get("correlation_id", ""),
        input_digest=digest(state.get("request", "")),
        lane=state.get("lane"),
        detail={
            "iterations_used": state.get("iterations", 0),
            "iteration_budget": settings.max_iterations,
            "stopped_because": "budget_exhausted" if exhausted else "work_complete",
            "findings": len(state.get("findings", [])),
            "answer_complete": complete,
            "research": gaps.as_detail(),
        },
    )

    # The whole run's trail, including this event, appended to the durable log. Written here
    # because this is the one node every completed run passes through, and written as one
    # append so a reader never sees half a run.
    #
    # A run that never reaches finalise -- one still paused at the approval gate, or one that
    # died -- has its trail in the checkpoint and not in the file. That is a real limitation
    # and it is recoverable rather than lost: the checkpoint holds the same events. It is
    # stated in the module docstring rather than left for someone to discover from an empty
    # log after an incident.
    write_trail([*state.get("audit_trail", []), event], settings)

    return {
        "finalised": True,
        "answer_complete": complete,
        "audit_trail": [event],
    }
