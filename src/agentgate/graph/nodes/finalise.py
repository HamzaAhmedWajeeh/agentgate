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
"""

from __future__ import annotations

from agentgate.audit.events import Decided, audit_event, digest
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

    return {
        "finalised": True,
        "answer_complete": complete,
        "audit_trail": [
            audit_event(
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
        ],
    }
