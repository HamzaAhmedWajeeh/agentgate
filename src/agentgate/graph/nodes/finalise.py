"""Close the run, recording why it ended.

"Finished the work" and "ran out of budget" produce the same absence of further activity and
are not the same event. A trail that cannot distinguish them cannot answer the only question
anyone asks about a truncated result.
"""

from __future__ import annotations

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.routing import budget_exhausted
from agentgate.graph.state import AgentState

NODE = "finalise"


def finalise(state: AgentState, settings: Settings) -> AgentState:
    """Mark the run complete and record the reason."""
    exhausted = budget_exhausted(state, settings)
    return {
        "finalised": True,
        "audit_trail": [
            audit_event(
                node=NODE,
                decided=Decided.BUDGET_EXCEEDED if exhausted else Decided.FINALISED,
                correlation_id=state.get("correlation_id", ""),
                input_digest=digest(state.get("request", "")),
                lane=state.get("lane"),
                detail={
                    "iterations_used": state.get("iterations", 0),
                    "iteration_budget": settings.max_iterations,
                    "stopped_because": "budget_exhausted" if exhausted else "work_complete",
                    "findings": len(state.get("findings", [])),
                },
            )
        ],
    }
