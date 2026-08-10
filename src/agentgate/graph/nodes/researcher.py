"""Dispatch research over sub-questions, bounded before anything runs.

This node does not do the research. It decides *how much* research is about to happen, records
that decision, and lets the fan-out edge open one branch per surviving question.

**The width cap is the point of this node.** A ``Send`` fan-out is the one place in the graph
where a model chooses how many model calls get paid for: the list being fanned out over is
model output. The iteration budget counts hand-offs that already happened and the token ceiling
counts tokens that were already spent -- both are rear-view mirrors, which is enough while work
arrives one call at a time and useless against a single super-step that opens forty branches
at once. Width has to be decided going in.

So the cap is applied here, before a single ``Send`` exists, and the truncation is an audit
event rather than a silently shorter list. :func:`cap_fan_out` is also applied at the fan-out
edge itself, which is belt and braces on purpose: this node records *why* the list shrank, and
the edge guarantees the width regardless of what is in state by the time it runs.
"""

from __future__ import annotations

from langgraph.types import Send

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.state import AgentState

NODE = "researcher"
BRANCH = "research_branch"


def cap_fan_out(questions: list[str], settings: Settings) -> list[str]:
    """The questions a dispatch is allowed to open branches for.

    Truncates rather than refuses. A request that produced eleven sub-questions is not
    malicious and its first five are still worth answering; refusing outright would turn a
    verbose plan into a failed run. What must not happen is the truncation going unrecorded --
    the caller has to be able to tell a five-branch answer to a five-branch question from a
    five-branch answer to an eleven-branch one, and :data:`AgentState.answer_complete` carries
    that.

    Pure, and used in two places, so the width guarantee does not depend on which one ran.
    """
    return [question for question in questions if question.strip()][: settings.max_fan_out]


def research(state: AgentState, settings: Settings) -> AgentState:
    """Record what is about to be dispatched, and how much of it was dropped."""
    requested = [question for question in state.get("sub_questions", []) if question.strip()]
    dispatching = cap_fan_out(requested, settings)
    dropped = len(requested) - len(dispatching)

    events = [
        audit_event(
            node=NODE,
            decided=Decided.DISPATCHED,
            correlation_id=state.get("correlation_id", ""),
            input_digest=digest(state.get("request", "")),
            lane=state.get("lane"),
            detail={
                "requested": len(requested),
                "dispatching": len(dispatching),
                "width_limit": settings.max_fan_out,
            },
        )
    ]

    if dropped:
        # Its own event, not a field on the one above. "Dispatched five" and "dispatched five
        # of eleven" are different facts, and a reviewer scanning for the second should not
        # have to know to open the first.
        events.append(
            audit_event(
                node=NODE,
                decided=Decided.FAN_OUT_CAPPED,
                correlation_id=state.get("correlation_id", ""),
                input_digest=digest(state.get("request", "")),
                lane=state.get("lane"),
                detail={
                    "requested": len(requested),
                    "width_limit": settings.max_fan_out,
                    "dropped": dropped,
                    "dropped_questions": requested[settings.max_fan_out :],
                },
            )
        )

    return {
        "sub_questions": dispatching,
        "dispatched": len(dispatching),
        "audit_trail": events,
    }


def dispatch(state: AgentState, settings: Settings) -> list[Send]:
    """The fan-out edge: one ``Send`` per question, and never more than the cap.

    A module-level function rather than a closure inside ``build_graph`` so that it can be
    called directly by a test. The width guarantee is worth being able to check without
    standing up a graph, because the guarantee is the whole point: this runs at the moment
    ``Send`` objects come into existence, which is the last instant before a fan-out becomes
    a bill.

    It caps again rather than trusting ``research`` to have done it. That is not defensive
    padding -- ``research`` explains the truncation and this enforces it, and a guarantee that
    holds only when an earlier node ran first is a convention, not a guarantee.
    """
    return [
        Send(
            BRANCH,
            {
                "question": question,
                "correlation_id": state.get("correlation_id", ""),
                "lane": state.get("lane", ""),
            },
        )
        for question in cap_fan_out(list(state.get("sub_questions", [])), settings)
    ]
