"""The human gate: nothing irreversible happens on the far side of this without a person.

Built on ``interrupt()``, called from inside the node body, rather than on ``interrupt_before``
in the graph definition. That is not a style preference -- ``interrupt_before`` passed in the
invoke config is silently ignored (leak inventory, item 4), and a gate that does not gate looks
exactly like a gate that does. ``interrupt()`` cannot be dropped by being passed in the wrong
place, because it is a call, not a configuration value.

**Nothing before the ``interrupt()`` may have a side effect.** This is the property the whole
node is arranged around, and the reason is mechanical rather than stylistic: on resume,
LangGraph re-executes the interrupted node *from its top*. Every statement above the
``interrupt()`` runs a second time. A node that appended an audit event, incremented a counter,
or sent anything before pausing would do it once per resume, and a run resumed three times
would have three of whatever it was.

So the shape here is: read state, build the summary, pause. All three are pure. Everything with
an effect -- the audit event, the decision, the feedback -- happens strictly after the
``interrupt()`` returns, which is code that only runs once because it only runs on the resume
side.

``tests/integration/test_approval_gate.py`` proves the re-execution rather than describing it:
a counter incremented above the ``interrupt()`` is observed going up on every resume, which is
what makes the rule real rather than folklore.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Command, interrupt

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.completeness import research_gaps
from agentgate.graph.state import AgentState, Decision

NODE = "approval_gate"

Destination = Literal["execute", "drafter"]


def review_packet(state: AgentState) -> dict[str, Any]:
    """What the human is shown.

    Pure, and computed before the pause, so it is safe to recompute on every resume. It carries
    the completeness of the research as well as the draft: approving a deliverable without
    being told that a third of its evidence never arrived is not informed approval, and the
    gate exists to be informed.

    Completeness is **computed here, not read off state**. ``answer_complete`` is written by
    ``finalise``, which runs on the far side of this gate, so reading the field would have the
    reviewer see the default -- ``True`` -- on precisely the runs where it is false. That is
    the same failure this system spends a whole channel guarding against, aimed at the one
    person whose job is to catch it.
    """
    gaps = research_gaps(state)
    return {
        "request": state.get("request", ""),
        "draft": state.get("draft", ""),
        "findings": len(state.get("findings", [])),
        "answer_complete": gaps.complete,
        "research": gaps.as_detail(),
        "revision": state.get("revisions", 0),
        "correlation_id": state.get("correlation_id", ""),
    }


def approval_gate(state: AgentState, settings: Settings) -> Command[Destination]:
    """Pause for a human decision, then route on what they said.

    Returns a ``Command`` so the decision and its consequence are one value. An approval that
    updated state and left the routing to a conditional edge reading it back would have a
    window in which the run is approved and control has not moved -- and that window is exactly
    where a crash would resume into the wrong branch.

    Args:
        state: Read only above the pause.
        settings: Supplies the revision budget.
    """
    # --- above the interrupt: pure only. This block re-runs on every resume. -------------
    packet = review_packet(state)
    correlation_id = state.get("correlation_id", "")
    revisions = state.get("revisions", 0)

    verdict = interrupt(packet)

    # --- below the interrupt: runs once, on the resume side. -----------------------------
    decision, feedback = _read_verdict(verdict)

    if decision is Decision.APPROVED:
        return Command(
            update={
                "decision": Decision.APPROVED.value,
                "audit_trail": [
                    audit_event(
                        node=NODE,
                        decided=Decided.APPROVED,
                        correlation_id=correlation_id,
                        input_digest=digest(state.get("draft", "")),
                        lane=state.get("lane"),
                        detail={
                            "revision": revisions,
                            "approved_partial": not packet["answer_complete"],
                        },
                    )
                ],
            },
            goto="execute",
        )

    return Command(
        update={
            "decision": Decision.REJECTED.value,
            "feedback": feedback,
            "revisions": revisions + 1,
            # Cleared so the supervisor routes back to the drafter. The draft is the thing
            # being rejected; leaving it in place would have the next turn treat the run as
            # already drafted and walk straight back to the gate with the same text.
            "draft": "",
            "audit_trail": [
                audit_event(
                    node=NODE,
                    decided=Decided.REJECTED,
                    correlation_id=correlation_id,
                    input_digest=digest(state.get("draft", "")),
                    lane=state.get("lane"),
                    detail={
                        "revision": revisions,
                        "revision_budget": settings.max_iterations,
                        "feedback_given": bool(feedback),
                    },
                )
            ],
        },
        goto="drafter",
    )


def _read_verdict(verdict: Any) -> tuple[Decision, str]:
    """Interpret whatever the resume supplied.

    Fails closed. Anything this does not recognise as an explicit approval is a rejection,
    because the cost of misreading a rejection as approval is an irreversible action nobody
    sanctioned, and the cost of the opposite is one more revision.
    """
    if isinstance(verdict, dict):
        raw = str(verdict.get("decision", "")).strip().lower()
        feedback = str(verdict.get("feedback", ""))
    else:
        raw = str(verdict).strip().lower()
        feedback = ""

    if raw == Decision.APPROVED.value:
        return Decision.APPROVED, ""
    return Decision.REJECTED, feedback
