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
from agentgate.effects.proposals import digest_of
from agentgate.graph.completeness import research_gaps
from agentgate.graph.state import AgentState, Decision, proposals_of

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
    proposals = proposals_of(state)
    return {
        "request": state.get("request", ""),
        "draft": state.get("draft", ""),
        # The actions, exactly, and a hash of exactly them. An approval has to carry this hash
        # back, so it can only ever authorise what was on this packet (leak inventory item 24).
        "proposed_actions": [proposal.as_channel() for proposal in proposals],
        "proposals_digest": digest_of(proposals),
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
    decision, feedback, approved_digest = _read_verdict(verdict)

    # Approving has to name what it approves. The hash is recomputed from state here, on the
    # resume side, and compared with the one the approval carries from the packet the human saw:
    # a proposal changed after the pause no longer matches, and the approval does not transfer.
    # With no proposals there is nothing to act on, so an approval without a hash is the old,
    # draft-only approval; with any, the hash is required.
    shown = packet["proposals_digest"]
    matches = (
        approved_digest == shown if approved_digest is not None else not packet["proposed_actions"]
    )
    if decision is Decision.APPROVED and not matches:
        return _refused(state, settings, packet, revisions, approved_digest)

    if decision is Decision.APPROVED:
        return Command(
            update={
                "decision": Decision.APPROVED.value,
                "approved_digest": shown,
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
                            "proposals_digest": shown,
                            "actions_approved": len(packet["proposed_actions"]),
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
            "proposed_actions": [],
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


def _refused(
    state: AgentState,
    settings: Settings,
    packet: dict[str, Any],
    revisions: int,
    approved_digest: str | None,
) -> Command[Destination]:
    """An approval that does not match what was shown. Treated as a rejection: the run does not
    act, the draft and its proposals go back for revision, and the refusal is recorded as itself
    rather than as a reviewer's rejection."""
    return Command(
        update={
            "decision": Decision.REJECTED.value,
            "feedback": "the approval did not match the actions that were shown",
            "revisions": revisions + 1,
            "draft": "",
            "proposed_actions": [],
            "audit_trail": [
                audit_event(
                    node=NODE,
                    decided=Decided.APPROVAL_REFUSED,
                    correlation_id=state.get("correlation_id", ""),
                    input_digest=digest(state.get("draft", "")),
                    lane=state.get("lane"),
                    detail={
                        "revision": revisions,
                        "revision_budget": settings.max_iterations,
                        "proposals_digest": packet["proposals_digest"],
                        "approval_carried": approved_digest,
                        "actions_proposed": len(packet["proposed_actions"]),
                    },
                )
            ],
        },
        goto="drafter",
    )


def _read_verdict(verdict: Any) -> tuple[Decision, str, str | None]:
    """Interpret whatever the resume supplied.

    Fails closed. Anything this does not recognise as an explicit approval is a rejection,
    because the cost of misreading a rejection as approval is an irreversible action nobody
    sanctioned, and the cost of the opposite is one more revision.
    """
    approved_digest: str | None = None
    if isinstance(verdict, dict):
        raw = str(verdict.get("decision", "")).strip().lower()
        feedback = str(verdict.get("feedback", ""))
        carried = verdict.get("approved_digest")
        approved_digest = str(carried) if carried is not None else None
    else:
        raw = str(verdict).strip().lower()
        feedback = ""

    if raw == Decision.APPROVED.value:
        return Decision.APPROVED, "", approved_digest
    return Decision.REJECTED, feedback, approved_digest
