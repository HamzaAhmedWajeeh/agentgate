"""The irreversible action, reachable only past the gate.

The topology already guarantees that: the only edge into this node comes out of
``approval_gate``, and only on the approved branch. This node checks anyway, and the
duplication is deliberate.

A topological guarantee is a statement about the graph as currently drawn. It is true until
somebody adds an edge, and the person adding that edge will be thinking about the feature they
are adding rather than about this invariant. The check here is a statement about the node
itself, and it holds regardless of what the graph looks like. That is the same argument the
tool allowlist makes about binding versus authorisation, and it applies for the same reason:
the expensive failure is silent, and cheap redundancy against a silent failure is worth having.
"""

from __future__ import annotations

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.errors import AgentgateError
from agentgate.graph.state import AgentState, Decision

NODE = "execute"


class UnapprovedExecutionError(AgentgateError):
    """``execute`` was reached without an approval on the state it was reached with.

    Raised rather than returned. Every other failure in this graph is caught and summarised,
    because the alternative is losing work; this one aborts the run, because the alternative is
    performing an irreversible action nobody sanctioned. A run that dies here is a run that did
    not do the thing.
    """


def execute(state: AgentState, settings: Settings) -> AgentState:  # noqa: ARG001 - node signature
    """Perform the approved action and record it.

    Raises:
        UnapprovedExecutionError: if the decision on state is anything but approved.
    """
    decision = state.get("decision", Decision.PENDING)
    if decision is not Decision.APPROVED:
        msg = (
            f"execute reached with decision={decision.value!r}. The only edge into this node "
            "comes from the approved branch of the approval gate, so arriving here without an "
            "approval means the topology changed and this invariant did not. Refusing."
        )
        raise UnapprovedExecutionError(msg)

    # Nothing irreversible is wired yet: the tools that move money and send mail raise if their
    # handlers are reached (tools/registry.py), and the executor's allowlist is declared but no
    # agent holds it. What this node does today is record that the gate was passed, which is
    # the part the audit trail needs and the part Phase 5 is responsible for.
    return {
        "audit_trail": [
            audit_event(
                node=NODE,
                decided=Decided.EXECUTED,
                correlation_id=state.get("correlation_id", ""),
                input_digest=digest(state.get("draft", "")),
                lane=state.get("lane"),
                detail={
                    "revisions_before_approval": state.get("revisions", 0),
                    "answer_complete": state.get("answer_complete", True),
                    "irreversible_effects": [],
                },
            )
        ],
    }
