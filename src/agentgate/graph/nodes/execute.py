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

**It is the executor.** It holds the ``EXECUTOR`` allowlist and performs the approved proposals --
only those, only if they hash to what the human approved, each screened again against that
allowlist and the tool's schema, and each through the effect sink, whose only implementation is an
append-only outbox that performs nothing real. Until this existed the node recorded
``irreversible_effects: []`` because no effect had ever been proposed: leak inventory item 24.

**Exactly once.** Each effect is keyed by the run, its position and its arguments, and the sink
does not write a key twice. That matters here more than anywhere: the node can be re-run -- a
crash after an effect is written and before the checkpoint records it resumes into this node
again -- and a second write of a refund is a customer charged twice.

**Charged to the run ledger**, though nothing it does costs anything today. It requires the run's
ledger and checks the ceiling before any effect, so a run already over budget does not get to
perform its effects on the way out, and so this is not a fourth call site connected to nothing.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.effects.proposals import digest_of, effect_key, screen_proposals
from agentgate.effects.sink import EffectSink, build_effect_sink
from agentgate.errors import AgentgateError
from agentgate.graph.state import AgentState, Decision, decision_of, proposals_of
from agentgate.guardrails.run_ledger import ledger_of
from agentgate.tools.registry import Agent, is_allowed

NODE = "execute"


class UnapprovedExecutionError(AgentgateError):
    """``execute`` was reached without an approval on the state it was reached with.

    Raised rather than returned. Every other failure in this graph is caught and summarised,
    because the alternative is losing work; this one aborts the run, because the alternative is
    performing an irreversible action nobody sanctioned. A run that dies here is a run that did
    not do the thing.
    """


def execute(
    state: AgentState,
    settings: Settings,
    config: RunnableConfig,
    effect_sink_factory: Callable[[Settings], EffectSink] = build_effect_sink,
) -> AgentState:
    """Perform the approved actions, exactly once each, and record them.

    Raises:
        UnapprovedExecutionError: if the decision on state is anything but approved, or the
            proposals no longer hash to what was approved, or one is not the executor's to run.
        LedgerMissingError: if the run was started without a ledger.
        SpendCeilingExceededError: if the run is already over its ceiling.
    """
    decision = decision_of(state)
    if decision is not Decision.APPROVED:
        msg = (
            f"execute reached with decision={decision.value!r}. The only edge into this node "
            "comes from the approved branch of the approval gate, so arriving here without an "
            "approval means the topology changed and this invariant did not. Refusing."
        )
        raise UnapprovedExecutionError(msg)

    ledger = ledger_of(config)
    ledger.check()

    proposals = proposals_of(state)
    approved = state.get("approved_digest")
    if proposals and approved != digest_of(proposals):
        msg = (
            "execute reached with proposals that do not hash to what was approved; the gate "
            "checks this, so reaching here means a route to this node skipped it. Refusing."
        )
        raise UnapprovedExecutionError(msg)

    correlation_id = state.get("correlation_id", "")
    sink = effect_sink_factory(settings)
    effects = []
    for index, proposal in enumerate(proposals):
        # Screened again, as the executor, against its own allowlist -- the drafter screened
        # before the gate, and this is the same reasoning as checking the decision above.
        if (
            not is_allowed(Agent.EXECUTOR, proposal.tool)
            or screen_proposals([proposal.as_channel()]).dropped
        ):
            msg = f"proposal {index} ({proposal.tool}) is not the executor's to run. Refusing."
            raise UnapprovedExecutionError(msg)
        effects.append(sink.record(effect_key(correlation_id, index, proposal), proposal))

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
                    "irreversible_effects": effects,
                },
            )
        ],
    }
