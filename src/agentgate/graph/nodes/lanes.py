"""The three targets of the policy gate.

Each binds a lane and tier onto state and records that the binding happened. They are thin on
purpose: the decision was made by ``route_by_policy``, and a node that re-derived it would give
the policy two homes and eventually two answers.

They exist as separate nodes rather than one node with a parameter because that is what makes
the policy gate visible in the compiled graph. Rendering the topology shows three edges leaving
the router, which is the honest picture of what the system can do with a request.
"""

from __future__ import annotations

from typing import Protocol

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Lane, Settings, Tier
from agentgate.graph.state import AgentState


class LaneNode(Protocol):
    """A lane-binding node.

    Declared as a Protocol rather than a bare Callable so the ``settings`` parameter keeps
    its name. ``functools.partial(node, settings=...)`` cannot be type-checked against a
    Callable alias, which erases argument names.
    """

    def __call__(self, state: AgentState, *, settings: Settings) -> AgentState: ...


def bind_lane(lane: Lane, tier: Tier) -> LaneNode:
    """Build the node that records a lane and tier selection.

    Args:
        lane: Where this request is now allowed to go.
        tier: Which capability tier within that lane.
    """
    node_name = f"bind_{lane.value}_{tier.value}"

    def bind(state: AgentState, *, settings: Settings) -> AgentState:
        classification = state.get("classification")
        return {
            # Stored as a plain string: state is serialised into checkpoints, and an enum
            # that round-trips through JSON as a string but is compared as an enum is a
            # resume-time surprise waiting to happen.
            "lane": lane.value,
            "audit_trail": [
                audit_event(
                    node=node_name,
                    decided=Decided.LANE_SELECTED,
                    correlation_id=state.get("correlation_id", ""),
                    input_digest=digest(state.get("request", "")),
                    model=settings.model_for(tier),
                    lane=lane.value,
                    detail={
                        "tier": tier.value,
                        "because": (
                            classification.sensitivity.value
                            if classification is not None
                            else "unclassified"
                        ),
                    },
                )
            ],
        }

    return bind


# The three targets `route_by_policy` can return, named to match its Literal exactly. A
# mismatch between these keys and that Literal is a graph that compiles and then dead-ends at
# runtime, so they are defined together and asserted equal in the tests.
LANE_NODES: dict[str, LaneNode] = {
    "cloud_capable": bind_lane(Lane.CLOUD, Tier.CAPABLE),
    "cloud_cheap": bind_lane(Lane.CLOUD, Tier.CHEAP),
    "sovereign": bind_lane(Lane.SOVEREIGN, Tier.CHEAP),
}
