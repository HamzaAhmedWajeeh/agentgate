"""Routing functions: the policy gate and the budget gate.

Both are conditional edges rather than nodes, and that is the design. A conditional edge
returns the name of the next node and cannot modify state, so a routing decision cannot
quietly change anything on its way past. Reading these two functions tells you every path the
graph can take.

The policy gate is the one that matters. It decides which lane -- which family of endpoints --
a request is allowed to reach, based on how its content was classified. It is the whole reason
this system exists, so it lives in one small function that returns a closed set of literals and
never calls a model.
"""

from __future__ import annotations

from typing import Literal

from agentgate.config import Settings
from agentgate.graph.state import AgentState, Complexity, Sensitivity, classification_of

PolicyRoute = Literal["cloud_capable", "cloud_cheap", "sovereign"]
BudgetRoute = Literal["continue", "finalise"]


def route_by_policy(state: AgentState) -> PolicyRoute:
    """Choose the lane and tier a request may reach.

    The rule, in order of precedence:

    1. **Restricted content goes to the sovereign lane.** No exceptions, and complexity does
       not enter into it. A restricted request that would be answered better by a cloud model
       is still not allowed to reach one -- that trade is the operator's to make in policy,
       not the graph's to make per request.
    2. Anything else goes to the cloud lane, at the capable tier only if the request looks
       involved enough to need it.

    Returning a ``Literal`` rather than a string means a typo in a branch name is a type error
    rather than a runtime dead end.

    Note that both cloud tiers commonly resolve to the same model. The split is a policy
    boundary that exists so tier selection is explicit and auditable, not a claim that one
    tier is more expensive.
    """
    classification = classification_of(state)
    if classification is None:
        # Unclassified means unknown sensitivity, and unknown sensitivity is treated as the
        # most restrictive. Failing open here would make the classifier a single point of
        # policy failure: one bad parse and restricted content reaches a cloud model.
        return "sovereign"

    if classification.sensitivity is Sensitivity.RESTRICTED:
        return "sovereign"

    if classification.complexity is Complexity.INVOLVED:
        return "cloud_capable"
    return "cloud_cheap"


def route_by_budget(state: AgentState, settings: Settings) -> BudgetRoute:
    """Decide whether the run may take another turn.

    Compared against an explicit counter in state rather than against LangGraph's own
    ``recursion_limit``. The two are different things: this is a policy that stops a run
    cleanly and records why, while the recursion limit is a backstop that aborts. Configuration
    enforces that the backstop sits above this, so this trips first.
    """
    if state.get("finalised"):
        return "finalise"
    if state.get("iterations", 0) >= settings.max_iterations:
        return "finalise"
    return "continue"


def budget_exhausted(state: AgentState, settings: Settings) -> bool:
    """Whether the iteration budget is the reason a run is stopping.

    Distinct from :func:`route_by_budget` so the audit trail can say *why* a run finalised.
    "Finished" and "ran out of budget" look identical in the output and are not the same
    event.
    """
    return state.get("iterations", 0) >= settings.max_iterations
