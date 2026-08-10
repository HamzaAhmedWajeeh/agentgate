"""Each node as a pure function of state.

Nodes are tested without a graph, because a node that only works inside a compiled graph is a
node whose behaviour nobody can reason about. Every one of these takes a state dict and returns
a state update.

The policy properties get the most attention. A routing bug does not look like a bug -- the run
succeeds, the output is plausible, and the only symptom is that content reached somewhere it
should not have.
"""

from __future__ import annotations

import pytest

from agentgate.audit.events import Decided
from agentgate.config import CallClass, Lane, Settings, Tier
from agentgate.graph.nodes.classify import classify
from agentgate.graph.nodes.finalise import finalise
from agentgate.graph.nodes.lanes import LANE_NODES
from agentgate.graph.nodes.supervisor import supervise
from agentgate.graph.routing import route_by_budget, route_by_policy
from agentgate.graph.state import (
    AgentState,
    Classification,
    Complexity,
    Sensitivity,
    initial_state,
)
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")


def settings_with(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def factory_returning(*responses: str) -> object:
    def make(
        _settings: Settings, _tier: Tier, _call_class: CallClass, *, lane: Lane | None = None
    ) -> FakeChatModel:
        return FakeChatModel(responses=list(responses))

    return make


def classified(**overrides: object) -> Classification:
    base: dict[str, object] = {
        "sensitivity": Sensitivity.INTERNAL,
        "complexity": Complexity.SIMPLE,
        "contains_pii": False,
        "reason": "test",
    }
    return Classification(**{**base, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- classify


def test_classify_returns_a_validated_verdict() -> None:
    state = initial_state("Summarise the refund policy.", "run-1")
    verdict = scripted_json(
        {
            "sensitivity": "internal",
            "complexity": "simple",
            "contains_pii": False,
            "reason": "ordinary business request",
        }
    )

    update = classify(state, settings_with(), model_factory=factory_returning(verdict))  # type: ignore[arg-type]

    assert update["classification"]["sensitivity"] == Sensitivity.INTERNAL.value
    assert update["audit_trail"][0]["decided"] == Decided.CLASSIFIED.value


def test_classify_records_the_input_as_a_hash_not_as_content() -> None:
    """The trail has to be readable by people who may not read the request itself."""
    sensitive_request = "Client Foo owes 12,000 and their contact is jane@example.com"
    state = initial_state(sensitive_request, "run-1")

    update = classify(state, settings_with(), model_factory=factory_returning("not json"))  # type: ignore[arg-type]

    rendered = str(update["audit_trail"])
    assert sensitive_request not in rendered
    assert "jane@example.com" not in rendered


# ------------------------------------------------------ fail closed: the policy property


def test_unparseable_classification_is_treated_as_restricted() -> None:
    """Fail closed. This is a policy decision, not an artefact of the fake model.

    An unparseable verdict means the sensitivity is *unknown*, and unknown sensitivity must be
    handled as the most restrictive case. The alternative -- defaulting to something permissive
    when the classifier stumbles -- makes a JSON parse error sufficient to send restricted
    content to a cloud provider.

    If anyone ever makes this fail open, this test goes red.
    """
    state = initial_state("Anything at all.", "run-1")

    update = classify(
        state,
        settings_with(),
        model_factory=factory_returning("I'm sorry, I can't help with that."),  # type: ignore[arg-type]
    )

    assert update["classification"]["sensitivity"] == Sensitivity.RESTRICTED.value
    assert update["classification"]["contains_pii"] is True


def test_a_failed_classification_routes_to_the_most_restrictive_lane() -> None:
    """The property that actually matters: the verdict is only interesting if routing honours it."""
    state = initial_state("Anything at all.", "run-1")
    update = classify(state, settings_with(), model_factory=factory_returning("not json"))  # type: ignore[arg-type]

    destination = route_by_policy({**state, **update})  # type: ignore[typeddict-item]

    assert destination == "sovereign"
    assert destination != "cloud_capable"


def test_a_failure_is_distinguishable_from_a_genuine_restricted_verdict() -> None:
    """ "Judged restricted" and "could not judge, so assumed restricted" are different facts.

    A reviewer looking at a trail full of restricted verdicts needs to know which kind they
    are, or the classifier could be entirely broken and look merely cautious.
    """
    failed = classify(
        initial_state("x", "r"), settings_with(), model_factory=factory_returning("not json")
    )  # type: ignore[arg-type]
    genuine = classify(
        initial_state("x", "r"),
        settings_with(),
        model_factory=factory_returning(
            scripted_json(
                {
                    "sensitivity": "restricted",
                    "complexity": "simple",
                    "contains_pii": True,
                    "reason": "contains client identifiers",
                }
            )
        ),  # type: ignore[arg-type]
    )

    assert failed["audit_trail"][0]["detail"]["classification_failed"] is not None
    assert genuine["audit_trail"][0]["detail"]["classification_failed"] is None


def test_an_unclassified_state_also_routes_to_the_sovereign_lane() -> None:
    """Belt and braces: the router fails closed too, not only the classifier.

    Two independent chances to get this right, because one of them being wrong is enough.
    """
    assert route_by_policy(initial_state("x", "r")) == "sovereign"


# --------------------------------------------------------------------------- policy routing


@pytest.mark.parametrize(
    ("sensitivity", "complexity", "expected"),
    [
        (Sensitivity.RESTRICTED, Complexity.SIMPLE, "sovereign"),
        (Sensitivity.RESTRICTED, Complexity.INVOLVED, "sovereign"),
        (Sensitivity.INTERNAL, Complexity.SIMPLE, "cloud_cheap"),
        (Sensitivity.INTERNAL, Complexity.INVOLVED, "cloud_capable"),
        (Sensitivity.PUBLIC, Complexity.SIMPLE, "cloud_cheap"),
        (Sensitivity.PUBLIC, Complexity.INVOLVED, "cloud_capable"),
    ],
)
def test_the_policy_table_in_full(
    sensitivity: Sensitivity, complexity: Complexity, expected: str
) -> None:
    """Every combination, because the interesting cell is the one nobody thought about."""
    state: AgentState = {
        "classification": classified(sensitivity=sensitivity, complexity=complexity)
    }

    assert route_by_policy(state) == expected


def test_restricted_content_never_reaches_a_cloud_lane_however_complex() -> None:
    """Complexity must not be able to override sensitivity.

    A restricted request that would be answered better by a capable cloud model is still not
    allowed to reach one. That trade belongs to the operator, in policy, not to the graph on a
    per-request basis.
    """
    for complexity in Complexity:
        state: AgentState = {
            "classification": classified(sensitivity=Sensitivity.RESTRICTED, complexity=complexity)
        }
        assert not route_by_policy(state).startswith("cloud")


def test_every_route_the_policy_can_return_has_a_node() -> None:
    """A route with no node compiles fine and dead-ends at runtime."""
    for sensitivity in Sensitivity:
        for complexity in Complexity:
            state: AgentState = {
                "classification": classified(sensitivity=sensitivity, complexity=complexity)
            }
            assert route_by_policy(state) in LANE_NODES


# --------------------------------------------------------------------------- lane binding


def test_binding_a_lane_records_it_as_a_serialisable_string() -> None:
    """State is checkpointed. An enum that round-trips as a string is a resume-time surprise."""
    update = LANE_NODES["sovereign"](initial_state("x", "r"), settings=settings_with())

    assert update["lane"] == Lane.SOVEREIGN.value
    assert isinstance(update["lane"], str)
    assert update["audit_trail"][0]["decided"] == Decided.LANE_SELECTED.value


def test_the_lane_event_records_why() -> None:
    state = {**initial_state("x", "r"), "classification": classified()}

    update = LANE_NODES["cloud_cheap"](state, settings=settings_with())  # type: ignore[arg-type]

    assert update["audit_trail"][0]["detail"]["because"] == Sensitivity.INTERNAL.value


# --------------------------------------------------------------------------- supervisor


def test_the_supervisor_advances_the_counter_and_moves_control_together() -> None:
    """One object, so state and control flow cannot disagree about what happened."""
    command = supervise(initial_state("x", "r"), settings_with())

    assert command.update["iterations"] == 1
    assert command.goto == "budget_guard"


def test_the_supervisor_finishes_when_there_is_nothing_outstanding() -> None:
    command = supervise(initial_state("x", "r"), settings_with())

    assert command.update["finalised"] is True


def test_the_supervisor_keeps_going_while_work_remains() -> None:
    state = {**initial_state("x", "r"), "sub_questions": ["one", "two"]}

    command = supervise(state, settings_with())  # type: ignore[arg-type]

    assert command.update["finalised"] is False
    assert command.update["audit_trail"][0]["detail"]["outstanding"] == 2


# --------------------------------------------------------------------------- budget guard


def test_the_budget_guard_lets_a_fresh_run_continue() -> None:
    state = {**initial_state("x", "r"), "sub_questions": ["one"]}

    assert route_by_budget(state, settings_with(max_iterations=4)) == "continue"  # type: ignore[arg-type]


def test_the_budget_guard_stops_at_the_ceiling() -> None:
    state = {**initial_state("x", "r"), "iterations": 4}

    assert route_by_budget(state, settings_with(max_iterations=4)) == "finalise"  # type: ignore[arg-type]


def test_the_budget_guard_honours_an_early_finish() -> None:
    state = {**initial_state("x", "r"), "finalised": True}

    assert route_by_budget(state, settings_with()) == "finalise"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- finalise


def test_finalising_records_completion_rather_than_exhaustion() -> None:
    state = {**initial_state("x", "r"), "iterations": 1}

    update = finalise(state, settings_with(max_iterations=8))  # type: ignore[arg-type]

    assert update["finalised"] is True
    assert update["audit_trail"][0]["decided"] == Decided.FINALISED.value
    assert update["audit_trail"][0]["detail"]["stopped_because"] == "work_complete"


def test_finalising_after_exhaustion_says_so() -> None:
    """ "Finished" and "ran out of budget" look identical in the output and are not the same."""
    state = {**initial_state("x", "r"), "iterations": 8}

    update = finalise(state, settings_with(max_iterations=8))  # type: ignore[arg-type]

    assert update["audit_trail"][0]["decided"] == Decided.BUDGET_EXCEEDED.value
    assert update["audit_trail"][0]["detail"]["stopped_because"] == "budget_exhausted"
