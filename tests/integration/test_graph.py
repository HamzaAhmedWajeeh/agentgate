"""The whole graph, end to end, on the fake lane.

The node tests prove each part in isolation. This proves they are wired together: that the
classifier's verdict actually reaches the router, that the router's choice actually selects a
node, and that the budget guard actually stops a run rather than merely being able to.

Offline, deterministic, no key. Every one of these runs in CI.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from agentgate.config import CallClass, CheckpointerBackend, Lane, Settings, Tier
from agentgate.graph.build import build_checkpointer, build_graph, checkpointer_for
from agentgate.graph.state import initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")


def verdict(sensitivity: str, complexity: str = "simple", pii: bool = False) -> str:
    return scripted_json(
        {
            "sensitivity": sensitivity,
            "complexity": complexity,
            "contains_pii": pii,
            "reason": "test fixture",
        }
    )


def factory_for(response: str) -> Any:
    def make(
        _settings: Settings, _tier: Tier, _call_class: CallClass, *, lane: Lane | None = None
    ) -> FakeChatModel:
        return FakeChatModel(responses=[response])

    return make


def run(settings: Settings, request: str, response: str, **state: object) -> dict[str, Any]:
    graph = build_graph(settings, build_checkpointer(settings), model_factory=factory_for(response))
    return graph.invoke(  # type: ignore[no-any-return]
        {**initial_state(request, "corr-1"), **state},
        {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": settings.recursion_limit,
        },
    )


def settings_with(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- the happy path


def test_a_run_classifies_routes_and_finalises() -> None:
    result = run(settings_with(), "Summarise the refund policy.", verdict("internal"))

    decided = [event["decided"] for event in result["audit_trail"]]
    assert decided == ["classified", "lane_selected", "dispatched", "finalised"]
    assert result["finalised"] is True


def test_every_node_records_the_same_correlation_id() -> None:
    """A trail whose events cannot be tied together is a pile of events."""
    result = run(settings_with(), "A request.", verdict("public"))

    assert {event["correlation_id"] for event in result["audit_trail"]} == {"corr-1"}


def test_the_audit_trail_accumulates_rather_than_overwriting() -> None:
    """`operator.add` on the channel is what makes this true; last-write-wins would not."""
    result = run(settings_with(), "A request.", verdict("public"))

    assert len(result["audit_trail"]) > 1


# --------------------------------------------------------------------------- policy, end to end


@pytest.mark.parametrize(
    ("sensitivity", "complexity", "expected_lane"),
    [
        ("public", "simple", Lane.CLOUD.value),
        ("internal", "involved", Lane.CLOUD.value),
        ("restricted", "simple", Lane.SOVEREIGN.value),
        ("restricted", "involved", Lane.SOVEREIGN.value),
    ],
)
def test_the_classifiers_verdict_reaches_the_lane_binding(
    sensitivity: str, complexity: str, expected_lane: str
) -> None:
    """The wiring that matters: a verdict that never reaches the router decides nothing."""
    result = run(settings_with(), "A request.", verdict(sensitivity, complexity))

    assert result["lane"] == expected_lane


def test_restricted_content_reaches_the_sovereign_lane_in_a_real_run() -> None:
    """The thesis, executed rather than asserted about."""
    result = run(
        settings_with(),
        "Draft a note about account 4471 and its outstanding balance.",
        verdict("restricted", "involved", pii=True),
    )

    assert result["lane"] == Lane.SOVEREIGN.value
    lane_event = next(e for e in result["audit_trail"] if e["decided"] == "lane_selected")
    assert lane_event["detail"]["because"] == "restricted"


def test_an_unparseable_verdict_still_lands_on_the_sovereign_lane() -> None:
    """Fail-closed, proved through the compiled graph rather than against the node."""
    result = run(settings_with(), "A request.", "the model said something unhelpful")

    assert result["lane"] == Lane.SOVEREIGN.value


# --------------------------------------------------------------------------- the budget gate


def test_the_budget_guard_stops_a_run_that_will_not_stop_itself() -> None:
    """Seeded with outstanding work so the supervisor keeps going. The guard has to win."""
    settings = settings_with(max_iterations=3, recursion_limit=40)

    result = run(settings, "A request.", verdict("public"), sub_questions=["one", "two", "three"])

    assert result["iterations"] == 3
    assert result["audit_trail"][-1]["decided"] == "budget_exceeded"


def test_the_budget_guard_trips_before_the_recursion_limit() -> None:
    """The guard is the policy; recursion_limit is the backstop. Order matters.

    If the backstop fired first the run would abort instead of finalising cleanly, and there
    would be no audit event explaining why it stopped.
    """
    settings = settings_with(max_iterations=2, recursion_limit=40)

    result = run(settings, "A request.", verdict("public"), sub_questions=["one"])

    assert result["finalised"] is True
    assert result["audit_trail"][-1]["detail"]["stopped_because"] == "budget_exhausted"


def test_a_completed_run_records_completion_not_exhaustion() -> None:
    result = run(settings_with(max_iterations=8), "A request.", verdict("public"))

    assert result["audit_trail"][-1]["detail"]["stopped_because"] == "work_complete"


# --------------------------------------------------------------------------- checkpointer


def test_the_checkpointer_is_chosen_by_configuration_alone() -> None:
    """Three backends, one graph, no code change. The claim in build.py's docstring."""
    memory = settings_with()
    assert memory.checkpointer is CheckpointerBackend.MEMORY

    with checkpointer_for(memory) as saver:
        assert build_graph(memory, saver) is not None


def test_a_sqlite_configured_run_persists_to_disk(tmp_path: Any) -> None:
    settings = settings_with(checkpointer="sqlite", sqlite_path=tmp_path / "graph.db")
    config = {
        "configurable": {"thread_id": "persisted"},
        "recursion_limit": settings.recursion_limit,
    }

    with checkpointer_for(settings) as saver:
        graph = build_graph(settings, saver, model_factory=factory_for(verdict("public")))
        graph.invoke(initial_state("A request.", "corr-1"), config)

    assert (tmp_path / "graph.db").exists()


def test_the_same_graph_runs_under_every_available_checkpointer(tmp_path: Any) -> None:
    """Behaviour must not depend on which backend is configured.

    Postgres is excluded here because it needs a server; the Compose stack covers it. What
    this pins is that memory and SQLite produce the same result, so the crash-resume behaviour
    tested against SQLite is not a SQLite peculiarity.
    """
    outcomes = []
    for backend, extra in [
        ("memory", {}),
        ("sqlite", {"sqlite_path": tmp_path / "parity.db"}),
    ]:
        settings = settings_with(checkpointer=backend, **extra)
        with checkpointer_for(settings) as saver:
            graph = build_graph(settings, saver, model_factory=factory_for(verdict("internal")))
            result = graph.invoke(
                initial_state("A request.", "corr-1"),
                {
                    "configurable": {"thread_id": f"parity-{backend}"},
                    "recursion_limit": settings.recursion_limit,
                },
            )
        outcomes.append((result["lane"], result["finalised"], len(result["audit_trail"])))

    assert outcomes[0] == outcomes[1]
