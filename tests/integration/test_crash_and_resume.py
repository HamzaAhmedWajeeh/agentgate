"""What happens when the process dies mid-run and comes back.

The claim being tested is the one every durable-execution system makes and few people verify:
restart against the same ``thread_id`` and the run continues from the last checkpoint rather
than from the top.

The counted side effects are the point. They separate two things that sound the same:

- **Completed super-steps are not replayed.** A node that finished before the crash does not
  run again, so its side effects happen exactly once.
- **The super-step that was in flight *is* replayed from the top.** The node that was running
  when the process died runs again from its first line -- not from wherever it got to.

That second property is the one that bites. Everything a node does before its failure point
happens twice, so anything in there must be idempotent. A node that charged a card, appended to
a ledger, or sent a message before crashing would do it a second time on resume, and no amount
of checkpointing prevents that -- checkpoints record state, not side effects on the world.

This is why the approval gate in Phase 5 is structured so that nothing before its ``interrupt()``
touches anything outside state.

SQLite is used rather than the in-memory saver because in-memory state does not survive the
process it lives in, which makes it exactly the wrong thing to test durability against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from agentgate.config import Settings
from agentgate.graph.build import build_graph, checkpointer_for
from agentgate.graph.state import AgentState, initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

CLASSIFICATION = scripted_json(
    {
        "sensitivity": "internal",
        "complexity": "simple",
        "contains_pii": False,
        "reason": "ordinary business request",
    }
)


def sqlite_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings pointed at a real database file, so state outlives a graph object."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        checkpointer="sqlite",
        sqlite_path=tmp_path / "resume.db",
        **overrides,
    )


def scripted_factory(*_args: object, **_kwargs: object) -> FakeChatModel:
    return FakeChatModel(responses=[CLASSIFICATION])


class CrashedError(RuntimeError):
    """Stands in for the process dying. Raised once, then never again."""


# --------------------------------------------------------------------- the mechanism


class Tally:
    """Counts how many times each node body was entered.

    A plain object rather than module globals so tests cannot leak counts into each other.
    """

    def __init__(self) -> None:
        self.before_crash = 0
        self.at_crash = 0
        self.after_crash = 0
        self.side_effects: list[str] = []


class ResumeState(AgentState, total=False):
    """The project state plus a marker the crashing node writes."""

    progress: str


def crashing_graph(tally: Tally, *, crash: bool) -> Any:
    """Three nodes in sequence, the middle one able to die once.

    Deliberately sequential: each node is its own super-step, so the checkpoint boundary
    between them is unambiguous and the test is about durability rather than about merging.
    """

    def before(state: ResumeState) -> ResumeState:
        tally.before_crash += 1
        # Stands in for anything with a consequence outside state: a row inserted, a message
        # queued, a file written. Recorded so the test can prove it happened exactly once.
        tally.side_effects.append("charged")
        return {"progress": "before done"}

    def middle(state: ResumeState) -> ResumeState:
        tally.at_crash += 1
        if crash:
            # Everything above this line has now run twice across the two attempts. That is
            # the replay, and it is why this node must not do anything irreversible before
            # reaching here.
            raise CrashedError
        return {"progress": "middle done"}

    def after(state: ResumeState) -> ResumeState:
        tally.after_crash += 1
        return {"progress": "after done"}

    graph = StateGraph(ResumeState)
    graph.add_node("before", before)
    graph.add_node("middle", middle)
    graph.add_node("after", after)
    graph.add_edge(START, "before")
    graph.add_edge("before", "middle")
    graph.add_edge("middle", "after")
    graph.add_edge("after", END)
    return graph


def test_a_completed_super_step_is_not_replayed_after_a_crash(tmp_path: Path) -> None:
    """The core durability claim, and the reason checkpointing is worth its complexity."""
    settings = sqlite_settings(tmp_path)
    tally = Tally()
    config = {"configurable": {"thread_id": "crash-and-resume"}}

    with checkpointer_for(settings) as saver:
        crashed = crashing_graph(tally, crash=True).compile(checkpointer=saver)
        with pytest.raises(CrashedError):
            crashed.invoke({}, config)

    assert tally.before_crash == 1
    assert tally.at_crash == 1
    assert tally.after_crash == 0

    # A new graph object against the same thread id: the shape of a restarted process.
    with checkpointer_for(settings) as saver:
        recovered = crashing_graph(tally, crash=False).compile(checkpointer=saver)
        result = recovered.invoke(None, config)

    assert result["progress"] == "after done"
    assert tally.before_crash == 1, "a completed node was replayed; the checkpoint did nothing"
    assert tally.after_crash == 1


def test_the_interrupted_super_step_is_replayed_from_its_top(tmp_path: Path) -> None:
    """The property that makes idempotency a requirement rather than a nicety.

    The node that was in flight runs again from its first line. Anything it did before dying
    happens a second time, and no checkpoint can undo it, because checkpoints record state
    rather than effects on the world.
    """
    settings = sqlite_settings(tmp_path)
    tally = Tally()
    config = {"configurable": {"thread_id": "replay-from-top"}}

    with checkpointer_for(settings) as saver:
        crashed = crashing_graph(tally, crash=True).compile(checkpointer=saver)
        with pytest.raises(CrashedError):
            crashed.invoke({}, config)

    with checkpointer_for(settings) as saver:
        recovered = crashing_graph(tally, crash=False).compile(checkpointer=saver)
        recovered.invoke(None, config)

    assert tally.at_crash == 2, "the interrupted node did not re-execute on resume"


def test_a_side_effect_before_the_failure_point_happens_only_once_here(
    tmp_path: Path,
) -> None:
    """And would happen twice if it lived in the node that crashed.

    The side effect is in `before`, which completed, so it fires once. Move the same line
    into `middle` above the raise and it fires twice -- that is the whole design constraint,
    and the reason nothing irreversible may precede an interrupt.
    """
    settings = sqlite_settings(tmp_path)
    tally = Tally()
    config = {"configurable": {"thread_id": "side-effects"}}

    with checkpointer_for(settings) as saver, pytest.raises(CrashedError):
        crashing_graph(tally, crash=True).compile(checkpointer=saver).invoke({}, config)
    with checkpointer_for(settings) as saver:
        crashing_graph(tally, crash=False).compile(checkpointer=saver).invoke(None, config)

    assert tally.side_effects == ["charged"]


# --------------------------------------------------------------------- the real graph


def test_the_project_graph_resumes_on_the_same_thread(tmp_path: Path) -> None:
    """Not a toy topology: the actual graph, interrupted before it finalises and resumed."""
    settings = sqlite_settings(tmp_path)
    config = {
        "configurable": {"thread_id": "project-graph"},
        "recursion_limit": settings.recursion_limit,
    }

    with checkpointer_for(settings) as saver:
        # interrupt_before is a compile-time option. Passing it in the invoke config is
        # silently ignored, which would leave this test asserting against a completed run.
        graph = build_graph(
            settings, saver, model_factory=scripted_factory, interrupt_before=["finalise"]
        )
        paused = graph.invoke(initial_state("Summarise the refund policy.", "resume-run"), config)

    # `finalised` is already True in state -- the supervisor set it. What has *not* happened
    # is the finalise node, so the absence to assert on is the audit event, not the flag.
    paused_events = [event["decided"] for event in paused["audit_trail"]]
    assert "finalised" not in paused_events
    assert "classified" in paused_events

    with checkpointer_for(settings) as saver:
        resumed = build_graph(settings, saver, model_factory=scripted_factory)
        final = resumed.invoke(None, config)

    assert final["finalised"] is True
    # Classification survived the restart rather than being recomputed from the top.
    assert final["classification"].sensitivity.value == "internal"
    decided = [event["decided"] for event in final["audit_trail"]]
    assert decided.count("classified") == 1, "the run restarted from the top"


def test_state_persists_across_two_separate_checkpointer_connections(
    tmp_path: Path,
) -> None:
    """Proves the durability is in the database rather than in a live object."""
    settings = sqlite_settings(tmp_path)
    config = {
        "configurable": {"thread_id": "across-connections"},
        "recursion_limit": settings.recursion_limit,
    }

    with checkpointer_for(settings) as saver:
        build_graph(settings, saver, model_factory=scripted_factory).invoke(
            initial_state("A request worth remembering.", "persist-run"), config
        )

    with checkpointer_for(settings) as saver:
        snapshot = build_graph(settings, saver, model_factory=scripted_factory).get_state(config)

    assert snapshot.values["request"] == "A request worth remembering."
    assert snapshot.values["finalised"] is True
