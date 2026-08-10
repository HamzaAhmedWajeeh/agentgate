"""The human gate, and the loop it creates.

Three properties, and the first one is the reason the node is shaped the way it is.

**Resume re-executes the node from its top.** Not "may" -- does. Everything above the
``interrupt()`` runs again on every resume, so a side effect placed above it happens once per
resume. That is proven here by observation rather than asserted from the docs: a counter
incremented above the pause is watched going up.

**Rejection is a loop a person can drive.** Reject, revise, return to the gate, reject again.
It is the only loop in this graph that does not terminate on its own, which makes it the first
thing the iteration cap has ever had to cap.

**The cap actually stops it.** A reviewer who never approves must not be able to run forever,
and the run has to end on the budget rather than on the reviewer giving up. Exercised end to
end against the real loop, because an unexercised guard is the same category as everything in
the leak inventory.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentgate.config import CallClass, Settings
from agentgate.graph.build import build_checkpointer, build_graph
from agentgate.graph.nodes.approval import review_packet
from agentgate.graph.nodes.execute import UnapprovedExecutionError, execute
from agentgate.graph.state import Decision, initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

VERDICT = scripted_json(
    {
        "sensitivity": "internal",
        "complexity": "involved",
        "contains_pii": False,
        "reason": "internal analysis",
    }
)


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {"lane": "fake", "corpus_path": CORPUS}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def model_factory(_settings: Settings, _tier: object, call_class: CallClass, **_k: object) -> Any:
    if call_class is CallClass.SYNTHESIS:
        return FakeChatModel(responses=["A draft of the response."])
    return FakeChatModel(responses=[VERDICT])


class Run:
    """One thread, driven to the gate and resumed as many times as a test wants."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph = build_graph(
            settings, build_checkpointer(settings), model_factory=model_factory
        )
        self.config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": settings.recursion_limit,
        }

    def start(self, questions: list[str] | None = None) -> dict[str, Any]:
        state = initial_state("Draft a refund response.", str(uuid.uuid4()))
        state["sub_questions"] = questions if questions is not None else ["refund escalation"]
        return dict(self.graph.invoke(state, self.config))

    def resume(self, **verdict: Any) -> dict[str, Any]:
        return dict(self.graph.invoke(Command(resume=verdict), self.config))


def paused(result: dict[str, Any]) -> bool:
    return "__interrupt__" in result


def decided(result: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in result.get("audit_trail", []) if event["decided"] == kind]


# ------------------------------------------------------------------ 1. resume semantics


def test_the_interrupted_node_re_executes_from_its_top_on_resume() -> None:
    """Observed, not quoted. This is the fact the node's whole shape depends on.

    A bare counter above an ``interrupt()`` goes up once per resume. If this ever stops being
    true the rule about side effects becomes unnecessary; while it is true, any effect placed
    above the pause happens once per resume, and a run resumed three times gets three of it.
    """
    above: list[int] = []
    below: list[int] = []

    def gate(state: dict[str, Any]) -> dict[str, Any]:
        above.append(1)
        verdict = interrupt({"question": "again?"})
        below.append(1)
        return {"verdict": verdict}

    graph = StateGraph(dict[str, Any])
    graph.add_node("gate", gate)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    compiled.invoke({}, config)
    compiled.invoke(Command(resume="one"), config)

    assert above == [1, 1], "the code above the interrupt did not re-run; the rule is now wrong"
    assert below == [1], "the code below the interrupt ran more than once"


def test_nothing_above_the_pause_writes_an_audit_event() -> None:
    """The rule applied to this node: the review packet is pure, so recomputing it costs
    nothing. Approving after two rejections must not produce three approval events."""
    run = Run(settings_with())

    run.start()
    run.resume(decision="rejected", feedback="too short")
    run.resume(decision="rejected", feedback="still too short")
    final = run.resume(decision="approved")

    assert len(decided(final, "approved")) == 1
    assert len(decided(final, "rejected")) == 2
    assert len(decided(final, "executed")) == 1


def test_the_packet_is_a_pure_function_of_state() -> None:
    """Recomputed on every resume, so it had better not depend on anything else."""
    state = initial_state("A request.", "corr")
    state["draft"] = "text"

    assert review_packet(state) == review_packet(state)


def test_the_reviewer_is_told_when_the_evidence_is_partial() -> None:
    """Approving a deliverable without being told a third of its research never arrived is not
    informed approval, and an uninformed gate is a rubber stamp with extra steps.

    The first version of this test set ``answer_complete=False`` on the state and asserted the
    packet carried it through. It passed, and it was worthless: ``answer_complete`` is written
    by ``finalise``, which runs on the far side of this gate, so on a real run the field is
    absent here and the packet read the default -- ``True`` -- on exactly the runs where it is
    false. The test checked that a field is copied. The claim is that a human is correctly
    informed, and only a state that a real fan-out could produce can check it.
    """
    state = initial_state("A request.", "corr")
    state["dispatched"] = 3
    state["research_outcomes"] = []  # three branches sent, none reported

    packet = review_packet(state)

    assert packet["answer_complete"] is False
    assert packet["research"]["silent"] == 3
    assert "answer_complete" not in state, (
        "if finalise now writes this before the gate, the packet may read it directly -- but "
        "check the ordering before relying on it, because that is how this broke the first time"
    )


# ------------------------------------------------------------------ 2. the revision loop


def test_a_run_pauses_at_the_gate_before_anything_irreversible() -> None:
    run = Run(settings_with())

    first = run.start()

    assert paused(first)
    assert not decided(first, "executed")
    assert first["__interrupt__"][0].value["draft"]


def test_rejection_sends_the_draft_back_and_pauses_again() -> None:
    run = Run(settings_with())
    run.start()

    second = run.resume(decision="rejected", feedback="cite the retention schedule")

    assert paused(second)
    assert second["revisions"] == 1
    assert second["feedback"] == "cite the retention schedule"
    assert len(decided(second, "drafted")) == 2, "the rejection did not produce a new draft"


def test_approval_reaches_execute_and_rejection_never_does() -> None:
    approved = Run(settings_with())
    approved.start()
    final = approved.resume(decision="approved")

    rejected = Run(settings_with())
    rejected.start()
    still_going = rejected.resume(decision="rejected", feedback="no")

    assert final["decision"] is Decision.APPROVED
    assert decided(final, "executed")
    assert not decided(still_going, "executed")


def test_an_unrecognised_verdict_is_treated_as_rejection() -> None:
    """Fails closed. Misreading a rejection as approval costs an irreversible action nobody
    sanctioned; misreading the other way costs one more revision."""
    run = Run(settings_with())
    run.start()

    result = run.resume(decision="looks fine to me")

    assert paused(result), "an unparseable verdict let the run through the gate"
    assert not decided(result, "executed")
    assert result["revisions"] == 1


def test_execute_refuses_an_unapproved_state_whatever_the_topology_says() -> None:
    """The node's own check, independent of the edges. The topology is true until somebody
    draws another edge; this is true regardless."""
    state = initial_state("A request.", "corr")
    state["decision"] = Decision.REJECTED

    with pytest.raises(UnapprovedExecutionError, match="Refusing"):
        execute(state, settings_with())


# ------------------------------------------------------------------ 3. the cap, exercised


def test_a_reviewer_who_never_approves_is_stopped_by_the_iteration_cap() -> None:
    """The guard, finally doing the job it was written for.

    Before Phase 5 nothing in this graph looped, so the iteration cap had nothing to cap and
    was not verified end to end. This is the loop: a person rejecting forever. The run has to
    terminate on the budget rather than on the reviewer getting bored, and it has to say that
    is why it stopped.
    """
    settings = settings_with(max_iterations=6, recursion_limit=40)
    run = Run(settings)
    run.start()

    result: dict[str, Any] = {}
    for _ in range(20):
        result = run.resume(decision="rejected", feedback="again")
        if not paused(result):
            break
    else:  # pragma: no cover - only reached if the cap never fires
        pytest.fail("twenty rejections and the run was still going; the cap did not hold")

    assert result["finalised"] is True
    assert result["iterations"] >= settings.max_iterations
    assert decided(result, "budget_exceeded"), "it stopped, but not because of the budget"
    assert result["audit_trail"][-1]["detail"]["stopped_because"] == "budget_exhausted"


def test_the_cap_scales_with_the_budget_rather_than_being_a_fixed_number() -> None:
    """Two budgets, two lengths. A run that stopped after the same number of rejections under
    both would be stopping on something else."""

    def rejections_until_stopped(max_iterations: int) -> int:
        run = Run(settings_with(max_iterations=max_iterations, recursion_limit=60))
        run.start()
        for attempt in range(1, 40):
            if not paused(run.resume(decision="rejected", feedback="again")):
                return attempt
        raise AssertionError("never stopped")  # pragma: no cover

    assert rejections_until_stopped(10) > rejections_until_stopped(5)
