"""The failure modes a ``Send`` fan-out has that a sequential loop does not.

A loop that calls a function five times cannot lose the third call without raising. A
super-step that opens five branches can: one branch fails, the merge succeeds with four
contributions, and the result is a shorter list that looks exactly like a shorter list was
always expected. Nothing throws. Nothing logs. The answer is wrong in a way that reads as
complete.

That is the same shape as a gate that does not gate, and it gets the same treatment as the
parallel-write and crash-resume pairs: a named test per property, each one exercised against a
real compiled graph rather than against the node in isolation, because the properties are
about what the *merge* does.

Three properties, three questions a reviewer would actually ask:

1. If one branch dies, do the others still deliver?
2. Can you tell, afterwards, that it died?
3. Does anything downstream still call the result complete?
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from agentgate.config import Settings
from agentgate.graph.build import build_checkpointer, build_graph
from agentgate.graph.completeness import research_gaps
from agentgate.graph.nodes.researcher import BRANCH, dispatch
from agentgate.graph.state import AgentState, initial_state
from agentgate.models.fake import FakeChatModel, scripted_json
from agentgate.retrieval.index import build_index

pytestmark = pytest.mark.usefixtures("isolated_env")

QUESTIONS = [
    "refund escalation second approver",
    "most common customer complaints",
    "transaction record retention period",
]

CLASSIFICATION = scripted_json(
    {
        "sensitivity": "internal",
        "complexity": "involved",
        "contains_pii": False,
        "reason": "internal analysis",
    }
)


CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def settings_with(**overrides: object) -> Settings:
    # Absolute, because `isolated_env` moves the working directory and a relative corpus path
    # would resolve to somewhere empty.
    base: dict[str, object] = {"lane": "fake", "corpus_path": CORPUS}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def model_factory(*_args: object, **_kwargs: object) -> FakeChatModel:
    return FakeChatModel(responses=[CLASSIFICATION])


class OneBranchFails:
    """A retriever that raises for exactly one question and works for the rest.

    A whole-retriever failure would prove nothing about fan-out: every branch would fail and
    the run would be uniformly broken, which is the easy case. The interesting case is
    asymmetric, because that is the one where the surviving branches make the result look
    healthy.
    """

    def __init__(self, settings: Settings, poisoned: str) -> None:
        self.real = build_index(settings)
        self.poisoned = poisoned
        self.attempted: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.attempted.append(question)
        if question == self.poisoned:
            msg = "corpus shard unavailable"
            raise RuntimeError(msg)
        return [scored.document for scored in self.real.search(question, k=2)]


def run_graph(settings: Settings, questions: list[str], retriever: Any) -> dict[str, Any]:
    """Drive one full run with a supplied retriever, and hand back the final state."""
    graph = build_graph(
        settings,
        build_checkpointer(settings),
        model_factory=model_factory,
        retriever_factory=lambda _settings: retriever,
    )
    state = initial_state("Compare the refund policy against complaints", str(uuid.uuid4()))
    state["sub_questions"] = questions
    return dict(
        graph.invoke(
            state,
            {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": settings.recursion_limit,
            },
        )
    )


def decided(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in state.get("audit_trail", []) if event["decided"] == kind]


# ------------------------------------------------------------------ 1. partial failure


def test_a_failing_branch_does_not_take_the_survivors_with_it() -> None:
    """The survivors' findings still reach the fan-in.

    If the branch raised into the super-step instead of catching, the exception would abort
    the whole step and discard work that had already succeeded -- one branch's bad day costing
    every other branch's output.
    """
    settings = settings_with()
    retriever = OneBranchFails(settings, poisoned=QUESTIONS[1])

    final = run_graph(settings, QUESTIONS, retriever)

    assert len(retriever.attempted) == 3, "every branch should have been opened"
    surviving = {finding.question for finding in final["findings"]}
    assert surviving == {QUESTIONS[0], QUESTIONS[2]}
    assert final["findings"], "the fan-in lost everything, not just the failed branch"


def test_the_failure_is_recorded_in_the_audit_trail_rather_than_swallowed() -> None:
    """Catching a branch failure is only correct if it is also reported.

    A caught exception that leaves no trace is worse than an uncaught one: the run completes,
    the output looks reasonable, and nothing anywhere says a third of the research never
    happened.
    """
    settings = settings_with()
    retriever = OneBranchFails(settings, poisoned=QUESTIONS[1])

    final = run_graph(settings, QUESTIONS, retriever)

    failures = decided(final, "research_failed")
    assert len(failures) == 1
    assert failures[0]["detail"]["question"] == QUESTIONS[1]
    assert "corpus shard unavailable" in failures[0]["detail"]["error"]

    outcomes = {outcome.question: outcome.ok for outcome in final["research_outcomes"]}
    assert outcomes == {QUESTIONS[0]: True, QUESTIONS[1]: False, QUESTIONS[2]: True}


def test_a_partial_answer_is_not_presented_as_a_complete_one() -> None:
    """The property that makes the other two matter.

    Findings arrived, the run finalised, nothing raised. Without this the state a caller reads
    is indistinguishable from a healthy run that happened to find less.
    """
    settings = settings_with()

    partial = run_graph(settings, QUESTIONS, OneBranchFails(settings, poisoned=QUESTIONS[1]))
    whole = run_graph(settings, QUESTIONS, OneBranchFails(settings, poisoned="nothing matches"))

    assert partial["answer_complete"] is False
    assert whole["answer_complete"] is True

    event = decided(partial, "finalised_incomplete")[0]
    assert event["detail"]["research"]["failed"] == 1
    assert event["detail"]["research"]["failed_questions"] == [QUESTIONS[1]]
    assert not decided(whole, "finalised_incomplete")


def test_a_branch_that_reports_nothing_at_all_is_still_counted() -> None:
    """The loss a list of outcomes cannot see.

    Every other check here compares outcomes against each other. This one compares them
    against what was *sent*, which is the only way to notice a branch that produced neither a
    finding nor a failure. ``dispatched`` exists for this and nothing else.
    """
    state: AgentState = {"dispatched": 5, "research_outcomes": []}

    gaps = research_gaps(state)

    assert gaps.silent == 5
    assert gaps.failed == 0, "silent and failed are different losses and are counted apart"
    assert gaps.complete is False


# ------------------------------------------------------------------ 2. bounded width


def test_the_fan_out_is_capped_before_any_branch_runs() -> None:
    """Width is decided going in, not counted coming out.

    The model produced eleven sub-questions. Iterations and tokens would both have found out
    about eleven branches only after paying for them.
    """
    settings = settings_with(max_fan_out=4)
    retriever = OneBranchFails(settings, poisoned="nothing matches")
    questions = [f"question number {index}" for index in range(11)]

    final = run_graph(settings, questions, retriever)

    assert len(retriever.attempted) == 4, "more branches ran than the cap allows"
    assert final["dispatched"] == 4


def test_capping_the_fan_out_is_an_audit_event_not_a_shorter_list() -> None:
    """Truncation that leaves no record is indistinguishable from the model having produced
    four questions in the first place."""
    settings = settings_with(max_fan_out=4)
    questions = [f"question number {index}" for index in range(11)]

    final = run_graph(settings, questions, OneBranchFails(settings, poisoned="nothing matches"))

    capped = decided(final, "fan_out_capped")
    assert len(capped) == 1
    assert capped[0]["detail"]["requested"] == 11
    assert capped[0]["detail"]["dropped"] == 7
    assert capped[0]["detail"]["width_limit"] == 4


def test_the_cap_holds_at_the_dispatch_edge_regardless_of_what_state_says() -> None:
    """Enforced where ``Send`` objects are built, not only where they were explained.

    ``research`` caps the list and records why. This checks the guarantee independently of
    that node ever having run, by handing the edge a state that the node would never have
    produced. A width limit that depends on an earlier node is a convention.
    """
    settings = settings_with(max_fan_out=3)
    state: AgentState = {"sub_questions": [f"q{index}" for index in range(50)]}

    sends = dispatch(state, settings)

    assert len(sends) == 3
    assert all(send.node == BRANCH for send in sends)


def test_blank_questions_never_become_branches() -> None:
    """A branch per empty string is a call per empty string."""
    settings = settings_with(max_fan_out=5)

    sends = dispatch({"sub_questions": ["real question", "", "   ", "another"]}, settings)

    assert len(sends) == 2


# ------------------------------------------------------------------ 3. fan-in convergence


def test_the_fan_in_costs_one_supervisor_turn_no_matter_how_wide() -> None:
    """Every branch returns ``Command(graph=Command.PARENT, goto="supervisor")``. Three
    branches could plausibly mean three supervisor turns, which would make the iteration
    budget a function of fan-out width -- a wide run would exhaust its hand-offs without ever
    having made a second decision.

    Observed to converge to one. Pinned here because it is a property of LangGraph's
    scheduling rather than of anything in this repository, and an upgrade could change it
    without any of our code moving.

    Three turns, and the count is the point only insofar as it is the *same* three: dispatch
    research, decide to draft, finish. None of them is "another branch came back".
    """
    settings = settings_with()
    retriever = OneBranchFails(settings, poisoned="nothing matches")

    narrow = run_graph(settings, QUESTIONS[:1], retriever)
    wide = run_graph(settings, QUESTIONS, retriever)

    assert narrow["iterations"] == wide["iterations"] == 3
