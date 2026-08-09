"""Why reducers exist, demonstrated rather than asserted.

Both halves of the same situation, side by side, against real compiled graphs:

1. Two nodes writing the same **plain** field in one super-step raises ``InvalidUpdateError``.
2. The same two nodes writing a **reducer-annotated** field merge cleanly.

The reducer is not a style preference and not a way to quiet an error. LangGraph runs every
node scheduled in a super-step concurrently and then merges their partial updates. For a plain
channel there is no merge rule, so a tie has no resolution and LangGraph refuses rather than
picking a winner. Refusing is the right behaviour: silently keeping one of two answers is data
loss that surfaces much later as a mysteriously incomplete result.

Annotating a channel with a reducer answers the question the error is asking -- *what does
merging mean here* -- and that answer is a design decision about the channel, not a workaround.
Getting it wrong in either direction is a correctness bug:

- A reducer on a single-writer field hides a double-write that should have been caught.
- No reducer on a fan-out field turns a legitimate concurrent merge into a crash.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

from agentgate.graph.state import AgentState, Finding


class PlainFieldState(TypedDict, total=False):
    """One channel, no reducer. Last-write-wins, which cannot resolve a tie."""

    verdict: str


class ReducedFieldState(TypedDict, total=False):
    """The same channel, told how to merge."""

    verdict: Annotated[list[str], operator.add]


def fan_out_graph(state_type: type, left: Any, right: Any) -> Any:
    """Two nodes scheduled in the same super-step, then joined.

    Both branch from START, so LangGraph runs them concurrently and merges their updates at
    the end of the step. This is the shape produced by `Send` fan-out and by any diamond in
    the graph -- it is not contrived.
    """
    graph = StateGraph(state_type)
    graph.add_node("left", left)
    graph.add_node("right", right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    return graph.compile()


# --------------------------------------------------------------- 1. the failure


def test_two_nodes_writing_a_plain_field_in_one_super_step_raises() -> None:
    """The error that sends people looking for a workaround.

    It is not a bug in the graph and not a race to be retried. LangGraph is saying it has no
    rule for merging two values into one channel, and it is right to refuse: the alternative
    is discarding one branch's work with no record that it happened.
    """
    graph = fan_out_graph(
        PlainFieldState,
        lambda _state: {"verdict": "from the left branch"},
        lambda _state: {"verdict": "from the right branch"},
    )

    with pytest.raises(InvalidUpdateError) as caught:
        graph.invoke({})

    message = str(caught.value)
    assert "verdict" in message
    assert "one value per step" in message.lower()


def test_the_same_nodes_are_fine_when_they_do_not_collide() -> None:
    """Proves the failure is about concurrency, not about the field.

    Writing the same plain channel from two nodes in *different* super-steps is ordinary
    last-write-wins and entirely legal. Only the tie is a problem.
    """
    graph = StateGraph(PlainFieldState)
    graph.add_node("first", lambda _state: {"verdict": "written first"})
    graph.add_node("second", lambda _state: {"verdict": "written second"})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")  # sequential, so two separate super-steps
    graph.add_edge("second", END)

    result = graph.compile().invoke({})

    assert result["verdict"] == "written second"


# --------------------------------------------------------------- 2. the fix


def test_a_reducer_merges_what_last_write_wins_could_not() -> None:
    """The identical topology, with the channel told how to merge. Both branches survive."""
    graph = fan_out_graph(
        ReducedFieldState,
        lambda _state: {"verdict": ["from the left branch"]},
        lambda _state: {"verdict": ["from the right branch"]},
    )

    result = graph.invoke({"verdict": []})

    assert sorted(result["verdict"]) == ["from the left branch", "from the right branch"]


def test_the_reducer_keeps_every_branch_at_any_width() -> None:
    """Research fans out over however many sub-questions there are, not over exactly two."""

    def branch(index: int) -> Any:
        """Bind the index now; a bare closure over the loop variable would capture the last."""

        def contribute(_state: ReducedFieldState) -> dict[str, list[str]]:
            return {"verdict": [f"finding {index}"]}

        return contribute

    graph = StateGraph(ReducedFieldState)
    width = 5
    for index in range(width):
        name = f"branch_{index}"
        graph.add_node(name, branch(index))
        graph.add_edge(START, name)
        graph.add_edge(name, END)

    result = graph.compile().invoke({"verdict": []})

    assert len(result["verdict"]) == width


# --------------------------------------------------------------- the real state


def test_the_project_state_reduces_the_channels_that_fan_out() -> None:
    """The two demonstrations above, applied to the state this system actually uses.

    `findings` and `audit_trail` accumulate from parallel branches, so both carry reducers.
    """
    graph = StateGraph(AgentState)
    graph.add_node(
        "researcher_a",
        lambda _state: {"findings": [Finding(question="a", content="alpha")]},
    )
    graph.add_node(
        "researcher_b",
        lambda _state: {"findings": [Finding(question="b", content="beta")]},
    )
    graph.add_edge(START, "researcher_a")
    graph.add_edge(START, "researcher_b")
    graph.add_edge("researcher_a", END)
    graph.add_edge("researcher_b", END)

    result = graph.compile().invoke({"findings": []})

    assert {finding.question for finding in result["findings"]} == {"a", "b"}


def test_the_project_state_refuses_concurrent_writes_to_a_single_writer_field() -> None:
    """`draft` has no reducer because exactly one node writes it.

    If a future change makes two nodes write it concurrently, this fails loudly rather than
    losing one of the drafts. That is the protection the missing reducer buys, and the reason
    not to add one reflexively.
    """
    graph = StateGraph(AgentState)
    graph.add_node("drafter_a", lambda _state: {"draft": "one draft"})
    graph.add_node("drafter_b", lambda _state: {"draft": "another draft"})
    graph.add_edge(START, "drafter_a")
    graph.add_edge(START, "drafter_b")
    graph.add_edge("drafter_a", END)
    graph.add_edge("drafter_b", END)

    with pytest.raises(InvalidUpdateError, match="draft"):
        graph.compile().invoke({})
