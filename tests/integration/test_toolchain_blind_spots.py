"""Places where a check reports success without having checked.

This build keeps hitting the same shape of failure: a tool says yes, the runtime says no, and
the gap between them is invisible until something depends on it. A comment describing such a
gap is worth very little -- comments get deleted, and nothing goes red when they are wrong.

So each one gets pinned here. If a future version of mypy, LangGraph, or this codebase closes
one of these gaps, the corresponding test fails and someone has to notice. That failure is a
good outcome: it means a blind spot stopped being blind.

Recorded in the leak inventory in docs/adr/0004-provider-abstraction-and-lanes.md.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from agentgate.config import Settings
from agentgate.graph.build import build_graph, checkpointer_for
from agentgate.graph.state import AgentState, initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSIFICATION = scripted_json(
    {
        "sensitivity": "internal",
        "complexity": "simple",
        "contains_pii": False,
        "reason": "ordinary",
    }
)


def scripted_factory(*_args: object, **_kwargs: object) -> FakeChatModel:
    return FakeChatModel(responses=[CLASSIFICATION])


def run_mypy(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Type check a snippet with the project's strict settings.

    A real mypy invocation rather than a reasoned argument about what mypy would say. The
    entire point of these tests is that reasoning about what a checker accepts is exactly
    where the mistake was made in the first place.
    """
    module = tmp_path / "snippet.py"
    module.write_text(textwrap.dedent(source), encoding="utf-8")
    return subprocess.run(  # noqa: S603 - the input is a literal defined in this module
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
            str(module),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )


# ------------------------------------------------------------------ functools.partial erasure

# The two snippets differ in exactly one thing: whether the node is wrapped in `partial`.
# Neither annotates the graph as Any -- doing so erases `add_node` itself and makes both
# snippets pass, which is a way of proving nothing at all. (It did, on the first attempt.)

WRONG_NODE = """
    from functools import partial

    from langgraph.graph import StateGraph

    from agentgate.graph.state import AgentState


    def wrongly_shaped(state: AgentState, required: int) -> AgentState:
        '''Takes a second required argument that nothing will ever supply.'''
        return AgentState()


    graph = StateGraph(AgentState)
    graph.add_node("wrong", partial(wrongly_shaped))
"""

WRONG_NODE_WITHOUT_PARTIAL = """
    from langgraph.graph import StateGraph

    from agentgate.graph.state import AgentState


    def wrongly_shaped(state: AgentState, required: int) -> AgentState:
        return AgentState()


    graph = StateGraph(AgentState)
    graph.add_node("wrong", wrongly_shaped)
"""


def test_partial_hides_a_wrongly_shaped_node_from_mypy(tmp_path: Path) -> None:
    """The blind spot, demonstrated against a real mypy run.

    `functools.partial` types as `partial[T]`, whose parameter list is effectively `...`. That
    matches anything, so wrapping a node in `partial` silences the signature check entirely --
    including for a node that could never be called successfully.

    If this ever starts failing, mypy has got better and the workaround in build.py can go.
    """
    result = run_mypy(WRONG_NODE, tmp_path)

    assert result.returncode == 0, (
        "mypy rejected the partial-wrapped bad node, which would be an improvement. "
        f"Remove the closure workaround in build.py.\n{result.stdout}{result.stderr}"
    )


def test_without_partial_mypy_catches_the_same_node(tmp_path: Path) -> None:
    """The control. Proves the erasure is what hides it, not something about the node."""
    result = run_mypy(WRONG_NODE_WITHOUT_PARTIAL, tmp_path)

    assert result.returncode != 0
    assert "add_node" in result.stdout


def test_the_wrongly_shaped_node_really_does_fail_at_runtime() -> None:
    """The other half: what the type check let through does not work.

    Without this, "mypy accepts it" would be a curiosity. With it, mypy accepting it is a
    hole, because the runtime is unambiguous about the node being wrong.
    """

    def wrongly_shaped(state: AgentState, required: int) -> AgentState:
        return AgentState()

    graph: Any = StateGraph(AgentState)
    graph.add_node("wrong", wrongly_shaped)
    graph.add_edge(START, "wrong")
    graph.add_edge("wrong", END)

    with pytest.raises(TypeError, match="required"):
        graph.compile().invoke({})


def test_the_projects_own_nodes_are_checked_rather_than_erased(tmp_path: Path) -> None:
    """The workaround holds: build.py's closure keeps the signature check meaningful.

    A node with the wrong shape, passed the way build.py passes lane nodes, must be rejected.
    """
    source = """
        from typing import Any

        from langgraph.graph import StateGraph

        from agentgate.graph.build import GraphNode
        from agentgate.graph.state import AgentState


        def wrongly_shaped(state: AgentState, required: int) -> AgentState:
            return AgentState()


        node: GraphNode = wrongly_shaped
    """
    result = run_mypy(source, tmp_path)

    assert result.returncode != 0, "GraphNode accepted a node it should have rejected"


# ------------------------------------------------------------------ interrupt_before placement


def test_interrupt_before_in_the_invoke_config_is_silently_ignored(tmp_path: Path) -> None:
    """A safety-relevant setting that does nothing and says nothing.

    This is worse than an error. Asking a graph to pause before a node and having it run
    straight through, with no warning, is the failure mode Phase 5's approval gate cannot
    afford -- a gate that does not gate looks exactly like a gate that does.
    """
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, checkpointer="sqlite", sqlite_path=tmp_path / "ignored.db"
    )
    config = {
        "configurable": {"thread_id": "config-time-interrupt"},
        "recursion_limit": settings.recursion_limit,
        # Looks like it should pause. Does not.
        "interrupt_before": ["finalise"],
    }

    with checkpointer_for(settings) as saver:
        graph = build_graph(settings, saver, model_factory=scripted_factory)
        result = graph.invoke(initial_state("A request.", "run"), config)

    decided = [event["decided"] for event in result["audit_trail"]]
    assert "finalised" in decided, (
        "the invoke-config form of interrupt_before now works. That is an improvement -- "
        "update build.py's docstring and this test."
    )
    assert result["finalised"] is True


def test_interrupt_before_at_compile_time_actually_pauses(tmp_path: Path) -> None:
    """The form that works, asserted next to the form that does not."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, checkpointer="sqlite", sqlite_path=tmp_path / "honoured.db"
    )
    config = {
        "configurable": {"thread_id": "compile-time-interrupt"},
        "recursion_limit": settings.recursion_limit,
    }

    with checkpointer_for(settings) as saver:
        graph = build_graph(
            settings, saver, model_factory=scripted_factory, interrupt_before=["finalise"]
        )
        result = graph.invoke(initial_state("A request.", "run"), config)

    decided = [event["decided"] for event in result["audit_trail"]]
    assert "finalised" not in decided
    assert "classified" in decided
