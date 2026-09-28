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
from typing import Any, TypedDict

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import CallClass, Settings, Tier
from agentgate.graph.build import build_checkpointer, build_graph, checkpointer_for, run_config
from agentgate.graph.state import AgentState, initial_state
from agentgate.guardrails.run_ledger import accounted
from agentgate.guardrails.spend import Ceilings, SpendCeilingExceededError, SpendLedger
from agentgate.models.fake import FakeChatModel, scripted_json
from agentgate.models.registry import build_model

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
        **run_config(settings, "config-time-interrupt"),
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
    config = run_config(settings, "compile-time-interrupt")

    with checkpointer_for(settings) as saver:
        graph = build_graph(
            settings, saver, model_factory=scripted_factory, interrupt_before=["finalise"]
        )
        result = graph.invoke(initial_state("A request.", "run"), config)

    decided = [event["decided"] for event in result["audit_trail"]]
    assert "finalised" not in decided
    assert "classified" in decided


# -------------------------------------------- config injection, and a type hint that loses it


def _config_seen_by(node: Any) -> list[object]:
    """Run a one-node graph and report what its ``config`` parameter received."""
    seen: list[object] = []

    def wrapped(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen.append(node(state, **kwargs))
        return {}

    wrapped.__signature__ = __import__("inspect").signature(node)  # type: ignore[attr-defined]
    graph = StateGraph(dict[str, Any])
    graph.add_node("only", wrapped)
    graph.add_edge(START, "only")
    graph.add_edge("only", END)
    graph.compile().invoke({}, {"configurable": {"marker": "present"}})
    return seen


def _optional_union(state: dict[str, Any], config: RunnableConfig | None = None) -> object:
    return None if config is None else config.get("configurable", {}).get("marker")


def _required(state: dict[str, Any], config: RunnableConfig) -> object:
    return config.get("configurable", {}).get("marker")


def test_a_config_typed_as_an_optional_union_is_silently_not_injected() -> None:
    """Leak inventory item 22. Under ``from __future__ import annotations`` -- which every module
    here uses -- ``config: RunnableConfig | None`` is a string LangGraph does not recognise, so it
    passes **no config at all**, and only warns -- with a message recommending that exact spelling.

    Found because the run ledger's nodes fail closed: the ledger travels in the config, a node
    that did not receive it refused to call a model, and the first run stopped. A node reading
    anything optional from its config would have run on without it.
    """
    with pytest.warns(UserWarning, match="config"):
        assert _config_seen_by(_optional_union) == [None]


def test_a_required_config_is_injected() -> None:
    """The control, and the spelling every node here uses."""
    assert _config_seen_by(_required) == ["present"]


# -------------------- create_agent structured output: non-compliance is a paid loop, item 25


class _Proposal(BaseModel):
    tool: str
    arguments: dict[str, Any]


class _DraftWithProposals(BaseModel):
    draft: str
    proposed_actions: list[_Proposal] = Field(default_factory=list)


_PAYLOAD = {
    "draft": "A draft.",
    "proposed_actions": [{"tool": "issue_refund", "arguments": {"account": "1", "amount": 2}}],
}
_LIMIT = 8


def _structured_agent(model: Any) -> Any:
    return create_agent(model, tools=[], system_prompt="x", response_format=_DraftWithProposals)


def _stub_model(stub: StubServer) -> Any:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane="cloud",
        openai_api_key="not-required",
        openai_base_url=stub.base_url,
        cloud_capable_model="capable",
        cloud_cheap_model="cheap",
        model_prices_usd_per_million={
            "capable": {"input": 1.0, "output": 4.0},
            "cheap": {"input": 0.1, "output": 0.4},
        },
    )
    return settings, build_model(settings, Tier.CAPABLE, CallClass.SYNTHESIS)


def test_structured_output_the_fake_cannot_satisfy_loops_to_the_recursion_limit() -> None:
    """Leak inventory item 25. ``create_agent(response_format=...)`` gets structured output by
    having the model call a synthetic response tool. A model that answers with the right JSON as
    text -- which is what the fake is scripted to do -- never calls it, and nothing treats that as
    an error: the agent re-prompts until LangGraph's recursion limit stops it."""
    agent = _structured_agent(FakeChatModel(responses=[scripted_json(_PAYLOAD)]))

    with pytest.raises(GraphRecursionError):
        agent.invoke({"messages": [HumanMessage("hi")]}, {"recursion_limit": _LIMIT})


@pytest.mark.parametrize("native", [True, False])
def test_every_turn_of_that_loop_is_a_billed_request(native: bool) -> None:
    """The same loop against the OpenAI-compatible stub, with and without native structured output
    -- the stub answers correctly either way and never calls the response tool. Each turn is a
    request the provider would bill; only the recursion limit ends it."""
    with running_stub(
        StubBehaviour(reply=_PAYLOAD, supports_native_structured_output=native)
    ) as stub:
        _, model = _stub_model(stub)

        with pytest.raises(GraphRecursionError):
            _structured_agent(model).invoke(
                {"messages": [HumanMessage("hi")]}, {"recursion_limit": _LIMIT}
            )

        assert stub.behaviour.request_count == _LIMIT


def test_the_run_ledger_stops_that_loop_at_the_spend_ceiling_first() -> None:
    """With the model charged to a run ledger, the loop ends at the ceiling -- a refusal naming
    what was spent -- well before the recursion limit, and before the provider sees the rest of
    it. The ledger turning a silent runaway into a visible refusal, a second time."""
    with running_stub(StubBehaviour(reply=_PAYLOAD)) as stub:
        settings, model = _stub_model(stub)
        tight = settings.model_copy(update={"max_total_tokens": 200})
        ledger = SpendLedger(tight, Ceilings.for_run(tight))

        with pytest.raises(SpendCeilingExceededError):
            _structured_agent(accounted(model, ledger)).invoke(
                {"messages": [HumanMessage("hi")]}, {"recursion_limit": 100}
            )

        assert 0 < stub.behaviour.request_count < _LIMIT


# -------------------------------------- update_state against a paused interrupt, item 26


class _Paused(TypedDict, total=False):
    value: str
    verdict: str
    acted: bool


def _paused_graph(ran: list[str]) -> Any:
    """draft -> gate (interrupt) -> act, with a *static* edge out of the gate."""

    def draft(_state: _Paused) -> _Paused:
        return {"value": "drafted"}

    def gate(state: _Paused) -> _Paused:
        verdict = interrupt({"shown": state.get("value")})
        return {"verdict": str(verdict)}

    def act(_state: _Paused) -> _Paused:
        ran.append("act")
        return {"acted": True}

    graph = StateGraph(_Paused)
    graph.add_node("draft", draft)
    graph.add_node("gate", gate)
    graph.add_node("act", act)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "gate")
    graph.add_edge("gate", "act")
    graph.add_edge("act", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_a_state_update_written_as_the_paused_node_walks_past_its_interrupt() -> None:
    """Leak inventory item 26. ``update_state(..., as_node=<the node paused in interrupt()>)``
    is treated as that node having finished: the pause disappears, the resume value is discarded,
    and the static successor runs **without the human's decision** -- no error, no warning."""
    ran: list[str] = []
    graph = _paused_graph(ran)
    config = {"configurable": {"thread_id": "walk-past"}}
    graph.invoke({}, config)
    assert graph.get_state(config).interrupts, "precondition: paused at the gate"

    graph.update_state(config, {"value": "changed"}, as_node="gate")
    result = graph.invoke(Command(resume="approved"), config)

    assert ran == ["act"], "the step past the gate ran"
    assert result.get("verdict") is None, "the human's decision never arrived"


def test_without_the_update_the_same_resume_delivers_the_decision() -> None:
    """The control: the identical graph and resume, with no update, behave as a gate should."""
    ran: list[str] = []
    graph = _paused_graph(ran)
    config = {"configurable": {"thread_id": "control"}}
    graph.invoke({}, config)

    result = graph.invoke(Command(resume="approved"), config)

    assert result["verdict"] == "approved"
    assert ran == ["act"]


@pytest.mark.parametrize(
    ("values", "as_node"),
    [
        ({"proposed_actions": []}, None),
        ({"decision": "approved"}, "approval_gate"),
    ],
)
def test_in_agentgate_an_update_at_the_gate_ends_the_run_and_can_reach_nothing(
    tmp_path: Path, values: dict[str, Any], as_node: str | None
) -> None:
    """Why the project graph survives item 26: the approval gate leaves only by ``Command``, with
    no static edge to ``execute``. So the same update -- including one that writes an approval *as
    the gate* -- leaves no next node at all. The run ends, and the resume after it is silently a
    no-op: nothing executes, nothing is refused, nothing errors. Adding a static edge out of the
    gate would turn this red, which is the point of pinning it."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        outbox_path=tmp_path / "outbox.jsonl",
        corpus_path=Path(__file__).resolve().parents[2] / "corpus",
    )
    graph = build_graph(settings, build_checkpointer(settings))
    config = run_config(settings, "item-26")
    state = initial_state("A request.", "item-26")
    state["sub_questions"] = ["refund escalation"]
    graph.invoke(state, config)
    assert graph.get_state(config).next == ("approval_gate",), "precondition: paused at the gate"

    graph.update_state(config, values, as_node=as_node)
    after = graph.get_state(config)
    result = graph.invoke(Command(resume={"decision": "approved"}), config)

    assert after.next == (), "the update ended the run"
    assert not after.interrupts, "and dropped the pause with it"
    assert not [e for e in result["audit_trail"] if e["decided"] == "executed"]
    assert not (tmp_path / "outbox.jsonl").exists()
