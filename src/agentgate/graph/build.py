"""Graph assembly and checkpointer selection.

The topology is declared once, here, and reading this file tells you every path a request can
take. Nodes are bound to settings at build time with :func:`functools.partial` rather than
reaching for a global, so a test can compile a graph against different configuration without
mutating process state -- which matters because settings are frozen and cached on purpose.

**The checkpointer is chosen by configuration, never by editing code.** Which one you get is a
deployment property: in-memory for tests, SQLite for a local single-process run, Postgres for
the Compose stack. Every one of them persists the same state, so a graph that works under one
works under all three, and the crash-resume behaviour tested against SQLite is the behaviour
Postgres gives you.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentgate.config import CheckpointerBackend, Settings
from agentgate.decider.build import build_decider
from agentgate.effects.sink import EffectSink, build_effect_sink
from agentgate.errors import AgentgateError
from agentgate.graph.nodes.approval import approval_gate
from agentgate.graph.nodes.assess import DeciderFactory, assess
from agentgate.graph.nodes.classify import classify
from agentgate.graph.nodes.drafter import draft
from agentgate.graph.nodes.execute import execute
from agentgate.graph.nodes.finalise import finalise
from agentgate.graph.nodes.lanes import LANE_NODES, LaneNode
from agentgate.graph.nodes.researcher import dispatch, research
from agentgate.graph.nodes.supervisor import supervise
from agentgate.graph.routing import route_by_budget, route_by_policy
from agentgate.graph.state import AgentState
from agentgate.graph.subgraphs.retrieval import (
    NODE as RESEARCH_BRANCH,
)
from agentgate.graph.subgraphs.retrieval import (
    RetrieverFactory,
    build_retrieval_subgraph,
)
from agentgate.guardrails.run_ledger import RUN_LEDGER
from agentgate.guardrails.spend import Ceilings, SpendLedger
from agentgate.models.registry import ModelFactory, build_resilient_model
from agentgate.retrieval.index import build_retriever


class GraphNode(Protocol):
    """A node as LangGraph types it.

    LangGraph's internal node type is a Protocol whose ``__call__`` names its parameter
    ``state``. That naming is load-bearing: a bare ``Callable[[AgentState], AgentState]`` is
    positional-only as far as the type system is concerned, so it does not satisfy the
    protocol, and a function whose parameter is named anything else -- ``_state``, say -- does
    not either. Both failures surface as an unhelpful "no overload variant matches".
    """

    def __call__(self, state: AgentState) -> AgentState: ...


class CheckpointerUnavailableError(AgentgateError):
    """The configured checkpointer cannot be constructed from this configuration."""


def build_graph(  # noqa: PLR0913 - each factory is an injection point a test needs
    settings: Settings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    model_factory: ModelFactory = build_resilient_model,
    interrupt_before: Sequence[str] | None = None,
    retriever_factory: RetrieverFactory = build_retriever,
    effect_sink_factory: Callable[[Settings], EffectSink] = build_effect_sink,
    decider_factory: DeciderFactory = build_decider,
) -> Any:
    """Assemble and compile the graph.

    Args:
        settings: Bound into every node at build time.
        checkpointer: Supplied explicitly by tests and by the resume path. When omitted the
            graph compiles without persistence, which is only correct for a single
            fire-and-forget invocation.
        model_factory: How nodes obtain models. Injected rather than imported so a test can
            script replies without patching a global.
        retriever_factory: How a research branch obtains a retriever. Injected for the same
            reason, and called lazily -- compiling a graph does not read the corpus.
        decider_factory: How the assess node obtains the decider. Injected so a test can
            script a verdict; in the running system it is ``build_decider``.
        effect_sink_factory: Where ``execute`` records approved effects. Injected so a test can
            crash a sink after it writes; in the running system it is the outbox, the only sink.
        interrupt_before: Nodes to pause before. A compile-time option in LangGraph 1.x --
            passing it in the invoke config is silently ignored, which makes a paused-run
            test quietly assert nothing. Used for debugging and by the resume tests; the
            approval gate in Phase 5 uses ``interrupt()`` instead, which pauses from inside a
            node rather than from the graph definition.
    """
    graph = StateGraph(AgentState)

    def bound(node: LaneNode) -> GraphNode:
        """Close a lane node over settings.

        An explicit closure rather than ``partial``. ``partial`` does type-check here, but
        only because its type erases the parameter list entirely -- it would accept a node
        with the wrong signature just as happily. The closure keeps the check meaningful.
        """

        def run(state: AgentState) -> AgentState:
            return node(state, settings=settings)

        return run

    graph.add_node("classify", partial(classify, settings=settings, model_factory=model_factory))
    for name, node in LANE_NODES.items():
        graph.add_node(name, bound(node))
    graph.add_node("supervisor", partial(supervise, settings=settings))
    graph.add_node("researcher", partial(research, settings=settings))
    graph.add_node("drafter", partial(draft, settings=settings, model_factory=model_factory))
    graph.add_node("assess", partial(assess, settings=settings, decider_factory=decider_factory))
    graph.add_node("approval_gate", partial(approval_gate, settings=settings))
    graph.add_node(
        "execute",
        partial(execute, settings=settings, effect_sink_factory=effect_sink_factory),
    )
    graph.add_node(RESEARCH_BRANCH, build_retrieval_subgraph(settings, retriever_factory))
    graph.add_node("budget_guard", _budget_guard)
    graph.add_node("finalise", partial(finalise, settings=settings))

    graph.add_edge(START, "classify")

    # The policy gate. Three destinations, and the mapping is the identity because the
    # router's Literal and the node names are deliberately the same strings -- a divergence
    # would compile fine and dead-end at runtime.
    graph.add_conditional_edges(
        "classify",
        route_by_policy,
        {name: name for name in LANE_NODES},
    )

    for name in LANE_NODES:
        graph.add_edge(name, "supervisor")

    # The supervisor reaches budget_guard and researcher by returning Command(goto=...), so no
    # static edge is declared for either. The destinations are still declared to the compiler
    # by the nodes existing.

    # The fan-out. A conditional edge rather than a node return, because this is the one place
    # the width has to be guaranteed: `dispatch` caps here, at the point Send objects are
    # constructed, so the number of branches is bounded before any of them is scheduled. The
    # researcher node applied the same cap a step earlier and recorded why the list shrank --
    # this is the enforcement, that was the explanation.
    graph.add_conditional_edges(
        "researcher", partial(dispatch, settings=settings), [RESEARCH_BRANCH]
    )

    # No edge leaves RESEARCH_BRANCH. Every branch exits by Command(graph=Command.PARENT,
    # goto="supervisor") from inside the subgraph, which is the handoff, and declaring a static
    # edge as well would describe a second path that never runs.

    graph.add_edge("drafter", "supervisor")

    # The decider assesses once, then the gate reads the stored verdict. A static edge is right
    # here and would be wrong out of the gate: assess does not pause, and nothing it leads to
    # acts -- the gate still stands between it and `execute` (leak inventory item 26).
    graph.add_edge("assess", "approval_gate")

    # The approval gate leaves by Command(goto=...) -- "execute" on approval, "drafter" on
    # rejection. The rejection edge is the revision loop, and it is the only loop in the graph
    # that a person can drive indefinitely, which is what finally gives the iteration cap
    # something to cap.
    #
    # `execute` has exactly one edge in, from the approved branch. It checks the decision on
    # state as well: the topology is true until someone draws another edge, and the node's own
    # check is true regardless of what the topology looks like.
    graph.add_edge("execute", "supervisor")

    graph.add_conditional_edges(
        "budget_guard",
        partial(route_by_budget, settings=settings),
        {"continue": "supervisor", "finalise": "finalise"},
    )
    graph.add_edge("finalise", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before) if interrupt_before else None,
    )


def _budget_guard(state: AgentState) -> AgentState:  # noqa: ARG001 - name is part of the protocol
    """A pass-through node whose only job is to be the place the budget edge hangs off.

    A conditional edge cannot modify state, and the guard needs somewhere to sit in the
    topology that a ``Command`` can target. Keeping it empty means the decision stays entirely
    in ``route_by_budget``, where it can be read and tested as a pure function.
    """
    return AgentState()


def run_config(
    settings: Settings,
    thread_id: str,
    *,
    spent: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The config one run is invoked with, carrying that run's spend ledger.

    **Every run starts here**, because the model-calling nodes refuse to run without the ledger
    it carries. One ledger per run: a new one per invocation, starting from ``spent`` when the run
    has spent already -- which is what :func:`resume_config` passes. The ledger is never
    checkpointed (only primitives in ``configurable`` reach checkpoint metadata), so the only way
    its totals outlive a process is the ``spend`` channel the supervisor writes.
    """
    ledger = SpendLedger.resumed(settings, Ceilings.for_run(settings), spent)
    return {
        "configurable": {"thread_id": thread_id, RUN_LEDGER: ledger},
        "recursion_limit": settings.recursion_limit,
    }


def resume_config(graph: Any, settings: Settings, thread_id: str) -> dict[str, Any]:
    """The config for picking a paused run back up, its ledger starting from what it had spent.

    Read off the checkpoint, not carried in memory: the process that resumes a run is usually not
    the one that started it, and a ledger that restarted from zero on every approval would let a
    reviewer who kept rejecting drive spend past a ceiling that never saw it.
    """
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    spent = (snapshot.values or {}).get("spend") if snapshot else None
    return run_config(settings, thread_id, spent=spent)


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """Construct the checkpointer this configuration asks for.

    Only the in-memory saver is returned directly; the others hold connections and must be
    used through :func:`checkpointer_for`, which closes them.

    Raises:
        CheckpointerUnavailableError: for a backend that needs a context manager.
    """
    if settings.checkpointer is CheckpointerBackend.MEMORY:
        return InMemorySaver()
    msg = (
        f"the '{settings.checkpointer.value}' checkpointer holds a connection and must be "
        "used via checkpointer_for(), which closes it"
    )
    raise CheckpointerUnavailableError(msg)


@contextlib.contextmanager
def checkpointer_for(settings: Settings) -> Iterator[BaseCheckpointSaver[Any]]:
    """Yield the configured checkpointer, closing it afterwards.

    Imports are local to each branch so a deployment that uses SQLite does not need the
    Postgres driver importable, and vice versa. The alternative is a module that fails to
    import on a machine missing a driver it was never going to use.
    """
    match settings.checkpointer:
        case CheckpointerBackend.MEMORY:
            yield InMemorySaver()

        case CheckpointerBackend.SQLITE:
            import sqlite3  # noqa: PLC0415 - backend-specific, see docstring

            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

            settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False because LangGraph may touch the connection from a
            # worker thread; the saver serialises its own access.
            connection = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
            try:
                sqlite_saver = SqliteSaver(connection)
                sqlite_saver.setup()
                yield sqlite_saver
            finally:
                connection.close()

        case CheckpointerBackend.POSTGRES:
            from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415

            if settings.postgres_dsn is None:  # pragma: no cover - validation prevents this
                msg = "postgres checkpointer selected without a DSN"
                raise CheckpointerUnavailableError(msg)
            dsn = settings.postgres_dsn.get_secret_value()
            with PostgresSaver.from_conn_string(dsn) as postgres_saver:
                postgres_saver.setup()
                yield postgres_saver


def compiled_graph_for(
    settings: Settings, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Typed convenience wrapper for callers that want the compiled type."""
    return build_graph(settings, checkpointer)  # type: ignore[no-any-return]
