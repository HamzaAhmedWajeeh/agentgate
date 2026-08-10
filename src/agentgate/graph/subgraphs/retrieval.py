"""The retrieval subgraph: one research branch, compiled and used as a node.

A subgraph rather than two nodes in the parent, for a reason that is about blast radius rather
than tidiness. A branch runs on a *sub-question*, not on the request, and it has no business
reading the draft, the classification, or the other branches' findings. Giving it its own state
schema means it structurally cannot: the only things that cross the boundary are the ones named
here.

**The handoff out is a ``Command(graph=Command.PARENT)``.** The branch's last act is to write
its finding into the parent's accumulating channels and return control. Doing it this way
rather than ending the subgraph and letting a parent edge collect the result keeps the delivery
and the routing in the same return value -- the same argument the supervisor makes for using
``Command`` at all. There is no window in which a finding exists in the subgraph and the parent
has not been told where to go next.

**A branch that fails does not take the fan-out down with it.** The retriever is fallible: a
corpus can be missing, an embedding call can time out. That is caught here and turned into a
recorded outcome, because the alternative is one branch's exception aborting a super-step in
which four other branches had already succeeded. What must not happen is the failure becoming
invisible -- a fan-out that quietly loses a branch is the retrieval equivalent of a gate that
does not gate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import Settings
from agentgate.graph.state import Finding, ResearchOutcome
from agentgate.retrieval.index import build_retriever

NODE = "research_branch"
PARENT_RETURN = "supervisor"

RetrieverFactory = Callable[[Settings], Any]


class RetrievalState(TypedDict, total=False):
    """What one branch is allowed to see.

    Deliberately narrow. Everything absent from this schema is something a research branch
    cannot read even by accident, which is the containment argument for making this a subgraph
    at all.
    """

    question: str
    """The sub-question this branch is answering."""

    correlation_id: str
    """Carried through so the branch's audit events join the rest of the run."""

    lane: str
    """Recorded on the branch's events. The branch does not choose it and cannot change it."""

    retrieved: list[dict[str, str]]
    """Chunks found, as plain dicts. State is checkpointed, so this crosses a serialisation
    boundary and carries no object identity."""

    error: str
    """Set when the search failed. Its presence is what routes the branch to a failed outcome
    rather than to an empty one -- "found nothing" and "could not look" are different facts."""


def _memoised(settings: Settings, factory: RetrieverFactory) -> Callable[[], Any]:
    """Build the retriever once, on first use.

    Lazily, so compiling a graph does not read the corpus off disk and embed it. A graph that
    is never routed through research -- every test of the policy gate, for instance -- should
    not pay for an index it will not query.
    """
    cache: dict[str, Any] = {}

    def get() -> Any:
        if "retriever" not in cache:
            cache["retriever"] = factory(settings)
        return cache["retriever"]

    return get


def build_retrieval_subgraph(
    settings: Settings, retriever_factory: RetrieverFactory = build_retriever
) -> Any:
    """Compile the one-branch retrieval subgraph.

    Args:
        settings: Bound into both nodes at build time.
        retriever_factory: How the branch obtains a retriever. Injected rather than imported so
            a test can supply a corpus of its own, or one that raises.
    """
    retriever = _memoised(settings, retriever_factory)

    def search(state: RetrievalState) -> RetrievalState:
        """Look the sub-question up in the corpus.

        Failure is caught and recorded rather than raised. An exception here would abort the
        whole super-step, discarding the findings of every sibling branch that had already
        succeeded -- turning one branch's bad day into a total loss.
        """
        question = state.get("question", "")
        try:
            documents = retriever().invoke(question)
        except Exception as error:  # any retrieval failure means the same thing to the branch
            return {"error": f"{type(error).__name__}: {str(error)[:200]}"}

        return {
            "retrieved": [
                {
                    "source": str(document.metadata.get("source", "")),
                    "heading": str(document.metadata.get("heading", "")),
                    "content": document.page_content,
                }
                for document in documents
            ]
        }

    def deliver(state: RetrievalState) -> Command[Any]:
        """Hand the branch's result up to the parent and return control.

        The one ``Command(graph=Command.PARENT)`` in the system. The update lands in the
        parent's channels, where ``operator.add`` concatenates it with whatever the sibling
        branches produced -- which is the fan-in, and it is a property of the parent's state
        schema rather than of any code written here.
        """
        question = state.get("question", "")
        correlation_id = state.get("correlation_id", "")
        lane = state.get("lane")
        error = state.get("error")
        retrieved = state.get("retrieved", [])

        if error:
            return Command(
                graph=Command.PARENT,
                goto=PARENT_RETURN,
                update={
                    "research_outcomes": [
                        ResearchOutcome(question=question, ok=False, detail=error)
                    ],
                    "audit_trail": [
                        audit_event(
                            node=NODE,
                            decided=Decided.RESEARCH_FAILED,
                            correlation_id=correlation_id,
                            input_digest=digest(question),
                            lane=lane,
                            detail={"question": question, "error": error},
                        )
                    ],
                },
            )

        findings = [
            Finding(
                question=question,
                content=chunk["content"],
                source=f"{chunk['source']}#{chunk['heading']}",
            )
            for chunk in retrieved
        ]

        # An empty corpus hit is a success with nothing in it, not a failure. The branch did
        # its job; the corpus had no answer. Recorded as ok=True with the count, so a reviewer
        # can see a run that succeeded at finding nothing.
        return Command(
            graph=Command.PARENT,
            goto=PARENT_RETURN,
            update={
                "findings": findings,
                "research_outcomes": [
                    ResearchOutcome(
                        question=question, ok=True, detail=f"{len(findings)} chunks retrieved"
                    )
                ],
                "audit_trail": [
                    audit_event(
                        node=NODE,
                        decided=Decided.RESEARCHED,
                        correlation_id=correlation_id,
                        input_digest=digest(question),
                        lane=lane,
                        detail={
                            "question": question,
                            "chunks": len(findings),
                            "sources": sorted({finding.source for finding in findings}),
                        },
                    )
                ],
            },
        )

    graph = StateGraph(RetrievalState)
    graph.add_node("search", search)
    graph.add_node("deliver", deliver)
    graph.add_edge(START, "search")
    graph.add_edge("search", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()
