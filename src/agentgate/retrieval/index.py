"""Index construction and retriever wiring.

Dense, in-process, by default. The reasoning is recorded in ADR 0010; the short form is that a
service dependency in the default path costs every reader a Docker command before they can run
the tests, and this corpus is small enough that an in-memory index is not a compromise.

**The similarity search is written here rather than imported.** LangChain's own
`InMemoryVectorStore` needs numpy, which is not a dependency of this project and would be
added for a dot product over a few dozen short vectors. Cosine similarity on L2-normalised
vectors *is* the dot product; that is six lines, and six lines is cheaper than a dependency in
the path of every retrieval, in the image, and in the audit surface. If the corpus ever grows
to where this matters, the answer is Qdrant, not numpy.

Qdrant is a configuration value away and carries the hybrid-search story, but it is
**declared, not implemented** -- selecting it raises rather than silently falling back to the
in-memory index. A backend that quietly degrades to a different one is how a deployment ends
up believing it has hybrid search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from agentgate.config import Settings, VectorBackend
from agentgate.errors import AgentgateError
from agentgate.retrieval.corpus import load_corpus
from agentgate.retrieval.embeddings import build_embeddings


class VectorBackendUnavailableError(AgentgateError):
    """The configured vector backend cannot be built."""


@dataclass(frozen=True)
class Scored:
    """A document and how well it matched. The score is carried so a caller can see *how*
    weakly a weak match matched, rather than only that it was the best of a bad set."""

    document: Document
    score: float


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length, so a dot product is a cosine.

    Done here as well as in :class:`HashingEmbeddings` because a provider's embeddings are only
    approximately normalised, and "approximately" compounds across a ranking.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


class DenseIndex:
    """Every chunk, embedded once, searched by cosine similarity.

    Exhaustive rather than approximate. At this corpus size an approximate index would trade
    away exactness for a speedup nobody can measure, and it would make a retrieval test's
    expected output depend on index-construction parameters.
    """

    def __init__(self, embeddings: Embeddings, documents: list[Document]) -> None:
        self.documents = documents
        self.vectors = [
            _normalise(vector)
            for vector in embeddings.embed_documents([d.page_content for d in documents])
        ]
        self.embeddings = embeddings

    def search(self, query: str, k: int) -> list[Scored]:
        """The k best matches, best first.

        Ties break on corpus order, which is sorted by filename and then by position in the
        file. Deterministic on purpose: a retrieval test that asserts on a specific chunk
        should not become flaky because two chunks scored identically.
        """
        target = _normalise(self.embeddings.embed_query(query))
        scored = [
            Scored(document, sum(a * b for a, b in zip(target, vector, strict=True)))
            for document, vector in zip(self.documents, self.vectors, strict=True)
        ]
        scored.sort(key=lambda item: -item.score)
        return scored[:k]


class DenseRetriever(BaseRetriever):
    """The index behind LangChain's retriever interface, so it composes with everything else."""

    index: DenseIndex
    k: int

    # DenseIndex is a plain class, not a pydantic model, and BaseRetriever is one.
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002 - part of the interface
        **kwargs: Any,  # noqa: ARG002 - part of the interface
    ) -> list[Document]:
        return [scored.document for scored in self.index.search(query, self.k)]


def build_index(settings: Settings, documents: list[Document] | None = None) -> DenseIndex:
    """Index the corpus into the configured vector store.

    Args:
        settings: Supplies the corpus path, the backend, and the embeddings.
        documents: Pre-loaded chunks, for tests that want a corpus of their own. Loaded from
            ``settings.corpus_path`` when omitted.

    Raises:
        VectorBackendUnavailableError: for a backend that is declared but not built.
        CorpusUnavailableError: via :func:`load_corpus` if there is nothing to index.
    """
    if settings.vector_backend is not VectorBackend.MEMORY:
        msg = (
            f"vector backend '{settings.vector_backend.value}' is declared in configuration but "
            "not implemented. It is an optional Compose profile carrying the hybrid-search "
            "story (ADR 0010) and nothing in this repository claims it works. Refusing to fall "
            "back to the in-memory index, because a silent downgrade is how a deployment comes "
            "to believe it has hybrid search."
        )
        raise VectorBackendUnavailableError(msg)

    chunks = documents if documents is not None else load_corpus(settings.corpus_path)
    return DenseIndex(build_embeddings(settings), chunks)


def build_retriever(settings: Settings, documents: list[Document] | None = None) -> DenseRetriever:
    """Construct the retriever the retrieval subgraph uses.

    ``k`` comes from configuration rather than from a call site, so the number of chunks a
    model is shown is an operational knob and not a constant buried in a node.
    """
    return DenseRetriever(index=build_index(settings, documents), k=settings.retrieval_top_k)
