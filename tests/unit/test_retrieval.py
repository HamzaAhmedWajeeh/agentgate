"""Corpus loading, chunking, and the dense index.

The corpus is committed, so these assert against known content. A failure here means retrieval
changed, not that a fixture drifted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentgate.config import Settings
from agentgate.retrieval.corpus import CorpusUnavailableError, chunk_document, load_corpus
from agentgate.retrieval.index import (
    VectorBackendUnavailableError,
    build_index,
    build_retriever,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {"lane": "fake", "corpus_path": CORPUS}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


# --------------------------------------------------------------------------- chunking


def test_a_document_splits_on_its_headings(tmp_path: Path) -> None:
    file = tmp_path / "policy.md"
    file.write_text("# Title\n\nintro\n\n## One\n\nfirst\n\n## Two\n\nsecond\n", encoding="utf-8")

    chunks = chunk_document(file)

    assert [chunk.metadata["heading"] for chunk in chunks] == ["Title", "One", "Two"]
    assert "first" in chunks[1].page_content
    assert "second" not in chunks[1].page_content


def test_content_before_the_first_heading_is_kept(tmp_path: Path) -> None:
    """A preamble is content. Dropping it would make the corpus differ from what a reader of
    the file believes it contains, which is the kind of gap nobody goes looking for."""
    file = tmp_path / "notes.md"
    file.write_text("standalone preamble text\n\n## Section\n\nbody\n", encoding="utf-8")

    chunks = chunk_document(file)

    assert chunks[0].metadata["heading"] == "notes"
    assert "standalone preamble" in chunks[0].page_content


def test_every_chunk_records_where_it_came_from() -> None:
    for chunk in load_corpus(CORPUS):
        assert chunk.metadata["source"].endswith(".md")
        assert chunk.metadata["heading"]


def test_the_corpus_readme_is_not_indexed() -> None:
    """It describes the directory rather than belonging to it, and indexed it competes for
    retrieval slots -- answering a question about the repository when a policy was asked for."""
    sources = {chunk.metadata["source"] for chunk in load_corpus(CORPUS)}

    assert "README.md" not in sources
    assert len(sources) == 4


# --------------------------------------------------------------------------- refusals


def test_a_missing_corpus_directory_refuses(tmp_path: Path) -> None:
    """An empty index answers every query with silence, which reads downstream as 'nothing
    relevant found' -- a plausible result that is really a configuration failure."""
    with pytest.raises(CorpusUnavailableError, match="does not exist"):
        load_corpus(tmp_path / "nowhere")


def test_a_corpus_of_only_excluded_files_refuses(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Corpus\n\nabout\n", encoding="utf-8")

    with pytest.raises(CorpusUnavailableError, match="no indexable"):
        load_corpus(tmp_path)


def test_a_corpus_of_empty_files_refuses(tmp_path: Path) -> None:
    (tmp_path / "blank.md").write_text("   \n\n", encoding="utf-8")

    with pytest.raises(CorpusUnavailableError, match="no chunks"):
        load_corpus(tmp_path)


def test_an_unimplemented_vector_backend_refuses_rather_than_falling_back() -> None:
    """A silent downgrade is how a deployment comes to believe it has hybrid search."""
    settings = settings_with(vector_backend="qdrant", qdrant_url="http://localhost:6333")

    with pytest.raises(VectorBackendUnavailableError, match="not implemented"):
        build_index(settings)


# --------------------------------------------------------------------------- search


def test_search_returns_k_documents_best_first() -> None:
    index = build_index(settings_with())

    results = index.search("refund escalation second approver", k=3)

    assert len(results) == 3
    assert [scored.score for scored in results] == sorted(
        (scored.score for scored in results), reverse=True
    )


def test_the_retriever_honours_the_configured_k() -> None:
    """k is an operational knob, not a constant buried in a node."""
    retriever = build_retriever(settings_with(retrieval_top_k=2))

    assert len(retriever.invoke("refund")) == 2


def test_a_question_retrieves_the_section_that_answers_it() -> None:
    """The plumbing claim, and the only one made for this lane.

    The offline embedder matches on shared vocabulary and has no notion of synonymy. That it
    finds the right section for a question phrased in the corpus's own words is what the
    fan-out and subgraph tests rest on; retrieval *quality* is a live-lane question and is not
    claimed anywhere on the strength of this.
    """
    index = build_index(settings_with())

    best = index.search("transaction records retention period seven years", k=1)[0]

    assert best.document.metadata["source"] == "data-retention.md"
    assert best.document.metadata["heading"] == "Retention periods"


def test_the_same_corpus_indexes_identically_twice() -> None:
    """Ties break on corpus order. Without that, a retrieval test asserting on a specific
    chunk becomes flaky the moment two chunks score the same."""
    settings = settings_with()
    first = [s.document.metadata["heading"] for s in build_index(settings).search("policy", k=5)]
    second = [s.document.metadata["heading"] for s in build_index(settings).search("policy", k=5)]

    assert first == second


def test_an_index_over_supplied_documents_does_not_touch_the_corpus_directory() -> None:
    """Tests downstream build small corpora of their own; this is the seam they use."""
    documents = [
        Document(page_content="pineapple provisioning", metadata={"source": "x.md", "heading": "x"})
    ]

    index = build_index(settings_with(corpus_path=Path("nowhere-at-all")), documents)

    assert len(index.documents) == 1
