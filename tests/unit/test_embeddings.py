"""The offline embedder, checked for the properties retrieval actually rests on.

Two of these exist because the obvious version of the test passes while the code is broken,
which is the only kind of test worth writing about in a docstring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentgate.config import Settings
from agentgate.errors import AgentgateError
from agentgate.retrieval.corpus import load_corpus
from agentgate.retrieval.embeddings import (
    DIMENSIONS,
    STOPWORDS,
    HashingEmbeddings,
    _bucket,
    build_embeddings,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

# Above this, the space is not separating the vocabulary and every ranking degrades. Set from
# the measurement recorded on DIMENSIONS: 4096 buckets over this corpus collides at 7%.
MAX_COLLISION_RATE = 0.12


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {"lane": "fake"}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


# --------------------------------------------------------------------------- determinism


def test_vectors_are_stable_across_processes() -> None:
    """The test that catches the real bug, which is why it pays for a subprocess.

    Python's ``hash()`` is salted per interpreter. An embedder built on it is perfectly
    deterministic *within* one process -- so a same-process test passes -- and produces a
    different vector space in the next one, which would make an index built by
    `seed_corpus.py` disagree with every query the application later makes. The failure would
    present as poor relevance, not as an error.
    """
    snippet = (
        "from agentgate.retrieval.embeddings import HashingEmbeddings;"
        "print(HashingEmbeddings().embed_query('retention schedule for authentication logs'))"
    )
    runs = [
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    ]

    assert runs[0] == runs[1]
    assert runs[0].strip(), "the subprocess produced no vector, so this compared nothing"


def test_the_same_text_embeds_identically_within_a_process() -> None:
    embeddings = HashingEmbeddings()

    assert embeddings.embed_query("refund window") == embeddings.embed_query("refund window")


# --------------------------------------------------------------------------- the space


def test_the_committed_corpus_fits_the_vector_space() -> None:
    """Pins the measurement that DIMENSIONS was set from.

    At 256 buckets, 70% of this vocabulary collided and retrieval ranked on noise while
    looking entirely plausible. If the corpus grows past what 4096 can separate, that should
    fail the build rather than quietly degrade every ranking in the system.
    """
    terms = {
        token
        for chunk in load_corpus(CORPUS)
        for token in chunk.page_content.lower().split()
        if token.isalnum() and token not in STOPWORDS
    }
    buckets: dict[int, int] = {}
    for term in terms:
        buckets[_bucket(term)] = buckets.get(_bucket(term), 0) + 1
    collided = sum(count for count in buckets.values() if count > 1)

    rate = collided / len(terms)
    assert rate <= MAX_COLLISION_RATE, (
        f"{rate:.0%} of the corpus vocabulary shares a hash bucket at {DIMENSIONS} dimensions. "
        "Retrieval still returns results; they are just decided by collisions. Raise "
        "DIMENSIONS and re-measure."
    )


def test_vectors_are_unit_length_so_length_does_not_decide_the_ranking() -> None:
    """Without normalisation a long document outscores a short one against every query, and
    top-k becomes a length ranking wearing a similarity costume."""
    embeddings = HashingEmbeddings()
    short = embeddings.embed_query("refund")
    long = embeddings.embed_query("refund " * 200 + "policy escalation approver prorated")

    assert sum(value * value for value in short) == pytest.approx(1.0)
    assert sum(value * value for value in long) == pytest.approx(1.0)


def test_text_with_no_usable_terms_is_zero_rather_than_an_error() -> None:
    """It scores zero against everything, which is the honest answer for a chunk with no
    content -- not a failure to embed."""
    vector = HashingEmbeddings().embed_query("... --- ???")

    assert set(vector) == {0.0}


def test_stopwords_do_not_reach_the_vector() -> None:
    embeddings = HashingEmbeddings()

    assert embeddings.embed_query("the refund") == embeddings.embed_query("refund")


# --------------------------------------------------------------------------- lane selection


def test_a_non_cloud_lane_embeds_in_process() -> None:
    assert isinstance(build_embeddings(settings_with()), HashingEmbeddings)


def test_the_cloud_lane_refuses_to_guess_an_embedding_model() -> None:
    """The identifier determines the vector space. A wrong guess indexes cleanly and retrieves
    nonsense, which is the failure that looks least like a configuration error."""
    settings = settings_with(
        lane="cloud",
        openai_api_key="not-required",
        cloud_capable_model="m",
        cloud_cheap_model="m",
        model_prices_usd_per_million={"m": {"input": 1.0, "output": 1.0}},
    )

    with pytest.raises(AgentgateError, match="refusing to assume an identifier"):
        build_embeddings(settings)
