"""Embeddings, chosen by lane like everything else that reaches a network.

**Which lane: the most contained one the deployment can reach**, not the configured default and
not the routed one. That is :attr:`Settings.most_contained_lane`, the same rule classification
uses, and deliberately not a second copy of it. On that lane the cloud uses the configured
embedding model; every other lane uses :class:`HashingEmbeddings`, an in-process deterministic
vectoriser with no network and no cost, which is also what lets the whole test suite exercise real
retrieval rather than a stub that returns whatever the test wanted.

Why not the routed lane, as the chat calls use: **a vector index belongs to the model that built
it.** A query embedded by one model cannot search an index built by another, so honouring the
route would mean one index per lane -- and without a sovereign embedding endpoint, the sovereign
index would be built by the hashing embedder, making retrieval quality correlate exactly with
sensitivity. One index on the contained lane means no routing inside retrieval and, on a hybrid
deployment, no corpus or query text reaching a third party. It costs retrieval quality on every
request, public ones included. Leak inventory item 17 has the measurement and its limits.

**What it does not close.** A cloud-only deployment's most contained lane is the cloud, so the
corpus and every query are still embedded there -- the same residual egress as item 14's
classification. That spend is accounted: every call is billed to the run that made it, and checked
against its ceiling (item 19, closed).

**It is not a language model and does not pretend to be.** It has no notion of synonymy: a
query matches a document when they share vocabulary. That is enough to prove the plumbing --
chunking, indexing, top-k, the subgraph handoff, fan-in -- which is what the offline suite is
for. Retrieval *quality* is a live-lane question and is not claimed anywhere on the strength of
this.

The trap worth naming: Python's built-in ``hash()`` is salted per process. An index built in
one process and queried from another would silently return nothing in particular, and the
failure would look like bad relevance rather than like a bug. ``blake2b`` is used instead, and
``tests/unit/test_embeddings.py`` pins determinism across a subprocess boundary rather than
within one interpreter, because within one interpreter the salted version passes too.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Final

from langchain_core.embeddings import Embeddings

from agentgate.config import Lane, Settings
from agentgate.errors import AgentgateError
from agentgate.guardrails.run_ledger import charged_ledger
from agentgate.retrieval.accounting import AccountedEmbeddings, OpenAIEmbeddingsWithUsage

DIMENSIONS: Final = 4096
"""MEASURED, not chosen for looking reasonable.

The first attempt used 256, on the argument that a small corpus needs a small space. It does
not: 256 buckets over the committed corpus's 301 distinct terms put **70% of the vocabulary
into a shared bucket**, so most of the similarity being measured was collision noise. The
symptom was retrieval that looked plausible and ranked wrongly -- a query about log retention
returned a section on incident severity above the retention schedule, and dropping stopwords
made it no better, because the problem was never the stopwords.

Collision rate against this corpus, measured:

    dims    256     70% of terms share a bucket
    dims   1024     19%
    dims   4096      7%
    dims  16384      1%

4096 is where the curve flattens against a pure-Python dot product. ``test_embeddings.py``
pins the rate, so growing the corpus past what this space can separate fails the build instead
of quietly degrading every ranking.
"""

TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS: Final = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "within",
        "would",
        "you",
        "your",
    ]
)
"""Dropped before hashing.

Without this the ranking is decided by how many times two texts both said "the". Term
frequency over unfiltered text scores a long document highly against every query, and the
signal from the two or three words that actually carry the question is buried under it --
observed directly: "how long do we keep authentication logs" retrieved a section on incident
severity above the retention schedule.

A fixed list rather than inverse document frequency, deliberately. IDF is the better technique
and it needs corpus statistics, which would make the embedder *fitted* -- carrying state
learned from one corpus and silently wrong against another. The ``Embeddings`` interface has
nowhere honest to put that, and a stateful embedder that looks stateless is a worse problem
than an imperfect ranking on a lane whose purpose is to be free and deterministic.
"""


class EmbeddingsUnavailableError(AgentgateError):
    """The configured lane cannot produce embeddings from this configuration."""


def _bucket(token: str) -> int:
    """Map a token to a dimension, stably across processes.

    ``blake2b`` rather than ``hash()``: the built-in is salted per interpreter, so an index
    built in one process would not agree with a query made in another.
    """
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()) % (
        DIMENSIONS
    )


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: term frequency, hashed into a fixed space, L2
    normalised so cosine similarity is a dot product.

    Normalisation is not cosmetic. Without it a long document scores higher against every
    query purely for being long, and top-k degenerates into a length ranking.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in TOKEN.findall(text.lower()):
            if token in STOPWORDS:
                continue
            vector[_bucket(token)] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty or punctuation-only string. Returned as zeroes rather than as an error:
            # it scores zero against everything, which is the honest answer for a document
            # with no terms in it.
            return vector
        return [value / norm for value in vector]


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the embeddings for the most contained lane this deployment can reach.

    Raises:
        EmbeddingsUnavailableError: if that lane is the cloud and no embedding model is named.
            Indexing against an unnamed model is not something to guess at -- the identifier
            determines the vector space, and a wrong guess produces an index that retrieves
            confidently and wrongly.
    """
    # Not ``settings.lane``, which is the configured default and on a hybrid deployment is the
    # cloud: that sent restricted research queries, and the whole corpus, to the third party.
    if settings.most_contained_lane is not Lane.CLOUD:
        return HashingEmbeddings()

    if not settings.embedding_model:
        msg = (
            "the cloud lane needs AGENTGATE_EMBEDDING_MODEL to index a corpus; refusing to "
            "assume an identifier, because the model determines the vector space and a wrong "
            "one indexes cleanly and retrieves nonsense"
        )
        raise EmbeddingsUnavailableError(msg)

    if settings.openai_api_key is None:  # pragma: no cover - config validation prevents this
        msg = "the cloud lane needs OPENAI_API_KEY to embed"
        raise EmbeddingsUnavailableError(msg)

    # Accounted, and billed to whichever run is embedding: the index outlives any one run, so
    # the ledger is resolved per call from the research branch's charge rather than held here.
    # This is the wiring leak inventory item 19 found missing -- `AccountedEmbeddings` was built,
    # tested, and constructed by nothing, so embedding spend reached no ceiling.
    #
    # The client is called directly rather than through `OpenAIEmbeddings`, which discards the
    # usage block (item 11). It is pointed at the same endpoint as the cloud lane's chat calls,
    # which is what makes this egress observable on a stub at all (item 17).
    return AccountedEmbeddings(
        OpenAIEmbeddingsWithUsage(
            model=settings.embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            base_url=settings.openai_base_url,
        ),
        settings.embedding_model,
        charged_ledger,
    )
