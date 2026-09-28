"""Measure what each embedder costs to build an index, and how well it retrieves.

The two numbers that decide ADR 0004 item 17. Closing it means choosing what indexes the corpus,
and the options trade build cost against retrieval quality -- so both have to be measured rather
than argued about. ADR 0010's own dimension count came out of a measurement that contradicted the
reasoning behind it, which is the precedent this follows.

It is a script for the same reason ``measure_run.py`` and ``probe_capabilities.py`` are: **it
produces a fact.** The fact then gets recorded by a person, and the recorded value is what the
system enforces. Nothing here asserts anything.

## What can and cannot be measured offline

*Build time and call count* are real for both embedders. The stub-backed path uses the actual
client, the actual batching, and a real socket, so the call count is what a provider would see and
the wall time is everything except provider latency.

*Retrieval quality* is real for ``HashingEmbeddings`` and **meaningless for the stub**, which
returns a blake2b hash of the token ids it was sent. That is not a limitation to apologise for --
it makes the stub a **chance baseline**, and the baseline is what stops this measurement being
worthless. If the hashing embedder and a deliberately random one score the same, the questions are
answerable by anything and the hit rate measures nothing. Any real figure for a cloud embedder
needs a key and a live probe; this script says so and refuses to print a number it cannot observe.

Questions and expected chunks live in ``scripts/retrieval_questions.json``, committed so the
measurement is repeatable. A chunk is identified by ``(source, heading)``, which is what
``corpus.chunk_document`` writes into metadata -- not by text, which drifts when a document is
reworded.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO))

from agentgate.config import Lane, Settings  # noqa: E402
from agentgate.guardrails.run_ledger import charging  # noqa: E402
from agentgate.guardrails.spend import Ceilings, SpendLedger  # noqa: E402
from agentgate.retrieval.corpus import load_corpus  # noqa: E402
from agentgate.retrieval.embeddings import HashingEmbeddings, build_embeddings  # noqa: E402
from agentgate.retrieval.index import DenseIndex  # noqa: E402

EXIT_OK: Final = 0
EXIT_USAGE: Final = 1

CORPUS = REPO / "corpus"
QUESTIONS = Path(__file__).resolve().parent / "retrieval_questions.json"

STUB_MODEL: Final = "embedding-stub"
TOP_K: Final = 4
"""The configured default. Hit rate is reported at this k because it is the number a run uses."""

REPEATS: Final = 3
"""Build the index this many times and report the median.

One timing is a sample of one machine's mood. Three and a median is not a benchmark either, and
this does not pretend to be one -- it is enough to tell a tenth of a second from ten seconds, which
is the only distinction the decision turns on.
"""


@dataclass(frozen=True)
class Expected:
    question: str
    source: str
    heading: str
    overlap: str


@dataclass
class BuildCost:
    label: str
    seconds: float
    calls: int
    chunks: int


@dataclass
class Quality:
    label: str
    hits: list[tuple[Expected, bool, str]]

    @property
    def rate(self) -> float:
        return sum(1 for _, hit, _ in self.hits if hit) / len(self.hits) if self.hits else 0.0

    def rate_for(self, overlap: str) -> tuple[int, int]:
        subset = [hit for expected, hit, _ in self.hits if expected.overlap == overlap]
        return sum(subset), len(subset)


def expectations() -> list[Expected]:
    raw: dict[str, Any] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [
        Expected(
            question=entry["question"],
            source=entry["source"],
            heading=entry["heading"],
            overlap=entry["overlap"],
        )
        for entry in raw["questions"]
    ]


def stub_settings(base_url: str) -> Settings:
    """A cloud-lane configuration pointed at the loopback stub."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=base_url,
        cloud_capable_model="cloud-capable-stub",
        cloud_cheap_model="cloud-cheap-stub",
        embedding_model=STUB_MODEL,
        corpus_path=CORPUS,
        model_prices_usd_per_million={
            "cloud-capable-stub": {"input": 1.0, "output": 4.0},
            "cloud-cheap-stub": {"input": 0.1, "output": 0.4},
            STUB_MODEL: {"input": 0.02, "output": 0.0},
        },
    )


def time_build(label: str, embedder: Embeddings, chunks: list[Document], counter: Any) -> BuildCost:
    """Build the index ``REPEATS`` times and report the median wall time."""
    timings: list[float] = []
    calls = 0
    for _ in range(REPEATS):
        if counter is not None:
            counter.embedding_requests.clear()
        start = time.perf_counter()
        DenseIndex(embedder, chunks)
        timings.append(time.perf_counter() - start)
        if counter is not None:
            calls = len(counter.embedding_requests)
    return BuildCost(
        label=label, seconds=statistics.median(timings), calls=calls, chunks=len(chunks)
    )


def measure_quality(label: str, embedder: Embeddings, chunks: list[Document]) -> Quality:
    """Top-k hit rate: did the expected chunk appear in the k best matches."""
    index = DenseIndex(embedder, chunks)
    hits: list[tuple[Expected, bool, str]] = []
    for expected in expectations():
        results = index.search(expected.question, TOP_K)
        retrieved = [
            (
                str(scored.document.metadata.get("source")),
                str(scored.document.metadata.get("heading")),
            )
            for scored in results
        ]
        hit = (expected.source, expected.heading) in retrieved
        top = f"{retrieved[0][0]} / {retrieved[0][1]}" if retrieved else "(nothing)"
        hits.append((expected, hit, top))
    return Quality(label=label, hits=hits)


def running_stub_server() -> Iterator[Any]:
    """The committed double, imported lazily so the hashing half runs without it."""
    from tests.doubles.openai_compatible import (  # noqa: PLC0415 - script-side import
        StubBehaviour,
        running_stub,
    )

    return running_stub(StubBehaviour())  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    """Measure both embedders and print what item 17's options cost."""
    if argv:
        print(f"usage: python scripts/measure_retrieval.py  (got {argv})", file=sys.stderr)
        return EXIT_USAGE

    chunks = load_corpus(CORPUS)
    expected = expectations()

    print(f"\n  Corpus: {len(chunks)} chunks from {CORPUS.name}/")
    print(f"  Questions: {len(expected)} committed in {QUESTIONS.name}")
    print(
        f"  Hit rate is top-{TOP_K}, the configured default. Build time is the median of "
        f"{REPEATS}.\n"
    )

    # ------------------------------------------------------------------ hashing, in process
    hashing_cost = time_build("HashingEmbeddings (in process)", HashingEmbeddings(), chunks, None)
    hashing_quality = measure_quality("HashingEmbeddings", HashingEmbeddings(), chunks)

    # ------------------------------------------------------------------ the provider client
    with running_stub_server() as stub:  # type: ignore[attr-defined]
        settings = stub_settings(stub.base_url)
        provider = build_embeddings(settings)
        # The cloud embedder bills whichever ledger is charged; against a loopback stub that is
        # a ledger nobody reads, but an uncharged call is refused rather than made for free.
        with charging(SpendLedger(settings, Ceilings.for_run(settings))):
            provider_cost = time_build(
                "OpenAIEmbeddings -> stub (real HTTP)", provider, chunks, stub.behaviour
            )
            provider_quality = measure_quality("stub (chance baseline)", provider, chunks)

    # ------------------------------------------------------------------ build cost
    print("  BUILD COST\n")
    print(f"  {'embedder':<38} {'seconds':>9} {'calls':>7} {'chunks':>7}")
    for cost in (hashing_cost, provider_cost):
        print(f"  {cost.label:<38} {cost.seconds:>9.3f} {cost.calls:>7} {cost.chunks:>7}")
    ratio = provider_cost.seconds / hashing_cost.seconds if hashing_cost.seconds else float("nan")
    print(f"\n  The provider path is {ratio:.1f}x the in-process one against a loopback stub,")
    print("  which excludes provider latency entirely. A real endpoint adds a round trip per")
    print("  call, so treat this as the floor and not the figure.\n")

    # ------------------------------------------------------------------ quality
    print("  RETRIEVAL QUALITY\n")
    print(f"  {'question':<56} {'hashing':>9} {'baseline':>9}")
    for (exp, hashing_hit, hashing_top), (_, stub_hit, _) in zip(
        hashing_quality.hits, provider_quality.hits, strict=True
    ):
        mark = "HIT " if hashing_hit else "miss"
        base = "HIT " if stub_hit else "miss"
        print(f"  {exp.question[:54]:<56} {mark:>9} {base:>9}")
        if not hashing_hit:
            print(f"    {'':<54} expected {exp.source} / {exp.heading}")
            print(f"    {'':<54} top hit  {hashing_top}")

    print()
    # The number that makes the one above interpretable. Retrieving k of n chunks at random puts
    # the expected chunk in the results k/n of the time, so a hit rate is only meaningful against
    # it -- and on a corpus this small, that floor is not low.
    chance = TOP_K / len(chunks)
    label = f"chance (top-{TOP_K} of {len(chunks)} at random)"
    print(f"  {label:<28} {chance:>6.0%}")
    for quality in (hashing_quality, provider_quality):
        shared_hits, shared_total = quality.rate_for("shared")
        synonym_hits, synonym_total = quality.rate_for("synonym")
        print(
            f"  {quality.label:<28} {quality.rate:>6.0%} overall   "
            f"shared vocabulary {shared_hits}/{shared_total}   "
            f"synonym only {synonym_hits}/{synonym_total}"
        )

    # ----------------------------------------------------- what it does and does not say
    print(
        "\n  The baseline column is a deliberately meaningless embedder: the stub returns a hash\n"
        "  of the token ids it was sent. It is here because it is the control. If it scored as\n"
        "  well as the hashing embedder, these questions would be answerable by anything and the\n"
        "  hit rate above would measure nothing.\n"
    )
    print(
        "  NOT MEASURED: retrieval quality of a real cloud embedder. That needs a key and a live\n"
        "  probe, and no number for it is printed here or recorded anywhere, because this script\n"
        "  cannot observe one. The comparison this makes is hashing against chance, which is what\n"
        "  decides whether indexing on the contained lane is acceptable on its own terms.\n"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
