"""Index the committed corpus and show what it produced.

A script rather than a test, for the same reason as the others here: it produces a fact for a
person to read. What it prints -- how many chunks each file yielded, and what a sample query
actually retrieves -- is the only way to see whether the chunking is doing something sensible
before any of it becomes load-bearing.

On the default lane this costs nothing and touches no network: the fake lane's embeddings are
computed in-process. Against the cloud lane it embeds every chunk and is billed accordingly,
so it says which it is doing before it does it.

  python scripts/seed_corpus.py           index and show sample retrievals
  python scripts/seed_corpus.py --check   validate the corpus, index nothing
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Final

from agentgate.config import Lane, get_settings
from agentgate.errors import AgentgateError, ConfigurationError
from agentgate.guardrails.run_ledger import charging
from agentgate.guardrails.spend import Ceilings, SpendLedger
from agentgate.retrieval.corpus import load_corpus
from agentgate.retrieval.index import build_retriever

EXIT_OK: Final = 0
EXIT_BAD_USAGE: Final = 1
EXIT_BAD_CONFIG: Final = 2
EXIT_NO_CORPUS: Final = 3

SAMPLE_QUERIES: Final = [
    "how long do we keep authentication logs",
    "what do customers complain about most often",
    "when must the regulator be told about an incident",
]


def main(argv: list[str] | None = None) -> int:
    """Load, report, and unless asked not to, index and sample."""
    arguments = argv or []
    check_only = arguments == ["--check"]
    if arguments and not check_only:
        print(f"usage: python scripts/seed_corpus.py [--check]  (got {arguments})", file=sys.stderr)
        return EXIT_BAD_USAGE

    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        chunks = load_corpus(settings.corpus_path)
    except AgentgateError as error:
        print(str(error), file=sys.stderr)
        return EXIT_NO_CORPUS

    per_file = Counter(str(chunk.metadata["source"]) for chunk in chunks)
    print(f"\n  Corpus at {settings.corpus_path}\n")
    for source, count in sorted(per_file.items()):
        print(f"    {source:<28} {count:>3} chunks")
    print(f"    {'TOTAL':<28} {len(chunks):>3} chunks\n")

    if check_only:
        print("  --check: nothing indexed.\n")
        return EXIT_OK

    # Asked of the lane embeddings actually use -- the most contained one -- not the default. A
    # hybrid deployment defaults to the cloud and embeds in process for nothing (item 17).
    embedding_lane = settings.most_contained_lane
    if embedding_lane is Lane.CLOUD:
        print(f"  Embedding {len(chunks)} chunks via {settings.embedding_model}. This is billed.\n")
    else:
        print(
            f"  Embedding in-process on the '{embedding_lane.value}' lane. No network, no cost.\n"
        )

    # Billed to a ledger of its own and checked against the run ceiling, like any embedding. The
    # charge covers the index build as well as the queries: building the retriever is what embeds
    # the whole corpus.
    ledger = SpendLedger(settings, Ceilings.for_run(settings))
    with charging(ledger):
        try:
            retriever = build_retriever(settings, chunks)
        except AgentgateError as error:
            print(str(error), file=sys.stderr)
            return EXIT_BAD_CONFIG

        print(f"  Sample retrievals at k={settings.retrieval_top_k}:\n")
        for query in SAMPLE_QUERIES:
            print(f"    ? {query}")
            for document in retriever.invoke(query):
                print(f"        {document.metadata['source']} :: {document.metadata['heading']}")
            print()

    if ledger.calls:
        print(f"  Spent {ledger.total_tokens} tokens, ${ledger.total_usd:.6f}.\n")

    print(
        "  Read the samples rather than the counts. Chunking that looks fine by the numbers and\n"
        "  returns the wrong section is the failure worth catching here.\n"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
