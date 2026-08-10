"""Loading and chunking the committed corpus.

The corpus is a handful of synthetic Markdown documents describing a fictional organisation.
Committed rather than generated, so the retrieval tests assert against known content and a
failure means retrieval broke rather than that the fixture drifted.

**Chunked on Markdown headings, not on a character count.** A heading is an author's own
statement about where one idea ends, so a heading-shaped chunk answers a question without
dragging in the neighbouring section. Fixed-width splitting would cut mid-sentence and is
tuned by a number nobody can justify. The cost is that a very long section stays one chunk;
that is visible in the corpus and preferable to an arbitrary boundary.

This deliberately does not use a text-splitter library. The rule is worth stating: chunking
decides what the model is allowed to see, so it is application logic, not plumbing to delegate.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from agentgate.errors import AgentgateError

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
CORPUS_GLOB = "*.md"

EXCLUDED = frozenset({"README.md"})
"""Files that describe the corpus rather than belong to it.

`corpus/README.md` explains what the directory is and that the organisation in it is
fictional. Indexed, it competes with the documents for retrieval slots and answers questions
about the repository when it was asked a question about a policy."""


class CorpusUnavailableError(AgentgateError):
    """The corpus directory is missing, empty, or produced no chunks.

    Raised rather than returning nothing. An empty index answers every query with silence,
    which reads downstream as "no relevant documents" -- a plausible answer that happens to be
    a configuration failure wearing the costume of a result.
    """


def load_corpus(path: Path) -> list[Document]:
    """Read every Markdown file under ``path`` and split it into heading-sized chunks.

    Files are read in sorted order so an index built twice is identical twice, which is what
    lets a retrieval test assert on a specific chunk rather than on a set.

    Raises:
        CorpusUnavailableError: if the directory does not exist, holds no Markdown, or yields
            no chunks with content.
    """
    if not path.is_dir():
        msg = (
            f"corpus directory {path} does not exist; nothing to index. Run "
            "`python scripts/seed_corpus.py --check` to see what is expected."
        )
        raise CorpusUnavailableError(msg)

    files = sorted(file for file in path.glob(CORPUS_GLOB) if file.name not in EXCLUDED)
    if not files:
        msg = (
            f"corpus directory {path} contains no indexable {CORPUS_GLOB} files "
            f"(excluding {sorted(EXCLUDED)})"
        )
        raise CorpusUnavailableError(msg)

    documents = [chunk for file in files for chunk in chunk_document(file)]
    if not documents:
        msg = f"corpus at {path} produced no chunks; every file appears to be empty"
        raise CorpusUnavailableError(msg)
    return documents


def chunk_document(file: Path) -> list[Document]:
    """Split one Markdown file on its headings.

    Content appearing before the first heading is kept under the file's own name rather than
    discarded -- a preamble is content, and silently dropping it would make the corpus differ
    from what a reader of the file expects it to contain.
    """
    text = file.read_text(encoding="utf-8")
    matches = list(HEADING.finditer(text))

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()] if matches else text
    if preamble.strip():
        sections.append((file.stem, preamble))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(2).strip(), text[match.start() : end]))

    return [
        Document(
            page_content=body.strip(),
            metadata={"source": file.name, "heading": heading},
        )
        for heading, body in sections
        if body.strip()
    ]
