"""`docs/concept-map.md` is documentation, and documentation rots.

It is the one document here with nothing holding it to the repository, and it demonstrated why:
five concepts were listed as built in one table while the original rows sat in another still
marked pending, so the map claimed each of them both built and not built at once. Nobody
noticed, because nothing was looking.

Same shape as `test_env_example.py`, which holds `.env.example` to the settings model in both
directions. The rule the map runs on is that **a row is a claim about what exists**, so:

- a row marked *done* must name a file that exists and a symbol that is in it
- a row marked *not built* must not name a symbol that resolves, so a row that quietly becomes
  true fails here until someone updates it
- no concept may appear twice with different statuses

Every count is asserted non-empty. A parser that silently matches nothing turns all of this
into a test that passes by looking at an empty list, which is the failure this repository has
produced more than once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "docs" / "concept-map.md"
SOURCE = ROOT / "src" / "agentgate"

ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", re.M)
HEADER = {"Concept", "---"}
CODE = re.compile(r"`([^`]+)`")

# A backticked token counts as a path reference if it looks like one. Prose in a Location cell
# is allowed -- some rows explain rather than point -- but anything shaped like a file has to
# be real.
PATHISH = re.compile(r"^[\w./-]+\.(py|md)(:[\w.]+)?$")

MINIMUM_BUILT = 40
MINIMUM_NOT_BUILT = 8


class Row:
    def __init__(self, concept: str, status: str, location: str) -> None:
        self.concept = concept
        self.status = status
        self.location = location

    @property
    def built(self) -> bool:
        return "**done**" in self.status

    @property
    def not_built(self) -> bool:
        return "not built" in self.status

    def references(self) -> list[tuple[Path, str | None]]:
        """Every ``path`` or ``path:symbol`` this row points at, resolved against the repo."""
        found: list[tuple[Path, str | None]] = []
        for token in CODE.findall(self.location):
            if not PATHISH.match(token):
                continue
            path, _, symbol = token.partition(":")
            candidate = SOURCE / path if (SOURCE / path).exists() else ROOT / path
            found.append((candidate, symbol or None))
        return found

    def __repr__(self) -> str:  # pragma: no cover - failure output only
        return f"{self.concept!r} ({self.status})"


@pytest.fixture(scope="module")
def rows() -> list[Row]:
    parsed: list[Row] = []
    inside = False
    tables = 0
    for line in MAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Concept "):
            # The document also carries an ADR index, which is a table with different columns.
            # Only the Concept/Status/Location tables are claims about what exists.
            inside = True
            tables += 1
            continue
        if inside and not line.startswith("|"):
            inside = False
            continue
        match = ROW.match(line) if inside else None
        if match and match.group(1) not in HEADER:
            parsed.append(Row(*match.groups()))

    assert tables >= 3, f"found {tables} concept tables; the parser is reading the wrong document"
    assert parsed, "the row parser matched nothing; every assertion below would be vacuous"
    return parsed


def test_the_map_parses_into_rows_of_both_kinds(rows: list[Row]) -> None:
    """The guard for everything else here.

    Each check below filters the rows and asserts over what is left. If the parser or the
    status wording drifted, those filters would return empty and every one of them would pass
    while reading nothing.
    """
    built = [row for row in rows if row.built]
    not_built = [row for row in rows if row.not_built]

    assert len(built) >= MINIMUM_BUILT, f"only {len(built)} built rows parsed; the parser drifted"
    assert len(not_built) >= MINIMUM_NOT_BUILT, f"only {len(not_built)} not-built rows parsed"
    assert len(built) + len(not_built) == len(rows), (
        "some rows are neither built nor not built: "
        f"{[row for row in rows if not row.built and not row.not_built]}"
    )


def test_every_built_row_points_at_something_that_exists(rows: list[Row]) -> None:
    """A row marked done has a file behind it today, or it is not done."""
    checked = 0
    for row in (row for row in rows if row.built):
        for path, symbol in row.references():
            checked += 1
            assert path.exists(), f"{row}: {path} does not exist"
            if symbol:
                # Dotted, sometimes: `state.py:AgentState.messages` names a class and a field.
                # Each part has to be in the file; requiring the joined form would only pass
                # where the source happens to write it that way.
                body = path.read_text(encoding="utf-8")
                missing = [part for part in symbol.split(".") if part not in body]
                assert not missing, f"{row}: {path.name} does not contain {missing}"

    assert checked >= MINIMUM_BUILT, (
        f"only {checked} references resolved from the built rows; either the Location column "
        "stopped naming files or the extractor stopped recognising them"
    )


def test_no_not_built_row_names_a_symbol_that_resolves(rows: list[Row]) -> None:
    """A row that has quietly become true has to fail here.

    Bare file paths are allowed on a not-built row -- several name where the thing would live,
    which is a plan and says so. Naming a *symbol* is different: if it resolves, the concept
    exists and the row is now a lie.
    """
    for row in (row for row in rows if row.not_built):
        for path, symbol in row.references():
            if symbol is None:
                continue
            if not path.exists():
                continue
            body = path.read_text(encoding="utf-8")
            assert not all(part in body for part in symbol.split(".")), (
                f"{row}: {path.name} already contains {symbol!r}, so this row is out of date"
            )


def test_no_concept_appears_with_two_different_statuses(rows: list[Row]) -> None:
    """The failure that prompted this file.

    Five LangChain concepts were added to the LangGraph table as built while their original
    rows stayed pending in the table below, so the map asserted both at once. A reader taking
    either table at face value got a different answer.
    """
    statuses: dict[str, set[bool]] = {}
    for row in rows:
        statuses.setdefault(row.concept, set()).add(row.built)

    contradictory = sorted(concept for concept, seen in statuses.items() if len(seen) > 1)

    assert not contradictory, f"listed as both built and not built: {contradictory}"


def test_the_map_says_the_build_is_paused(rows: list[Row]) -> None:
    """The framing above the table is part of the claim, not decoration.

    Without it a table of built and not-built rows reads as a roadmap in progress. The line
    saying the build is paused is what makes "not built" mean "not built" rather than "coming".
    """
    text = MAP.read_text(encoding="utf-8")

    assert "paused at Phase 5" in text
    assert "claim about what exists" in text
