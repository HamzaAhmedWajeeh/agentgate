"""Citation provenance: the one output check that is a set comparison rather than a judgement.

A fabricated citation is worse than a missing one in a compliance domain. It names a document a
reader can go and look for, fail to find, and reasonably conclude was invented — and unlike a
vague claim, it survives review by looking rigorous.

The check is exact: every source the draft cites either came back from research or it did not.
No pattern matching against what a leak or an injection "looks like", because a guardrail that
is right most of the time converts *we do not check this* into *we check this*, and the second
is false in exactly the cases that matter.
"""

from __future__ import annotations

import pytest

from agentgate.graph.state import AgentState, Finding
from agentgate.guardrails.output import check_provenance

pytestmark = pytest.mark.usefixtures("isolated_env")

RETENTION = "data-retention.md#Retention periods"
REFUNDS = "refund-policy.md#Eligibility window"


def state_with(draft: str, *sources: str) -> AgentState:
    return {
        "draft": draft,
        "findings": [
            Finding(question="q", content="c", source=source).as_channel() for source in sources
        ],
    }


def test_a_citation_that_research_returned_is_clean() -> None:
    report = check_provenance(state_with(f"Records are kept seven years ({RETENTION}).", RETENTION))

    assert report.clean
    assert report.cited == {RETENTION}
    assert not report.fabricated


def test_a_citation_research_never_returned_is_caught() -> None:
    """The whole point. The draft names a real-looking document that this run never saw."""
    report = check_provenance(
        state_with(f"Approval needs two signatories ({REFUNDS}).", RETENTION),
    )

    assert not report.clean
    assert report.fabricated == {REFUNDS}


def test_the_real_citations_survive_alongside_a_fabricated_one() -> None:
    """A mixed draft must not be reported as wholly bad or wholly fine."""
    draft = f"Retention is seven years ({RETENTION}) and refunds run thirty days ({REFUNDS})."

    report = check_provenance(state_with(draft, RETENTION))

    assert report.cited == {RETENTION, REFUNDS}
    assert report.fabricated == {REFUNDS}


def test_a_draft_that_cites_nothing_is_clean_rather_than_suspicious() -> None:
    """Citing nothing and citing something invented are different failures, and this check is
    only about the second. How much evidence there was is `research_gaps`, and the review packet
    already carries it."""
    report = check_provenance(state_with("A short answer with no references.", RETENTION))

    assert report.clean
    assert report.cited == frozenset()


def test_a_finding_with_no_source_cannot_launder_a_citation() -> None:
    """An empty source string must not become a wildcard that matches anything."""
    report = check_provenance(state_with(f"See {RETENTION}.", ""))

    assert report.fabricated == {RETENTION}


def test_citations_are_found_next_to_ordinary_punctuation() -> None:
    """Guards the extractor. If the pattern stopped matching, every draft would read as clean
    and this file would pass while checking nothing."""
    for draft in (
        f"See {RETENTION}.",
        f"See {RETENTION}, which applies.",
        f"(see {RETENTION})",
        f"Sources:\n- {RETENTION}\n",
    ):
        assert check_provenance(state_with(draft, RETENTION)).cited == {RETENTION}, draft


def test_the_check_reports_what_it_compared() -> None:
    """A report that says "not clean" without saying against what is not reviewable."""
    detail = check_provenance(state_with(f"See {REFUNDS}.", RETENTION)).as_detail()

    assert detail == {"cited": 1, "retrieved": 1, "fabricated": [REFUNDS]}
