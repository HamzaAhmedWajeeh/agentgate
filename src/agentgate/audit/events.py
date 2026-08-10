"""Audit events.

Every node records what it decided. The trail is append-only -- the ``audit_trail`` channel
uses ``operator.add`` precisely so that concurrent branches concatenate rather than one
overwriting the other, which is the only merge that preserves a record.

An event answers four questions: what decided, what it decided, on what input, and as part of
which run. The input is recorded as a hash rather than as content, because the trail is meant
to be reviewable in environments where the content itself is regulated -- the point of an audit
log that copies the data it is auditing is hard to defend.

The durable writer arrives in Phase 5. This is the event shape, and it is deliberately a plain
dict: it lives in graph state, which is serialised into checkpoints, so it has to survive a
round trip through JSON without a custom encoder.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Decided(StrEnum):
    """The kind of decision an event records.

    A closed set so the trail can be queried. Free-text event names become unqueryable within
    about a month.
    """

    CLASSIFIED = "classified"
    LANE_SELECTED = "lane_selected"
    DISPATCHED = "dispatched"
    BUDGET_CHECKED = "budget_checked"
    BUDGET_EXCEEDED = "budget_exceeded"
    FINALISED = "finalised"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


def digest(content: str) -> str:
    """A short, stable hash of an input.

    Recorded instead of the content so the trail can be read by people who are not cleared to
    read the request. Short because it exists to correlate identical inputs across events, not
    to be cryptographically load-bearing.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def audit_event(  # noqa: PLR0913 - each field is one column of the record
    *,
    node: str,
    decided: Decided,
    correlation_id: str,
    input_digest: str,
    model: str | None = None,
    lane: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one audit event.

    Args:
        node: Which node decided.
        decided: What kind of decision it was.
        correlation_id: Ties every event in a run together.
        input_digest: Hash of what the decision was made on, from :func:`digest`.
        model: Which model produced the decision, where one did.
        lane: Which lane it ran on, where that applies.
        detail: Decision-specific fields. Must be JSON-serialisable, since state goes into
            checkpoints.
    """
    return {
        # Timezone-aware: an audit trail spanning hosts in different zones and compared by
        # naive timestamps is worse than no timestamps.
        "at": datetime.now(UTC).isoformat(),
        "node": node,
        "decided": decided.value,
        "correlation_id": correlation_id,
        "input_digest": input_digest,
        "model": model,
        "lane": lane,
        "detail": detail or {},
    }
