"""The durable audit trail: append-only JSON lines, readable without this codebase.

The trail is the artifact the thesis rests on. Every gate in this system decides something, and
the claim is that no gate decides silently. A record that only the producing system can read
does not support that claim — it supports a weaker one, that the system remembers what it did.

So the format is chosen for a reader who does not have the repository, is not running the code,
and may be reading it years later:

*Append-only JSON lines.* One event per line, complete in itself. A truncated file loses its
last line and nothing else, and `tail`, `grep`, `jq` and twenty lines of any language all work
on it. This is the same argument that decided the checkpoint boundary in ADR 0011, applied one
level out — durable formats are schemas, and schemas that need the producer's code to read are
not records.

*Self-describing field names.* `decided`, `node`, `correlation_id`, `at`. Not `d`, `n`, `cid`.
The file is the documentation.

*Timezone-aware timestamps.* An audit trail spanning hosts in different zones, compared by
naive timestamps, is worse than no timestamps: it looks orderable and is not.

*Input as a hash, never content.* The trail has to be readable by people who are not cleared to
read the requests it describes. An audit log that copies the regulated data it is auditing is
hard to defend, and impossible to hand to a reviewer.

The writer opens in append mode and never rewrites. There is no update path and no delete path,
because the property being claimed is append-only and a method that could edit a past line
would make that a convention rather than a fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentgate.config import Settings
from agentgate.errors import AgentgateError

REQUIRED_FIELDS = ("at", "node", "decided", "correlation_id", "input_digest")
"""What every line must carry to be worth reading on its own.

A line missing any of these is not a partial record, it is an unattributable one: you cannot
say when it happened, what decided, what it decided, or which run it belonged to.
"""


class AuditWriteError(AgentgateError):
    """The trail could not be written.

    Raised rather than swallowed. Every other failure in this system is caught and summarised
    so a run survives it; this one is not, because a run that proceeds past a gate whose
    decision was not recorded has produced exactly the thing the trail exists to make
    impossible.
    """


def write_events(events: list[dict[str, Any]], path: Path) -> int:
    """Append events to the trail, one JSON object per line.

    Args:
        events: Audit events, as produced by ``audit.events.audit_event``.
        path: Destination. Parent directories are created; the file is never truncated.

    Returns:
        How many lines were written.

    Raises:
        AuditWriteError: if an event is missing a required field, is not JSON-serialisable, or
            the file cannot be written. Each is a reason the record would be unreadable, and an
            unreadable record is worse than a loud failure.
    """
    for index, event in enumerate(events):
        missing = [field for field in REQUIRED_FIELDS if not event.get(field)]
        if missing:
            msg = f"audit event {index} is missing {missing}; it would not be attributable"
            raise AuditWriteError(msg)

    try:
        lines = [json.dumps(event, sort_keys=True, ensure_ascii=False) for event in events]
    except (TypeError, ValueError) as error:
        msg = (
            f"an audit event is not JSON-serialisable: {error}. The trail is a durable format "
            "and must not depend on this codebase to decode (ADR 0011)."
        )
        raise AuditWriteError(msg) from error

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    except OSError as error:
        msg = f"could not append to the audit trail at {path}: {error}"
        raise AuditWriteError(msg) from error

    return len(lines)


def write_trail(state_audit_trail: list[dict[str, Any]], settings: Settings) -> int:
    """Persist a run's accumulated trail to the configured destination."""
    return write_events(state_audit_trail, settings.audit_log_path)
