"""Where an approved effect goes. One implementation, and configuration refuses any other.

:class:`OutboxSink` appends one JSON line per effect to a file and does nothing else. An approved
refund is *recorded* there; no money moves and no mail is sent. There is no other sink -- not a
disabled one, not a stub for a payment provider -- because a sink that could perform something real
would make "this performs nothing real" a matter of configuration discipline rather than of what
the code can do. ``EffectSinkBackend`` has one member and ``Settings`` refuses any other name.

**Exactly once, by key.** :meth:`OutboxSink.record` reads the file for the effect's key before
writing and returns the existing record if it is there. The file is re-read every time rather than
cached, because the writer that matters is a *different* process: one that crashed after writing
an effect and before its checkpoint recorded that, whose resumed successor re-runs ``execute``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from agentgate.config import EffectSinkBackend, Settings
from agentgate.graph.state import Proposal


class EffectSink(Protocol):
    """Records an approved effect, at most once per key."""

    def record(self, key: str, proposal: Proposal) -> dict[str, Any]: ...


class OutboxSink:
    """Append-only JSON lines on disk. Performs nothing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _existing(self, key: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("key") == key:
                return dict(entry)
        return None

    def record(self, key: str, proposal: Proposal) -> dict[str, Any]:
        existing = self._existing(key)
        if existing is not None:
            return existing
        effect = {
            "key": key,
            "tool": proposal.tool,
            "arguments": proposal.arguments,
            # Stated on every record, so the outbox read on its own cannot be mistaken for a log
            # of things that happened in the world.
            "simulated": True,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as outbox:
            outbox.write(json.dumps(effect, sort_keys=True) + "\n")
            outbox.flush()
        return effect


def build_effect_sink(settings: Settings) -> EffectSink:
    """The sink configuration names. There is exactly one."""
    match settings.effect_sink:
        case EffectSinkBackend.OUTBOX:
            return OutboxSink(settings.outbox_path)
