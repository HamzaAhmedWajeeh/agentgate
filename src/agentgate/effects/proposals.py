"""Screening a proposal, hashing what was approved, and keying an effect.

**Screened before the gate.** A proposal the executor could not run never reaches the human: a tool
outside the ``EXECUTOR`` allowlist, arguments the tool's own schema rejects, or something that is
not a proposal at all. Each is dropped with its reason, and the drafter records the drops as an
audit event -- so the human is never shown an action that cannot happen, and the drop is not silent.

**Approval is of a hash.** :func:`digest_of` is a hash of the exact proposals, in order, as the
human was shown them. The gate records it; ``execute`` recomputes it from state and refuses a
mismatch. Approving a draft therefore cannot authorise a proposal the human did not see.

**Effects are keyed for exactly-once.** :func:`effect_key` is the run, the proposal's position,
and a hash of its arguments. It is what the outbox deduplicates on, so an ``execute`` re-run after
a crash between writing an effect and checkpointing it finds the key and does not write it again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from agentgate.graph.state import Proposal
from agentgate.tools.registry import ALLOWLISTS, TOOLS, Agent

EXECUTABLE: Final = ALLOWLISTS[Agent.EXECUTOR]


@dataclass
class Screened:
    """Proposals kept, and proposals dropped with the reason for each."""

    valid: list[Proposal] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)


def screen_proposals(raw: object) -> Screened:
    """Keep the proposals the executor could run; drop the rest, each with its reason."""
    screened = Screened()
    if not isinstance(raw, list):
        screened.dropped.append(
            {"proposal": raw, "reason": "proposed_actions is not a list of proposals (shape)"}
        )
        return screened

    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str):
            screened.dropped.append(
                {"proposal": item, "reason": "not a {tool, arguments} object (shape)"}
            )
            continue
        name = item["tool"]
        if name not in EXECUTABLE:
            screened.dropped.append(
                {"proposal": item, "reason": f"{name!r} is not an executor tool"}
            )
            continue
        arguments = item.get("arguments")
        schema = TOOLS[name].args_schema
        refused = {"proposal": item, "reason": f"arguments do not satisfy {name}'s schema"}
        if not isinstance(arguments, dict) or not (
            isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            screened.dropped.append(refused)
            continue
        try:
            checked = schema.model_validate(arguments)
        except ValidationError:
            screened.dropped.append(refused)
            continue
        screened.valid.append(Proposal(tool=name, arguments=checked.model_dump(mode="json")))
    return screened


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_of(proposals: Sequence[Proposal]) -> str:
    """A hash of exactly these proposals, in this order."""
    return hashlib.sha256(_canonical([p.as_channel() for p in proposals])).hexdigest()


def effect_key(correlation_id: str, index: int, proposal: Proposal) -> str:
    """The run, the position, and the arguments: what makes an effect the same effect."""
    arguments = hashlib.sha256(_canonical(proposal.as_channel())).hexdigest()[:16]
    return f"{correlation_id}:{index}:{arguments}"
