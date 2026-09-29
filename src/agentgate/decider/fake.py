"""A deterministic decider for the offline suite.

Returns exactly what it was scripted to and records every state it was shown, so a test can
assert both what the gate did with an answer and what the decider was asked. It spends nothing
and records nothing in a ledger: it is in process, and an in-process call has no cost to count.

**Unscripted, it asks a human.** A test that forgets to script an answer must not auto-approve
anything by accident, so the default is a confident ``human_review``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from agentgate.config import DeciderBackend
from agentgate.decider.assessment import AUTO_APPROVE, HUMAN_REVIEW, Assessment

FAKE_MODEL: Final = "fake-decider"

ASK_A_HUMAN: Final = Assessment(
    backend=DeciderBackend.FAKE.value,
    model=FAKE_MODEL,
    route=HUMAN_REVIEW,
    route_probabilities={AUTO_APPROVE: 0.0, HUMAN_REVIEW: 1.0},
    route_confidence=1.0,
    irreversibility=1.0,
)


class FakeDecider:
    backend = DeciderBackend.FAKE

    def __init__(self, scripted: Assessment | None = None) -> None:
        self.scripted = scripted or ASK_A_HUMAN
        self.states_seen: list[dict[str, Any]] = []

    def assess(self, state: Mapping[str, Any]) -> Assessment:
        self.states_seen.append(dict(state))
        return self.scripted
