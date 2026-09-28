"""What a decider is asked, and the shape of what it answers.

**The questions.** Two, asked of every backend in the same words:

- ``route`` -- a Choice between ``auto_approve`` and ``human_review``. Its answer carries the
  probability of each option and a ``confidence``, both as the API reports them.
- ``irreversibility`` -- a Noul: does any proposed action have an effect that cannot be undone.
  Its answer is one probability of "yes" and carries **no confidence** (the API documents that),
  so none is invented for it.

The wording refers to fields of the state by name in backticks, which is how TypeSafe's docs say
to point a question at structured state. **It asks only what needs judgement.** Facts code can
check -- whether provenance passed, whether a tool was denied -- are
:data:`DETERMINISTIC_PRECONDITIONS`, checked by the gate and named by no question. The state
itself is built by the assess node (B5) from facts the drafter did not author: the routed lane,
the finding count, the tools that were denied, whether the provenance check passed, and the
proposed actions.

**The answer.** :class:`Assessment` is what gets stored, as plain JSON (ADR 0011). A failed
assessment is still an :class:`Assessment`, with ``failure`` set and every answer field ``None``
-- so a reader that looks for a reason to auto-approve finds none, and the gate asks a human.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final, Protocol

from agentgate.config import DeciderBackend

ROUTE: Final = "route"
IRREVERSIBILITY: Final = "irreversibility"
AUTO_APPROVE: Final = "auto_approve"
HUMAN_REVIEW: Final = "human_review"
ROUTE_OPTIONS: Final = (AUTO_APPROVE, HUMAN_REVIEW)

DETERMINISTIC_PRECONDITIONS: Final = ("provenance_check_passed", "denied_tools")
"""State fields the gate checks **in code** before an auto-approve can count, and which no question
names. Each is a boolean or a list already in the state: code can check it exactly, and asking a
model to re-derive it would only add a way to get it wrong. The gate (B5) requires provenance to
have passed and no tool to have been denied before it reads the decider at all."""

QUESTIONS: Final[dict[str, dict[str, Any]]] = {
    ROUTE: {
        "type": "choice",
        "instructions": (
            "Can the `proposed_actions` be carried out without a human reviewing them first?"
        ),
        "criteria": {
            AUTO_APPROVE: (
                "Routine and unambiguous: every item in `proposed_actions` is one a careful "
                "reviewer would approve without needing to look at it first."
            ),
            HUMAN_REVIEW: (
                "Any item in `proposed_actions` that is unusual, ambiguous, or one a careful "
                "reviewer would want to look at before it runs."
            ),
        },
    },
    IRREVERSIBILITY: {
        "type": "noul",
        "instructions": "Would any of the `proposed_actions` have an effect that cannot be undone?",
        "criteria": {
            "true": (
                "At least one proposed action changes something outside this system in a way that "
                "cannot be reversed, such as moving money or sending a message to a customer."
            ),
            "false": "Every proposed action can be reversed, or there are none.",
        },
    },
}


@dataclass(frozen=True)
class Assessment:
    """One decider's answer to both questions, or the reason it has none.

    Attributes:
        backend: Which backend answered.
        model: The model that answered, as the response named it. ``None`` if nothing answered.
        route: ``auto_approve`` or ``human_review``, as chosen. ``None`` on failure.
        route_probabilities: Each route option's probability, as reported. ``None`` when the
            backend reports none, which the ``llm`` backend never does.
        route_confidence: The Choice's ``confidence`` field as reported -- never recomputed.
        irreversibility: The Noul's probability of "yes". ``None`` when not reported.
        failure: Why there is no usable answer. ``None`` only when every call succeeded.
    """

    backend: str
    model: str | None = None
    route: str | None = None
    route_probabilities: dict[str, float] | None = None
    route_confidence: float | None = None
    irreversibility: float | None = None
    failure: str | None = None

    @classmethod
    def failed(
        cls, backend: DeciderBackend, reason: str, *, model: str | None = None
    ) -> Assessment:
        """An assessment that carries only the reason it failed."""
        return cls(backend=backend.value, model=model, failure=reason)

    @property
    def auto_approve_probability(self) -> float | None:
        """The route Choice's probability for ``auto_approve``, or ``None`` if not reported."""
        if self.route_probabilities is None:
            return None
        return self.route_probabilities.get(AUTO_APPROVE)

    def as_channel(self) -> dict[str, Any]:
        """Plain JSON for a state channel (ADR 0011)."""
        return asdict(self)


class Decider(Protocol):
    """Anything that can assess a state. Never raises for a failed assessment -- it returns one.

    The single exception is a spend ceiling: crossing it aborts the run, as it does everywhere
    else, because budgets are deterministic code and not a decider's to overrule.
    """

    backend: DeciderBackend

    def assess(self, state: Mapping[str, Any]) -> Assessment: ...
