"""What each decider backend is known to report, how that was learned, and when.

The same discipline as ``CAPABILITY_MATRIX`` in :mod:`agentgate.models.registry`, and the same
:class:`Observation` type, in a separate table because a decider is not a lane: it is keyed by
backend, and its capabilities are about what a response *carries*, not how a chat call behaves.

**Every Jev row is ``STUB`` until a live probe runs.** The stub in ``tests/doubles`` is shaped from
TypeSafe's published API reference, so these rows say "the documented shape, as this repository
reads it" -- not "TypeSafe was observed to do this". ``scripts/probe_capabilities.py jev`` makes one
real call and emits ``LIVE_PROBE`` rows to replace them, and refuses to run against anything but the
official endpoint so that a stub cannot be recorded as the real thing.

**Calibration is not a row.** Whether Jev's probabilities are *calibrated* is a claim about accuracy
over many answers, and no single probe -- live or stub -- can observe it. It is what shadow mode
measures, and it stays out of this table until that measurement exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from typing import Final

from agentgate.config import DeciderBackend
from agentgate.models.registry import Observation, Provenance


class DeciderCapability(StrEnum):
    """Something a decider's response either carries or does not."""

    REPORTS_USAGE = "reports_usage"
    """A usage block with input and output token counts. What the ledger depends on."""

    REPORTS_CONFIDENCE = "reports_confidence"
    """A ``confidence`` field on a Choice answer. What the second auto-approve condition reads."""

    ECHOES_VERSIONED_MODEL = "echoes_versioned_model"
    """The response names the pinned version that answered. What the pin check depends on."""


NETWORKED_BACKENDS: Final = frozenset({DeciderBackend.JEV, DeciderBackend.LLM})

_STUB_NOTE: Final = (
    "Observed against tests/doubles/typesafe_systemone.py, which is shaped from the API "
    "reference at docs.typesafe.ai read 2026-09-28. Replace with scripts/probe_capabilities.py "
    "jev output once a live call has run. See "
)

DECIDER_CAPABILITY_MATRIX: Final[Mapping[tuple[DeciderBackend, DeciderCapability], Observation]] = {
    (DeciderBackend.JEV, DeciderCapability.REPORTS_USAGE): Observation(
        supported=True,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 9, 28),
        note=_STUB_NOTE + "tests/integration/test_jev_decider.py::"
        "test_usage_is_recorded_against_the_pinned_model_and_priced_on_input",
    ),
    (DeciderBackend.JEV, DeciderCapability.REPORTS_CONFIDENCE): Observation(
        supported=True,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 9, 28),
        note=_STUB_NOTE
        + "tests/integration/test_jev_decider.py::test_a_clean_answer_is_read_field_by_field",
    ),
    (DeciderBackend.JEV, DeciderCapability.ECHOES_VERSIONED_MODEL): Observation(
        supported=True,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 9, 28),
        note=_STUB_NOTE + "tests/integration/test_jev_decider.py::"
        "test_a_response_from_a_different_model_version_is_a_failure",
    ),
    (DeciderBackend.LLM, DeciderCapability.REPORTS_CONFIDENCE): Observation(
        supported=False,
        provenance=Provenance.IN_PROCESS,
        recorded_on=date(2026, 9, 28),
        note=(
            "By construction: the llm backend reports a route and nothing numeric, because a "
            "chat model's self-reported number is generated text, not a measurement. So it can "
            "never meet the auto-approve thresholds. See tests/integration/test_llm_decider.py::"
            "test_it_reports_a_route_and_nothing_numeric"
        ),
    ),
}


def unverified_networked_decider_entries() -> list[tuple[DeciderBackend, DeciderCapability]]:
    """Rows for a networked backend that rest on nothing but assumption. Must be empty."""
    return sorted(
        (backend, capability)
        for (backend, capability), observation in DECIDER_CAPABILITY_MATRIX.items()
        if backend in NETWORKED_BACKENDS and not observation.is_trustworthy_on_a_networked_lane
    )
