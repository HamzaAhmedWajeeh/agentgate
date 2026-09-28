"""The decider's pieces that need no network: usage, the fake, construction, the record."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentgate.config import DeciderBackend, Lane, Settings
from agentgate.decider.assessment import AUTO_APPROVE, HUMAN_REVIEW, Assessment
from agentgate.decider.build import build_decider
from agentgate.decider.capabilities import (
    DECIDER_CAPABILITY_MATRIX,
    DeciderCapability,
    unverified_networked_decider_entries,
)
from agentgate.decider.fake import FakeDecider
from agentgate.decider.jev import JevDecider
from agentgate.decider.llm import LlmDecider
from agentgate.guardrails.spend import Ceilings, MissingUsageError, SpendLedger, Usage, usage_of
from agentgate.models.registry import Provenance

pytestmark = pytest.mark.usefixtures("isolated_env")

KEY = "AGENTGATE_JEV_API_KEY"  # ADR 0004 item 21


def cloud(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "cloud_capable_model": "m",
        "cloud_cheap_model": "m",
        "model_prices_usd_per_million": {
            "m": {"input": 1.0, "output": 1.0},
            "jev-9.9.9": {"input": 0.042, "output": 0.0},
        },
        "jev_model": "jev-9.9.9",
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


def ledger_for(settings: Settings) -> SpendLedger:
    return SpendLedger(settings, Ceilings.for_run(settings))


# ------------------------------------------------------------------ usage from a raw block


def test_a_usage_block_is_read_as_reported() -> None:
    assert usage_of({"input_tokens": 296, "output_tokens": 20}) == Usage(296, 20)


@pytest.mark.parametrize(
    "block",
    [None, {}, {"output_tokens": 20}, {"input_tokens": 296}, {"input_tokens": "296"}],
)
def test_a_missing_or_partial_usage_block_is_refused_not_zeroed(block: object) -> None:
    """Input tokens are what Jev charges for, so a block without them is not "free" -- it is an
    unmeasured call, and the ledger refuses it the way it refuses a chat reply with no usage."""
    with pytest.raises(MissingUsageError):
        usage_of(block)  # type: ignore[arg-type]


# -------------------------------------------------------------------------------- the fake


def test_the_fake_returns_what_it_was_scripted_to_and_logs_the_state() -> None:
    scripted = Assessment(
        backend="fake",
        model="fake-decider",
        route=AUTO_APPROVE,
        route_probabilities={AUTO_APPROVE: 0.95, HUMAN_REVIEW: 0.05},
        route_confidence=0.9,
        irreversibility=0.02,
    )
    decider = FakeDecider(scripted)

    assert decider.assess({"finding_count": 1}) == scripted
    assert decider.assess({"finding_count": 2}) == scripted
    assert decider.states_seen == [{"finding_count": 1}, {"finding_count": 2}]


def test_an_unscripted_fake_asks_a_human() -> None:
    """The default has to be the safe answer, so a test that forgets to script one cannot
    auto-approve anything by accident."""
    assessment = FakeDecider().assess({})

    assert assessment.route == HUMAN_REVIEW
    assert assessment.auto_approve_probability == 0.0


# ------------------------------------------------------------------------ construction


def test_no_backend_builds_no_decider() -> None:
    settings = cloud()

    assert build_decider(settings, ledger_for(settings)) is None


@pytest.mark.parametrize(
    ("backend", "kind"),
    [("jev", JevDecider), ("llm", LlmDecider), ("fake", FakeDecider)],
)
def test_each_backend_builds_its_decider(backend: str, kind: type) -> None:
    settings = cloud(decider_backend=backend, **{KEY: "not-required"})

    decider = build_decider(settings, ledger_for(settings))

    assert isinstance(decider, kind)
    assert decider.backend is DeciderBackend(backend)


def test_a_fake_decider_needs_no_key_but_does_need_a_cloud_lane() -> None:
    assert cloud(decider_backend="fake").decider_backend is DeciderBackend.FAKE
    with pytest.raises(ValidationError, match="no cloud lane"):
        Settings(_env_file=None, decider_backend="fake")  # type: ignore[call-arg]


# ------------------------------------------------------------------------------ the record


def test_an_assessment_is_plain_json_for_a_state_channel() -> None:
    """ADR 0011: state channels hold JSON only."""
    assessment = Assessment(
        backend="jev",
        model="jev-9.9.9",
        route=AUTO_APPROVE,
        route_probabilities={AUTO_APPROVE: 0.9, HUMAN_REVIEW: 0.1},
        route_confidence=0.37,
        irreversibility=0.04,
    )

    channel = assessment.as_channel()

    assert json.loads(json.dumps(channel)) == channel
    assert channel["route_confidence"] == 0.37


def test_a_failed_assessment_carries_no_answer_at_all() -> None:
    assessment = Assessment.failed(DeciderBackend.JEV, "HTTP 529")

    assert assessment.failure == "HTTP 529"
    assert assessment.route is None
    assert assessment.auto_approve_probability is None
    assert assessment.route_confidence is None
    assert assessment.irreversibility is None


# ------------------------------------------------------------------ the capability matrix


def test_no_networked_decider_row_rests_on_assumption() -> None:
    assert DECIDER_CAPABILITY_MATRIX, "an empty matrix would make the check below vacuous"
    assert unverified_networked_decider_entries() == []


def test_every_jev_row_is_stub_provenance_until_a_live_probe_runs() -> None:
    jev_rows = {
        capability: observation
        for (backend, capability), observation in DECIDER_CAPABILITY_MATRIX.items()
        if backend is DeciderBackend.JEV
    }

    assert set(jev_rows) == set(DeciderCapability)
    assert all(row.provenance is Provenance.STUB for row in jev_rows.values())


def test_the_llm_backend_is_recorded_as_reporting_no_confidence() -> None:
    row = DECIDER_CAPABILITY_MATRIX[(DeciderBackend.LLM, DeciderCapability.REPORTS_CONFIDENCE)]

    assert row.supported is False


def test_the_llm_backend_rides_the_cloud_lane() -> None:
    """Its questions go through the chat lane, so it is the cloud lane's egress with a different
    prompt -- which is why the no-cloud-lane refusal covers it."""
    settings = cloud(decider_backend="llm")

    assert Lane.CLOUD in settings.routable_lanes
