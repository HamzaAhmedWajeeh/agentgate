"""The Jev decider, asserted on the wire against a stub over real HTTP.

Every assertion here reads what reached the stub -- path, headers, body -- or what the decider
returned after a real HTTP exchange. None reads a client attribute.

The contract under test is narrow and total: the decider returns an :class:`Assessment` for every
outcome, and every outcome other than a clean, correctly-versioned, fully-accounted answer is an
assessment with ``failure`` set and no route -- which the gate reads as "ask a human". The one
thing allowed to escape is a spend ceiling, because budgets are deterministic code and not the
decider's to swallow.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from tests.doubles.typesafe_systemone import JevBehaviour, JevStub, running_jev_stub

from agentgate.config import Settings
from agentgate.decider.assessment import (
    AUTO_APPROVE,
    HUMAN_REVIEW,
    IRREVERSIBILITY,
    QUESTIONS,
    ROUTE,
)
from agentgate.decider.jev import MAX_RETRY_WAIT_SECONDS, JevDecider
from agentgate.guardrails.spend import Ceilings, SpendCeilingExceededError, SpendLedger, Usage

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-test"
CLOUD_CHEAP = "cloud-cheap-test"
PINNED = "jev-9.9.9"
KEY = "AGENTGATE_JEV_API_KEY"  # passed by its declared name: ADR 0004 item 21
JEV_KEY = "jev-wire-key-canary"

STATE: dict[str, Any] = {
    "routed_lane": "cloud",
    "finding_count": 3,
    "denied_tools": [],
    "provenance_check_passed": True,
    "proposed_actions": [{"tool": "send_summary_email", "arguments": {"to": "ops"}}],
}

CONFIDENT = {
    "choice_probabilities": {ROUTE: {AUTO_APPROVE: 0.9, HUMAN_REVIEW: 0.1}},
    # Deliberately not what the docs' example formula would give for 0.9/0.1 (that is 0.8). If
    # the decider recomputed confidence instead of reading the field, the assertion would see it.
    "choice_confidence": {ROUTE: 0.37},
    "noul": {IRREVERSIBILITY: 0.04},
}


def settings_for(base_url: str, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "cloud_capable_model": CLOUD_CAPABLE,
        "cloud_cheap_model": CLOUD_CHEAP,
        "decider_backend": "jev",
        KEY: JEV_KEY,
        "jev_model": PINNED,
        "jev_base_url": base_url,
        "request_timeout_seconds": 5.0,
        "model_prices_usd_per_million": {
            CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
            CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
            PINNED: {"input": 0.042, "output": 0.0},
        },
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


@pytest.fixture
def stub() -> Iterator[JevStub]:
    with running_jev_stub(JevBehaviour(**CONFIDENT)) as server:  # type: ignore[arg-type]
        yield server


class Sleeps:
    """Records every wait instead of waiting, so retry timing is asserted, not endured."""

    def __init__(self) -> None:
        self.seconds: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


def decider_for(settings: Settings, sleeps: Sleeps | None = None) -> tuple[JevDecider, SpendLedger]:
    ledger = SpendLedger(settings, Ceilings.for_run(settings))
    return JevDecider(settings, ledger, sleep=sleeps or Sleeps()), ledger


# ------------------------------------------------------------------------------ the request


def test_the_request_on_the_wire_is_the_documented_shape(stub: JevStub) -> None:
    decider, _ = decider_for(settings_for(stub.base_url))

    decider.assess(STATE)

    assert stub.behaviour.paths_seen == ["/v1/systemone"]
    headers = stub.behaviour.headers_seen[0]
    assert headers["authorization"] == f"Bearer {JEV_KEY}"
    assert headers["content-type"].startswith("application/json")

    body = stub.behaviour.requests_seen[0]
    assert set(body) == {"state", "model", "questions"}, "nothing beyond the documented fields"
    assert body["state"] == STATE
    assert body["model"] == PINNED, "the pinned version, never an alias"
    assert body["questions"] == QUESTIONS


@pytest.mark.parametrize("suffix", ["", "/", "//"])
def test_a_trailing_slash_on_the_base_url_cannot_double_the_path(
    stub: JevStub, suffix: str
) -> None:
    """``/v1`` and ``/v1/`` must both reach ``/v1/systemone`` -- not ``/v1//systemone``, and not
    ``/v1/v1/systemone``, which is what appending the documented full path would produce."""
    decider, _ = decider_for(settings_for(stub.base_url + suffix))

    assessment = decider.assess(STATE)

    assert stub.behaviour.paths_seen == ["/v1/systemone"]
    assert assessment.failure is None


# ------------------------------------------------------------------------------ the answer


def test_a_clean_answer_is_read_field_by_field(stub: JevStub) -> None:
    decider, _ = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert assessment.failure is None
    assert assessment.model == PINNED
    assert assessment.route == AUTO_APPROVE
    assert assessment.route_probabilities == {AUTO_APPROVE: 0.9, HUMAN_REVIEW: 0.1}
    assert assessment.auto_approve_probability == 0.9
    assert assessment.route_confidence == 0.37, "the reported field, not a recomputation"
    assert assessment.irreversibility == 0.04


def test_a_response_from_a_different_model_version_is_a_failure(stub: JevStub) -> None:
    """An alias that moved would answer as a new version. Thresholds tuned on shadow data from
    the pinned one mean nothing against it, so the answer is discarded and a human asked."""
    stub.behaviour.answer_model = "jev-9.9.10"
    decider, _ = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == 1, "precondition: the call was made and answered"
    assert assessment.route is None
    assert assessment.failure is not None
    assert "jev-9.9.10" in assessment.failure
    assert PINNED in assessment.failure


# ------------------------------------------------------------------------------- the ledger


def test_usage_is_recorded_against_the_pinned_model_and_priced_on_input(stub: JevStub) -> None:
    decider, ledger = decider_for(settings_for(stub.base_url))

    decider.assess(STATE)

    assert ledger.usage_by_model == {PINNED: Usage(input_tokens=296, output_tokens=20)}
    assert ledger.total_usd == pytest.approx(296 * 0.042 / 1_000_000)


def test_a_response_with_no_usage_is_refused_rather_than_recorded_as_free(stub: JevStub) -> None:
    """The usage block is documented as required and has never been seen on a real wire. If it
    is absent the call cost money nobody can count, so it is a failure -- never a zero."""
    stub.behaviour.omit_usage = True
    decider, ledger = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == 1, "precondition: a billed call was made"
    assert ledger.usage_by_model == {}, "nothing recorded, and in particular not a zero"
    assert ledger.calls == 0
    assert assessment.route is None
    assert assessment.failure is not None
    assert "usage" in assessment.failure


def test_a_spend_ceiling_is_not_swallowed_into_a_human_review(stub: JevStub) -> None:
    """Budgets are deterministic code. Crossing one aborts the run, as it does for every other
    call, rather than quietly becoming "ask a human" and letting the run carry on."""
    decider, _ = decider_for(settings_for(stub.base_url, max_total_tokens=100))

    with pytest.raises(SpendCeilingExceededError):
        decider.assess(STATE)


# ---------------------------------------------------------------------- every failure path


@pytest.mark.parametrize(
    ("statuses", "requests", "recovered"),
    [
        ([422], 1, False),  # validation: our fault, retrying sends the same bad body
        ([401], 1, False),  # authentication: retrying cannot help
        ([429], 2, True),  # rate limited, then answered
        ([529], 2, True),  # overloaded, then answered
        ([500], 2, True),  # a server fault, then answered
        ([429, 429], 2, False),  # at most one retry
        ([529, 500], 2, False),
        ([500, 529], 2, False),
    ],
)
def test_each_status_is_retried_at_most_once_and_otherwise_asks_a_human(
    stub: JevStub, statuses: list[int], requests: int, recovered: bool
) -> None:
    stub.behaviour.fail_with = list(statuses)
    decider, _ = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == requests
    if recovered:
        assert assessment.failure is None
        assert assessment.route == AUTO_APPROVE
    else:
        assert assessment.route is None
        assert assessment.failure is not None
        assert str(statuses[-1]) in assessment.failure


def test_retry_after_is_honoured_but_bounded(stub: JevStub) -> None:
    stub.behaviour.fail_with = [429]
    stub.behaviour.retry_after = "3600"
    sleeps = Sleeps()
    decider, _ = decider_for(settings_for(stub.base_url), sleeps)

    decider.assess(STATE)

    assert sleeps.seconds == [MAX_RETRY_WAIT_SECONDS], "an hour-long retry-after is capped"


def test_a_short_retry_after_is_used_as_given(stub: JevStub) -> None:
    stub.behaviour.fail_with = [529]
    stub.behaviour.retry_after = "1"
    sleeps = Sleeps()
    decider, _ = decider_for(settings_for(stub.base_url), sleeps)

    decider.assess(STATE)

    assert sleeps.seconds == [1.0]


def test_a_timeout_asks_a_human_and_is_not_retried(stub: JevStub) -> None:
    """A timeout is not in the retry set: the request may have been processed and billed, and a
    second one doubles the latency in front of a human who is waiting anyway."""
    stub.behaviour.delay_seconds = 1.0
    decider, _ = decider_for(settings_for(stub.base_url, request_timeout_seconds=0.2))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == 1
    assert assessment.route is None
    assert assessment.failure is not None
    assert "timed out" in assessment.failure


def test_a_body_that_is_not_json_asks_a_human(stub: JevStub) -> None:
    stub.behaviour.malformed = True
    decider, _ = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == 1
    assert assessment.route is None
    assert assessment.failure is not None


@pytest.mark.parametrize(
    "missing",
    [
        (ROUTE, "confidence"),
        (ROUTE, "probabilities"),
        (ROUTE, "choice"),
        (IRREVERSIBILITY, "noul"),
    ],
)
def test_an_answer_missing_a_field_asks_a_human(stub: JevStub, missing: tuple[str, str]) -> None:
    stub.behaviour.omit_answer_field = missing
    decider, _ = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert stub.behaviour.request_count == 1
    assert assessment.route is None
    assert assessment.failure is not None
    assert missing[0] in assessment.failure


def test_a_failed_assessment_still_records_what_was_spent(stub: JevStub) -> None:
    """Money spent on an answer that is then discarded is still money spent."""
    stub.behaviour.answer_model = "jev-9.9.10"
    decider, ledger = decider_for(settings_for(stub.base_url))

    assessment = decider.assess(STATE)

    assert assessment.failure is not None
    assert ledger.usage_by_model == {PINNED: Usage(input_tokens=296, output_tokens=20)}


def test_the_endpoint_strips_a_trailing_slash_even_past_validation() -> None:
    """Configuration strips the slash, and so does the endpoint, because a ``Settings`` built by
    ``model_copy(update=...)`` skips validation. Without this, removing either strip would survive
    every test above, since the other one catches it."""
    settings = settings_for("http://127.0.0.1:9/v1").model_copy(
        update={"jev_base_url": "http://127.0.0.1:9/v1/"}
    )
    decider, _ = decider_for(settings)

    assert decider.endpoint == "http://127.0.0.1:9/v1/systemone"
