"""The ``llm`` decider backend: the cloud chat lane asked the route question, on the wire.

It reports a route and nothing numeric. No probabilities, no confidence, no irreversibility --
a chat model's self-reported number is generated text, not a measurement, and none is invented
for it. So under the three auto-approve conditions it can never auto-approve in enforce mode.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import Settings
from agentgate.decider.assessment import AUTO_APPROVE
from agentgate.decider.llm import LlmDecider
from agentgate.guardrails.spend import Ceilings, SpendLedger

pytestmark = pytest.mark.usefixtures("isolated_env")

CAPABLE = "cloud-capable-test"
CHEAP = "cloud-cheap-test"
SOVEREIGN = "sovereign-test"

STATE = {
    "routed_lane": "cloud",
    "finding_count": 7031,
    "denied_tools": [],
    "provenance_check_passed": True,
    "proposed_actions": [],
}


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply={"route": AUTO_APPROVE})) as server:
        yield server


@pytest.fixture
def sovereign() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply={"route": AUTO_APPROVE})) as server:
        yield server


def settings_for(cloud: StubServer, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "openai_base_url": cloud.base_url,
        "cloud_capable_model": CAPABLE,
        "cloud_cheap_model": CHEAP,
        "decider_backend": "llm",
        "model_prices_usd_per_million": {
            CAPABLE: {"input": 1.0, "output": 4.0},
            CHEAP: {"input": 0.1, "output": 0.4},
            SOVEREIGN: {"input": 0.0, "output": 0.0},
        },
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


def test_it_reports_a_route_and_nothing_numeric(cloud: StubServer) -> None:
    settings = settings_for(cloud)
    ledger = SpendLedger(settings, Ceilings.for_run(settings))

    assessment = LlmDecider(settings, ledger).assess(STATE)

    assert cloud.behaviour.request_count >= 1, "precondition: the chat lane was asked"
    assert assessment.failure is None
    assert assessment.route == AUTO_APPROVE
    assert assessment.model == CHEAP
    assert assessment.route_probabilities is None
    assert assessment.route_confidence is None
    assert assessment.irreversibility is None


def test_the_state_is_what_it_sends(cloud: StubServer) -> None:
    settings = settings_for(cloud)

    LlmDecider(settings, SpendLedger(settings, Ceilings.for_run(settings))).assess(STATE)

    assert any("7031" in str(body) for body in cloud.behaviour.requests_seen)


def test_its_spend_goes_into_the_ledger_against_the_model_that_answered(cloud: StubServer) -> None:
    settings = settings_for(cloud)
    ledger = SpendLedger(settings, Ceilings.for_run(settings))

    LlmDecider(settings, ledger).assess(STATE)

    assert set(ledger.usage_by_model) == {CHEAP}
    assert ledger.total_tokens > 0


def test_it_rides_the_cloud_lane_even_where_classification_does_not(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """On a hybrid deployment the most contained lane is sovereign, and classification runs
    there. The decider only ever runs on a request the router sent to the cloud, so it asks the
    cloud -- and nothing reaches the operator's endpoint on its behalf."""
    settings = settings_for(cloud, sovereign_base_url=sovereign.base_url, sovereign_model=SOVEREIGN)

    LlmDecider(settings, SpendLedger(settings, Ceilings.for_run(settings))).assess(STATE)

    assert cloud.behaviour.request_count >= 1
    assert sovereign.behaviour.request_count == 0


def test_an_answer_that_never_parses_asks_a_human(cloud: StubServer) -> None:
    cloud.behaviour.reply = {"verdict": "looks fine to me"}
    settings = settings_for(cloud)

    assessment = LlmDecider(settings, SpendLedger(settings, Ceilings.for_run(settings))).assess(
        STATE
    )

    assert cloud.behaviour.request_count >= 1
    assert assessment.route is None
    assert assessment.failure is not None


def test_an_http_failure_asks_a_human(cloud: StubServer) -> None:
    cloud.behaviour.fail_first_n = 10
    settings = settings_for(cloud, max_retries=0)

    assessment = LlmDecider(settings, SpendLedger(settings, Ceilings.for_run(settings))).assess(
        STATE
    )

    assert assessment.route is None
    assert assessment.failure is not None
