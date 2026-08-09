"""Retry and fallback, against a server that really returns HTTP errors.

Two different failures need two different answers. A reset connection or a rate limit is
transient: the same request will probably work in a moment, so retry. A model that is not going
to answer is durable: retrying costs time and money for nothing, so fall back to another one.
Conflating them produces either a system that gives up too early or one that hammers a dead
endpoint.

The stub is driven through the cloud lane here rather than the sovereign one, purely so the two
tiers resolve to different identifiers and the request log can show *which* model answered.
Both lanes are the same client pointed at different endpoints, so the chain being exercised is
identical.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import CallClass, Settings
from agentgate.models.registry import build_resilient_model

pytestmark = pytest.mark.usefixtures("isolated_env")

CAPABLE = "capable-tier-stub"
CHEAP = "cheap-tier-stub"


def settings_for(stub: StubServer, *, max_retries: int) -> Settings:
    """Point both tiers at the stub, with distinguishable identifiers."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane="cloud",
        openai_api_key="not-required",
        openai_base_url=stub.base_url,
        cloud_capable_model=CAPABLE,
        cloud_cheap_model=CHEAP,
        max_retries=max_retries,
        model_prices_usd_per_million={
            CAPABLE: {"input": 1.0, "output": 4.0},
            CHEAP: {"input": 0.1, "output": 0.4},
        },
    )


@pytest.fixture
def stub() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply={"answer": "ok"})) as server:
        yield server


def models_called(stub: StubServer) -> list[str]:
    """Which model each request asked for, in order."""
    return [str(request.get("model", "")) for request in stub.behaviour.requests_seen]


def on_the_wire(stub: StubServer, field: str) -> object:
    """Read a field from the first request body as it was actually sent.

    Asserted against the wire rather than against the client object on purpose. The output
    ceiling is configured as ``max_tokens`` but langchain-openai 1.4.2 emits it as
    ``max_completion_tokens``, following the current OpenAI API. A test that checked the
    client attribute would have passed while the wire carried something else entirely.
    """
    return stub.behaviour.requests_seen[0].get(field)


# ------------------------------------------------------------------- transient failures


def test_a_transient_error_is_retried_on_the_same_tier(stub: StubServer) -> None:
    """The capable tier failing once must not cost a downgrade."""
    stub.behaviour.fail_first_n = 1
    chain = build_resilient_model(settings_for(stub, max_retries=1), CallClass.RESEARCH)

    reply = chain.invoke("a question")

    assert reply.content
    assert models_called(stub) == [CAPABLE, CAPABLE]


def test_a_rate_limit_is_retried_rather_than_treated_as_fatal(stub: StubServer) -> None:
    """429 is the most common real failure and the most clearly transient."""
    stub.behaviour.fail_first_n = 1
    stub.behaviour.status_for_failures = 429
    chain = build_resilient_model(settings_for(stub, max_retries=1), CallClass.RESEARCH)

    reply = chain.invoke("a question")

    assert reply.content
    assert stub.behaviour.request_count == 2


def test_retries_are_bounded_by_configuration(stub: StubServer) -> None:
    """Each attempt is a paid call, so the ceiling is honoured exactly.

    max_retries=2 means three attempts on the capable tier; the fourth request is the cheap
    tier taking over.
    """
    stub.behaviour.fail_first_n = 3
    chain = build_resilient_model(settings_for(stub, max_retries=2), CallClass.RESEARCH)

    chain.invoke("a question")

    assert models_called(stub) == [CAPABLE, CAPABLE, CAPABLE, CHEAP]


# ------------------------------------------------------------------- durable failures


def test_an_exhausted_tier_falls_back_to_the_cheaper_one(stub: StubServer) -> None:
    """The point of the fallback: a degraded answer beats no answer."""
    stub.behaviour.fail_first_n = 1
    chain = build_resilient_model(settings_for(stub, max_retries=0), CallClass.RESEARCH)

    reply = chain.invoke("a question")

    assert reply.content
    assert models_called(stub) == [CAPABLE, CHEAP]


def test_the_fallback_is_not_used_when_the_first_tier_answers(stub: StubServer) -> None:
    """A fallback that fires unnecessarily doubles the cost of every healthy call."""
    chain = build_resilient_model(settings_for(stub, max_retries=2), CallClass.RESEARCH)

    chain.invoke("a question")

    assert models_called(stub) == [CAPABLE]


def test_both_tiers_failing_surfaces_an_error_rather_than_an_empty_answer(
    stub: StubServer,
) -> None:
    """Silently returning nothing would let a graph proceed on a non-answer."""
    stub.behaviour.fail_first_n = 99
    chain = build_resilient_model(settings_for(stub, max_retries=0), CallClass.RESEARCH)

    with pytest.raises(Exception, match=r"(?i)error|500"):
        chain.invoke("a question")

    assert models_called(stub) == [CAPABLE, CHEAP]


# ------------------------------------------------------------------- call-class budgets


@pytest.mark.parametrize(
    "call_class", [CallClass.ROUTING, CallClass.CLASSIFICATION, CallClass.SYNTHESIS]
)
def test_the_output_ceiling_travels_with_the_call_class(
    stub: StubServer, call_class: CallClass
) -> None:
    """The cap has to reach the provider, not just exist in configuration.

    A routing decision arriving with a synthesis-sized ceiling is the budget guard failing
    quietly -- nothing errors, the calls just cost several times what they should.
    """
    settings = settings_for(stub, max_retries=0)

    build_resilient_model(settings, call_class).invoke("do the thing")

    assert on_the_wire(stub, "max_completion_tokens") == settings.max_tokens_for(call_class)


def test_temperature_reaches_the_provider_as_zero(stub: StubServer) -> None:
    """Determinism is a property of the request, not an intention in the config file."""
    build_resilient_model(settings_for(stub, max_retries=0), CallClass.ROUTING).invoke("x")

    assert on_the_wire(stub, "temperature") == 0.0
