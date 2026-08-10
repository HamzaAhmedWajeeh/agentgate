"""The spend guard is the only thing standing between a loop and a bill.

Tested for the properties that make it a guard rather than a gauge: it refuses to treat an
unmeasured call as free, it raises rather than reporting, and the session total catches the
shape of failure a per-run ceiling structurally cannot.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agentgate.config import Settings
from agentgate.errors import ConfigurationError
from agentgate.guardrails.spend import (
    Ceilings,
    MissingUsageError,
    SpendCeilingExceededError,
    SpendLedger,
    Usage,
    usage_of,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

MODEL = "a-priced-model"


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "cloud_capable_model": MODEL,
        "cloud_cheap_model": MODEL,
        "model_prices_usd_per_million": {MODEL: {"input": 1.0, "output": 10.0}},
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def ledger_for(settings: Settings, session: SpendLedger | None = None) -> SpendLedger:
    """A ledger accounting one run, which is what most of these cases are about."""
    return SpendLedger(settings, Ceilings.for_run(settings), session=session)


def reply(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


# --------------------------------------------------------------------------- usage


def test_a_reply_without_usage_is_an_error_not_a_zero() -> None:
    """A silent zero disarms every ceiling built on top of it.

    If a provider stopped reporting usage, treating that as free would make every run look
    costless and no limit would ever be reached -- while real money was spent.
    """
    with pytest.raises(MissingUsageError, match="refusing to treat"):
        usage_of(AIMessage(content="no usage here"))


def test_usage_adds_by_direction() -> None:
    assert (Usage(10, 2) + Usage(5, 3)) == Usage(15, 5)


# --------------------------------------------------------------------------- pricing


def test_input_and_output_are_priced_separately() -> None:
    """Output is typically several times input; averaging them understates every run."""
    ledger = ledger_for(settings_with())
    ledger.record(MODEL, reply(1_000_000, 0))
    input_only = ledger.total_usd

    ledger_out = ledger_for(settings_with())
    ledger_out.record(MODEL, reply(0, 1_000_000))

    assert input_only == pytest.approx(1.0)
    assert ledger_out.total_usd == pytest.approx(10.0)


def test_spend_accumulates_across_calls() -> None:
    ledger = ledger_for(settings_with())

    ledger.record(MODEL, reply(500_000, 0))
    ledger.record(MODEL, reply(500_000, 0))

    assert ledger.total_usd == pytest.approx(1.0)
    assert ledger.calls == 2


def test_a_model_with_no_price_refuses_rather_than_costing_nothing() -> None:
    ledger = ledger_for(settings_with())
    ledger.record("a-model-nobody-priced", reply(1000, 1000))

    with pytest.raises(ConfigurationError, match="refusing to guess"):
        _ = ledger.total_usd


# --------------------------------------------------------------------------- enforcement


def test_crossing_the_run_ceiling_raises() -> None:
    """Checked, not observed. A ledger that only recorded would be a gauge, not a guard."""
    ledger = ledger_for(
        settings_with(max_spend_usd=0.5, max_session_spend_usd=100.0, max_total_tokens=10_000_000)
    )
    ledger.record(MODEL, reply(1_000_000, 0))

    with pytest.raises(SpendCeilingExceededError, match="over the ceiling") as caught:
        ledger.check()

    assert caught.value.scope == "run"
    assert caught.value.spent_usd == pytest.approx(1.0)


def test_staying_under_the_ceiling_does_not_raise() -> None:
    ledger = ledger_for(settings_with(max_spend_usd=10.0, max_session_spend_usd=100.0))
    ledger.record(MODEL, reply(1_000, 1_000))

    ledger.check()


def test_the_token_ceiling_trips_independently_of_cost() -> None:
    """A cheap model can still run away; tokens bound that even when dollars do not."""
    ledger = ledger_for(
        settings_with(max_total_tokens=1_000, max_spend_usd=1000.0, max_session_spend_usd=1000.0)
    )
    ledger.record(MODEL, reply(2_000, 0))

    with pytest.raises(SpendCeilingExceededError) as caught:
        ledger.check()

    assert caught.value.scope == "run_tokens"


def test_the_session_ceiling_catches_a_loop_of_individually_cheap_runs() -> None:
    """The failure a per-run ceiling structurally cannot see.

    Each run stays comfortably under its own limit; together they are the bill.
    """
    settings = settings_with(
        max_spend_usd=1.0, max_session_spend_usd=2.0, max_total_tokens=10_000_000
    )
    session = ledger_for(settings)

    for _ in range(3):
        run = ledger_for(settings, session=session)
        run.record(MODEL, reply(900_000, 0))  # $0.90, under the run ceiling every time
        if session.total_usd > settings.max_session_spend_usd:
            break

    run = ledger_for(settings, session=session)
    run.record(MODEL, reply(1, 0))

    with pytest.raises(SpendCeilingExceededError) as caught:
        run.check()

    assert caught.value.scope == "session"


def test_a_run_rolls_up_into_its_session() -> None:
    settings = settings_with()
    session = ledger_for(settings)
    run = ledger_for(settings, session=session)

    run.record(MODEL, reply(1000, 1000))

    assert session.total_tokens == run.total_tokens
    assert session.calls == 1


# --------------------------------------------------------------------------- scope


def test_the_live_suite_is_not_accounted_against_the_run_ceiling() -> None:
    """A suite is not a run, so the run ceiling is the wrong instrument to measure it with.

    Six independent cases sharing one book will pass any per-run budget sized for one request
    through the graph. Charged against it, the suite aborts for being a suite -- a budget
    failure where nothing is wrong, whose obvious remedy is to raise the run ceiling, which
    weakens the guard that was working.
    """
    settings = settings_with(
        max_total_tokens=1_000,
        max_spend_usd=0.0001,
        max_session_spend_usd=1_000.0,
        max_live_suite_tokens=100_000,
        max_live_suite_spend_usd=1_000.0,
    )
    suite = SpendLedger(settings, Ceilings.for_live_suite(settings))

    suite.record(MODEL, reply(20_000, 0))

    suite.check()  # both run ceilings are far behind us, and neither one applies


def test_the_live_suite_trips_its_own_token_ceiling() -> None:
    """Its own ceiling, and it is a real one -- separating the scopes must not disarm it."""
    settings = settings_with(max_live_suite_tokens=1_000, max_live_suite_spend_usd=1_000.0)
    suite = SpendLedger(settings, Ceilings.for_live_suite(settings))
    suite.record(MODEL, reply(2_000, 0))

    with pytest.raises(SpendCeilingExceededError, match="live_suite consumed") as caught:
        suite.check()

    assert caught.value.scope == "live_suite_tokens"


def test_the_abort_figure_the_operator_consented_to_tightens_the_suite_ceiling() -> None:
    """The gatekeeper's estimate times the tolerance, enforced during the suite.

    It used to be exported into the subprocess environment and read by nothing, so the only
    enforcement happened after every case had already run and spent. A threshold that can only
    be crossed in hindsight stops nothing.
    """
    settings = settings_with(max_live_suite_spend_usd=10.0, live_spend_abort_usd=0.5)

    assert Ceilings.for_live_suite(settings).max_spend_usd == pytest.approx(0.5)


def test_the_configured_suite_ceiling_wins_when_it_is_the_lower_of_the_two() -> None:
    """Tightening only. The gatekeeper's figure may not raise a ceiling set in configuration."""
    settings = settings_with(max_live_suite_spend_usd=0.25, live_spend_abort_usd=5.0)

    assert Ceilings.for_live_suite(settings).max_spend_usd == pytest.approx(0.25)


# --------------------------------------------------------------------------- reporting


def test_the_summary_shows_each_model_and_a_total() -> None:
    ledger = ledger_for(settings_with())
    ledger.record(MODEL, reply(1000, 500))

    summary = ledger.summary()

    assert MODEL in summary
    assert "TOTAL" in summary
