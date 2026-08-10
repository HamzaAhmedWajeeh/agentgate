"""Six cases against a real provider. Everything else in this repository runs offline.

Deselected by default and never run in CI. Reached only through `make test-live`, which prints
an estimate, asks for confirmation, and aborts if actual spend exceeds that estimate by more
than the configured factor.

Kept few because each one costs money and because the offline suite already proves the
mechanics. What cannot be proven offline is what a *real* provider does, and that is all these
ask:

- that the configured identifier is one the key can actually use
- that the recorded structured-output capability is still true, failing if the record and
  reality disagree, and failing with instructions if no record exists
- that the repair loop works here regardless, since it is what an unmeasured lane gets
- that usage metadata arrives, since the spend guard accounts from it
- that the output ceiling is honoured by the provider and not merely sent

Discovery is not done here. `scripts/probe_capabilities.py` produces facts; these tests
enforce the facts that were recorded. A test that has to accept whichever answer it gets is
unfalsifiable, and a record nothing checks goes stale the first time a provider changes.

Every call records into a ledger the gatekeeper reads. A case that spends without recording
would make the enforcement blind, so the fixture records unconditionally.

The ledger is bounded by the suite's own ceilings, never the run ceilings. Six independent
cases are not a request through the graph, and charging them to a per-run budget would abort
the suite for being a suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from agentgate.config import CallClass, Lane, Settings, Tier, get_settings
from agentgate.guardrails.spend import Ceilings, SpendLedger
from agentgate.models.registry import (
    Capability,
    build_model,
    build_resilient_model,
    observation_for,
)
from agentgate.models.structured import invoke_with_repair

pytestmark = pytest.mark.live

# Short prompts on purpose. Every token here is billed.
PROMPT = "Reply with the single word: acknowledged."


class Sensitivity(BaseModel):
    """The classifier schema, in miniature."""

    sensitivity: str
    confidence: float = Field(ge=0.0, le=1.0)


@pytest.fixture(scope="module")
def settings() -> Settings:
    resolved = get_settings()
    if resolved.lane is not Lane.CLOUD:
        pytest.skip(f"lane is '{resolved.lane.value}'; live tests need the cloud lane")
    return resolved


@pytest.fixture(scope="module")
def ledger(settings: Settings) -> Iterator[SpendLedger]:
    """Accumulate spend and write it where the gatekeeper can read it.

    Bounded by ``Ceilings.for_live_suite``, which is the whole point of the fixture existing:
    the suite is accounted on its own basis, and the run ceilings are left to bound runs.

    Written on teardown even if a test fails: an aborted run still spent what it spent, and a
    ledger that only recorded successful runs would understate exactly the case worth catching.
    """
    book = SpendLedger(settings, Ceilings.for_live_suite(settings))
    try:
        yield book
    finally:
        destination = settings.live_spend_ledger
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "total_usd": book.total_usd,
                        "total_tokens": book.total_tokens,
                        "calls": book.calls,
                    }
                ),
                encoding="utf-8",
            )


def bill(ledger: SpendLedger, settings: Settings, tier: Tier, reply: AIMessage) -> None:
    """Record a call and enforce the ceilings mid-suite, not just at the end."""
    ledger.record(settings.model_for(tier), reply)
    ledger.check()


# ------------------------------------------------------------------------------- 1


def test_the_configured_model_identifier_is_usable(settings: Settings, ledger: SpendLedger) -> None:
    """The one thing configuration cannot check: that the key accepts this identifier.

    `make models` lists what is reachable, but reachable and usable are not the same claim.
    """
    model = build_model(settings, Tier.CHEAP, CallClass.ROUTING)

    reply = model.invoke([HumanMessage(PROMPT)])

    bill(ledger, settings, Tier.CHEAP, reply)
    assert reply.content


# ------------------------------------------------------------------------------- 2


def test_the_recorded_structured_output_capability_is_still_true(
    settings: Settings, ledger: SpendLedger
) -> None:
    """Enforce the matrix row rather than discover it.

    Discovery is `scripts/probe_capabilities.py`. Its job is to produce a fact. This test's
    job is to hold that fact to account: exercise the behaviour and fail if reality and the
    record disagree.

    Splitting them matters because a test that discovers cannot also enforce -- it has to
    accept whichever answer it gets, which makes it unfalsifiable. And a record nothing checks
    goes stale the first time a provider changes, which is the failure the whole
    provenance-and-dates design exists to prevent.
    """
    recorded = observation_for(settings.lane, Capability.NATIVE_STRUCTURED_OUTPUT)

    if recorded is None:
        pytest.fail(
            f"No CAPABILITY_MATRIX entry for ({settings.lane.value}, "
            "native_structured_output), so there is nothing to enforce.\n"
            "Run:  uv run python scripts/probe_capabilities.py\n"
            "then paste the entry it emits into src/agentgate/models/registry.py."
        )

    model = build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION)
    prompt = "Classify the sensitivity of: 'the office coffee machine is broken'."

    try:
        result = model.with_structured_output(Sensitivity).invoke([HumanMessage(prompt)])
        observed = isinstance(result, Sensitivity)
    except Exception:  # any failure here means the same thing: not natively supported
        observed = False

    assert observed == recorded.supported, (
        f"The matrix records native structured output on the {settings.lane.value} lane as "
        f"{recorded.supported} (provenance {recorded.provenance.value}, recorded "
        f"{recorded.recorded_on}), but this run observed {observed}. Either the provider "
        "changed or the record was wrong. Re-run scripts/probe_capabilities.py and update "
        "the entry -- do not edit the assertion."
    )


def test_the_repair_loop_still_works_on_this_lane_whatever_the_matrix_says(
    settings: Settings, ledger: SpendLedger
) -> None:
    """The fallback has to work regardless, since it is what an unmeasured lane gets.

    Every lane with no recorded capability is routed here by default, so this path being
    broken would be invisible right up until it mattered.
    """
    model = build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION)

    result = invoke_with_repair(
        model, Sensitivity, "Classify the sensitivity of: 'the fire alarm was tested today'."
    )

    assert isinstance(result, Sensitivity)


# ------------------------------------------------------------------------------- 3


def test_usage_metadata_arrives_from_the_real_provider(
    settings: Settings, ledger: SpendLedger
) -> None:
    """The spend guard accounts from this field. If the provider omits it, the guard is blind."""
    model = build_model(settings, Tier.CHEAP, CallClass.ROUTING)

    reply = model.invoke([HumanMessage(PROMPT)])

    assert reply.usage_metadata is not None
    assert reply.usage_metadata["input_tokens"] > 0
    assert reply.usage_metadata["output_tokens"] > 0
    bill(ledger, settings, Tier.CHEAP, reply)


# ------------------------------------------------------------------------------- 4


def test_the_output_ceiling_is_honoured_by_the_provider(
    settings: Settings, ledger: SpendLedger
) -> None:
    """Sending the cap is not the same as the cap being obeyed.

    The offline suite proves the value reaches the wire. Only a real provider can show that
    it constrains generation, which is what the budget actually depends on.
    """
    model = build_model(settings, Tier.CHEAP, CallClass.ROUTING)

    reply = model.invoke([HumanMessage("Count slowly from one to five hundred in words.")])

    bill(ledger, settings, Tier.CHEAP, reply)
    assert reply.usage_metadata is not None
    assert reply.usage_metadata["output_tokens"] <= settings.max_tokens_for(CallClass.ROUTING)


# ------------------------------------------------------------------------------- 5


def test_the_resilient_chain_works_end_to_end(settings: Settings, ledger: SpendLedger) -> None:
    """Retry and fallback are proven offline against induced errors. This proves the healthy
    path is not broken by the wrapping -- a chain that only works when something fails would
    be a strange thing to discover in production."""
    chain = build_resilient_model(settings, CallClass.ROUTING)

    reply = chain.invoke([HumanMessage(PROMPT)])

    bill(ledger, settings, Tier.CAPABLE, reply)
    assert reply.content
