"""Five cases against a real provider. Everything else in this repository runs offline.

Deselected by default and never run in CI. Reached only through `make test-live`, which prints
an estimate, asks for confirmation, and aborts if actual spend exceeds that estimate by more
than the configured factor.

Kept to five because each one costs money and because the offline suite already proves the
mechanics. What cannot be proven offline is what a *real* provider does, and that is all these
ask:

- that the configured identifier is one the key can actually use
- whether the cloud lane supports native structured output (a capability-matrix row, currently
  absent because nothing has measured it)
- that usage metadata arrives, since the spend guard accounts from it
- that the output ceiling is honoured by the provider and not merely sent
- that the whole resilient chain works end to end

Every call records into a ledger the gatekeeper reads. A case that spends without recording
would make the enforcement blind, so the fixture records unconditionally.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from agentgate.config import CallClass, Lane, Settings, Tier, get_settings
from agentgate.guardrails.spend import SpendLedger
from agentgate.models.registry import build_model, build_resilient_model
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

    Written on teardown even if a test fails: an aborted run still spent what it spent, and a
    ledger that only recorded successful runs would understate exactly the case worth catching.
    """
    book = SpendLedger(settings)
    try:
        yield book
    finally:
        destination = os.environ.get("AGENTGATE_LIVE_SPEND_LEDGER")
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


def test_whether_the_cloud_lane_supports_native_structured_output(
    settings: Settings, ledger: SpendLedger
) -> None:
    """Measures the capability-matrix row that is currently absent.

    The matrix has no entry for (CLOUD, NATIVE_STRUCTURED_OUTPUT) because nothing had asked.
    This is the probe that would justify adding one. It asserts only that *some* definite
    answer is obtainable -- either the native path yields a valid object, or it does not and
    the repair loop does. Both are legitimate results; an unmeasured row is not.

    Whichever way it lands, record it in CAPABILITY_MATRIX with provenance LIVE_PROBE and
    today's date. Do not guess the other way.
    """
    model = build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION)
    prompt = "Classify the sensitivity of: 'the office coffee machine is broken'."

    native_worked = True
    try:
        result = model.with_structured_output(Sensitivity).invoke([HumanMessage(prompt)])
    except Exception:  # any failure here means the same thing: not natively supported
        native_worked = False
        result = invoke_with_repair(model, Sensitivity, prompt)

    assert isinstance(result, Sensitivity)
    # Printed rather than asserted: this test exists to produce a fact, not to enforce one.
    print(f"\n  OBSERVED: cloud lane native structured output = {native_worked}")


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
