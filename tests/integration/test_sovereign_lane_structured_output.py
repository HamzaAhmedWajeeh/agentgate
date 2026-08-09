"""Where the provider abstraction leaks, and what catches it.

The cloud lane constrains decoding and returns an object. A self-hosted OpenAI-compatible
endpoint generally does not: asked for JSON it returns the right object wrapped in prose and a
code fence. One interface, two behaviours -- and the difference is invisible to the node doing
the classifying, which is exactly why it has to be handled here rather than discovered later.

These run against a real stub server on a real socket, reached through the real client library
by `base_url`, so what is asserted is what the plumbing actually does.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.models.structured import (
    StructuredOutputError,
    invoke_structured,
    invoke_with_repair,
)


class Sensitivity(BaseModel):
    """The classifier's schema, in miniature."""

    sensitivity: str
    confidence: float = Field(ge=0.0, le=1.0)


VERDICT = {"sensitivity": "restricted", "confidence": 0.82}


@pytest.fixture
def stub() -> Iterator[StubServer]:
    """A sovereign-lane stand-in: OpenAI dialect, no native structured output."""
    with running_stub(StubBehaviour(reply=VERDICT)) as server:
        yield server


def model_for(stub: StubServer) -> ChatOpenAI:
    """A client pointed at the stub exactly as the sovereign lane points at a local server.

    The model identifier is whatever the endpoint answers to. No real model name appears
    here, in keeping with the rule that identifiers are operator input rather than code.
    """
    return ChatOpenAI(
        model="stub",
        base_url=stub.base_url,
        api_key="not-required",
        temperature=0,
    )


# ------------------------------------------------------------------ the leak, demonstrated


def test_native_structured_output_fails_against_the_sovereign_lane(stub: StubServer) -> None:
    """The documented difference between lanes, asserted rather than assumed.

    This endpoint ignores `response_format` and answers with prose-wrapped JSON. The native
    path expects the body to be the object, so it fails. If this ever starts passing, the
    capability matrix is out of date and the repair loop is dead weight.
    """
    with pytest.raises(ValidationError, match=r"(?i)invalid json"):
        model_for(stub).with_structured_output(Sensitivity).invoke("classify this request")


# ------------------------------------------------------------------ the repair loop rescues it


def test_repair_loop_rescues_prose_wrapped_json_from_the_sovereign_lane(
    stub: StubServer,
) -> None:
    """The named case. A lane that cannot do native structured output still yields an object.

    This is the whole justification for the validate-and-repair fallback existing, proved
    against a server that really behaves this way rather than a mock that we told to.
    """
    result = invoke_with_repair(model_for(stub), Sensitivity, "classify this request")

    assert isinstance(result, Sensitivity)
    assert result.sensitivity == "restricted"
    assert result.confidence == pytest.approx(0.82)


def test_repair_succeeds_on_the_first_attempt_without_a_second_call(
    stub: StubServer,
) -> None:
    """Extraction is tried before repair. Prose-wrapped but valid JSON must not cost two calls.

    Getting this wrong doubles the token cost of every classification on this lane.
    """
    invoke_with_repair(model_for(stub), Sensitivity, "classify this request")

    assert stub.behaviour.request_count == 1


def test_dispatch_by_capability_routes_this_lane_to_repair(stub: StubServer) -> None:
    """`native=False` is what the capability matrix records for this lane."""
    result = invoke_structured(model_for(stub), Sensitivity, "classify this request", native=False)

    assert result.sensitivity == "restricted"


def test_a_lane_wrongly_recorded_as_native_still_recovers(stub: StubServer) -> None:
    """The matrix holds observations, and observations go stale when a provider changes.

    Claiming native support this lane does not have should degrade to the repair loop, not
    surface a provider error to a node that has no idea what a lane is.
    """
    result = invoke_structured(model_for(stub), Sensitivity, "classify this request", native=True)

    assert result.sensitivity == "restricted"


# ------------------------------------------------------------------ when repair cannot win


def test_repair_gives_up_and_says_what_it_saw() -> None:
    """A model that never complies must fail loudly and diagnosably, not loop forever."""
    behaviour = StubBehaviour(reply={"wrong_field": "nothing like the schema"})
    with running_stub(behaviour) as server, pytest.raises(StructuredOutputError) as caught:
        invoke_with_repair(model_for(server), Sensitivity, "classify this request", max_attempts=2)

    assert caught.value.attempts == 2
    assert "wrong_field" in caught.value.raw
    assert behaviour.request_count == 2


def test_repair_attempts_are_bounded_by_the_budget() -> None:
    """Each repair is another paid call, so the ceiling is honoured exactly."""
    behaviour = StubBehaviour(reply={"nope": True})
    with running_stub(behaviour) as server, pytest.raises(StructuredOutputError):
        invoke_with_repair(model_for(server), Sensitivity, "classify this", max_attempts=3)

    assert behaviour.request_count == 3


def test_the_repair_prompt_quotes_the_validation_error_back() -> None:
    """Repeating the bare instruction produces the same wrong answer again."""
    behaviour = StubBehaviour(reply={"nope": True})
    with running_stub(behaviour) as server, pytest.raises(StructuredOutputError):
        invoke_with_repair(model_for(server), Sensitivity, "classify this", max_attempts=2)

    second_request = behaviour.requests_seen[1]
    conversation = " ".join(str(m.get("content", "")) for m in second_request["messages"])
    assert "did not validate" in conversation
    assert "sensitivity" in conversation
