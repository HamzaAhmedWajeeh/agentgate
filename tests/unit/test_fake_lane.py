"""The fake lane is load-bearing: every other test trusts it.

If it is not deterministic, failures elsewhere become intermittent and unexplainable. If its
usage figures are wrong, the spend guard is tested against a fiction. So it gets tested
directly rather than only through the things that use it.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agentgate.models.fake import (
    FakeChatModel,
    ScriptedFailureError,
    estimate_tokens,
    failing_after,
    scripted_json,
)


class Verdict(BaseModel):
    """Minimal schema, used only to drive with_structured_output."""

    sensitivity: str
    confidence: float


# --------------------------------------------------------------------------- determinism


def test_scripted_responses_are_returned_in_order() -> None:
    model = FakeChatModel(responses=["first", "second"])

    assert model.invoke([HumanMessage("a")]).content == "first"
    assert model.invoke([HumanMessage("b")]).content == "second"


def test_unscripted_replies_are_stable_across_instances() -> None:
    """Two fresh models given the same conversation must answer identically.

    Without this, a test that under-scripts passes locally and fails in CI for reasons that
    look like the code under test.
    """
    first = FakeChatModel(model_name="x")
    second = FakeChatModel(model_name="x")

    conversation = [HumanMessage("what is the policy on this?")]

    assert first.invoke(conversation).content == second.invoke(conversation).content


def test_unscripted_replies_differ_by_conversation() -> None:
    model = FakeChatModel()

    one = model.invoke([HumanMessage("question one")]).content
    two = model.invoke([HumanMessage("question two")]).content

    assert one != two


def test_running_out_of_script_does_not_raise() -> None:
    """An under-scripted test should get a stable answer, not an IndexError elsewhere."""
    model = FakeChatModel(responses=["only one"])

    model.invoke([HumanMessage("a")])
    overflow = model.invoke([HumanMessage("b")])

    assert "unscripted" in str(overflow.content)


def test_a_scripted_message_is_not_mutated_by_use() -> None:
    """Reusing one AIMessage across two models must not leak usage between them."""
    shared = AIMessage(content="shared")
    first = FakeChatModel(responses=[shared])
    second = FakeChatModel(responses=[shared])

    first.invoke([HumanMessage("a")])
    second.invoke([HumanMessage("a longer prompt entirely")])

    assert shared.usage_metadata is None


# --------------------------------------------------------------------------- usage accounting


def test_every_reply_reports_usage() -> None:
    """The spend guard accounts from this field. A model that omits it cannot test the guard."""
    model = FakeChatModel(responses=["a reply"])

    reply = model.invoke([HumanMessage("a prompt")])

    assert reply.usage_metadata is not None
    assert reply.usage_metadata["input_tokens"] > 0
    assert reply.usage_metadata["output_tokens"] > 0


def test_usage_totals_are_self_consistent() -> None:
    model = FakeChatModel(responses=["a somewhat longer reply than the prompt"])

    usage = model.invoke([HumanMessage("short")]).usage_metadata

    assert usage is not None
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_longer_prompts_cost_more() -> None:
    model = FakeChatModel(responses=["x", "x"])

    short = model.invoke([HumanMessage("hi")]).usage_metadata
    long = model.invoke([HumanMessage("hi " * 200)]).usage_metadata

    assert short is not None
    assert long is not None
    assert long["input_tokens"] > short["input_tokens"]


@pytest.mark.parametrize(("text", "expected"), [("", 1), ("abcd", 1), ("a" * 400, 100)])
def test_token_estimate_is_stable_and_never_zero(text: str, expected: int) -> None:
    """Zero-token accounting would let an unbounded number of calls pass a spend ceiling."""
    assert estimate_tokens(text) == expected


# --------------------------------------------------------------------------- failure injection


def test_scripted_failure_raises_on_the_named_call() -> None:
    model = FakeChatModel(responses=["ok", "ok"], failures=frozenset({1}))

    assert model.invoke([HumanMessage("a")]).content == "ok"
    with pytest.raises(ScriptedFailureError):
        model.invoke([HumanMessage("b")])


def test_a_failed_call_still_counts() -> None:
    """Retry and fallback logic reasons about attempts, so a failure must be observable."""
    model = FakeChatModel(failures=frozenset({0}))

    with pytest.raises(ScriptedFailureError):
        model.invoke([HumanMessage("a")])

    assert model.call_count == 1


def test_failing_after_models_a_provider_that_degrades() -> None:
    model = FakeChatModel(responses=["a", "b", "c"], failures=failing_after(2, 3))

    assert model.invoke([HumanMessage("1")]).content == "a"
    assert model.invoke([HumanMessage("2")]).content == "b"
    with pytest.raises(ScriptedFailureError):
        model.invoke([HumanMessage("3")])


def test_calls_are_recorded_for_assertions() -> None:
    model = FakeChatModel(responses=["a"])

    model.invoke([HumanMessage("the exact prompt")])

    assert model.call_count == 1
    assert model.calls[0][0].content == "the exact prompt"


# --------------------------------------------------------------------------- structured output


def test_a_lane_can_declare_that_it_lacks_native_structured_output() -> None:
    """This is how a self-hosted endpoint with no structured-output support is represented."""
    model = FakeChatModel(supports_native_structured_output=False)

    with pytest.raises(NotImplementedError, match="validate-and-repair"):
        model.with_structured_output(Verdict)


def test_native_structured_output_parses_a_scripted_reply() -> None:
    model = FakeChatModel(
        responses=[scripted_json({"sensitivity": "restricted", "confidence": 0.91})],
        supports_native_structured_output=True,
    )

    result = model.with_structured_output(Verdict).invoke([HumanMessage("classify this")])

    assert isinstance(result, Verdict)
    assert result.sensitivity == "restricted"
    assert result.confidence == pytest.approx(0.91)


def test_native_structured_output_chokes_on_prose_wrapped_json() -> None:
    """The whole reason the repair loop exists, pinned as a fact rather than an assumption.

    A native structured-output path is strict: it expects the response body to be the object.
    Wrap that object in the conversational padding a model habitually adds and it fails.
    """
    model = FakeChatModel(
        responses=[
            scripted_json({"sensitivity": "public", "confidence": 0.5}, wrapped_in_prose=True)
        ],
        supports_native_structured_output=True,
    )

    with pytest.raises(json.JSONDecodeError):
        model.with_structured_output(Verdict).invoke([HumanMessage("classify this")])


def test_include_raw_reports_the_parse_failure_instead_of_raising() -> None:
    model = FakeChatModel(
        responses=[scripted_json({"sensitivity": "public"}, wrapped_in_prose=True)],
        supports_native_structured_output=True,
    )

    outcome = model.with_structured_output(Verdict, include_raw=True).invoke(
        [HumanMessage("classify this")]
    )

    assert isinstance(outcome, dict)
    assert outcome["parsed"] is None
    assert outcome["parsing_error"] is not None


# --------------------------------------------------------------------------- prose-wrapped JSON


def test_scripted_json_is_bare_by_default() -> None:
    assert json.loads(scripted_json({"a": 1})) == {"a": 1}


def test_scripted_json_can_be_wrapped_the_way_models_actually_reply() -> None:
    """Bare JSON is the case that already works. Prose-wrapped is the case that breaks."""
    wrapped = scripted_json({"a": 1}, wrapped_in_prose=True)

    assert "```json" in wrapped
    with pytest.raises(json.JSONDecodeError):
        json.loads(wrapped)
