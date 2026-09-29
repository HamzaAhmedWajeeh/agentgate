"""The rule a retry policy needs and does not come with: a guard is not a transient fault.

``with_retry`` retries on every exception by default. This system raises its own refusals as
exceptions too -- a crossed spend ceiling, a reply with no usage to account, a lane that cannot
be built -- and several of them are raised from inside a callback on the call that triggered
them. To a retry policy that reads every exception as "the provider is having a bad minute",
a budget ceiling is indistinguishable from a 500, and the answer to it is to call the provider
again. Leak inventory item 27.

Asserted here at the level of the rule rather than of one guard. The ceiling is the instance
that was found, and `tests/integration/test_resilience_wiring.py` pins it end to end through the
graph; what this file holds is the general shape, so a second guard added later inherits it
instead of having to rediscover it.

Every case pairs the absence with a presence. "It was not retried" is true of a chain that
never retries anything, so each refusal case is run against the same model and the same
settings as a provider failure that *is* retried.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentgate.errors import AgentgateError
from agentgate.guardrails.spend import MissingUsageError, SpendCeilingExceededError
from agentgate.models.registry import LaneUnavailableError
from agentgate.models.resilient import ResilientChatModel


class ScriptedFailure(BaseChatModel):
    """A leaf that raises what it is told to, and counts how often it was asked.

    A counting double rather than a stub server: the question here is how many times the
    composite *called* its leaf, which is a fact about the composite. The wire assertions
    belong to the integration tests, where what reached a provider is the thing in doubt.
    """

    error: Exception | None = None
    on_call: Any = None
    calls: list[str] = []  # noqa: RUF012 - replaced per instance below
    label: str = "leaf"

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.calls = []

    @property
    def _llm_type(self) -> str:
        return "scripted-failure"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002 - signature imposed by BaseChatModel
        stop: list[str] | None = None,  # noqa: ARG002 - ditto
        run_manager: Any = None,  # noqa: ARG002 - ditto
        **kwargs: Any,  # noqa: ARG002 - ditto
    ) -> ChatResult:
        self.calls.append(self.label)
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.label))])


class ProviderFailedError(Exception):
    """Stands in for a 500 or a reset connection: something retrying might fix."""


def chain(primary: BaseChatModel, fallback: BaseChatModel, *, attempts: int) -> ResilientChatModel:
    return ResilientChatModel(primary=primary, fallback=fallback, max_attempts=attempts)


def ceiling_crossed() -> SpendCeilingExceededError:
    """The refusal the ledger raises, with the figures it really carries."""
    return SpendCeilingExceededError(
        "run consumed 2627 tokens, over the ceiling of 2029",
        spent_usd=0.42,
        ceiling_usd=0.30,
        scope="run",
    )


REFUSALS: list[Exception] = [
    ceiling_crossed(),
    MissingUsageError("a reply arrived without usage metadata"),
    LaneUnavailableError("sovereign lane requested but no base URL is configured"),
]


@pytest.mark.parametrize("refusal", REFUSALS, ids=lambda error: type(error).__name__)
def test_a_refusal_by_this_systems_own_guards_is_attempted_once(refusal: Exception) -> None:
    """One attempt, no fallback, and the refusal reaches the caller unchanged.

    The fallback matters as much as the retry. A guard that stopped the capable tier and then
    let the cheap one answer would have been overruled just as thoroughly, and more quietly --
    the run would carry on, on a model nobody chose, past a ceiling that had already tripped.
    """
    assert isinstance(refusal, AgentgateError), "precondition: these are the deliberate refusals"
    primary = ScriptedFailure(label="primary", error=refusal)
    fallback = ScriptedFailure(label="fallback")

    with pytest.raises(type(refusal)):
        chain(primary, fallback, attempts=3).invoke("a question")

    assert primary.calls == ["primary"], (
        f"a refusal was retried {len(primary.calls)} time(s). Every attempt after the first is "
        "a real, billed call made after this system had already decided to stop"
    )
    assert fallback.calls == [], (
        "the refusal was answered by the fallback, so the guard stopped one model and the run "
        "continued on another"
    )


def test_a_provider_failure_is_retried_and_falls_back() -> None:
    """The control, and it is not decoration.

    Without it, every assertion above holds just as well against a composite that retries
    nothing at all -- which is precisely the state this repository spent four phases in.
    """
    primary = ScriptedFailure(label="primary", error=ProviderFailedError("500"))
    fallback = ScriptedFailure(label="fallback")

    reply = chain(primary, fallback, attempts=3).invoke("a question")

    assert primary.calls == ["primary"] * 3, (
        f"the provider failure was attempted {len(primary.calls)} time(s) of 3 configured; "
        "this chain does not retry, so the refusal assertions above prove nothing"
    )
    assert fallback.calls == ["fallback"], "the exhausted primary did not reach the fallback"
    assert reply.content == "fallback"


def test_a_refusal_part_way_through_a_chain_stops_it_where_it_stands() -> None:
    """The realistic shape, and the one a first-attempt test cannot reach.

    A ceiling is crossed *by* a call, so the run that crosses it is usually not its first: the
    provider hiccups, the retry succeeds, and the reply that comes back is the one that takes
    the ledger over. The chain has attempts left and a fallback below it at that moment, and
    both must go unused.

    Written this way after mutation-checking showed the obvious version -- a refusal raised by
    the fallback, the last attempt of all -- could not fail. "It was not retried" is free when
    there is nothing left to retry with, so that test passed with the fix removed and pinned
    nothing at all.
    """
    primary = ScriptedFailure(label="primary", error=ProviderFailedError("500"))
    fallback = ScriptedFailure(label="fallback")
    chain_under_test = chain(primary, fallback, attempts=3)

    # The first attempt is a provider failure; the second crosses the ceiling. The hook runs
    # after the call is recorded, so the count it reads includes the call in progress.
    def fail_then_refuse() -> None:
        if len(primary.calls) >= 2:
            primary.error = ceiling_crossed()

    primary.on_call = fail_then_refuse

    with pytest.raises(SpendCeilingExceededError):
        chain_under_test.invoke("a question")

    assert primary.calls == ["primary"] * 2, (
        f"the primary was called {len(primary.calls)} time(s). It had three attempts: one "
        "provider failure, then a refusal that should have ended the chain with one unused"
    )
    assert fallback.calls == [], (
        "the refusal was answered by the fallback. The guard stopped one model and the run "
        "continued on another, past a ceiling that had already tripped"
    )


def test_the_same_rule_holds_on_the_streaming_path() -> None:
    """The CLI streams every call, so a rule that only held for ``invoke`` would hold rarely.

    ``_stream`` has its own attempt loop -- it has to, because a stream that has begun cannot
    be retried without splicing two replies together -- and a second loop is a second place to
    forget this.
    """
    primary = ScriptedFailure(label="primary", error=ceiling_crossed())
    fallback = ScriptedFailure(label="fallback")

    with pytest.raises(SpendCeilingExceededError):
        list(chain(primary, fallback, attempts=3).stream("a question"))

    assert primary.calls == ["primary"], f"retried {len(primary.calls)} time(s) while streaming"
    assert fallback.calls == [], "the refusal was answered by the fallback while streaming"


def test_a_provider_failure_still_falls_back_while_streaming() -> None:
    """The streaming control, for the same reason the invoke one exists."""
    primary = ScriptedFailure(label="primary", error=ProviderFailedError("500"))
    fallback = ScriptedFailure(label="fallback")

    chunks = list(chain(primary, fallback, attempts=2).stream("a question"))

    assert primary.calls == ["primary"] * 2, "the streaming path does not retry"
    assert fallback.calls == ["fallback"], "the streaming path does not fall back"
    assert "".join(str(chunk.content) for chunk in chunks) == "fallback"
