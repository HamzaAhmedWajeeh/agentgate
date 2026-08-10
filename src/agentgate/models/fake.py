"""A deterministic scripted chat model.

The entire test suite and all of CI run on this. It never touches the network and never costs
anything, which is the only reason `make test` can be a thing you run without thinking about it.

Three properties matter:

*Deterministic.* The same input produces the same output, every run, on every machine. Where a
response is not scripted, it is derived from a stable hash of the conversation rather than from
anything that varies -- no clocks, no randomness, no iteration order of a set.

*Honest about usage.* Every reply carries ``usage_metadata``, because the spend guard accounts
from that field and a guard tested against a model that reports nothing is not tested at all.

*Able to misbehave on demand.* Real providers time out, rate-limit, and return content that
does not parse. A double that only ever succeeds cannot exercise the retry chain, the fallback
chain, or the repair loop, so failures are scriptable too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any, Final

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

# Deterministic stand-in for a tokeniser. Four characters per token is roughly right for
# English and, more importantly, is stable -- the exact number does not matter as long as
# every run agrees on it.
CHARS_PER_TOKEN: Final = 4


def estimate_tokens(text: str) -> int:
    """Return a stable, deterministic token estimate for a string."""
    return max(1, len(text) // CHARS_PER_TOKEN)


class ScriptedFailureError(RuntimeError):
    """Raised by the fake lane on demand, to exercise retry and fallback paths."""


class FakeChatModel(BaseChatModel):
    """A chat model whose every response is decided before the test runs.

    Args:
        responses: Replies handed out in order. A plain string becomes an ``AIMessage``.
            Once exhausted, replies fall back to a deterministic echo derived from the
            conversation, so a test that under-scripts gets a stable answer rather than an
            ``IndexError`` in an unrelated place.
        failures: Zero-based call indices that raise :class:`ScriptedFailureError` instead of
            replying. Used to drive the retry and fallback chains.
        supports_native_structured_output: When ``False``, :meth:`with_structured_output`
            raises, which is how a lane that cannot do native structured output is simulated
            without standing up a server.
        name: Identifies this instance in :attr:`calls` when several are wired into a chain,
            such as the capable and cheap tiers of a fallback.
    """

    responses: list[str | AIMessage] = Field(default_factory=list)
    failures: frozenset[int] = Field(default_factory=frozenset)
    supports_native_structured_output: bool = True
    model_name: str = "fake"

    # Mutable call log. Excluded from equality and serialisation; it is test scaffolding,
    # not model configuration.
    calls: list[list[BaseMessage]] = Field(default_factory=list, exclude=True)

    bound_tools: list[str] = Field(default_factory=list, exclude=True)
    """Names most recently passed to :meth:`bind_tools`.

    Recorded because "which tools was this model given" is a claim worth being able to check
    directly. An allowlist that is enforced downstream is still worth binding correctly, and a
    test that can only observe the enforcement cannot tell the difference between a tool that
    was withheld and one that was offered and refused."""

    @property
    def _llm_type(self) -> str:
        return "agentgate-fake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @property
    def call_count(self) -> int:
        """How many times this instance has been asked to generate."""
        return len(self.calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002 - signature imposed by BaseChatModel
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002 - ditto
        **kwargs: Any,  # noqa: ARG002 - ditto
    ) -> ChatResult:
        index = len(self.calls)
        self.calls.append(list(messages))

        if index in self.failures:
            msg = f"{self.model_name}: scripted failure on call {index}"
            raise ScriptedFailureError(msg)

        reply = self._reply_for(index, messages)
        prompt_text = "".join(str(message.content) for message in messages)
        reply.usage_metadata = {
            "input_tokens": estimate_tokens(prompt_text),
            "output_tokens": estimate_tokens(str(reply.content)),
            "total_tokens": estimate_tokens(prompt_text) + estimate_tokens(str(reply.content)),
        }
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _reply_for(self, index: int, messages: Sequence[BaseMessage]) -> AIMessage:
        if index < len(self.responses):
            scripted = self.responses[index]
            # Copied, so a test reusing the same AIMessage across two models does not see
            # usage_metadata from one run leak into the assertions of another.
            return (
                AIMessage(content=scripted)
                if isinstance(scripted, str)
                else scripted.model_copy(deep=True)
            )
        return AIMessage(content=self._deterministic_echo(messages))

    def _deterministic_echo(self, messages: Sequence[BaseMessage]) -> str:
        """Derive a reply from the conversation, stably across runs and machines."""
        transcript = "\n".join(f"{m.type}:{m.content}" for m in messages)
        digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:12]
        return f"[{self.model_name}] unscripted reply {digest}"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept a tool binding, record what was bound, and otherwise stay scripted.

        ``BaseChatModel.bind_tools`` raises by default, so without this the fake lane cannot
        be used with ``create_agent`` at all -- which is how this arrived: the drafter is the
        first thing here to build an agent, and the model the whole suite runs on could not be
        given tools.

        The binding does not change what the model replies. Tool *calls* are scripted the same
        way everything else is, as an ``AIMessage`` carrying ``tool_calls``. That keeps the
        double honest about the one thing it is for: a test that wants the model to demand an
        irreversible tool says so explicitly, rather than hoping a real model would.
        """
        self.bound_tools = [
            str(getattr(item, "name", None) or getattr(item, "__name__", None) or item)
            for item in tools
        ]
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,  # noqa: ARG002 - signature imposed by BaseChatModel
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Return a structured-output runnable, or refuse if this lane cannot do it natively.

        Refusing is the interesting case. It is how a lane whose endpoint has no native
        structured-output support is represented, so the validate-and-repair fallback can be
        tested without standing up a server that behaves badly.

        The supported path parses the scripted reply as JSON rather than emulating OpenAI's
        tool-call protocol. LangChain's default implementation routes through ``bind_tools``,
        which would mean scripting wire-format tool calls in every test -- coupling the suite
        to a provider's protocol details for no gain. Fidelity to that protocol is the stub
        server's job; determinism is this model's job. The two are tested separately.
        """
        if not self.supports_native_structured_output:
            msg = (
                f"{self.model_name} does not support native structured output; "
                "callers must use the validate-and-repair fallback"
            )
            raise NotImplementedError(msg)

        def parse(inputs: LanguageModelInput) -> dict[str, Any] | BaseModel:
            reply = self.invoke(inputs)
            payload = json.loads(str(reply.content))
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(payload)
            return dict(payload)

        def parse_with_raw(inputs: LanguageModelInput) -> dict[str, Any]:
            reply = self.invoke(inputs)
            try:
                payload = json.loads(str(reply.content))
                parsed: dict[str, Any] | BaseModel = (
                    schema.model_validate(payload)
                    if isinstance(schema, type) and issubclass(schema, BaseModel)
                    else dict(payload)
                )
            except (ValueError, ValidationError) as error:
                return {"raw": reply, "parsed": None, "parsing_error": error}
            return {"raw": reply, "parsed": parsed, "parsing_error": None}

        if include_raw:
            return RunnableLambda(parse_with_raw)
        return RunnableLambda(parse)


def scripted_json(payload: object, *, wrapped_in_prose: bool = False) -> str:
    """Render a payload the way a model would return it.

    Args:
        payload: The object to serialise.
        wrapped_in_prose: When ``True``, surround the JSON with the conversational padding
            that models habitually add, so the parser is tested against what arrives rather
            than against what was hoped for.
    """
    body = json.dumps(payload, sort_keys=True)
    if not wrapped_in_prose:
        return body
    return (
        "Sure -- here is the JSON you asked for:\n\n"
        f"```json\n{body}\n```\n\n"
        "Let me know if you need anything else."
    )


def failing_after(successes: int, total_calls: int) -> frozenset[int]:
    """Indices that fail once ``successes`` calls have gone through.

    Models a provider that works until it does not -- a rate limit reached partway through a
    fan-out, rather than an endpoint that was down from the start.
    """
    return frozenset(range(successes, total_calls))
