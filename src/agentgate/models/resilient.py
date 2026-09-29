"""Retry and fallback that are still a chat model.

``with_retry`` and ``with_fallbacks`` compose a ``Runnable``, and a ``Runnable`` is not a chat
model. ``create_agent`` takes ``str | BaseChatModel`` and calls ``bind_tools`` on it; a
``RunnableRetry`` has no such method, so the drafter -- the most expensive call in the system
and the one with the most to gain from a fallback -- cannot be handed a composed chain at all.
That is why wiring leak-inventory item 15 needed a chat model rather than a one-line swap: the
resilience has to survive being bound, copied and streamed like any other model.

**The fallback may never widen the lane.** That is not enforced here by a second lane decision,
because a second home for that policy is how the repository got item 13 in the first place.
Both leaves are built by :func:`~agentgate.models.registry.build_resilient_model` from one
``lane`` argument, each through ``build_model``, so they resolve through ``narrower_of`` to the
same lane by construction. A sovereign endpoint that times out is answered by retrying the
sovereign endpoint, or by nothing at all -- never by a third party. An outage is a worse trigger
for the item 13 mistake than a routing bug, because it fires when nobody is watching.

**A retry is for the provider's failures, never for this system's own.** Anything deriving from
:class:`~agentgate.errors.AgentgateError` is a deliberate refusal -- a ceiling crossed, a reply
with no usage to account, a lane that cannot be built -- and it leaves immediately. The ledger
raises its ceiling error from inside the callback of the call that crossed it, so a chain that
treated every exception as transient would answer the budget guard by calling the provider
again, and again, and then falling back and calling it once more. That is not a retry policy
meeting a guard; it is a guard being overruled by one, and it is what the two ceiling tests in
``tests/integration/test_run_ledger.py`` caught the moment this was wired. ``with_retry``
retries on every exception by default, so the composed chain this replaced had the same defect
for as long as it existed -- latent only because nothing called it.

**Each leaf bills itself.** The ledger callback is pushed down into the leaves by
:func:`~agentgate.guardrails.run_ledger.accounted` rather than attached here, so a call the
fallback answered is recorded against *the model that answered it*. A callback on the composite
would read the primary's identifier from its own metadata and price a cheap-tier reply at
capable-tier rates -- the item 11 pattern, a client convenience losing the number the budget
depends on, with the failure showing up only during an outage.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from agentgate.errors import AgentgateError


class ResilientChatModel(BaseChatModel):
    """``primary``, retried, falling back to ``fallback``, still a ``BaseChatModel``.

    Retry handles the transient case -- a reset connection, a rate limit -- where the same
    request will probably work in a moment. Fallback handles the durable case, where this model
    is not going to answer and a smaller one answering is better than nothing.

    Attributes:
        primary: The model the caller asked for.
        fallback: The model to try once the primary is exhausted, or ``None`` when there is
            nothing cheaper to degrade to. The cheap tier has no fallback for that reason:
            falling back *up* a tier is an escalation in cost, not a degradation in quality.
        max_attempts: Total attempts on the primary, so ``max_retries + 1``.
        bound_tools: Tools bound by :meth:`bind_tools`, applied to whichever leaf answers.
        bound_tool_kwargs: The keyword arguments that came with them.
    """

    primary: BaseChatModel
    fallback: BaseChatModel | None = None
    max_attempts: int = 1
    bound_tools: Sequence[Any] | None = None
    bound_tool_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return f"resilient:{self.primary._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "primary": self.primary._identifying_params,
            "fallback": self.fallback._identifying_params if self.fallback else None,
            "max_attempts": self.max_attempts,
        }

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Remember the tools and stay a chat model.

        ``BaseChatModel.bind_tools`` returns a ``Runnable``, which would put the composite back
        in the shape ``create_agent`` rejects. The binding is stored and applied to the leaf
        that is about to be asked, so both leaves see the same tools and the result of binding
        is still something an agent can hold.
        """
        return self.model_copy(update={"bound_tools": list(tools), "bound_tool_kwargs": kwargs})

    def _leaf(self, model: BaseChatModel) -> Runnable[Any, Any]:
        """One leaf, with whatever was bound to the composite applied to it."""
        if self.bound_tools is None:
            return model
        return model.bind_tools(self.bound_tools, **self.bound_tool_kwargs)

    def _attempts(self) -> Iterator[BaseChatModel]:
        """The models to try, in order: the primary N times, then the fallback once.

        Both come from one ``lane`` argument upstream, so the sequence cannot widen the lane.
        """
        for _ in range(max(1, self.max_attempts)):
            yield self.primary
        if self.fallback is not None:
            yield self.fallback

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002 - leaves own their runs
        **kwargs: Any,
    ) -> ChatResult:
        """Ask each attempt in turn; the last failure is what the caller hears about.

        The leaves are invoked through their public ``invoke``, not their ``_generate``, so the
        callbacks attached to *them* fire -- which is how each attempt that reaches a provider
        reaches the ledger. The run manager for this composite is deliberately not passed down:
        the leaf's own run is the one that gets billed, and handing it this run's children as
        well would bill the same answer twice.
        """
        last_error: Exception | None = None
        for model in self._attempts():
            try:
                reply = self._leaf(model).invoke(messages, stop=stop, **kwargs)
            except AgentgateError:
                raise
            except Exception as error:
                last_error = error
                continue
            if not isinstance(reply, AIMessage):  # pragma: no cover - defensive
                reply = AIMessage(content=str(reply))
            return ChatResult(generations=[ChatGeneration(message=reply)])

        raise _exhausted(last_error)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002 - leaves own their runs
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """The same order, with one rule: a stream that has begun is never retried.

        Once a chunk has been handed to the caller, retrying would replay a partial answer from
        the top and the caller would splice two different replies together. So an attempt is
        only replaced while it has produced nothing, and a failure part-way through a stream is
        raised rather than papered over. The CLI streams every call, so this is not a
        hypothetical path.
        """
        last_error: Exception | None = None
        for model in self._attempts():
            started = False
            try:
                for chunk in self._leaf(model).stream(messages, stop=stop, **kwargs):
                    started = True
                    yield _as_chunk(chunk)
            except AgentgateError:
                raise
            except Exception as error:
                if started:
                    raise
                last_error = error
                continue
            else:
                return

        raise _exhausted(last_error)


def _as_chunk(chunk: Any) -> ChatGenerationChunk:
    """Whatever the leaf yielded, as the chunk type this model's contract promises.

    A leaf that implements ``_stream`` yields message *chunks* and passes straight through. A
    leaf that does not -- the fake lane, and any provider client without a streaming path --
    has its whole reply handed over by ``BaseChatModel.stream`` as an ordinary ``AIMessage``,
    which ``ChatGenerationChunk`` rejects. Converting it is not cosmetic: ``usage_metadata``
    and any tool calls ride on that message, and a conversion that dropped them would lose the
    number the ledger bills on and the calls the drafter's agent is waiting for.
    """
    if isinstance(chunk, ChatGenerationChunk):
        return chunk
    if isinstance(chunk, BaseMessageChunk):
        return ChatGenerationChunk(message=chunk)
    if isinstance(chunk, AIMessage):
        fields = chunk.model_dump(exclude={"type", "tool_calls", "invalid_tool_calls"})
        return ChatGenerationChunk(
            message=AIMessageChunk(
                **fields,
                tool_call_chunks=[
                    ToolCallChunk(
                        name=call.get("name"),
                        args=json.dumps(call.get("args", {})),
                        id=call.get("id"),
                        index=index,
                    )
                    for index, call in enumerate(chunk.tool_calls)
                ],
            )
        )
    return ChatGenerationChunk(message=AIMessageChunk(content=str(chunk)))


def _exhausted(last_error: Exception | None) -> Exception:
    """The error to raise when nothing answered.

    ``_attempts`` always yields at least once, so ``last_error`` is set in practice. The
    explicit alternative exists so that a future edit which makes the loop yield nothing
    surfaces as a clear failure rather than as a silent ``None`` return.
    """
    if last_error is not None:
        return last_error
    return RuntimeError("no model was attempted, so there is no reply and no error to report")
