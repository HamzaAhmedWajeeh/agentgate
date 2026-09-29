"""The run ledger: one per run, reaching every model call the graph makes.

**How it reaches them.** A run starts through :func:`agentgate.graph.build.run_config`, which puts a
fresh :class:`SpendLedger` in ``config["configurable"]``. LangGraph hands that config to every node
and every subgraph, and never checkpoints it -- only strings, numbers and booleans from
``configurable`` reach checkpoint metadata, so a live object there is simply not persisted. Nodes
read it with :func:`ledger_of`, which **fails closed**: a model-calling node with no ledger raises
before it builds a model, so no call can happen unaccounted.

**Chat calls** are accounted by :class:`LedgerCallback`, attached to the model a node builds with
:func:`accounted`. Attached to the model rather than to the invocation, so that removing it from
a call site is a visible change to that call site -- and so that it survives ``bind_tools``, which
the drafter's agent calls on the model it is given.

**Embeddings** are accounted by ``AccountedEmbeddings``, whose ledger is resolved per call from
:func:`charged_ledger`. The corpus index outlives any one run -- it is built once per compiled
graph -- so the embedder inside it cannot hold a run's ledger. The research branch sets the charge
with :func:`charging` for the duration of its search, in its own thread, and the index build or
query that happens inside is billed to the run that caused it.

**Ceilings are checked after every call** and the error is raised, not logged: LangChain swallows
a callback's exception unless the handler sets ``raise_error``, and a ceiling that tripped into a
log line would be decoration.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any, Final
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agentgate.errors import AgentgateError
from agentgate.guardrails.spend import MissingUsageError, SpendLedger, usage_of
from agentgate.models.resilient import ResilientChatModel

RUN_LEDGER: Final = "run_ledger"
"""The ``configurable`` key the run's ledger travels under."""


class LedgerMissingError(AgentgateError):
    """A model call was about to happen with no run ledger to account it."""


def ledger_of(config: Mapping[str, Any] | None) -> SpendLedger:
    """The run's ledger, from the config LangGraph handed a node.

    Raises:
        LedgerMissingError: if the run was not started through ``run_config``.
    """
    ledger = ((config or {}).get("configurable") or {}).get(RUN_LEDGER)
    if not isinstance(ledger, SpendLedger):
        msg = (
            "this run has no spend ledger, so its model calls could not be accounted; start runs "
            "through agentgate.graph.build.run_config (or resume_config)"
        )
        raise LedgerMissingError(msg)
    return ledger


class LedgerCallback(BaseCallbackHandler):
    """Records each chat call's usage against the model that was asked for, then checks."""

    raise_error = True
    run_inline = True

    def __init__(self, ledger: SpendLedger) -> None:
        self.ledger = ledger
        self._models: dict[UUID | str, str] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002 - part of the callback interface
        messages: list[list[BaseMessage]],  # noqa: ARG002 - part of the callback interface
        *,
        run_id: UUID | str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002 - part of the callback interface
    ) -> None:
        # The configured identifier, which is what the price table is keyed by. A provider's
        # response names a dated snapshot the table has never heard of.
        model = (metadata or {}).get("ls_model_name")
        with self._lock:
            self._models[run_id] = str(model) if model else ""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | str,
        **kwargs: Any,  # noqa: ARG002 - part of the callback interface
    ) -> None:
        with self._lock:
            model = self._models.pop(run_id, "")
        if not model:
            msg = "a chat call finished without naming its model, so it cannot be priced"
            raise MissingUsageError(msg)
        generation = response.generations[0][0] if response.generations else None
        if not isinstance(generation, ChatGeneration):
            msg = f"{model} returned no chat message, so its usage cannot be read"
            raise MissingUsageError(msg)
        self.ledger.record_usage(model, usage_of(generation.message))  # type: ignore[arg-type]
        self.ledger.check()


def accounted(model: BaseChatModel, ledger: SpendLedger) -> BaseChatModel:
    """The same model, charging ``ledger`` for every call it makes.

    A resilient model is accounted *through*, not around. Its leaves are the things that reach
    a provider, so each of them carries the callback and the composite carries none. Attaching
    it to the composite instead would record one call per ``invoke`` no matter how many
    attempts it took, and would price every one of them under the primary's identifier -- so a
    reply the cheap fallback produced during an outage would be billed at capable-tier rates.
    That is the item 11 pattern again: the number the budget depends on, lost to a convenience.
    """
    if isinstance(model, ResilientChatModel):
        return model.model_copy(
            update={
                "primary": accounted(model.primary, ledger),
                "fallback": (
                    accounted(model.fallback, ledger) if model.fallback is not None else None
                ),
            }
        )
    existing = list(model.callbacks) if isinstance(model.callbacks, list) else []
    return model.model_copy(update={"callbacks": [*existing, LedgerCallback(ledger)]})


_CHARGED: ContextVar[SpendLedger | None] = ContextVar("agentgate_charged_ledger", default=None)


@contextlib.contextmanager
def charging(ledger: SpendLedger) -> Iterator[SpendLedger]:
    """Bill embedding calls made inside this block, in this thread, to ``ledger``."""
    token = _CHARGED.set(ledger)
    try:
        yield ledger
    finally:
        _CHARGED.reset(token)


def charged_ledger() -> SpendLedger:
    """The ledger an embedding call is billed to right now.

    Raises:
        LedgerMissingError: outside :func:`charging`. An embedding call nobody is paying for is
            refused, for the same reason a chat call with no ledger is.
    """
    ledger = _CHARGED.get()
    if ledger is None:
        msg = (
            "an embedding call was made outside any run's charge, so its spend could not be "
            "accounted; the research branch sets it, and nothing else should embed"
        )
        raise LedgerMissingError(msg)
    return ledger
