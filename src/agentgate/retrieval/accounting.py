"""Embedding spend, accounted the same way chat spend is.

Leak-inventory item 9 was that it was not. The ledger reads ``usage_metadata`` off chat replies;
embeddings do not produce one, so on the cloud lane indexing and querying the corpus cost money
that the run ceiling, the session ceiling and the live-suite ceiling were all blind to. The
decision, recorded in ADR 0004 item 9: **the run budget means all spend, not chat spend.**

The reason it matters more than the amount suggests is the shape of the gap rather than its
size. Embedding cost scales with fan-out width, and width is the one quantity a model chooses
rather than the system -- so the unaccounted path was precisely the path with model-controlled
multiplication in it.

Same three rules as chat calls, and none of them is new:

*A response with no usage is an error, not a zero.* A provider that stops reporting would
otherwise make every embedding free and no ceiling would ever be reached.

*An unpriced embedding model refuses to start.* Enforced in ``Settings``, alongside the chat
tiers, because a model that cannot be costed cannot be bounded.

*Recorded per model.* The ledger's summary says where the money went, and "embeddings" is not
a model -- ``text-embedding-3-small`` is.

**This is why the provider client is called directly.** ``langchain_openai.OpenAIEmbeddings``
discards the ``usage`` block the API returns, which is the one field the budget depends on.
Wrapping it would mean either estimating tokens -- a guess presented as a measurement -- or
accounting zero, which is the bug being fixed. So the embedding call goes through the client
underneath it and reads what the API actually reported. Recorded as leak-inventory item 11.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.embeddings import Embeddings

from agentgate.errors import AgentgateError
from agentgate.guardrails.spend import SpendLedger, Usage


class MissingEmbeddingUsageError(AgentgateError):
    """An embedding response arrived without usage, so it could not be accounted for."""


class UsageReportingEmbeddings(Protocol):
    """An embedder that can say what it cost.

    A protocol rather than a base class so a test can supply one without importing a provider,
    and so the accounting can be exercised offline against a double that reports honestly --
    and against one that reports nothing, which is the case worth testing.
    """

    def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Return the vectors and the number of tokens the provider charged for."""
        ...


class AccountedEmbeddings(Embeddings):
    """Wraps a usage-reporting embedder and books every call into a ledger.

    Args:
        inner: The embedder that actually calls the provider.
        model: The identifier the spend is recorded against.
        ledger: Where it is recorded. ``check()`` is called after every batch, so a runaway
            index trips the ceiling partway through rather than at the end -- the same
            mid-flight enforcement the live suite gets.

    Raises:
        MissingEmbeddingUsageError: if the provider reported no usage. Treated as a failure
            rather than as zero, for the reason in the module docstring.
    """

    def __init__(self, inner: UsageReportingEmbeddings, model: str, ledger: SpendLedger) -> None:
        self.inner = inner
        self.model = model
        self.ledger = ledger

    def _account(self, texts: list[str]) -> list[list[float]]:
        vectors, tokens = self.inner.embed_with_usage(texts)

        if tokens <= 0:
            msg = (
                f"embedding call to {self.model} reported {tokens} tokens; refusing to treat an "
                "unmeasured call as free. An embedder that stops reporting usage disarms every "
                "ceiling built on top of it."
            )
            raise MissingEmbeddingUsageError(msg)

        # Input only. An embedding response has no output tokens -- the vector is not billed as
        # generation -- so recording any would overstate the cost at the output rate, which is
        # typically several times the input rate.
        self.ledger.record_usage(self.model, Usage(input_tokens=tokens, output_tokens=0))
        self.ledger.check()
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._account(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._account([text])[0]


class OpenAIEmbeddingsWithUsage:
    """The cloud lane's embedder, calling the client directly so the usage survives.

    ``OpenAIEmbeddings.embed_documents`` returns vectors and drops the ``usage`` block. The
    OpenAI client underneath it does not, so the call is made there and the reported token
    count is read off the response rather than estimated from the text.
    """

    def __init__(self, model: str, api_key: str, timeout: float) -> None:
        from openai import OpenAI  # noqa: PLC0415 - lane-specific import

        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        response: Any = self.client.embeddings.create(model=self.model, input=texts)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return [item.embedding for item in response.data], tokens
