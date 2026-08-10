"""Embedding spend goes through the ledger, on the same three rules as chat spend.

Leak-inventory item 9 was that it did not: the ledger reads `usage_metadata` off chat replies
and embeddings do not produce one, so on the cloud lane indexing and querying the corpus cost
money no ceiling could see. The decision was that the run budget means *all* spend.

Exercised against doubles rather than a provider, because the interesting cases are the ones a
real provider will not perform on demand -- one that reports nothing, and one that pushes the
ledger past its ceiling mid-index.
"""

from __future__ import annotations

import pytest

from agentgate.config import Settings
from agentgate.errors import ConfigurationError
from agentgate.guardrails.spend import Ceilings, SpendCeilingExceededError, SpendLedger
from agentgate.retrieval.accounting import AccountedEmbeddings, MissingEmbeddingUsageError

pytestmark = pytest.mark.usefixtures("isolated_env")

EMBEDDER = "an-embedding-model"
CHAT = "a-chat-model"


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "cloud_capable_model": CHAT,
        "cloud_cheap_model": CHAT,
        "embedding_model": EMBEDDER,
        "model_prices_usd_per_million": {
            CHAT: {"input": 1.0, "output": 10.0},
            EMBEDDER: {"input": 2.0, "output": 0.0},
        },
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def ledger_for(settings: Settings) -> SpendLedger:
    return SpendLedger(settings, Ceilings.for_run(settings))


class Reporting:
    """An embedder that reports what it charged for."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.batches: list[list[str]] = []

    def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.batches.append(list(texts))
        return [[0.0] for _ in texts], self.tokens


# --------------------------------------------------------------------------- accounting


def test_embedding_tokens_reach_the_ledger() -> None:
    settings = settings_with()
    ledger = ledger_for(settings)

    AccountedEmbeddings(Reporting(1_000), EMBEDDER, ledger).embed_documents(["a", "b"])

    assert ledger.total_tokens == 1_000
    assert ledger.total_usd == pytest.approx(0.002)


def test_spend_is_recorded_against_the_model_not_a_category() -> None:
    """The summary has to say where the money went, and "embeddings" is not a model."""
    settings = settings_with()
    ledger = ledger_for(settings)

    AccountedEmbeddings(Reporting(500), EMBEDDER, ledger).embed_query("a query")

    assert set(ledger.usage_by_model) == {EMBEDDER}
    assert EMBEDDER in ledger.summary()


def test_embedding_usage_is_input_only() -> None:
    """An embedding response has no generation in it. Recording output tokens would price the
    call at the output rate, which is typically several times the input rate."""
    settings = settings_with()
    ledger = ledger_for(settings)

    AccountedEmbeddings(Reporting(1_000), EMBEDDER, ledger).embed_documents(["a"])

    assert ledger.usage_by_model[EMBEDDER].output_tokens == 0


def test_chat_and_embedding_spend_share_one_budget() -> None:
    """The ruling, expressed as arithmetic: the run budget means all spend."""
    settings = settings_with(
        max_total_tokens=10_000_000, max_spend_usd=100.0, max_session_spend_usd=100.0
    )
    ledger = ledger_for(settings)

    AccountedEmbeddings(Reporting(1_000_000), EMBEDDER, ledger).embed_documents(["a"])

    assert ledger.total_usd == pytest.approx(2.0), "embedding cost is in the run total"


# --------------------------------------------------------------------------- the refusals


def test_an_embedder_that_reports_no_usage_is_an_error_not_a_free_call() -> None:
    """The same rule as a chat reply with no usage metadata, and for the same reason: a silent
    zero disarms every ceiling built on top of it."""
    settings = settings_with()

    with pytest.raises(MissingEmbeddingUsageError, match="refusing to treat"):
        AccountedEmbeddings(Reporting(0), EMBEDDER, ledger_for(settings)).embed_documents(["a"])


def test_a_networked_lane_with_an_unpriced_embedding_model_refuses_to_start() -> None:
    """Item 9's other half. A model that cannot be costed cannot be bounded, and the embedding
    model was outside this check until the run budget was ruled to mean all spend."""
    with pytest.raises((ConfigurationError, ValueError), match="no price configured"):
        settings_with(model_prices_usd_per_million={CHAT: {"input": 1.0, "output": 10.0}})


def test_the_fake_lane_still_starts_without_an_embedding_price() -> None:
    """The refusal applies to lanes that reach a network. The offline lane embeds in-process
    and spends nothing, and requiring a price for it would break every offline test to guard
    against a cost that cannot occur."""
    assert Settings(_env_file=None, lane="fake", embedding_model=EMBEDDER)  # type: ignore[call-arg]


def test_a_runaway_index_trips_the_ceiling_partway_through() -> None:
    """Checked after every batch, not at the end.

    An index large enough to blow the budget should stop while it is doing it. Enforcement that
    only happens afterwards has told you the bill, not prevented it.
    """
    settings = settings_with(max_spend_usd=0.001, max_session_spend_usd=1.0)
    ledger = ledger_for(settings)
    embedder = AccountedEmbeddings(Reporting(1_000), EMBEDDER, ledger)

    with pytest.raises(SpendCeilingExceededError):
        for _ in range(10):
            embedder.embed_documents(["a chunk"])

    assert len(embedder.inner.batches) < 10, "it ran the whole index before noticing"  # type: ignore[union-attr]
