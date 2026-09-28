"""The run ledger's parts: the callback, the state record, the embedding charge, the lock."""

from __future__ import annotations

import json
import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agentgate.config import CallClass, Settings, Tier
from agentgate.guardrails.run_ledger import (
    LedgerCallback,
    LedgerMissingError,
    accounted,
    charged_ledger,
    charging,
)
from agentgate.guardrails.spend import (
    Ceilings,
    MissingUsageError,
    SpendCeilingExceededError,
    SpendLedger,
    Usage,
)
from agentgate.models.registry import build_model
from agentgate.retrieval.accounting import AccountedEmbeddings
from agentgate.retrieval.embeddings import build_embeddings

pytestmark = pytest.mark.usefixtures("isolated_env")


def fake(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def ledger(settings: Settings | None = None) -> SpendLedger:
    settings = settings or fake()
    return SpendLedger(settings, Ceilings.for_run(settings))


def reply(usage: dict[str, int] | None) -> LLMResult:
    message = AIMessage("x")
    if usage is not None:
        message.usage_metadata = {**usage, "total_tokens": sum(usage.values())}  # type: ignore[typeddict-item]
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# ----------------------------------------------------------------------- the state record


def test_a_ledger_round_trips_through_plain_json() -> None:
    """ADR 0011: the spend so far lives in a state channel, so it has to be plain data."""
    book = ledger()
    book.record_usage("m", Usage(3, 4))
    book.record_usage("e", Usage(5, 0))

    channel = json.loads(json.dumps(book.as_channel()))
    restored = SpendLedger.resumed(fake(), Ceilings.for_run(fake()), channel)

    assert restored.usage_by_model == book.usage_by_model
    assert restored.total_tokens == 12


def test_a_resumed_ledger_from_nothing_is_empty() -> None:
    assert SpendLedger.resumed(fake(), Ceilings.for_run(fake()), None).total_tokens == 0


# ------------------------------------------------------------------------- the callback


def test_the_callback_records_against_the_model_that_was_asked_for() -> None:
    book = ledger()
    callback = LedgerCallback(book)
    run_id = "00000000-0000-0000-0000-000000000001"

    callback.on_chat_model_start({}, [[]], run_id=run_id, metadata={"ls_model_name": "asked"})
    callback.on_llm_end(reply({"input_tokens": 7, "output_tokens": 2}), run_id=run_id)

    assert book.usage_by_model == {"asked": Usage(7, 2)}


def test_the_callback_refuses_a_call_that_reported_no_usage() -> None:
    """A streamed call with no ``stream_options.include_usage`` arrives exactly like this --
    item 18 -- and it is an unmeasured call, not a free one."""
    book = ledger()
    callback = LedgerCallback(book)
    run_id = "00000000-0000-0000-0000-000000000002"
    callback.on_chat_model_start({}, [[]], run_id=run_id, metadata={"ls_model_name": "asked"})

    with pytest.raises(MissingUsageError):
        callback.on_llm_end(reply(None), run_id=run_id)
    assert book.calls == 0


def test_the_callback_raises_the_ceiling_rather_than_logging_it() -> None:
    """LangChain swallows callback exceptions unless the handler says otherwise. A ceiling that
    tripped into a log line would be decoration."""
    book = ledger(fake(max_total_tokens=5))
    callback = LedgerCallback(book)
    run_id = "00000000-0000-0000-0000-000000000003"
    callback.on_chat_model_start({}, [[]], run_id=run_id, metadata={"ls_model_name": "asked"})

    assert callback.raise_error is True
    with pytest.raises(SpendCeilingExceededError):
        callback.on_llm_end(reply({"input_tokens": 7, "output_tokens": 2}), run_id=run_id)


def test_an_accounted_model_charges_its_ledger_on_a_real_invoke() -> None:
    """Through the model, not the handler alone: the callback has to be attached where the call
    actually fires, including through ``bind_tools``, which the drafter's agent uses."""
    settings = fake()
    book = ledger(settings)
    model = accounted(build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION), book)

    model.invoke("hello")
    model.bind_tools([]).invoke("hello again")

    assert book.calls == 2
    assert set(book.usage_by_model) == {settings.model_for(Tier.CHEAP)}


# ---------------------------------------------------------------------------- embeddings


def test_cloud_embeddings_are_accounted() -> None:
    """Item 19 closed: the embedder the index builds is the accounted one."""
    settings = fake(
        lane="cloud",
        openai_api_key="not-required",
        cloud_capable_model="m",
        cloud_cheap_model="m",
        embedding_model="e",
        model_prices_usd_per_million={
            "m": {"input": 1.0, "output": 1.0},
            "e": {"input": 0.02, "output": 0.0},
        },
    )

    assert isinstance(build_embeddings(settings), AccountedEmbeddings)


def test_embeddings_charge_the_ledger_of_the_run_that_is_embedding() -> None:
    book = ledger()
    with charging(book):
        assert charged_ledger() is book


def test_embedding_outside_a_run_is_refused() -> None:
    with pytest.raises(LedgerMissingError):
        charged_ledger()


def test_the_charge_does_not_leak_between_threads() -> None:
    """Research branches run in parallel threads, each inside its own run's charge."""
    outer = ledger()
    seen: list[BaseException | SpendLedger] = []

    def other_thread() -> None:
        try:
            seen.append(charged_ledger())
        except LedgerMissingError as error:
            seen.append(error)

    with charging(outer):
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()

    assert isinstance(seen[0], LedgerMissingError)


# ------------------------------------------------------------------------------ the lock


def test_concurrent_records_are_not_lost() -> None:
    """Fan-out branches record into one run ledger at once. Without the lock, read-add-write on
    the per-model total drops updates, and the ceiling reads low."""
    book = ledger(fake(max_total_tokens=10_000_000))
    workers = [
        threading.Thread(target=lambda: [book.record_usage("m", Usage(1, 0)) for _ in range(2000)])
        for _ in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert book.usage_by_model["m"].input_tokens == 16_000
    assert book.calls == 16_000
