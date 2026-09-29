"""The run ledger, asserted against what reached the endpoints.

One ledger per run, created by :func:`agentgate.graph.build.run_config`, reaching every model call
the graph makes: classification, the drafter's whole ``create_agent`` loop, and the embeddings
behind retrieval. Its ceilings are checked after every call, so a crossing stops the run *at that
call* -- the tests here assert that no further request reached any endpoint afterwards, which is
what "trips mid-run" means on the wire.

A run spans processes: it pauses at the approval gate and a separate invocation resumes it. So
the spend so far is written to state at every supervisor turn, and a resumed run's ledger starts
from it. Without that, a ceiling would reset at every approval and a reviewer who kept rejecting
would never reach it.

Two stubs, as in ``test_routed_lane_enforcement.py``, with their own request logs. Counts are read
off the stubs; the ledger is what is being tested, so it is never the evidence for itself.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import CallClass, Settings, Tier
from agentgate.graph.build import build_checkpointer, build_graph, resume_config, run_config
from agentgate.graph.state import Classification, initial_state
from agentgate.guardrails.run_ledger import LedgerMissingError, accounted, ledger_of
from agentgate.guardrails.spend import Ceilings, SpendCeilingExceededError, SpendLedger
from agentgate.models.registry import build_model
from agentgate.models.structured import invoke_structured

pytestmark = pytest.mark.usefixtures("isolated_env")

CAPABLE = "cloud-capable-stub"
CHEAP = "cloud-cheap-stub"
EMBEDDING = "embedding-stub"
CORPUS = Path(__file__).resolve().parents[2] / "corpus"

PUBLIC_REQUEST = "Summarise our published refund window for the website."
QUESTION = "What is the published refund window?"
# Involved, so the router sends the draft to the capable tier and this file exercises both:
# the classifier on cheap, the drafter on capable. Since leak inventory item 16 closed, a
# *simple* public request is drafted on the cheap tier too -- which would leave every
# per-model assertion below comparing one model against itself.
PUBLIC = {"sensitivity": "public", "complexity": "involved", "contains_pii": False, "reason": "t"}


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply=PUBLIC)) as server:
        yield server


def cloud_only(cloud: StubServer, **overrides: object) -> Settings:
    """Every kind of spend on one endpoint: chat on two tiers, and embeddings."""
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "openai_base_url": cloud.base_url,
        "cloud_capable_model": CAPABLE,
        "cloud_cheap_model": CHEAP,
        "embedding_model": EMBEDDING,
        "corpus_path": CORPUS,
        "model_prices_usd_per_million": {
            CAPABLE: {"input": 1.0, "output": 4.0},
            CHEAP: {"input": 0.1, "output": 0.4},
            EMBEDDING: {"input": 0.02, "output": 0.0},
        },
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


def run(settings: Settings, request: str = PUBLIC_REQUEST) -> tuple[Any, dict[str, Any]]:
    """A run as far as the approval gate, researching one real question."""
    graph = build_graph(settings, build_checkpointer(settings))
    config = run_config(settings, str(uuid.uuid4()))
    state = initial_state(request, config["configurable"]["thread_id"])
    state["sub_questions"] = [QUESTION]
    graph.invoke(state, config)
    return graph, config


def chat_requests(stub: StubServer, model: str) -> int:
    return sum(1 for body in stub.behaviour.requests_seen if body.get("model") == model)


# ---------------------------------------------------------------- every call, every kind


def test_every_request_that_reached_the_endpoint_is_in_the_ledger(cloud: StubServer) -> None:
    """The count on the stub and the count in the ledger agree, kind by kind. A call site that
    did not account would show up as a request the ledger never heard of."""
    settings = cloud_only(cloud)

    _, config = run(settings)
    ledger = ledger_of(config)

    assert chat_requests(cloud, CHEAP) >= 1, "precondition: classification reached the stub"
    assert chat_requests(cloud, CAPABLE) >= 1, "precondition: the drafter reached the stub"
    assert cloud.behaviour.embedding_requests, "precondition: retrieval embedded on the cloud"

    assert set(ledger.usage_by_model) == {CHEAP, CAPABLE, EMBEDDING}
    assert ledger.calls == len(cloud.behaviour.requests_seen) + len(
        cloud.behaviour.embedding_requests
    ), "a request reached the endpoint that the ledger never recorded"


def test_a_run_started_without_a_ledger_makes_no_model_call_at_all(cloud: StubServer) -> None:
    """Fail closed. A run that could proceed without a ledger could spend without one."""
    settings = cloud_only(cloud)
    graph = build_graph(settings, build_checkpointer(settings))
    config = {"configurable": {"thread_id": "no-ledger"}, "recursion_limit": 40}

    with pytest.raises(LedgerMissingError):
        graph.invoke(initial_state(PUBLIC_REQUEST, "no-ledger"), config)

    assert cloud.behaviour.requests_seen == []
    assert cloud.behaviour.embedding_requests == []


# ------------------------------------------------------------------ the ceiling, mid-run


def spent_before_drafting(cloud: StubServer) -> int:
    """Tokens a full run spends before its first drafter call, measured rather than assumed."""
    probe = cloud_only(cloud)
    _, config = run(probe)
    ledger = ledger_of(config)
    cloud.behaviour.reset()
    return ledger.total_tokens - ledger.usage_by_model[CAPABLE].total_tokens


def test_the_ceiling_trips_at_classification_and_nothing_follows(cloud: StubServer) -> None:
    settings = cloud_only(cloud, max_total_tokens=1)

    with pytest.raises(SpendCeilingExceededError):
        run(settings)

    assert chat_requests(cloud, CHEAP) == 1, "the classification call that crossed it"
    assert cloud.behaviour.embedding_requests == [], "no research after the ceiling"
    assert chat_requests(cloud, CAPABLE) == 0, "no drafting after the ceiling"


def test_the_ceiling_trips_inside_retrieval_and_is_not_swallowed(cloud: StubServer) -> None:
    """The research branch catches its own failures so one bad branch cannot sink its siblings.
    A crossed ceiling is not a bad branch: it has to stop the run, not become a failed outcome
    that the drafter then writes up."""
    classification = cloud_only(cloud)
    _, config = run(classification)
    cheap_tokens = ledger_of(config).usage_by_model[CHEAP].total_tokens
    cloud.behaviour.reset()

    settings = cloud_only(cloud, max_total_tokens=cheap_tokens)

    with pytest.raises(SpendCeilingExceededError):
        run(settings)

    assert cloud.behaviour.embedding_requests, "the embedding call that crossed it"
    assert chat_requests(cloud, CAPABLE) == 0, "no drafting after the ceiling"


def test_the_ceiling_trips_inside_the_drafter_loop(cloud: StubServer) -> None:
    settings = cloud_only(cloud, max_total_tokens=spent_before_drafting(cloud))

    with pytest.raises(SpendCeilingExceededError):
        run(settings)

    assert chat_requests(cloud, CHEAP) >= 1, "classification completed first"
    assert cloud.behaviour.embedding_requests, "research completed first"
    assert chat_requests(cloud, CAPABLE) == 1, "exactly the drafter call that crossed it"


# --------------------------------------------------------------- across a pause and resume


def test_the_spend_so_far_is_in_state_when_the_run_pauses(cloud: StubServer) -> None:
    settings = cloud_only(cloud)

    graph, config = run(settings)
    snapshot = graph.get_state(config)

    assert snapshot.interrupts, "precondition: paused at the approval gate"
    assert snapshot.values["spend"] == ledger_of(config).as_channel()
    assert snapshot.values["spend"], "and it is not an empty record"


def test_a_resumed_run_starts_from_what_was_already_spent(cloud: StubServer) -> None:
    settings = cloud_only(cloud)
    graph, config = run(settings)
    spent = ledger_of(config).total_tokens

    resumed = resume_config(graph, settings, config["configurable"]["thread_id"])

    assert ledger_of(resumed).total_tokens == spent
    assert ledger_of(resumed) is not ledger_of(config), "a new ledger, as in a new process"


def test_the_ceiling_counts_the_whole_run_across_a_rejection(cloud: StubServer) -> None:
    """A reviewer who keeps rejecting drives drafter calls in fresh invocations. Set the ceiling
    just above what the first invocation spent: only a ledger that carried that spend forward
    trips on the revision. One that started from zero would sail through."""
    probe = cloud_only(cloud)
    _, probe_config = run(probe)
    first_invocation = ledger_of(probe_config).total_tokens
    cloud.behaviour.reset()

    settings = cloud_only(cloud, max_total_tokens=first_invocation + 1)
    graph, config = run(settings)
    resumed = resume_config(graph, settings, config["configurable"]["thread_id"])

    with pytest.raises(SpendCeilingExceededError):
        graph.invoke(Command(resume={"decision": "rejected", "feedback": "shorter"}), resumed)


# ------------------------------------------- a billed call whose reply does not parse


def test_a_native_structured_call_that_fails_to_parse_is_still_accounted(
    cloud: StubServer,
) -> None:
    """Leak inventory item 23. The cloud lane is recorded as native, so classification asks for
    structured output first -- and this stub, like a model that misbehaves, answers in prose.

    ``with_structured_output`` hands the OpenAI client a pydantic class, and the client parses
    inside the call: the reply fails validation, the call raises, and a request that reached the
    provider and was billed reports no usage. The ledger saw five requests at the stub and four
    in the book. The native path now binds a plain JSON schema and parses the reply itself, so
    the call returns -- usage and all -- before anything decides it was unusable.
    """
    settings = cloud_only(cloud)
    ledger = SpendLedger(settings, Ceilings.for_run(settings))
    model = accounted(build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION), ledger)

    result = invoke_structured(model, Classification, "Classify: hello.", native=True)

    assert result.sensitivity.value == "public"
    assert len(cloud.behaviour.requests_seen) == 2, "precondition: native attempt, then repair"
    assert "response_format" in cloud.behaviour.requests_seen[0], "the first really was native"
    assert ledger.calls == 2, "the native attempt that failed to parse was billed and not booked"
