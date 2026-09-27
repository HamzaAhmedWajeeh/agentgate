"""The policy gate, enforced on the wire rather than in a state field.

This is the test the project's thesis rests on, and it did not exist until leak 13 was found.
Every other test of the policy gate asserts `result["lane"]`, which is a **label the graph wrote
about itself**. A label is not an endpoint. The whole class of defect this repository catalogues
is the gap between the two, so the only assertion worth making here reads what a server
actually received.

Two stubs run at once, on separate ports with separate request logs: one standing in for the
cloud provider, one for the operator's own endpoint. That is the only way to express the
property that matters, which is an **absence** -- restricted content appearing in no request
body the cloud endpoint ever saw. A single stub cannot express it, because "the sovereign
endpoint got the call" is true in a system that also sent it to the cloud.

Two canaries, and the split is deliberate:

- `REQUEST_CANARIES` sit in the request text. The classifier necessarily sees them, because it
  runs before the policy decision exists -- that is a separate, still-open gap, pinned below by
  `test_the_classifier_sends_the_raw_request_to_the_configured_lane` and recorded as item 14.
- `DRAFT_CANARIES` sit only in the findings, which nothing but the drafter is shown. They are
  therefore the ones whose appearance on the cloud endpoint can only mean the routed lane was
  ignored, and they are what makes the primary assertion specific rather than merely true.

Offline and free: every endpoint here is a loopback stub, and no test reaches a real provider.

The last section is the exception to "no research branch runs". Everything above seeds findings,
which keeps the run away from retrieval -- and that turned out to be where the next leak was, so
one test now runs the real branch and records what the embedder sends. Leak inventory item 17.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import Lane, Settings, Tier
from agentgate.graph.build import build_checkpointer, build_graph
from agentgate.graph.state import Finding, initial_state
from agentgate.models.registry import LaneUnavailableError
from agentgate.retrieval.embeddings import build_embeddings

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-stub"
CLOUD_CHEAP = "cloud-cheap-stub"
SOVEREIGN_MODEL = "sovereign-stub"

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

# Invented, and shaped like the real thing so a substring match cannot succeed by accident.
# The NHS format is ten digits in 3-3-4 groups; neither number belongs to anybody.
REQUEST_CANARIES: tuple[str, ...] = ("Jane Doe", "4929-1123-8876", "485 777 3456")
DRAFT_CANARIES: tuple[str, ...] = ("Ravi Chandrasekaran", "999 111 2222")

RESTRICTED_REQUEST = (
    "Draft a refund letter for Jane Doe, account 4929-1123-8876, "
    "NHS number 485 777 3456, who was overcharged 240 GBP."
)

PUBLIC_REQUEST = "Summarise our published refund window for the website."

# Only ever shown to the drafter, via the brief it builds from the findings.
RESTRICTED_FINDING = Finding(
    question="What does the account record say?",
    content=(
        "Case notes: handler Ravi Chandrasekaran confirmed the overcharge against "
        "NHS number 999 111 2222 on the 14th."
    ),
    source="case-notes.md",
)


def verdict(sensitivity: str, complexity: str = "simple", pii: bool = True) -> dict[str, Any]:
    """The classification the stub standing in for the classifier's lane will return."""
    return {
        "sensitivity": sensitivity,
        "complexity": complexity,
        "contains_pii": pii,
        "reason": "test fixture",
    }


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    """The third-party endpoint. Its request log is the thing under scrutiny."""
    with running_stub(StubBehaviour(reply=verdict("restricted"))) as server:
        yield server


@pytest.fixture
def sovereign() -> Iterator[StubServer]:
    """The operator's own endpoint.

    Replies with a classification verdict rather than a draft, because on a hybrid deployment
    this is the lane classification runs on. A stub that answered the classifier with prose sent
    the classifier into its fail-closed branch, which routes restricted -- the same destination
    the test was checking for, reached without a verdict being parsed at all. The trail assertion
    caught it; nothing else would have.

    Each stub returns one reply to every request, so the draft on this lane is that verdict
    wrapped in prose. Harmless here: no assertion in this file reads draft *content*, only which
    endpoint was asked and what the request body contained.
    """
    with running_stub(StubBehaviour(reply=verdict("restricted"))) as server:
        yield server


def hybrid_settings(cloud: StubServer, sovereign: StubServer, *, default: str) -> Settings:
    """A deployment with both lanes constructible, defaulting to one of them.

    This is the configuration the policy gate exists for. A deployment with only one lane
    reachable cannot demonstrate routing at all, and a deployment whose second lane is
    *configured but unbuildable* would fail loudly rather than leak, which is a different and
    much easier problem.
    """
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=default,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        sovereign_base_url=sovereign.base_url,
        sovereign_model=SOVEREIGN_MODEL,
        model_prices_usd_per_million={
            CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
            CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
            SOVEREIGN_MODEL: {"input": 0.0, "output": 0.0},
        },
    )


def run_to_the_gate(settings: Settings, request: str) -> dict[str, Any]:
    """Drive a run as far as the approval gate, which is one node past the drafter.

    ``dispatched`` and ``findings`` are seeded rather than researched. Research is not what is
    under test, and skipping it keeps the run away from the embeddings path -- which dispatches
    on the *configured* lane and cannot be pointed at a stub at all, so exercising it here would
    either reach a real provider or prove nothing.
    """
    graph = build_graph(settings, build_checkpointer(settings))
    state = initial_state(request, str(uuid.uuid4()))
    state["dispatched"] = 1
    state["findings"] = [RESTRICTED_FINDING.as_channel()]
    return dict(
        graph.invoke(
            state,
            {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": settings.recursion_limit,
            },
        )
    )


def bodies(stub: StubServer) -> list[str]:
    """Every request body the endpoint received, as the JSON text it was sent as.

    Serialised rather than walked, because a canary can turn up anywhere in a body -- a system
    prompt, a tool description, a message the client assembled -- and a walk that only looks in
    the places currently expected would go quiet the moment the client library rearranged
    something.

    Reads the chat log *and* the decoded embedding texts. Both halves were learned the hard way.
    Reading the chat log alone leaves every absence assertion here blind to retrieval egress, which
    is leak inventory item 17 -- so that blindness would have been load bearing. And reading the
    embedding bodies *raw* is no better than not reading them: `OpenAIEmbeddings` tokenises before
    sending, so a canary is present as token ids and absent as a string, and the assertion passes
    while the content leaves. See ``decode_embedding_input``.
    """
    return [
        *(json.dumps(body, sort_keys=True) for body in stub.behaviour.requests_seen),
        *stub.behaviour.embedding_texts(),
    ]


def canaries_seen_by(stub: StubServer, canaries: Sequence[str]) -> list[str]:
    """Which canaries appear in anything this endpoint received."""
    return sorted({canary for canary in canaries for body in bodies(stub) if canary in body})


def models_asked_of(stub: StubServer) -> list[str]:
    return [str(body.get("model", "")) for body in stub.behaviour.requests_seen]


# ------------------------------------------------------------------- the primary assertion


def test_restricted_content_never_reaches_the_cloud_endpoint(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The thesis, read off two request logs.

    The presence half is not decoration. An absence assertion passes perfectly against a run
    that never drafted anything, so the draft canaries are first shown to have gone *somewhere*.
    Only then does their absence from the cloud log mean what it appears to mean.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    result = run_to_the_gate(settings, RESTRICTED_REQUEST)

    assert result["lane"] == Lane.SOVEREIGN.value, "precondition: the router chose sovereign"
    assert canaries_seen_by(sovereign, DRAFT_CANARIES) == sorted(DRAFT_CANARIES), (
        "the drafter did not send the findings anywhere; an absence assertion on the cloud "
        "endpoint would pass for the wrong reason"
    )
    assert canaries_seen_by(cloud, DRAFT_CANARIES) == [], (
        "restricted content reached the third-party endpoint after the router sent the "
        "request to the sovereign lane"
    )


def test_the_draft_is_asked_of_the_sovereign_model_not_merely_the_sovereign_endpoint(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Half a fix is its own defect.

    Passing the routed lane to ``build_model`` chooses the endpoint. The model *identifier*
    comes from ``model_for``, which reads the configured lane, so a lane-aware endpoint with a
    lane-blind identifier sends a request for `cloud-capable-stub` to the operator's own server
    -- a name that endpoint has never heard of, recorded in the audit trail as the model that
    answered.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    run_to_the_gate(settings, RESTRICTED_REQUEST)

    asked = models_asked_of(sovereign)
    assert asked, "nothing reached the sovereign endpoint"
    assert set(asked) == {SOVEREIGN_MODEL}, (
        f"the sovereign endpoint was asked for {sorted(set(asked))}, which includes a model "
        "identifier belonging to another lane"
    )


def test_a_sovereign_default_never_reaches_the_cloud_endpoint_for_public_content(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The fix's own failure mode, pinned before it can be written.

    The obvious repair -- pass the routed lane straight through -- is worse than the bug on a
    deployment that defaults to sovereign. The router sends *public* content to the cloud lane
    by design, so honouring the route unconditionally would start calling a third party on
    behalf of an operator who configured their own endpoint as the default. The route narrows
    where a request may go; it does not widen it.
    """
    cloud.behaviour.reply = verdict("public", pii=False)
    sovereign.behaviour.reply = verdict("public", pii=False)
    settings = hybrid_settings(cloud, sovereign, default=Lane.SOVEREIGN.value)

    result = run_to_the_gate(settings, PUBLIC_REQUEST)

    assert result["draft"], "precondition: the run produced a draft, so models were called"
    assert bodies(cloud) == [], (
        "a deployment defaulting to the sovereign lane called the third-party endpoint "
        "because the policy route was allowed to widen the configured boundary"
    )


def test_the_audit_trail_names_the_endpoint_that_actually_answered(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The trail is checked against the request log rather than against itself.

    Every other assertion about the trail in this repository reads the trail alone, which is
    enough to prove an event was written and nothing at all about whether it is true. This one
    ties each claim to an observation: the model the drafted event names must be a model the
    sovereign endpoint was actually asked for, and must not be one the cloud endpoint was.

    Before item 13 this event named the configured lane's model beside the routed lane's label --
    `lane='sovereign' model='gpt-4.1-nano'` on a request that went to OpenAI. Neither field was
    a lie anybody had to tell, and no test could tell either.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    result = run_to_the_gate(settings, RESTRICTED_REQUEST)

    asked_of_sovereign = models_asked_of(sovereign)
    assert asked_of_sovereign, "precondition: the sovereign endpoint was called"

    drafted = next(e for e in result["audit_trail"] if e["decided"] == "drafted")
    assert drafted["lane"] == Lane.SOVEREIGN.value
    assert drafted["model"] in asked_of_sovereign, (
        "the drafted event names a model the endpoint it claims to have used was never asked for"
    )
    assert drafted["model"] not in models_asked_of(cloud)
    assert drafted["detail"]["policy_route"] == Lane.SOVEREIGN.value
    assert drafted["detail"]["lane_narrowed_by_deployment"] is False

    # The lane-selection event records the same model resolution one node earlier, and had the
    # same defect: a sovereign binding annotated with the cloud lane's cheap model.
    lane_event = next(e for e in result["audit_trail"] if e["decided"] == "lane_selected")
    assert lane_event["model"] == SOVEREIGN_MODEL
    assert lane_event["detail"]["effective_lane"] == Lane.SOVEREIGN.value

    # The classifier's event was the one that was always true -- it ran on the configured lane
    # and said so. Since the classification lane was narrowed it says something different, and
    # this asserts the new claim against the same request log: it ran on the sovereign lane, and
    # the model it names is one that endpoint was asked for.
    classified = next(e for e in result["audit_trail"] if e["decided"] == "classified")
    assert classified["lane"] == Lane.SOVEREIGN.value
    assert classified["model"] in asked_of_sovereign
    assert classified["detail"]["classified_on_lane"] == Lane.SOVEREIGN.value
    assert classified["detail"]["deployment_default_lane"] == Lane.CLOUD.value
    assert classified["detail"]["classification_failed"] is None, (
        "the verdict was parsed, so the routing under test came from a classification rather "
        "than from failing closed"
    )


# ------------------------------------------------------------------- the classification lane


def test_the_classifier_never_shows_the_raw_request_to_the_cloud_endpoint(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The inversion of what this file first recorded, and the reason to have recorded it.

    The first version of this test asserted that all three request canaries *did* reach the
    cloud endpoint, because they did: classification ran on the configured lane, so a
    cloud-default deployment showed a third party the raw request in order to decide whether it
    was allowed to see it. Writing that down as a passing assertion is what makes closing it a
    dated, visible change to a named test rather than an improvement nobody can point at.

    On a hybrid deployment the cloud endpoint should now see nothing whatsoever for a restricted
    request: not the draft, and not the classification either.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    run_to_the_gate(settings, RESTRICTED_REQUEST)

    assert canaries_seen_by(sovereign, REQUEST_CANARIES) == sorted(REQUEST_CANARIES), (
        "the request reached no endpoint at all; the absence assertion below would pass for "
        "the wrong reason"
    )
    assert canaries_seen_by(cloud, REQUEST_CANARIES) == []
    assert bodies(cloud) == [], (
        "a restricted request on a hybrid deployment reached the third-party endpoint; after "
        "the classification lane was narrowed there is no node left that should call it"
    )


def test_classification_does_not_take_the_native_path_on_a_lane_recorded_as_non_native(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The capability lookup has to follow the classification lane too.

    ``supports()`` was asked about the configured lane. Left that way, moving classification to
    the sovereign lane would ask for native structured output from an endpoint measured as not
    having it -- which does not fail, it falls through to the repair loop after wasting a call.
    Measured against this stub: 2 calls and ~733 prompt tokens where 1 and ~537 would do, on
    every classification, silently.

    ``response_format`` is the discriminator because it is what the native path puts on the
    wire; the repair path sends the schema as a system message instead.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    run_to_the_gate(settings, RESTRICTED_REQUEST)

    assert sovereign.behaviour.requests_seen, "nothing reached the sovereign endpoint"
    offenders = [body for body in sovereign.behaviour.requests_seen if "response_format" in body]
    assert offenders == [], (
        "classification asked a lane recorded as lacking native structured output for it; "
        "the capability lookup is still reading the configured lane"
    )


# ---------------------------------------------- the egress that remains, recorded as one


def test_a_cloud_only_deployment_classifies_on_the_cloud_lane_as_documented_egress(
    cloud: StubServer,
) -> None:
    """Item 14, narrowed rather than closed, and this is the part that stays open.

    A deployment with no sovereign endpoint has nowhere else to send a request to be judged, so
    the raw content still goes to the third party before any policy decision exists. That is not
    fixable in code -- it is a property of having one lane -- so it is recorded here as a
    passing assertion and in ADR 0004 item 14, rather than described as solved because the
    hybrid case improved.

    The ``response_format`` assertion is the control for the test above. The cloud lane is
    recorded as *having* native structured output, so it must appear here. Without this, a
    rename of that wire field by `langchain-openai` would make the absence assertion above pass
    against a field that no longer exists -- which is leak inventory item 2, exactly.
    """
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        model_prices_usd_per_million={
            CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
            CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
        },
    )
    assert settings.classification_lane is Lane.CLOUD, "precondition: nowhere else to classify"

    # And then it refuses, which is the other half of the story. Policy sends the request
    # somewhere more contained, there is no such lane, and the only alternative to stopping is
    # serving it from the lane policy just ruled out -- which is what used to happen.
    with pytest.raises(LaneUnavailableError, match="has not configured it"):
        run_to_the_gate(settings, RESTRICTED_REQUEST)

    assert canaries_seen_by(cloud, REQUEST_CANARIES) == sorted(REQUEST_CANARIES), (
        "the pre-classification egress described by item 14 no longer happens on a cloud-only "
        "deployment; if that is deliberate, this test and item 14 are what to rewrite"
    )
    assert any("response_format" in body for body in cloud.behaviour.requests_seen), (
        "the native path put no response_format on the wire, so the absence assertion in "
        "test_classification_does_not_take_the_native_path... is watching a field that is no "
        "longer sent -- see leak inventory item 2"
    )


def test_a_hybrid_deployment_still_uses_the_cloud_lane_for_public_content(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Non-vacuity for every absence assertion in this file.

    All of them would pass just as happily against a deployment whose cloud endpoint was
    unreachable, misconfigured, or never called for any reason. This is the one test that
    requires the cloud endpoint to be genuinely in use: public content is classified on the
    sovereign lane, because that is the more contained one, and then drafted on the cloud lane,
    because that is what policy chose and the deployment permits.
    """
    cloud.behaviour.reply = {"answer": "A summary of the published refund window."}
    sovereign.behaviour.reply = verdict("public", pii=False)
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    result = run_to_the_gate(settings, PUBLIC_REQUEST)

    assert result["lane"] == Lane.CLOUD.value, "precondition: public content routes to cloud"
    assert models_asked_of(sovereign) == [SOVEREIGN_MODEL], (
        "classification should have run on the sovereign lane and nothing else should have"
    )
    assert models_asked_of(cloud) == [CLOUD_CAPABLE], (
        "the draft should have been asked of the cloud lane's capable tier"
    )

    # Note which tier answered. `route_by_policy` returned "cloud_cheap" for this request and the
    # drafter asked for the capable one, because the routed *tier* is not wired through either --
    # `bind_lane` binds one and no channel carries it. Invisible in a deployment where both cloud
    # tiers name the same model, which is the reference configuration. Leak inventory item 16;
    # asserted here as the current truth so that wiring it has to come through this line.
    lane_event = next(e for e in result["audit_trail"] if e["decided"] == "lane_selected")
    assert lane_event["detail"]["tier"] == Tier.CHEAP.value
    assert models_asked_of(cloud) == [CLOUD_CAPABLE]


# ------------------------------------------------ retrieval, which is a third egress entirely


RETRIEVAL_QUESTION = "What is the refund window for Jane Doe, account 4929-1123-8876?"

EMBEDDING_MODEL = "embedding-stub"


def hybrid_with_embeddings(cloud: StubServer, sovereign: StubServer) -> Settings:
    """A hybrid deployment that also indexes and queries a corpus.

    ``embedding_model`` is what switches retrieval from the in-process hashing embedder to the
    provider, which is the only configuration in which the leak below exists at all -- and it is
    the configuration any real cloud deployment has.
    """
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        sovereign_base_url=sovereign.base_url,
        sovereign_model=SOVEREIGN_MODEL,
        embedding_model=EMBEDDING_MODEL,
        corpus_path=CORPUS,
        model_prices_usd_per_million={
            CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
            CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
            SOVEREIGN_MODEL: {"input": 0.0, "output": 0.0},
            EMBEDDING_MODEL: {"input": 0.02, "output": 0.0},
        },
    )


def run_with_research(settings: Settings, request: str, question: str) -> dict[str, Any]:
    """A run that actually researches, rather than being handed its findings.

    No ``retriever_factory`` is injected and no ``findings`` are seeded, so the retrieval subgraph
    builds the real index from the real corpus through the real embedder. That is the whole point:
    every other test in this file skips the path under test here.
    """
    graph = build_graph(settings, build_checkpointer(settings))
    state = initial_state(request, str(uuid.uuid4()))
    state["sub_questions"] = [question]
    return dict(
        graph.invoke(
            state,
            {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": settings.recursion_limit,
            },
        )
    )


def test_a_restricted_research_query_is_embedded_by_the_cloud_provider(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Leak inventory item 17, recorded as a passing assertion because it is what happens.

    `build_embeddings` dispatches on ``settings.lane`` -- the configured default -- and never sees
    the route. So on a hybrid deployment a request the policy gate sent to the sovereign lane has
    its research queries embedded by the third party anyway. The chat calls were fixed by item 13;
    this is the same defect in the function next door, and item 13's guards could not see it
    because they seed findings and never research.

    The canary is in the sub-question rather than only in the request, which is where it lives in
    reality: the sub-questions of a restricted request carry the identifiers that made it
    restricted. A question about a refund window is not answerable without naming the account.

    Written as an assertion that the leak *happens*, exactly like item 14's documented egress, so
    closing it is a visible inversion of a named test.
    """
    settings = hybrid_with_embeddings(cloud, sovereign)

    result = run_with_research(settings, RESTRICTED_REQUEST, RETRIEVAL_QUESTION)

    assert result["lane"] == Lane.SOVEREIGN.value, "precondition: the router chose sovereign"
    assert cloud.behaviour.embedding_requests, (
        "no embedding request reached the cloud endpoint, so this test is asserting about an "
        "empty log -- check that embedding_model is set and that a research branch ran"
    )

    # Decoded, not grepped. The raw body carries token ids, so the first version of this
    # assertion looked for "4929-1123-8876" in the JSON and failed -- for the right reason, and
    # it is the reason `decode_embedding_input` exists. An absence assertion written the naive
    # way would have passed forever.
    embedded = "\n".join(cloud.behaviour.embedding_texts())
    assert "4929-1123-8876" in embedded, (
        "the account number no longer reaches the cloud embedding endpoint; if that is "
        "deliberate, this test and ADR 0004 item 17 are what to rewrite"
    )
    assert "Jane Doe" in embedded

    # And the chat half is still correct, which is what makes this a *separate* leak rather than
    # a regression of item 13. No chat request reached the cloud endpoint at all.
    assert cloud.behaviour.requests_seen == [], (
        "a chat call reached the cloud endpoint; item 13 has regressed and this test is "
        "describing the wrong failure"
    )


def test_the_embedding_decoder_recovers_text_the_corpus_actually_contains(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The control for every canary assertion that reads an embedding request.

    `decode_embedding_input` is the only thing standing between "restricted content reached the
    embedding endpoint" and an assertion that greps token ids for a string and finds nothing. If
    the tokeniser is wrong, or a client upgrade changes the encoding, the decode returns plausible
    rubbish and every one of those assertions goes quiet.

    So this checks the decode against something independently known: a line that is in the
    committed corpus on disk. It is not asserting that retrieval works -- it is asserting that the
    instrument reads.
    """
    settings = hybrid_with_embeddings(cloud, sovereign)

    run_with_research(settings, RESTRICTED_REQUEST, RETRIEVAL_QUESTION)

    decoded = "\n".join(cloud.behaviour.embedding_texts())
    assert decoded.strip(), "the decoder returned nothing at all"

    corpus_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(CORPUS.glob("*.md"))
    )
    assert corpus_text.strip(), "the corpus is empty, so this control proves nothing"

    # A distinctive run of words from a chunk that was embedded, found in the file it came from.
    sample = next(
        (
            line.strip()
            for line in decoded.splitlines()
            if len(line.strip()) > 25 and line.strip() in corpus_text
        ),
        None,
    )
    assert sample is not None, (
        "nothing the decoder produced appears in the corpus on disk, so the token ids are being "
        "decoded with the wrong encoding and every canary assertion reading them is vacuous"
    )


def test_the_sovereign_lane_is_not_where_the_research_query_went(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The absence that makes the presence above mean something.

    If both endpoints saw the query, the finding would be "retrieval embeds everywhere" rather
    than "retrieval ignores the route". The operator's own endpoint received no embedding request
    at all, because nothing ever asks it for one -- there is no sovereign embedding path.
    """
    settings = hybrid_with_embeddings(cloud, sovereign)

    run_with_research(settings, RESTRICTED_REQUEST, RETRIEVAL_QUESTION)

    assert cloud.behaviour.embedding_requests, "precondition: embeddings happened somewhere"
    assert sovereign.behaviour.embedding_requests == [], (
        "the sovereign endpoint served an embedding request, which no code path constructs"
    )


def test_the_corpus_itself_is_indexed_through_the_same_egress(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Querying is not the only cost, and item 9 said so before it was closed.

    Every chunk of the corpus is embedded through the provider as well, in one batch per index
    build. Recorded because it bears directly on the options for closing item 17: an approach that
    embeds the query somewhere contained still has to decide what indexed the corpus, and a corpus
    indexed in one vector space cannot be searched with a query embedded in another.
    """
    settings = hybrid_with_embeddings(cloud, sovereign)

    run_with_research(settings, RESTRICTED_REQUEST, RETRIEVAL_QUESTION)

    batched = [
        body
        for body in cloud.behaviour.embedding_requests
        if isinstance(body.get("input"), list) and len(body["input"]) > 1
    ]
    assert batched, (
        "no multi-text embedding request arrived, so the corpus was not indexed through this "
        "endpoint and the premise of the index question in ADR 0004 item 17 is wrong"
    )
    assert all(body.get("model") == EMBEDDING_MODEL for body in cloud.behaviour.embedding_requests)


def test_no_embedding_call_is_accounted_for_anywhere(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Leak inventory item 18, and the reason item 9 is reopened.

    Item 9 is recorded as **closed**, by `AccountedEmbeddings`. Nothing constructs it.
    `build_embeddings` returns a plain `OpenAIEmbeddings`, which -- as `accounting.py`'s own
    docstring says -- discards the usage block the budget depends on. So embedding spend is still
    invisible to every ceiling, which is precisely the state item 9 describes as fixed.

    The endpoint reports usage; this asserts that the *client* throws it away, which is item 11
    still live for the same reason. Read off the response the double sent rather than inferred:
    the number is there on the wire and absent by the time anything could account for it.
    """
    settings = hybrid_with_embeddings(cloud, sovereign)

    run_with_research(settings, RESTRICTED_REQUEST, RETRIEVAL_QUESTION)

    assert cloud.behaviour.embedding_requests, "precondition: embeddings happened"

    # The path a wired ledger would have to run through. `AccountedEmbeddings` is what item 9
    # names as its closer, and `build_embeddings` is what the index actually calls.
    embedder = build_embeddings(settings)
    assert type(embedder).__name__ == "OpenAIEmbeddings", (
        f"the index now embeds through {type(embedder).__name__}; if that is AccountedEmbeddings "
        "then item 9 is genuinely closed and this test and items 9 and 18 are what to rewrite"
    )
