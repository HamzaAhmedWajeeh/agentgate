"""Resilience as the graph actually builds models, not as a helper composes them.

``tests/integration/test_resilience.py`` proves ``build_resilient_model`` retries and falls
back. It proves it by constructing the chain itself, which is evidence about the function and
none at all about the system -- leak inventory item 15, and the same shape as item 13. The
tests here build nothing. They start a run through ``build_graph`` and read what a server
received, so the only thing that can make them pass is a node reaching the resilient path.

The endpoints are the two-stub harness from ``test_routed_lane_enforcement``: one standing in
for the third party, one for the operator's own server, each with its own request log. A single
stub cannot express the property that matters here either, because **the constraint on a
fallback is an absence**. A sovereign endpoint that fails must not be rescued by the cloud one.
A fallback chain is a new place to make the item 13 mistake, and an outage is a worse trigger
for it than a routing bug: it fires exactly when nobody is watching.

Offline and free. Every endpoint is a loopback stub.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import Lane, Settings
from agentgate.graph.build import build_checkpointer, build_graph, run_config
from agentgate.graph.state import Finding, initial_state
from agentgate.guardrails.run_ledger import RUN_LEDGER
from agentgate.guardrails.spend import SpendCeilingExceededError

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-stub"
CLOUD_CHEAP = "cloud-cheap-stub"
SOVEREIGN_MODEL = "sovereign-stub"

PRICES = {
    CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
    CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
    SOVEREIGN_MODEL: {"input": 0.0, "output": 0.0},
}

PUBLIC_REQUEST = "Summarise our published refund window for the website."

SEEDED_FINDING = Finding(
    question="What does the published policy say?",
    content="Refunds are available within 30 days of purchase.",
    source="policy.md",
)


def verdict(sensitivity: str) -> dict[str, Any]:
    return {
        "sensitivity": sensitivity,
        "complexity": "simple",
        "contains_pii": False,
        "reason": "test fixture",
    }


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply=verdict("public"))) as server:
        yield server


def cloud_only_settings(cloud: StubServer, *, max_retries: int, **overrides: object) -> Settings:
    """A deployment with one lane and two distinguishable tier identifiers."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        max_retries=max_retries,
        model_prices_usd_per_million=PRICES,
        **overrides,
    )


def run_to_the_gate(settings: Settings, request: str) -> dict[str, Any]:
    """Drive a run as far as the approval gate.

    Findings are seeded for the same reason the lane-enforcement harness seeds them: research
    is not what is under test, and skipping it keeps the run away from the embeddings path.
    """
    graph = build_graph(settings, build_checkpointer(settings))
    state = initial_state(request, str(uuid.uuid4()))
    state["dispatched"] = 1
    state["findings"] = [SEEDED_FINDING.as_channel()]
    return dict(graph.invoke(state, run_config(settings, str(uuid.uuid4()))))


def models_asked_of(stub: StubServer) -> list[str]:
    return [str(body.get("model", "")) for body in stub.behaviour.requests_seen]


# ------------------------------------------------------------------- the pin


def test_a_transient_failure_is_retried_on_a_model_the_graph_built(cloud: StubServer) -> None:
    """One 500, one retry, and the run finishes -- read off the endpoint's log.

    Red before item 15 was wired: the first node's model is built by ``build_model``, which
    sets ``max_retries=0`` and nothing adds them back, so a single transient error ends the
    run. The assertion is the request log rather than the absence of an exception, because a
    run that survived by some other route -- a fail-closed branch, a swallowed error -- would
    otherwise read as a pass.

    **The wiring point is ``build_graph``, not the node.** Mutation-checking this found it:
    putting ``build_model`` back as ``classify``'s own default leaves the test green, because
    ``build_graph`` passes its ``model_factory`` into every node and the node default is dead
    on that path. Reverting ``build_graph``'s default turns it red. Worth knowing before
    someone changes a node signature and believes they have changed the system -- the whole
    subject of item 15.
    """
    cloud.behaviour.fail_first_n = 1
    settings = cloud_only_settings(cloud, max_retries=1)

    result = run_to_the_gate(settings, PUBLIC_REQUEST)

    asked = models_asked_of(cloud)
    assert len(asked) >= 2, (
        f"the endpoint saw {len(asked)} request(s); a retried call sends a second one. "
        f"Models asked, in order: {asked}"
    )
    assert asked[:2] == [CLOUD_CHEAP, CLOUD_CHEAP], (
        "the first node's failed call was not retried on the same tier. A retry repeats the "
        f"request it just lost; models asked, in order: {asked}"
    )
    assert result.get("draft"), "the run did not reach a draft, so it did not recover at all"


# ------------------------------------------------------------------- the fallback's constraint


@pytest.fixture
def sovereign() -> Iterator[StubServer]:
    """The operator's own endpoint."""
    with running_stub(StubBehaviour(reply=verdict("restricted"))) as server:
        yield server


def hybrid_settings(cloud: StubServer, sovereign: StubServer, *, max_retries: int) -> Settings:
    """Both lanes constructible, defaulting to the cloud.

    The shape that makes the constraint expressible: a deployment where falling back to the
    third party is *possible*, so a chain that did it would succeed rather than error. On a
    deployment with no cloud lane the assertion below would hold for a reason that has nothing
    to do with the fallback.
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
        max_retries=max_retries,
        model_prices_usd_per_million=PRICES,
    )


RESTRICTED_REQUEST = (
    "Draft a refund letter for Jane Doe, account 4929-1123-8876, who was overcharged 240 GBP."
)

RESTRICTED_FINDING = Finding(
    question="What does the account record say?",
    content="Case notes: handler Ravi Chandrasekaran confirmed the overcharge on the 14th.",
    source="case-notes.md",
)

DRAFT_CANARIES: tuple[str, ...] = ("Ravi Chandrasekaran",)
"""Invented, and shown only to the drafter, so its appearance anywhere else means one thing."""


def run_restricted_to_the_gate(settings: Settings) -> dict[str, Any]:
    graph = build_graph(settings, build_checkpointer(settings))
    state = initial_state(RESTRICTED_REQUEST, str(uuid.uuid4()))
    state["dispatched"] = 1
    state["findings"] = [RESTRICTED_FINDING.as_channel()]
    return dict(graph.invoke(state, run_config(settings, str(uuid.uuid4()))))


def everything_sent_to(stub: StubServer) -> list[str]:
    """Every body the endpoint received, chat and embedding alike, as the text it was sent as.

    Both logs, for the reason the lane-enforcement harness gives: an absence assertion that
    reads only the chat log is blind to retrieval egress.
    """
    return [
        *(json.dumps(body, sort_keys=True) for body in stub.behaviour.requests_seen),
        *stub.behaviour.embedding_texts(),
    ]


def canaries_seen_by(stub: StubServer, canaries: Sequence[str]) -> list[str]:
    return sorted({c for c in canaries for body in everything_sent_to(stub) if c in body})


def test_a_sovereign_retry_stays_on_the_sovereign_endpoint(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """A transient failure on the operator's endpoint is retried there, not elsewhere.

    The presence half comes first: the retry has to have happened at all, or the absence
    assertion is about a run that never recovered and passes for the wrong reason.
    """
    sovereign.behaviour.fail_first_n = 1
    settings = hybrid_settings(cloud, sovereign, max_retries=1)

    result = run_restricted_to_the_gate(settings)

    assert result["lane"] == Lane.SOVEREIGN.value, "precondition: the router chose sovereign"
    assert sovereign.behaviour.request_count >= 2, (
        "the sovereign endpoint was not retried, so this run never exercised the retry path"
    )
    assert result.get("draft"), "the run did not recover, so it proves nothing about where to"
    assert cloud.behaviour.every_request == [], (
        "a retry after a sovereign failure reached the third-party endpoint"
    )


def test_an_exhausted_sovereign_lane_never_falls_back_to_the_cloud_endpoint(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """The canary. An outage must not do what the router refused to do.

    The sovereign endpoint answers the classifier and then goes down, so the failure lands on
    the drafter -- the capable tier, the one call in the system that *has* a fallback. Every
    attempt is spent and the run fails. Failing is the correct outcome: a draft produced by a
    third party out of restricted case notes would be leak 13, triggered by an outage instead
    of by a routing bug, at the moment nobody is watching for it.

    Three assertions, and the first two are what stop the third being vacuous:

    - the sovereign endpoint saw the classifier's call *and* the drafter's whole chain, so a
      fallback really was attempted rather than skipped;
    - the restricted findings reached that endpoint, so there was content available to leak;
    - nothing whatsoever reached the cloud endpoint.
    """
    settings = hybrid_settings(cloud, sovereign, max_retries=1)
    # Down from the second request on: the classifier is answered, the drafter is not.
    sovereign.behaviour.reject = lambda _body: sovereign.behaviour.request_count > 1

    with pytest.raises(Exception, match=r"(?i)error|500"):
        run_restricted_to_the_gate(settings)

    attempts_after_classification = sovereign.behaviour.request_count - 1
    assert attempts_after_classification == settings.max_retries + 2, (
        f"the drafter made {attempts_after_classification} attempt(s) on the sovereign "
        f"endpoint; {settings.max_retries + 1} retried attempts plus one fallback were due. "
        "A fallback that was never attempted cannot demonstrate where it would have gone"
    )
    assert canaries_seen_by(sovereign, DRAFT_CANARIES) == sorted(DRAFT_CANARIES), (
        "the drafter never sent the restricted findings anywhere, so there was nothing for a "
        "fallback to leak and the assertion below would pass for the wrong reason"
    )
    assert cloud.behaviour.every_request == [], (
        "the sovereign lane failed and the fallback crossed to the third-party endpoint. "
        "A fallback may narrow the lane and never widen it -- leak inventory item 13, "
        "triggered by an outage"
    )


# ------------------------------------------------------------------- every attempt is billed


def seeded_public_run(settings: Settings) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """A public run ready to invoke, with the config held so its ledger can be read."""
    graph = build_graph(settings, build_checkpointer(settings))
    config = run_config(settings, str(uuid.uuid4()))
    state = initial_state(PUBLIC_REQUEST, str(uuid.uuid4()))
    state["dispatched"] = 1
    state["findings"] = [SEEDED_FINDING.as_channel()]
    return graph, state, config


def test_the_model_that_answered_is_the_model_that_is_billed(cloud: StubServer) -> None:
    """A fallback reply is charged to the fallback, at the fallback's price.

    The trap is accounting the *composite*. One callback on the outer model records one call
    per invocation, reads the primary's identifier out of its own metadata, and prices a
    cheap-tier reply at capable-tier rates. Nothing errors -- the ledger is simply wrong, and
    only during an outage. Same shape as leak inventory item 11, which is why the callback is
    pushed down into the leaves instead.

    Asserted on the per-model book rather than on a total, because the two tiers are priced
    alike in the reference configuration and a total cannot tell them apart.
    """
    settings = cloud_only_settings(cloud, max_retries=0)
    # Involved, so the router sends the draft to the capable tier. Since leak inventory item 16
    # closed, a simple request is drafted on the cheap tier -- and a run where the capable tier
    # is never asked cannot demonstrate a fallback away from it.
    cloud.behaviour.reply = {**verdict("public"), "complexity": "involved"}
    # The capable tier is down; the cheap tier is healthy. The drafter falls back.
    cloud.behaviour.reject = lambda body: body.get("model") == CLOUD_CAPABLE

    graph, state, config = seeded_public_run(settings)
    ledger = config["configurable"][RUN_LEDGER]

    result = dict(graph.invoke(state, config))

    assert CLOUD_CAPABLE in models_asked_of(cloud), (
        "the capable tier was never asked, so nothing fell back and this bills nothing"
    )
    assert result.get("draft"), "the fallback produced no draft, so nothing was billed either"
    assert CLOUD_CHEAP in ledger.usage_by_model, (
        "the fallback answered and its usage was not recorded against it. The ledger holds: "
        f"{sorted(ledger.usage_by_model)}"
    )
    assert ledger.usage_by_model[CLOUD_CHEAP].output_tokens > 0, (
        "the fallback reply was recorded with no output tokens, so it was billed at zero"
    )


def test_a_failed_attempt_is_not_billed_and_the_answer_is_billed_once(cloud: StubServer) -> None:
    """Each attempt that reaches a provider is a real call; only the one that answers has usage.

    Both directions matter. Counting a failed attempt would charge for tokens nobody received;
    counting the successful one twice -- which is what leaving a callback on the composite as
    well as the leaves would do -- trips a ceiling on a healthy run.
    """
    cloud.behaviour.fail_first_n = 1
    settings = cloud_only_settings(cloud, max_retries=1)

    graph, state, config = seeded_public_run(settings)
    ledger = config["configurable"][RUN_LEDGER]

    graph.invoke(state, config)

    reached_the_provider = cloud.behaviour.request_count
    answered = reached_the_provider - cloud.behaviour.fail_first_n
    assert reached_the_provider > answered, (
        "no attempt failed, so this run cannot tell a double count from an honest one"
    )
    assert ledger.calls == answered, (
        f"the ledger counted {ledger.calls} call(s) against {answered} that came back with a "
        f"reply, out of {reached_the_provider} that reached the provider. A failed attempt "
        "reports no usage and must not be counted; an answer must be counted exactly once"
    )


# ------------------------------------------------------------- a guard is not a transient fault


def test_a_ceiling_crossed_is_not_retried_past(cloud: StubServer) -> None:
    """The system's own refusals leave immediately. A retried guard is not a guard.

    The ledger raises from inside the callback of the call that crossed the ceiling, so to a
    retry policy that treats every exception as transient it looks exactly like a 500. The
    answer to a budget ceiling would then be to call the provider again, and again, and then
    to fall back and call it once more -- every one of those a real, billed call made *after*
    the run was supposed to have stopped.

    ``with_retry`` retries on every exception by default, so the chain this replaced had the
    same defect. It was invisible for four phases because nothing called it. Wiring is what
    made it a bug rather than a curiosity, and this is the assertion that keeps it fixed.
    """
    settings = cloud_only_settings(cloud, max_retries=2, max_total_tokens=1)

    graph, state, config = seeded_public_run(settings)

    with pytest.raises(SpendCeilingExceededError):
        graph.invoke(state, config)

    assert cloud.behaviour.request_count == 1, (
        f"the ceiling was crossed by request 1 and {cloud.behaviour.request_count} reached the "
        "provider. Every one after the first was billed to a run that had already stopped"
    )
