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

Offline and free: both endpoints are loopback stubs, and no research branch runs, so nothing
embeds and nothing reaches a real provider.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import Lane, Settings
from agentgate.graph.build import build_checkpointer, build_graph
from agentgate.graph.state import Finding, initial_state

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-stub"
CLOUD_CHEAP = "cloud-cheap-stub"
SOVEREIGN_MODEL = "sovereign-stub"

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
    """The operator's own endpoint."""
    with running_stub(StubBehaviour(reply={"answer": "A draft of the refund letter."})) as server:
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
    """
    return [json.dumps(body, sort_keys=True) for body in stub.behaviour.requests_seen]


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
    asked_of_cloud = models_asked_of(cloud)
    assert asked_of_sovereign and asked_of_cloud, "precondition: both endpoints were called"

    drafted = next(e for e in result["audit_trail"] if e["decided"] == "drafted")
    assert drafted["lane"] == Lane.SOVEREIGN.value
    assert drafted["model"] in asked_of_sovereign, (
        "the drafted event names a model the endpoint it claims to have used was never asked for"
    )
    assert drafted["model"] not in asked_of_cloud
    assert drafted["detail"]["policy_route"] == Lane.SOVEREIGN.value
    assert drafted["detail"]["lane_narrowed_by_deployment"] is False

    # The lane-selection event records the same model resolution one node earlier, and had the
    # same defect: a sovereign binding annotated with the cloud lane's cheap model.
    lane_event = next(e for e in result["audit_trail"] if e["decided"] == "lane_selected")
    assert lane_event["model"] == SOVEREIGN_MODEL
    assert lane_event["detail"]["effective_lane"] == Lane.SOVEREIGN.value

    # The classifier's event is the one that was always true: it really does run on the
    # configured lane, and says so. Asserted so that a future change which starts routing the
    # classifier has to come through here.
    classified = next(e for e in result["audit_trail"] if e["decided"] == "classified")
    assert classified["lane"] == Lane.CLOUD.value
    assert classified["model"] in asked_of_cloud


# ------------------------------------------------- the gap this test file does not close


def test_the_classifier_sends_the_raw_request_to_the_configured_lane(
    cloud: StubServer, sovereign: StubServer
) -> None:
    """Recorded as a passing assertion because it is true, not because it is acceptable.

    Classification runs before the policy decision exists, so on a cloud-default deployment the
    raw request -- including anything restricted in it -- is sent to the third party in order to
    decide whether it was allowed to go there. Leak inventory item 14.

    It is pinned here so that closing it is a visible change to a test that asserts the
    opposite, rather than a silent improvement nobody can date.
    """
    settings = hybrid_settings(cloud, sovereign, default=Lane.CLOUD.value)

    run_to_the_gate(settings, RESTRICTED_REQUEST)

    assert canaries_seen_by(cloud, REQUEST_CANARIES) == sorted(REQUEST_CANARIES), (
        "the pre-classification egress described by item 14 no longer happens; if that is "
        "deliberate, this test is the one to rewrite"
    )
