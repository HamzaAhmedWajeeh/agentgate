"""The routed tier, read off the wire rather than off the trail.

Leak inventory item 16. ``bind_lane`` binds a tier as well as a lane and records it in the
audit trail; no channel carried it, and the drafter always asked for the capable one. So
``route_by_policy`` distinguishing a simple request from an involved one changed the trail and
nothing else, for as long as nobody checked.

It survived because the one deployment that would reveal it is the one nobody runs: in the
reference configuration both cloud tiers name the same model, so the routed tier and the
constant were indistinguishable on the wire. These tests give the two tiers different
identifiers, which is the whole trick -- the same trick item 13 needed, for the same reason.

**The assertions read request logs, never state.** ``result["tier"]`` is a label the graph
wrote about itself, and a label is not a model identifier. That confusion is exactly what hid
item 13 for two phases, and item 16 is its sibling.

Offline and free. Every endpoint is a loopback stub.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from langgraph.types import Command
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

from agentgate.config import Lane, Settings
from agentgate.graph.build import build_checkpointer, build_graph, resume_config, run_config
from agentgate.graph.state import Decision, Finding, initial_state

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-stub"
CLOUD_CHEAP = "cloud-cheap-stub"
SOVEREIGN_MODEL = "sovereign-stub"

PRICES = {
    CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
    CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
    SOVEREIGN_MODEL: {"input": 0.0, "output": 0.0},
}

SIMPLE_PUBLIC_REQUEST = "Summarise our published refund window for the website."

# Only ever shown to the drafter, in the brief it builds from the findings. Its presence in a
# request body is what identifies that body as the drafter's rather than the classifier's --
# read rather than inferred from position, because a position depends on how many calls each
# node happens to make and would quietly stop identifying anything if that changed.
DRAFTER_MARKER = "Ravi Chandrasekaran"

SEEDED_FINDING = Finding(
    question="What does the published policy say?",
    content=f"Handler {DRAFTER_MARKER} confirms refunds are available within 30 days.",
    source="policy.md",
)


def verdict(sensitivity: str, complexity: str) -> dict[str, Any]:
    """What the stub standing in for the classifier's lane returns."""
    return {
        "sensitivity": sensitivity,
        "complexity": complexity,
        "contains_pii": False,
        "reason": "test fixture",
    }


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    """The third-party endpoint, answering the classifier with public and simple."""
    with running_stub(StubBehaviour(reply=verdict("public", "simple"))) as server:
        yield server


@pytest.fixture
def sovereign() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply=verdict("restricted", "simple"))) as server:
        yield server


def cloud_only_settings(cloud: StubServer, **overrides: object) -> Settings:
    """One lane, two tiers, two different identifiers.

    The identifiers differing is the entire experiment. With both tiers naming one model --
    which is the documented reference configuration -- every assertion in this file would pass
    against the defect.
    """
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        model_prices_usd_per_million=PRICES,
        **overrides,
    )


def hybrid_settings(cloud: StubServer, sovereign: StubServer, **overrides: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane=Lane.CLOUD.value,
        openai_api_key="not-required",
        openai_base_url=cloud.base_url,
        cloud_capable_model=CLOUD_CAPABLE,
        cloud_cheap_model=CLOUD_CHEAP,
        sovereign_base_url=sovereign.base_url,
        sovereign_model=SOVEREIGN_MODEL,
        model_prices_usd_per_million=PRICES,
        **overrides,
    )


def seeded(request: str) -> Any:
    """A run seeded past research, so the only model calls are the classifier's and drafter's."""
    state = initial_state(request, str(uuid.uuid4()))
    state["dispatched"] = 1
    state["findings"] = [SEEDED_FINDING.as_channel()]
    return state


def drafter_requests(stub: StubServer) -> list[dict[str, Any]]:
    """Every request body that carries the drafter's brief.

    Identified by the marker only the drafter is shown, not by position in the log. The
    classifier and the drafter both talk to this endpoint, and a test that assumed "request
    two onwards" would silently start asserting about the wrong call the moment either node
    made a different number of them.
    """
    return [
        body
        for body in stub.behaviour.requests_seen
        if DRAFTER_MARKER in json.dumps(body, sort_keys=True)
    ]


def models_asked_of(stub: StubServer) -> list[str]:
    return [str(body.get("model", "")) for body in stub.behaviour.requests_seen]


def lane_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in result.get("audit_trail", [])
        if str(event.get("decided", "")) == "lane_selected"
    ]


def drafting_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The drafter's events that name a model.

    It writes others -- a dropped proposal, a fabricated citation -- which carry no model
    because no model call produced them. Including those would put a ``None`` in the set and
    make this assertion fail for a reason that has nothing to do with the tier.
    """
    return [
        event
        for event in result.get("audit_trail", [])
        if event.get("node") == "drafter" and str(event.get("decided", "")) == "drafted"
    ]


# ------------------------------------------------------------------- the pin


def test_a_cheap_routed_request_is_drafted_by_the_cheap_model(cloud: StubServer) -> None:
    """The defect, stated on the wire.

    A public, simple request routes to ``cloud_cheap``. Before item 16 was wired the trail
    said so and the drafter asked the endpoint for the capable model anyway. The presence
    assertion comes first: without it, "no request asked for the capable model" is true of a
    run that never drafted at all.
    """
    settings = cloud_only_settings(cloud)
    graph = build_graph(settings, build_checkpointer(settings))

    result = dict(
        graph.invoke(seeded(SIMPLE_PUBLIC_REQUEST), run_config(settings, str(uuid.uuid4())))
    )

    assert [event.get("node") for event in lane_events(result)] == ["cloud_cheap"], (
        "precondition: the router did not send this request to the cheap tier"
    )
    asked = drafter_requests(cloud)
    assert asked, (
        "no request carried the drafter's brief, so the endpoint was never asked to draft and "
        f"there is nothing to assert about. Models asked: {models_asked_of(cloud)}"
    )
    assert [str(body.get("model", "")) for body in asked] == [CLOUD_CHEAP] * len(asked), (
        "the request was routed to the cheap tier and the drafter asked the endpoint for "
        f"something else. The drafter's calls asked for: "
        f"{[str(body.get('model', '')) for body in asked]}"
    )


def test_the_trail_names_the_model_that_actually_answered(cloud: StubServer) -> None:
    """Half a fix is its own defect, and this is the half that fails quietly.

    The drafter builds its model and writes its audit event from two separate reads of the
    tier. Move one and not the other and every wire assertion above passes while the trail
    names a model the endpoint was never asked for -- which is item 13's half-fix exactly,
    and the one way this change goes wrong without anything turning red.
    """
    settings = cloud_only_settings(cloud)
    graph = build_graph(settings, build_checkpointer(settings))

    result = dict(
        graph.invoke(seeded(SIMPLE_PUBLIC_REQUEST), run_config(settings, str(uuid.uuid4())))
    )

    asked = drafter_requests(cloud)
    assert asked, "the drafter made no call, so the trail has nothing to agree or disagree with"
    on_the_wire = {str(body.get("model", "")) for body in asked}

    events = drafting_events(result)
    assert events, "the drafter wrote no audit event, so there is no claim to check"
    claimed = {str(event.get("model", "")) for event in events}

    assert claimed == on_the_wire, (
        f"the trail says the draft was made by {sorted(claimed)} and the endpoint was asked "
        f"for {sorted(on_the_wire)}. A trail naming a model that did not answer is worse than "
        "no trail: it is a record that reads as evidence"
    )


def test_an_involved_request_still_reaches_the_capable_tier(cloud: StubServer) -> None:
    """The control. Without it the pin above passes against a drafter hardwired to cheap.

    Swapping one constant for another is not wiring, and a suite that only ever checks the
    cheap route cannot tell the difference.
    """
    cloud.behaviour.reply = verdict("public", "involved")
    settings = cloud_only_settings(cloud)
    graph = build_graph(settings, build_checkpointer(settings))

    result = dict(
        graph.invoke(
            seeded("Compare our refund policy with three competitors."),
            run_config(settings, str(uuid.uuid4())),
        )
    )

    assert [event.get("node") for event in lane_events(result)] == ["cloud_capable"], (
        "precondition: the router did not send this request to the capable tier"
    )
    asked = drafter_requests(cloud)
    assert asked, "the drafter made no call"
    assert [str(body.get("model", "")) for body in asked] == [CLOUD_CAPABLE] * len(asked), (
        f"an involved request was drafted by {[str(body.get('model', '')) for body in asked]}"
    )


# ------------------------------------------------------------------- resuming an old checkpoint


def test_a_checkpoint_written_before_the_channel_resumes_on_the_capable_tier(
    cloud: StubServer,
) -> None:
    """A run paused before the tier channel existed must resume, and resume unchanged.

    The tier is a cost decision, so the default is today's behaviour rather than the cheaper
    one: fail-closed belongs on the containment axis, where there is a safe direction, and
    there is none on cost. ``lane_of`` defaults to ``SOVEREIGN`` for that reason and this
    deliberately does not copy it.

    The pause is simulated by emptying the channel, which is what
    ``state.get("tier", "")`` sees for a key that was never written.

    **The update names no node, and that is load bearing** -- leak inventory item 26. Naming
    ``drafter`` was the first attempt and it moved ``next`` from ``approval_gate`` to
    ``supervisor``, consuming the pause: the resume then re-entered the gate without drafting
    anything, and the assertion below failed for a reason that had nothing to do with tiers.
    With no node named the update is credited to the last writer, ``assess``, which has a
    static edge into the gate, so the pending resume applies to a fresh pass through it. The
    absence of ``as_node`` here is the fix, not an omission.
    """
    settings = cloud_only_settings(cloud)
    graph = build_graph(settings, build_checkpointer(settings))
    thread = str(uuid.uuid4())

    graph.invoke(seeded(SIMPLE_PUBLIC_REQUEST), run_config(settings, thread))
    before = graph.get_state({"configurable": {"thread_id": thread}})
    assert (before.values or {}).get("tier") == "cheap", (
        "precondition: this run wrote a cheap tier, so clearing it is a real change"
    )

    graph.update_state({"configurable": {"thread_id": thread}}, {"tier": ""})
    after = graph.get_state({"configurable": {"thread_id": thread}})
    assert after.next == ("approval_gate",), (
        f"the update moved the run to {after.next}, so the pause was consumed and the resume "
        "below will not reach the drafter. See leak inventory item 26"
    )
    cloud.behaviour.reset()

    result = dict(
        graph.invoke(
            Command(resume={"decision": Decision.REJECTED.value, "feedback": "shorter, please"}),
            resume_config(graph, settings, thread),
        )
    )

    asked = drafter_requests(cloud)
    assert asked, (
        "the revision never reached the drafter, so this proves nothing about what an old "
        "checkpoint resumes on"
    )
    assert [str(body.get("model", "")) for body in asked] == [CLOUD_CAPABLE] * len(asked), (
        "a checkpoint with no tier channel did not resume on the capable tier. It asked for "
        f"{[str(body.get('model', '')) for body in asked]}. Defaulting the other way would "
        "make an old run answer worse than it started"
    )
    assert result.get("draft"), "the resumed run did not complete"


# ------------------------------------------------------------------- the tier and max_retries


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_a_cheap_routed_request_makes_exactly_the_configured_number_of_calls(
    cloud: StubServer, max_retries: int
) -> None:
    """``max_retries`` means retries, including when it means none.

    This is the reason the cheap tier has no fallback, and it is a behaviour defect rather
    than a naming one. A fallback beneath the cheap tier is another call to the same model
    through the same connection pool, so a cheap-routed request would make
    ``max_retries + 2`` provider calls. At ``max_retries=0`` -- an operator saying *do not
    retry* -- that is two. The setting would not mean what it says, and the extra call would be
    billed, on a request the deployment had asked to be tried once.

    The capable tier keeps its fallback because the call beneath it is a **different model**:
    a degradation, not a spare attempt. ADR 0004 item 16.

    Parametrised deliberately. A single case at ``max_retries=1`` cannot tell "one retry plus a
    fallback" from "two attempts and no fallback", and zero is the case that makes the point.
    """
    settings = cloud_only_settings(cloud, max_retries=max_retries)
    # The drafter's every attempt fails, so the log shows the whole chain rather than however
    # much of it the first success happened to use.
    cloud.behaviour.reject = lambda body: DRAFTER_MARKER in json.dumps(body, sort_keys=True)

    graph = build_graph(settings, build_checkpointer(settings))
    with pytest.raises(Exception, match=r"(?i)error|500"):
        graph.invoke(seeded(SIMPLE_PUBLIC_REQUEST), run_config(settings, str(uuid.uuid4())))

    asked = drafter_requests(cloud)
    assert asked, "the drafter never reached the endpoint, so no count was measured"
    assert len(asked) == max_retries + 1, (
        f"the drafter made {len(asked)} provider call(s) with max_retries={max_retries}; "
        f"{max_retries + 1} were due. An extra call here is a fallback to the same model "
        "through the same connection pool -- one more retry, billed, on a request the "
        "operator configured not to retry that many times"
    )
    assert [str(body.get("model", "")) for body in asked] == [CLOUD_CHEAP] * len(asked), (
        "a cheap-routed attempt asked for another tier. Escalating on exhaustion costs more "
        f"at exactly the moment nobody is watching. Asked for: "
        f"{[str(body.get('model', '')) for body in asked]}"
    )


def test_a_capable_routed_request_still_degrades_to_the_cheap_model(cloud: StubServer) -> None:
    """The control, and the distinction the decision above rests on.

    Without it, "the cheap tier makes exactly its retries" holds just as well against a build
    with no fallbacks at all -- and the capable tier's fallback is the thing that makes the
    cheap tier's absence a considered asymmetry rather than an oversight.
    """
    cloud.behaviour.reply = verdict("public", "involved")
    settings = cloud_only_settings(cloud, max_retries=1)
    cloud.behaviour.reject = lambda body: (
        body.get("model") == CLOUD_CAPABLE and DRAFTER_MARKER in json.dumps(body, sort_keys=True)
    )

    graph = build_graph(settings, build_checkpointer(settings))
    result = dict(
        graph.invoke(
            seeded("Compare our refund policy with three competitors."),
            run_config(settings, str(uuid.uuid4())),
        )
    )

    asked = [str(body.get("model", "")) for body in drafter_requests(cloud)]
    assert asked == [CLOUD_CAPABLE] * (settings.max_retries + 1) + [CLOUD_CHEAP], (
        f"the capable tier did not exhaust its retries and then degrade. Asked for: {asked}"
    )
    assert result.get("draft"), "the degraded call did not answer"


def test_no_tier_decision_widens_the_lane(cloud: StubServer, sovereign: StubServer) -> None:
    """The tier is chosen inside a lane and can never be a way out of one.

    A restricted request binds the sovereign lane at the cheap tier. Both the retries and the
    fallback stay there, and the cloud endpoint -- which this deployment *can* reach -- sees
    nothing at all.
    """
    settings = hybrid_settings(cloud, sovereign, max_retries=1)
    sovereign.behaviour.reject = lambda body: DRAFTER_MARKER in json.dumps(body, sort_keys=True)

    graph = build_graph(settings, build_checkpointer(settings))
    with pytest.raises(Exception, match=r"(?i)error|500"):
        graph.invoke(
            seeded("Draft a refund letter for account 4929-1123-8876."),
            run_config(settings, str(uuid.uuid4())),
        )

    attempts = drafter_requests(sovereign)
    # The sovereign lane binds the cheap tier, which has no fallback, so the whole chain is
    # its retries -- see ADR 0004 item 16. Spending all of them is the precondition: a chain
    # with attempts left cannot show where it would have gone when it ran out.
    assert len(attempts) == settings.max_retries + 1, (
        f"the drafter made {len(attempts)} attempt(s) on the sovereign endpoint; "
        f"{settings.max_retries + 1} were due, and the chain has to be spent before its "
        "next destination means anything"
    )
    assert cloud.behaviour.every_request == [], (
        "a cheap-tier chain on the sovereign lane reached the third-party endpoint"
    )


def canaries_seen_by(stub: StubServer, canaries: Sequence[str]) -> list[str]:
    bodies = [
        *(json.dumps(body, sort_keys=True) for body in stub.behaviour.requests_seen),
        *stub.behaviour.embedding_texts(),
    ]
    return sorted({c for c in canaries for body in bodies if c in body})
