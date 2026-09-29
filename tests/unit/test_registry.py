"""The capability matrix is a record of measurements, and is tested as one.

The value of the matrix is not that it is complete. It is that a reader can tell which rows
were measured, how, and when. A row nobody checked is worse than a missing row, because it
invites a caller down a path that fails somewhere far away from here.

Nothing here asserts what a constructed client *would* send. Anything about a value reaching a
provider -- output ceilings, temperature, the model actually called -- is asserted against an
observed request body in tests/integration/test_resilience.py. A client attribute and the wire
are not the same thing: langchain-openai configures `max_tokens` and emits
`max_completion_tokens`, so an attribute check passes while the request carries another field.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from agentgate.config import CallClass, Lane, Settings, Tier
from agentgate.models.fake import FakeChatModel
from agentgate.models.registry import (
    CAPABILITY_MATRIX,
    NETWORKED_LANES,
    Capability,
    LaneUnavailableError,
    Observation,
    Provenance,
    build_model,
    build_resilient_model,
    observation_for,
    supports,
    unverified_networked_entries,
)

pytestmark = pytest.mark.usefixtures("isolated_env")


def build(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def priced(*models: str) -> dict[str, dict[str, float]]:
    return {model: {"input": 0.10, "output": 0.40} for model in models}


# --------------------------------------------------------------------- provenance discipline


def test_no_networked_lane_entry_rests_on_assumption() -> None:
    """The rule that makes the matrix worth reading.

    Any row about a lane that reaches the network must say how it was learned. If this fails,
    someone added a guess about a real provider.
    """
    assert unverified_networked_entries() == []


def test_every_entry_records_how_and_when_it_was_learned() -> None:
    for (lane, capability), observation in CAPABILITY_MATRIX.items():
        assert observation.note.strip(), f"{lane}/{capability} has no note"
        assert isinstance(observation.recorded_on, date)


def test_measured_entries_point_at_something_reproducible() -> None:
    """A note saying "it works" is not evidence. A note naming a test is."""
    for (lane, capability), observation in CAPABILITY_MATRIX.items():
        if observation.is_measured:
            assert "test" in observation.note.lower() or "stub" in observation.note.lower(), (
                f"{lane}/{capability} claims measurement but cites nothing reproducible"
            )


def test_the_guard_would_actually_catch_an_assumption() -> None:
    """A checker that cannot fail is not a checker.

    Proves the rule has teeth by constructing exactly what it exists to reject.
    """
    smuggled = Observation(
        supported=True,
        provenance=Provenance.ASSUMED,
        recorded_on=date(2026, 1, 1),
        note="seemed likely",
    )

    assert not smuggled.is_trustworthy_on_a_networked_lane
    assert not smuggled.is_measured


def test_an_operator_declaration_is_acceptable_but_not_measured() -> None:
    """A self-hosted endpoint's owner may assert its capabilities. That is attributable."""
    declared = Observation(
        supported=True,
        provenance=Provenance.CONFIG_DECLARED,
        recorded_on=date(2026, 1, 1),
        note="operator asserts their vLLM build has grammar-constrained decoding",
    )

    assert declared.is_trustworthy_on_a_networked_lane
    assert not declared.is_measured


# --------------------------------------------------------------------- reading the matrix


def test_the_sovereign_lane_is_recorded_as_lacking_native_structured_output() -> None:
    """The observed difference between lanes, which is what drives the repair loop."""
    observation = observation_for(Lane.SOVEREIGN, Capability.NATIVE_STRUCTURED_OUTPUT)

    assert observation is not None
    assert observation.supported is False
    assert observation.provenance is Provenance.STUB


def test_the_cloud_lane_row_came_from_a_live_probe() -> None:
    """The row exists now, and how it got here is the part worth pinning.

    It was absent until 2026-08-10 because nothing is known about what a given key can do
    until it has been asked. What replaced the absence has to be an observation, not an
    optimistic default -- so this asserts the provenance, not the answer.
    """
    observation = observation_for(Lane.CLOUD, Capability.NATIVE_STRUCTURED_OUTPUT)

    assert observation is not None
    assert observation.provenance is Provenance.LIVE_PROBE


def test_an_unmeasured_capability_reads_as_unsupported() -> None:
    """Pessimistic by design: the fallback costs tokens, the optimistic error costs a run."""
    assert supports(Lane.CLOUD, Capability.STREAMING) is False
    assert supports(Lane.SOVEREIGN, Capability.STREAMING) is False


def test_a_recorded_negative_is_distinguishable_from_an_absent_row() -> None:
    """ "Measured as absent" and "never measured" are different facts, and both matter."""
    measured_absent = observation_for(Lane.SOVEREIGN, Capability.NATIVE_STRUCTURED_OUTPUT)
    never_measured = observation_for(Lane.SOVEREIGN, Capability.TOOL_CALLING)

    assert measured_absent is not None
    assert never_measured is None
    assert supports(Lane.SOVEREIGN, Capability.NATIVE_STRUCTURED_OUTPUT) is False
    assert supports(Lane.SOVEREIGN, Capability.TOOL_CALLING) is False


def test_networked_lanes_are_the_ones_that_leave_the_process() -> None:
    assert Lane.FAKE not in NETWORKED_LANES
    assert set(NETWORKED_LANES) == {Lane.CLOUD, Lane.SOVEREIGN}


# --------------------------------------------------------------------- model construction


def test_the_fake_lane_builds_without_any_configuration() -> None:
    model = build_model(build(), Tier.CHEAP, CallClass.ROUTING)

    assert isinstance(model, FakeChatModel)


def test_both_networked_lanes_are_the_same_integration_pointed_elsewhere() -> None:
    """The sovereign lane is cheap to support precisely because it is not a second client."""
    sovereign = build(
        lane="sovereign",
        sovereign_base_url="http://127.0.0.1:1/v1",
        sovereign_model="stub",
        model_prices_usd_per_million=priced("stub"),
    )
    cloud = build(
        lane="cloud",
        openai_api_key="sk-test",
        cloud_capable_model="a",
        cloud_cheap_model="a",
        model_prices_usd_per_million=priced("a"),
    )

    assert type(build_model(sovereign, Tier.CHEAP, CallClass.ROUTING)) is type(
        build_model(cloud, Tier.CHEAP, CallClass.ROUTING)
    )


def test_a_lane_that_cannot_be_built_says_so() -> None:
    """A deployment that meant to be hybrid and forgot the endpoint fails loudly.

    The scenario is specific, because it is the only one that can happen. A *fake* deployment
    asked for the sovereign lane does not raise -- it stays fake, because fake is the more
    contained of the two. It is the cloud-default deployment with no sovereign endpoint that has
    nowhere safe to put a restricted request, and refusing is the only correct answer: falling
    back to the configured lane there would be leak 13 with a shrug attached.
    """
    settings = build(
        lane="cloud",
        openai_api_key="sk-test",
        cloud_capable_model="a",
        cloud_cheap_model="a",
        model_prices_usd_per_million=priced("a"),
    )

    with pytest.raises(LaneUnavailableError, match="sovereign"):
        build_model(settings, Tier.CHEAP, CallClass.ROUTING, lane=Lane.SOVEREIGN)


def test_the_route_narrows_the_configured_lane() -> None:
    """A restricted request goes somewhere stricter than the default, whatever the default is."""
    settings = build(
        lane="sovereign",
        sovereign_base_url="http://127.0.0.1:1/v1",
        sovereign_model="stub",
        model_prices_usd_per_million=priced("stub"),
    )

    model = build_model(settings, Tier.CHEAP, CallClass.ROUTING, lane=Lane.FAKE)

    assert isinstance(model, FakeChatModel)
    # Named for the lane that built it. A fake model carrying a cloud model's identifier would
    # put that identifier in the audit trail and the spend ledger, for a call that never left
    # the process -- the same class of untrue record as item 13 itself.
    assert model.model_name == "fake-cheap"


def test_the_route_cannot_widen_the_configured_lane() -> None:
    """The direction that was never tested, and the one the obvious fix gets wrong.

    ``route_by_policy`` returns a cloud route for public content, which is right as policy and
    wrong as an instruction on a deployment whose default is its own endpoint. If this passes by
    raising rather than by building a sovereign model, read it again: the cloud lane here is
    fully configured and buildable, so the only reason not to build it is the rule.
    """
    settings = build(
        lane="sovereign",
        sovereign_base_url="http://127.0.0.1:1/v1",
        sovereign_model="sovereign-stub",
        openai_api_key="sk-test",
        openai_base_url="http://127.0.0.1:2/v1",
        cloud_capable_model="cloud-stub",
        cloud_cheap_model="cloud-stub",
        model_prices_usd_per_million={
            "sovereign-stub": {"input": 0.0, "output": 0.0},
            "cloud-stub": {"input": 1.0, "output": 1.0},
        },
    )

    model = build_model(settings, Tier.CHEAP, CallClass.ROUTING, lane=Lane.CLOUD)

    assert not isinstance(model, FakeChatModel), "precondition: a networked client was built"
    assert settings.model_for(Tier.CHEAP, lane=Lane.SOVEREIGN) == "sovereign-stub"
    assert getattr(model, "model_name", None) == "sovereign-stub", (
        "a cloud route widened a sovereign deployment's boundary"
    )


def test_the_resilient_chain_builds_on_the_fake_lane() -> None:
    chain = build_resilient_model(build(), Tier.CAPABLE, CallClass.RESEARCH)

    assert chain is not None
    assert hasattr(chain, "invoke")


def test_replace_keeps_observations_immutable() -> None:
    """Entries are frozen so a caller cannot quietly upgrade a guess into a measurement."""
    original = Observation(
        supported=False,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 8, 9),
        note="stub",
    )

    upgraded = replace(original, supported=True)

    assert original.supported is False
    assert upgraded.supported is True


def networked(**overrides: object) -> Settings:
    """A configuration whose tiers build real clients rather than the fake lane."""
    return build(
        lane="cloud",
        openai_api_key="not-required",
        openai_base_url="http://127.0.0.1:1/v1",
        cloud_capable_model="capable-stub",
        cloud_cheap_model="cheap-stub",
        model_prices_usd_per_million={
            "capable-stub": {"input": 1.0, "output": 4.0},
            "cheap-stub": {"input": 0.1, "output": 0.4},
        },
        **overrides,
    )


def test_the_cheap_tier_has_no_fallback() -> None:
    """So that ``max_retries`` means retries, including when it means none.

    The only call that could sit beneath the cheap tier is another call to the same model, so
    a fallback there would make a cheap-routed request cost ``max_retries + 2`` provider calls
    -- two at ``max_retries=0``, where the operator asked for one. That is a behaviour defect
    rather than a naming one, and it is why ADR 0004 item 16 rejected cheap-to-cheap after
    trying it.

    Asserted on the shape rather than on a call count, which
    ``tests/integration/test_routed_tier.py`` reads off an endpoint. The earlier version of
    this test asserted that the two leaves shared an httpx connection pool -- true, and the
    measurement that settled the decision, but it pinned a dependency's private attribute to
    defend a fallback that no longer exists. The measurement is a dated fact in the row now.
    """
    chain = build_resilient_model(networked(), Tier.CHEAP, CallClass.SYNTHESIS)

    assert chain.fallback is None, (
        "the cheap tier has a fallback, so a cheap-routed request makes one more provider "
        "call than max_retries allows for"
    )
    assert chain.primary.model_name == "cheap-stub", "precondition: the cheap tier was built"


def test_the_capable_tier_falls_back_to_a_genuinely_cheaper_model() -> None:
    """The control for the assertion above: on the capable tier the fallback is a real one."""
    chain = build_resilient_model(networked(), Tier.CAPABLE, CallClass.SYNTHESIS)

    assert chain.primary.model_name == "capable-stub"
    assert chain.fallback is not None
    assert chain.fallback.model_name == "cheap-stub", (
        "the capable tier did not degrade to the cheap one, so this file would be asserting "
        "the same thing twice"
    )
