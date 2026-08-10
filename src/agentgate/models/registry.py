"""Lane definitions, the capability matrix, and model construction.

One interface, three lanes. A node asks for "the cheap tier" and gets a model; it never learns
which lane answered, because that is a policy decision made upstream by the router.

**The capability matrix records observations, not beliefs.** Every entry carries how it was
learned and when. A row that says a lane supports something because it seemed likely is worth
less than no row at all: it invites callers to take a path that fails in production, and it
cannot be audited. So provenance is part of the data, and the suite refuses to accept a bare
assumption about any lane that reaches the network.

A capability with no recorded entry is treated as absent. That is the pessimistic reading, and
it is deliberate: the fallback path costs more tokens but works everywhere, whereas assuming a
capability that is missing produces a provider error in a node with no idea what a lane is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agentgate.config import CallClass, Lane, Settings, Tier
from agentgate.errors import AgentgateError
from agentgate.models.fake import FakeChatModel


class LaneUnavailableError(AgentgateError):
    """A lane was requested that this configuration cannot construct."""


class Capability(StrEnum):
    """Something a lane either can or cannot do.

    Kept to behaviours that change how calling code is written. "Supports long contexts" is a
    number, not a capability; "returns a schema-valid object without post-processing" changes
    which code path runs.
    """

    NATIVE_STRUCTURED_OUTPUT = "native_structured_output"
    TOOL_CALLING = "tool_calling"
    USAGE_METADATA = "usage_metadata"
    STREAMING = "streaming"


class Provenance(StrEnum):
    """How a capability entry came to be known.

    Ordered loosely by strength of evidence. ``ASSUMED`` exists so the checker has something
    to catch -- it is never a valid state for a lane that reaches the network.
    """

    LIVE_PROBE = "live_probe"
    """Measured against the real provider by a live test."""

    STUB = "stub"
    """Measured against the committed OpenAI-compatible stub server over real HTTP."""

    IN_PROCESS = "in_process"
    """Measured against the in-process fake, whose behaviour is defined by this repository."""

    CONFIG_DECLARED = "config_declared"
    """Asserted by the operator for their own endpoint. Trusted, but attributed to them."""

    ASSUMED = "assumed"
    """Nobody checked. Rejected on any networked lane."""


NETWORKED_LANES: Final = frozenset({Lane.CLOUD, Lane.SOVEREIGN})
MEASURED: Final = frozenset({Provenance.LIVE_PROBE, Provenance.STUB, Provenance.IN_PROCESS})


@dataclass(frozen=True)
class Observation:
    """What was seen, how, and when.

    Attributes:
        supported: Whether the capability was present.
        provenance: How this was learned.
        recorded_on: The date the observation was made. Providers change; an old date is a
            reason to re-probe rather than to trust harder.
        note: Where to look to reproduce it -- ideally the test that measures it.
    """

    supported: bool
    provenance: Provenance
    recorded_on: date
    note: str

    @property
    def is_measured(self) -> bool:
        return self.provenance in MEASURED

    @property
    def is_trustworthy_on_a_networked_lane(self) -> bool:
        return self.provenance is not Provenance.ASSUMED


# ---------------------------------------------------------------------------------------
# The matrix.
#
# Rows exist only where something was actually observed. An absent row means "not measured",
# which callers read as "not supported" -- see supports(). Adding a row is a claim, so it
# needs a provenance and a date, and for a networked lane it needs to be more than a guess.
#
# The cloud lane stayed absent until a live probe had run, because nothing is known about what
# a given key can do until it has been asked. Its one row now is what the probe observed, and
# the live suite fails if the provider stops agreeing with it.
# ---------------------------------------------------------------------------------------

CAPABILITY_MATRIX: Final[Mapping[tuple[Lane, Capability], Observation]] = {
    (Lane.CLOUD, Capability.NATIVE_STRUCTURED_OUTPUT): Observation(
        supported=True,
        provenance=Provenance.LIVE_PROBE,
        recorded_on=date(2026, 8, 10),
        note=(
            "Probed against gpt-4.1-nano on 2026-08-10 via scripts/probe_capabilities.py. "
            "Native path returned a valid ProbeSchema without post-processing. Enforced by "
            "tests/live/test_cloud_lane.py::"
            "test_the_recorded_structured_output_capability_is_still_true"
        ),
    ),
    (Lane.FAKE, Capability.NATIVE_STRUCTURED_OUTPUT): Observation(
        supported=True,
        provenance=Provenance.IN_PROCESS,
        recorded_on=date(2026, 8, 9),
        note="tests/unit/test_fake_lane.py::test_native_structured_output_parses_a_scripted_reply",
    ),
    (Lane.FAKE, Capability.USAGE_METADATA): Observation(
        supported=True,
        provenance=Provenance.IN_PROCESS,
        recorded_on=date(2026, 8, 9),
        note="tests/unit/test_fake_lane.py::test_every_reply_reports_usage",
    ),
    (Lane.SOVEREIGN, Capability.NATIVE_STRUCTURED_OUTPUT): Observation(
        supported=False,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 8, 9),
        note=(
            "Measured against tests/doubles/openai_compatible.py over real HTTP. The endpoint "
            "ignores response_format and returns JSON wrapped in prose and a code fence, so "
            "the native path raises and callers must use validate-and-repair. See "
            "tests/integration/test_sovereign_lane_structured_output.py::"
            "test_native_structured_output_fails_against_the_sovereign_lane"
        ),
    ),
    (Lane.SOVEREIGN, Capability.USAGE_METADATA): Observation(
        supported=True,
        provenance=Provenance.STUB,
        recorded_on=date(2026, 8, 9),
        note="The stub returns an OpenAI-shaped usage block and the client surfaces it.",
    ),
}


def observation_for(lane: Lane, capability: Capability) -> Observation | None:
    """Return what is known about a lane's capability, or ``None`` if nothing was measured."""
    return CAPABILITY_MATRIX.get((lane, capability))


def supports(lane: Lane, capability: Capability) -> bool:
    """Whether a lane supports a capability.

    An unmeasured capability reads as unsupported. The cost of that being wrong is some extra
    tokens on a fallback path; the cost of the opposite error is a provider exception in a
    node that cannot interpret it.
    """
    observation = observation_for(lane, capability)
    return observation is not None and observation.supported


def unverified_networked_entries() -> list[tuple[Lane, Capability]]:
    """Entries on a networked lane that rest on nothing but assumption.

    A matrix is only worth having if a reader can tell which rows were measured, so the suite
    treats any such entry as a failure rather than a warning.
    """
    return sorted(
        (lane, capability)
        for (lane, capability), observation in CAPABILITY_MATRIX.items()
        if lane in NETWORKED_LANES and not observation.is_trustworthy_on_a_networked_lane
    )


# ---------------------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------------------


class ModelFactory(Protocol):
    """How a node obtains a model.

    Exists so a test can script the model a node will use without patching a global or
    reaching inside the node. The default is :func:`build_model`; the fake lane needs
    scripting because an unscripted reply is a hash, which is not valid JSON -- and a
    classifier that cannot parse its own model's output fails closed to `restricted`, which
    would make every test look like a policy test.
    """

    def __call__(
        self, settings: Settings, tier: Tier, call_class: CallClass, *, lane: Lane | None = ...
    ) -> BaseChatModel: ...


def build_model(
    settings: Settings,
    tier: Tier,
    call_class: CallClass,
    *,
    lane: Lane | None = None,
) -> BaseChatModel:
    """Construct the model for a tier on a lane.

    Args:
        settings: Resolved configuration.
        tier: Capable or cheap. Both may point at the same model; the split is a policy
            boundary, not a promise that one is more expensive.
        call_class: Determines the output ceiling, so a routing decision cannot spend a
            synthesis-sized budget.
        lane: Overrides the configured default, for when the policy router has sent an
            individual request somewhere stricter.

    Raises:
        LaneUnavailableError: if the lane cannot be built from this configuration.
    """
    chosen = lane or settings.lane
    model_id = settings.model_for(tier)
    max_tokens = settings.max_tokens_for(call_class)

    if chosen is Lane.FAKE:
        return FakeChatModel(model_name=model_id)

    if chosen is Lane.CLOUD:
        if settings.openai_api_key is None:
            msg = "cloud lane requested but no API key is configured"
            raise LaneUnavailableError(msg)
        return _init_openai_compatible(
            model_id,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            settings=settings,
            max_tokens=max_tokens,
        )

    if settings.sovereign_base_url is None:
        msg = "sovereign lane requested but no base URL is configured"
        raise LaneUnavailableError(msg)
    return _init_openai_compatible(
        model_id,
        api_key=settings.sovereign_api_key.get_secret_value(),
        base_url=settings.sovereign_base_url,
        settings=settings,
        max_tokens=max_tokens,
    )


def _init_openai_compatible(
    model_id: str,
    *,
    api_key: str,
    base_url: str | None,
    settings: Settings,
    max_tokens: int,
) -> BaseChatModel:
    """Both networked lanes speak the same dialect; only the endpoint differs.

    That is the whole reason the sovereign lane is cheap to support: it is not a second
    integration, it is the same one pointed somewhere else.
    """
    return init_chat_model(
        model_id,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=settings.temperature,
        max_tokens=max_tokens,
        timeout=settings.request_timeout_seconds,
        max_retries=0,  # retries are applied by build_resilient_model, in one place
    )


def build_resilient_model(
    settings: Settings,
    call_class: CallClass,
    *,
    lane: Lane | None = None,
) -> Runnable[LanguageModelInput, AIMessage]:
    """The capable tier, retried, falling back to the cheap tier, then failing clearly.

    Retry handles the transient case -- a reset connection, a rate limit -- where the same
    request will probably work in a moment. Fallback handles the durable case, where this
    model is not going to answer and a smaller one answering is better than nothing.

    Both tiers commonly resolve to the same model, in which case the fallback is a second
    attempt with fresh state rather than a downgrade. That is still worth having, and it
    costs nothing when the first attempt succeeds.
    """
    capable = build_model(settings, Tier.CAPABLE, call_class, lane=lane).with_retry(
        stop_after_attempt=settings.max_retries + 1,
        wait_exponential_jitter=True,
    )
    cheap = build_model(settings, Tier.CHEAP, call_class, lane=lane).with_retry(
        stop_after_attempt=settings.max_retries + 1,
        wait_exponential_jitter=True,
    )
    return capable.with_fallbacks([cheap])
