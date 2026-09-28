"""Configuration for agentgate.

Every tunable in the system lives here. There are no feature flags in application code, no
module-level constants that a deployment might want to change, and no model identifiers
compiled into logic. If behaviour varies between environments, it varies through this file.

Two rules shape the design:

*The safe default is the one that cannot spend money or leak data.* An unconfigured process
comes up on the ``fake`` lane with an in-memory checkpointer. Reaching a real provider is
something you opt into, which is why the test suite needs no API key and CI needs no secrets.

*A bad configuration stops the process at startup, not at the first model call.* Every entry
point calls :func:`get_settings` as the first statement inside its handler, so a missing key
or a nonsense budget fails in the first second with a message naming the variable, rather than
thirty seconds into a graph run with a provider stack trace.

Importing this module has no side effects and cannot raise. Validation used to run at import,
which meant the import statement itself was the thing that failed -- a landmine every future
entry point had to know to step around. See
``docs/adr/0007-configuration-validated-at-startup.md``.
"""

from __future__ import annotations

import os
import re
from difflib import get_close_matches
from enum import StrEnum
from functools import lru_cache, reduce
from pathlib import Path
from typing import Annotated, Any, Final

from dotenv import dotenv_values
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentgate.errors import ConfigurationError

ENV_PREFIX: Final = "AGENTGATE_"

# Re-exported so callers can keep importing it from here, but it is *defined* in
# agentgate.errors: importing this module is itself the thing that can fail, and an entry
# point cannot catch that using a class it imports from this module.
__all__ = ["ENV_PREFIX", "ConfigurationError", "Settings", "get_settings"]


class Lane(StrEnum):
    """Which family of model endpoints a request is allowed to reach.

    The lane is a *policy* boundary, not a performance one: ``SOVEREIGN`` exists so that data
    classified as restricted never leaves infrastructure the operator controls.
    """

    CLOUD = "cloud"
    SOVEREIGN = "sovereign"
    FAKE = "fake"


CONTAINMENT: Final[dict[Lane, int]] = {
    Lane.FAKE: 0,
    Lane.SOVEREIGN: 1,
    Lane.CLOUD: 2,
}
"""How far data travels on each lane. Lower is more contained.

``FAKE`` never leaves the process. ``SOVEREIGN`` leaves it for infrastructure the operator
controls. ``CLOUD`` leaves the organisation. This is an ordering, not a ranking of quality, and
it exists so that combining two constraints has one obvious answer.
"""


def narrower_of(first: Lane, second: Lane) -> Lane:
    """The more contained of two lanes.

    The policy router and the deployment are both constraints on where a request may go, and a
    constraint can only ever *narrow*. Taking the minimum is what makes that true in code:

    - A cloud-default deployment routing restricted content to ``sovereign`` honours the route.
      That is leak-inventory item 13, and the reason this function exists.
    - A sovereign-default deployment routing public content to ``cloud`` stays on ``sovereign``.
      Honouring that route would start calling a third party on behalf of an operator who
      configured their own endpoint as the default -- a worse bug than the one being fixed, and
      the one the obvious repair introduces.
    - A ``fake`` deployment stays fake whatever the router says, which is what lets the whole
      offline suite exercise real routing without an endpoint in sight.
    """
    return first if CONTAINMENT[first] <= CONTAINMENT[second] else second


class Tier(StrEnum):
    """Capability tier within a lane. Cheap handles routing and classification."""

    CAPABLE = "capable"
    CHEAP = "cheap"


class CallClass(StrEnum):
    """What a model call is for.

    Output ceilings are set per class rather than globally. Classification and routing produce
    a handful of tokens and are capped tightly; only final synthesis gets a generous budget.
    A single global ``max_tokens`` would have to be sized for synthesis, which means every
    cheap call would carry a synthesis-sized worst case.
    """

    ROUTING = "routing"
    CLASSIFICATION = "classification"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    REPAIR = "repair"


class ModelPrice(BaseModel):
    """Price in USD per million tokens, input and output priced separately."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Annotated[float, Field(ge=0)]
    output: Annotated[float, Field(ge=0)]

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost of one call, in USD."""
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


class CheckpointerBackend(StrEnum):
    """Where thread state is persisted between super-steps."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class StoreBackend(StrEnum):
    """Where long-term, cross-thread memory is persisted.

    Deliberately separate from :class:`CheckpointerBackend`. A checkpoint is the state of one
    conversation; the store holds durable facts about a user that outlive every thread.
    """

    MEMORY = "memory"
    POSTGRES = "postgres"


class VectorBackend(StrEnum):
    """Where the retrieval corpus is indexed."""

    MEMORY = "memory"
    QDRANT = "qdrant"


class EffectSinkBackend(StrEnum):
    """Where an approved action's effect goes. See ``effects/sink.py``.

    **One member, on purpose.** The only sink is an append-only outbox on disk: an approved refund
    or email is *recorded* there and nothing real happens. There is no sink that moves money or
    sends mail, so there is no value that could select one -- and ``Settings`` refuses any other
    name at startup with a message saying so, rather than falling back to the outbox. "This
    performs nothing real" is enforced by configuration, not by a sentence in a README.
    """

    OUTBOX = "outbox"


class TracingBackend(StrEnum):
    """Where OpenTelemetry spans are exported.

    The instrumentation never changes; only the exporter behind it does. The environments this
    runtime targets treat prompt and document content as regulated data, so where traces land
    is a deployment decision rather than a default. See ``docs/adr/0008``.
    """

    NONE = "none"
    """Off. The process still logs structurally; nothing leaves it."""

    LANGSMITH = "langsmith"
    """Managed backend. Correct when the data is allowed to leave the boundary."""

    OTLP = "otlp"
    """Any OTLP collector, including one inside your own network. The self-hostable path."""


class DeciderBackend(StrEnum):
    """What, if anything, assesses a draft before the approval gate. See docs/adr/0012.

    The decider can only auto-approve or ask a human; it never rejects. It runs only when the
    routed lane is cloud, because every backend here is a cloud egress.
    """

    NONE = "none"
    """Off. Every draft goes to a human, as it did before the decider existed."""

    JEV = "jev"
    """TypeSafe's decision model: typed questions in, calibrated probabilities out."""

    LLM = "llm"
    """The cloud chat lane asked the same questions. No calibrated confidence."""


class DeciderMode(StrEnum):
    """Whether the decider's verdict is acted on."""

    SHADOW = "shadow"
    """Recorded beside the human's decision and never acted on. The default, because the
    thresholds enforce mode needs come from measuring this agreement."""

    ENFORCE = "enforce"
    """An auto-approve verdict that meets every threshold skips the human."""


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    DOCKER = "docker"


Port = Annotated[int, Field(ge=1, le=65535)]
Positive = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=2.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

EVEN_ODDS: Final = 0.5
"""Where a two-outcome answer stops leaning either way: the route Choice's winner is always at or
above it, and a Noul above it says "more likely yes". The vacuity bounds on the auto-approve
thresholds are stated against this rather than against a number chosen for looking safe."""

EXACT_VERSION: Final = re.compile(r"-\d+\.\d+\.\d+$")
"""A versioned model identifier ends in major.minor.patch, as ``jev-1.13.0`` does. Aliases such
as ``jev-latest`` do not, and neither does ``jev-1.13``, which names a line, not a release."""


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and a local ``.env``.

    Frozen, so nothing can mutate configuration mid-run and leave two nodes disagreeing about
    the budget. ``extra="forbid"`` turns a typo'd ``AGENTGATE_*`` variable into a startup
    failure rather than a silently ignored line in a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # `.env` is a shared namespace. Other tools legitimately keep their own keys there,
        # and rejecting them would make this application the owner of a file it merely reads.
        # Typo protection for the AGENTGATE_ namespace comes from
        # _reject_unrecognised_variables below, which is stricter than extra="forbid" was --
        # it catches names the settings source silently drops, and suggests the intended one.
        extra="ignore",
        frozen=True,
    )

    # ---------------------------------------------------------------- application

    environment: Environment = Environment.LOCAL
    service_name: str = "agentgate"
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = True

    # ---------------------------------------------------------------- lane selection

    lane: Lane = Lane.FAKE
    """Default lane. The classifier may route an individual request to a stricter one."""

    temperature: UnitInterval = 0.0
    request_timeout_seconds: PositiveFloat = 30.0
    max_retries: Annotated[int, Field(ge=0, le=5)] = 2

    # Output ceilings per call class. Tight where the answer is a label or a route, generous
    # only where the answer is the deliverable.
    max_tokens_routing: Positive = 128
    max_tokens_classification: Positive = 256
    max_tokens_research: Positive = 1_024
    max_tokens_synthesis: Positive = 4_096
    max_tokens_repair: Positive = 512

    model_prices_usd_per_million: dict[str, ModelPrice] = Field(default_factory=dict)
    """Per-model prices, keyed by model identifier.

    Deliberately empty by default. Prices change, this repository cannot fetch them, and a
    default of zero would silently disarm the spend guard -- an unpriced model would look
    free and no ceiling would ever be crossed. An unpriced model on a networked lane is a
    startup error instead. ``.env.example`` documents the format and where the numbers come
    from.
    """

    # ---------------------------------------------------------------- cloud lane

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str | None = None
    """Override for an OpenAI-compatible gateway or proxy in front of the cloud lane."""

    cloud_capable_model: str | None = None
    """Synthesis tier. No default: model identifiers are an operator decision, and guessing a
    name that has since been retired fails at request time instead of startup."""

    cloud_cheap_model: str | None = None
    """Routing and classification tier. Populate from a real listing of the models the key can
    reach -- see ``agentgate models list`` -- rather than assuming a name."""

    embedding_model: str | None = None

    # ---------------------------------------------------------------- sovereign lane

    sovereign_base_url: str | None = None
    """OpenAI-compatible endpoint. Ollama serves this at ``http://localhost:11434/v1``;
    vLLM at ``http://<host>:8000/v1``."""

    sovereign_model: str | None = None
    sovereign_api_key: SecretStr = SecretStr("not-required")
    """Most self-hosted OpenAI-compatible servers ignore this but reject a missing header."""

    # ---------------------------------------------------------------- the decider
    #
    # An egress point with the authority to skip a human, so every setting either says where the
    # egress goes or bounds when the authority applies. Configuration decides what is AVAILABLE;
    # the policy router still decides what is ALLOWED per request, and the decider runs only when
    # the routed lane is cloud. See docs/adr/0012.

    decider_backend: DeciderBackend = DeciderBackend.NONE
    decider_mode: DeciderMode = DeciderMode.SHADOW

    jev_base_url: str = "https://api.typesafe.ai/v1"
    """TypeSafe's official API. Overridable so a stub can observe what is sent, which is the only
    way an assertion about this egress can read the wire."""

    jev_model: str = Field(default="jev-1.13.0", validate_default=True)
    """An exact version, never an alias.

    The one model identifier with a default, and deliberately. The auto-approve thresholds are
    measured against a specific model's answers, so the version is part of what those numbers
    mean: ``jev-latest`` moves on a release and would silently invalidate every threshold tuned
    against the version it used to point at. Moving it is an operator decision taken with a new
    shadow-mode measurement, not a side effect of TypeSafe shipping.
    """

    jev_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}JEV_API_KEY", "TYPESAFE_API_KEY"),
    )

    # The three conditions an auto-approve must meet, each read off a field the API returns.
    # No defaults: they come from measured shadow-mode agreement, and a default would be a number
    # nobody measured, in force for everyone who did not override it. Required in enforce mode.

    auto_approve_min_probability: Probability | None = None
    """The route Choice's probability for ``auto_approve`` must be at or above this."""

    auto_approve_min_confidence: Probability | None = None
    """The route Choice's reported ``confidence`` must be at or above this. The field as
    returned, never recomputed from the probabilities."""

    auto_approve_max_irreversibility: Probability | None = None
    """The irreversibility Noul must be at or below this. A Noul carries no confidence, so this
    reads its probability and nothing is invented to stand in for one."""

    # ---------------------------------------------------------------- budget gates

    max_iterations: Positive = 8
    """Supervisor hand-offs allowed before the budget guard forces finalisation."""

    max_fan_out: Positive = 5
    """How many research branches one dispatch may open.

    **The one budget decided before the spending, rather than counted after it.** Iterations
    and tokens are both measured from what already happened, which is enough when work arrives
    one call at a time. A ``Send`` fan-out is different: the list being fanned out over was
    produced by a model, so without this the model chooses how many calls get paid for, and the
    token ceiling only finds out once they have all run.

    A policy limit, like ``max_iterations`` -- not a measurement. Its relationship to the token
    ceiling is arithmetic (width times per-branch cost has to fit under the run budget), and
    that arithmetic cannot be done honestly until a fan-out run has been measured. Re-derive
    both together with ``make measure``.
    """

    max_total_tokens: Positive = 120_000

    max_spend_usd: PositiveFloat = 0.50
    """Hard ceiling for a single run. Crossing it aborts the graph; it is not a warning."""

    max_session_spend_usd: PositiveFloat = 5.00
    """Hard ceiling across every run in this process. Bounds a loop of cheap runs, which no
    per-run ceiling can catch."""

    live_spend_tolerance: Annotated[float, Field(gt=1.0, le=20.0)] = 3.0
    """How far the live suite may exceed its pre-run estimate before aborting.

    An estimate that is only advisory stops nothing. Must exceed 1.0, since an estimate is a
    guess and a threshold at exactly the guess would abort on rounding.
    """

    # ---------------------------------------------------------------- the live suite
    #
    # A suite is not a run. Its cases are independent, they share one book only so the total is
    # visible, and accounting them against ``max_total_tokens`` measures the wrong thing: the
    # suite trips for being a suite, and the obvious fix -- raising the run ceiling -- weakens
    # the guard that was working. So the suite gets its own ceilings, on their own basis.

    max_live_suite_tokens: Positive = 24_750
    """Token ceiling for one complete live suite.

    DERIVED, not chosen: the estimate at the top of ``scripts/run_live.py`` (15 calls at 400
    input and 150 output apiece, so 8,250 tokens) times ``live_spend_tolerance`` of 3. The same
    basis as the dollar abort the gatekeeper already computes, because it is bounding the same
    thing by the other unit.

    ``tests/unit/test_live_suite_ceilings.py`` recomputes this from those constants, so editing
    the estimate without re-deriving the ceiling fails the build rather than leaving a ceiling
    describing a suite that no longer exists.
    """

    max_live_suite_spend_usd: PositiveFloat = 0.005
    """Dollar ceiling for one complete live suite.

    Derived from the token budget above the same way ``max_spend_usd`` is derived from
    ``max_total_tokens``: 18,000 input and 6,750 output priced at the $0.10 / $0.40 per million
    used throughout ``.env.example`` comes to $0.0045, rounded up so rounding alone cannot trip
    it.
    """

    live_spend_ledger: Path | None = None
    """Where the live suite writes what it spent, for the gatekeeper to read back.

    Set by ``scripts/run_live.py`` on the subprocess it launches; unset otherwise, and the
    suite then records nothing to disk. Declared here rather than read from ``os.environ``
    because the namespace guard rejects every unrecognised ``AGENTGATE_*`` variable -- which,
    until this field existed, included the two the gatekeeper set itself.
    """

    live_spend_abort_usd: PositiveFloat | None = None
    """Per-invocation dollar ceiling injected by the gatekeeper: the estimate it printed times
    the tolerance.

    Tightens ``max_live_suite_spend_usd`` when present, so the suite aborts mid-flight against
    the figure the operator consented to instead of only being told about it once the money was
    already gone.
    """

    recursion_limit: Positive = 40
    """LangGraph's own super-step ceiling. A backstop behind ``max_iterations``, not a
    substitute for it: hitting this one is a bug, hitting the other one is a policy."""

    # ---------------------------------------------------------------- persistence

    checkpointer: CheckpointerBackend = CheckpointerBackend.MEMORY
    sqlite_path: Path = Path("data/agentgate.db")
    postgres_dsn: SecretStr | None = None
    store_backend: StoreBackend = StoreBackend.MEMORY

    # ---------------------------------------------------------------- retrieval

    corpus_path: Path = Path("corpus")
    vector_backend: VectorBackend = VectorBackend.MEMORY
    qdrant_url: str | None = None
    retrieval_top_k: Positive = 4

    # ---------------------------------------------------------------- effects

    effect_sink: EffectSinkBackend = EffectSinkBackend.OUTBOX
    outbox_path: Path = Path("data/outbox.jsonl")
    """Where approved effects are recorded, one JSON line each, append-only. Idempotent per
    effect: a key already in the file is never written again, which is what makes a resumed
    ``execute`` safe after a crash."""

    # ---------------------------------------------------------------- audit

    audit_log_path: Path = Path("data/audit.jsonl")

    # ---------------------------------------------------------------- surfaces

    api_host: str = "127.0.0.1"
    api_port: Port = 8000

    # ---------------------------------------------------------------- observability

    tracing_backend: TracingBackend = TracingBackend.NONE
    """Where spans go. Off by default: see docs/adr/0008.

    OpenTelemetry is the instrumentation in every case. This chooses only the exporter behind
    it, which is why switching backends is a deployment decision and not a code change.
    """

    otel_exporter_endpoint: str | None = None
    """OTLP collector. Required when the backend is ``otlp``."""

    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}LANGSMITH_API_KEY", "LANGSMITH_API_KEY"),
    )
    """Read from the conventional unprefixed name too, the same way the OpenAI key is.

    Declared explicitly rather than left to chance: without an alias this field still picked
    up an unprefixed value from a shared ``.env`` by accident, and a credential being read by
    accident is exactly the kind of thing that should be written down.
    """

    langsmith_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}LANGSMITH_PROJECT", "LANGSMITH_PROJECT"),
    )

    metrics_enabled: bool = True

    # ---------------------------------------------------------------- validation

    @model_validator(mode="before")
    @classmethod
    def _reject_unrecognised_variables(cls, data: Any) -> Any:
        """Fail on an ``AGENTGATE_*`` variable that matches no field.

        ``extra="forbid"`` does not cover this. pydantic-settings builds its environment source
        by looking up each *field*, so a variable nothing asks for is dropped without comment.
        That is the quietest possible failure: the operator writes AGENTGATE_MAX_ITERATION,
        believes a budget is in force, and the process runs on defaults.
        """
        _reject_unknown_variables()
        return data

    @field_validator(
        "openai_base_url",
        "sovereign_base_url",
        "qdrant_url",
        "otel_exporter_endpoint",
        "jev_base_url",
    )
    @classmethod
    def _must_look_like_a_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            msg = f"expected an http:// or https:// URL, got {cleaned!r}"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def _cloud_lane_is_fully_specified(self) -> Settings:
        if self.lane is not Lane.CLOUD:
            return self
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                (f"{ENV_PREFIX}CLOUD_CAPABLE_MODEL", self.cloud_capable_model),
                (f"{ENV_PREFIX}CLOUD_CHEAP_MODEL", self.cloud_cheap_model),
            )
            if not value
        ]
        if missing:
            msg = (
                f"lane is 'cloud' but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not set"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _sovereign_lane_is_fully_specified(self) -> Settings:
        """Complete if it is the default, and equally if the router can merely reach it.

        The second half is new with leak-inventory item 13. Before the routed lane was wired
        through, a base URL with no model was harmless: nothing ever built a sovereign model on a
        cloud-default deployment, so the missing identifier was never asked for. Now the policy
        gate can send restricted content there, which turns that configuration into a run that
        classifies correctly, routes correctly, and then dies inside the lane node -- at the
        first restricted request rather than at startup.
        """
        if self.lane is Lane.SOVEREIGN:
            missing = [
                name
                for name, value in (
                    (f"{ENV_PREFIX}SOVEREIGN_BASE_URL", self.sovereign_base_url),
                    (f"{ENV_PREFIX}SOVEREIGN_MODEL", self.sovereign_model),
                )
                if not value
            ]
            if missing:
                msg = (
                    f"lane is 'sovereign' but {', '.join(missing)} "
                    f"{'is' if len(missing) == 1 else 'are'} not set"
                )
                raise ValueError(msg)
            return self

        if self.sovereign_base_url and not self.sovereign_model:
            msg = (
                f"{ENV_PREFIX}SOVEREIGN_BASE_URL is set, so the policy gate can route "
                f"restricted content to the sovereign lane, but {ENV_PREFIX}SOVEREIGN_MODEL is "
                "not set. A lane the router can choose needs a model identifier, or the first "
                "restricted request fails inside the graph instead of here"
            )
            raise ValueError(msg)
        return self

    @field_validator("effect_sink", mode="before")
    @classmethod
    def _only_the_outbox_exists(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() != EffectSinkBackend.OUTBOX.value:
            msg = (
                f"{ENV_PREFIX}EFFECT_SINK is {value!r}, and no effect sink but 'outbox' exists. "
                "Approved actions are recorded to an append-only outbox and nothing real happens; "
                "there is no sink that moves money or sends mail, so none can be selected"
            )
            raise ValueError(msg)
        return value

    @field_validator("jev_model")
    @classmethod
    def _jev_model_is_an_exact_version(cls, value: str) -> str:
        if not EXACT_VERSION.search(value):
            msg = (
                f"{ENV_PREFIX}JEV_MODEL is {value!r}; it must name an exact version such as "
                "jev-1.13.0. An alias moves when TypeSafe ships, and every auto-approve threshold "
                "was measured against the version it used to point at"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _decider_can_run_where_it_is_configured(self) -> Settings:
        """A decider that can never run, or cannot authenticate, is refused here.

        Both fail later and worse otherwise: one silently never assesses anything, the other
        fails at the first draft. The first is also a contradiction in intent -- the decider is a
        cloud egress, so a deployment with no routable cloud lane (fake, or sovereign by default,
        which cannot widen) has chosen not to send data to a third party, and configuring a
        decider says it will send data to TypeSafe anyway.
        """
        if self.decider_backend is DeciderBackend.NONE:
            return self
        if Lane.CLOUD not in self.routable_lanes:
            routes = ", ".join(sorted(lane.value for lane in self.routable_lanes))
            msg = (
                f"{ENV_PREFIX}DECIDER_BACKEND is {self.decider_backend.value!r} but this "
                f"deployment has no cloud lane (it can route to: {routes}). The decider runs "
                "only when a request is routed to the cloud, so here it would never run -- and a "
                "deployment that does not send data to OpenAI should not send it to a decider"
            )
            raise ValueError(msg)
        if self.decider_backend is DeciderBackend.JEV and self.jev_api_key is None:
            msg = (
                f"{ENV_PREFIX}DECIDER_BACKEND is 'jev' but {ENV_PREFIX}JEV_API_KEY "
                "(or TYPESAFE_API_KEY) is not set"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _enforce_mode_has_thresholds_that_can_fail(self) -> Settings:
        """Enforcement is only as strict as its weakest condition.

        Each threshold is refused where it is *vacuous* -- met by every answer the API can
        return -- and nowhere else. A stricter floor would be a number chosen here, and the
        thresholds are meant to come from measured shadow-mode agreement.

        - The route Choice has two options, so its winner is always at 0.5 or above: a minimum
          probability at or below 0.5 is met by ``route == auto_approve`` alone.
        - Confidence is reported in [0, 1], so a minimum of 0 is met by every answer.
        - A maximum irreversibility at or above 0.5 admits a Noul that says "more likely
          irreversible than not".
        """
        if self.decider_mode is not DeciderMode.ENFORCE:
            return self
        if self.decider_backend is DeciderBackend.NONE:
            msg = (
                f"{ENV_PREFIX}DECIDER_MODE is 'enforce' but {ENV_PREFIX}DECIDER_BACKEND is "
                "'none': there is nothing to enforce, and an operator reading the mode would "
                "believe drafts were being assessed"
            )
            raise ValueError(msg)

        minimum_probability = self.auto_approve_min_probability
        minimum_confidence = self.auto_approve_min_confidence
        maximum_irreversibility = self.auto_approve_max_irreversibility
        missing = [
            f"{ENV_PREFIX}{name.upper()}"
            for name, value in (
                ("auto_approve_min_probability", minimum_probability),
                ("auto_approve_min_confidence", minimum_confidence),
                ("auto_approve_max_irreversibility", maximum_irreversibility),
            )
            if value is None
        ]
        if missing:
            msg = (
                f"{ENV_PREFIX}DECIDER_MODE is 'enforce' but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not set. Measure them in shadow mode "
                "first: each is a condition an auto-approve has to meet"
            )
            raise ValueError(msg)
        assert minimum_probability is not None  # noqa: S101 - narrowed by `missing` above
        assert minimum_confidence is not None  # noqa: S101
        assert maximum_irreversibility is not None  # noqa: S101

        vacuous = []
        if minimum_probability <= EVEN_ODDS:
            vacuous.append(
                f"{ENV_PREFIX}AUTO_APPROVE_MIN_PROBABILITY={minimum_probability} must exceed "
                "0.5, because a two-option Choice's winner is always at or above it"
            )
        if minimum_confidence <= 0.0:
            vacuous.append(
                f"{ENV_PREFIX}AUTO_APPROVE_MIN_CONFIDENCE={minimum_confidence} must exceed 0, "
                "because every reported confidence meets it"
            )
        if maximum_irreversibility >= EVEN_ODDS:
            vacuous.append(
                f"{ENV_PREFIX}AUTO_APPROVE_MAX_IRREVERSIBILITY={maximum_irreversibility} must be "
                "below 0.5, or an answer of 'more likely irreversible' passes"
            )
        if vacuous:
            msg = "enforce mode has a threshold no answer can fail: " + "; ".join(vacuous)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _backends_have_their_connection_details(self) -> Settings:
        if self.checkpointer is CheckpointerBackend.POSTGRES and not self.postgres_dsn:
            msg = f"checkpointer is 'postgres' but {ENV_PREFIX}POSTGRES_DSN is not set"
            raise ValueError(msg)
        if self.store_backend is StoreBackend.POSTGRES and not self.postgres_dsn:
            msg = f"store_backend is 'postgres' but {ENV_PREFIX}POSTGRES_DSN is not set"
            raise ValueError(msg)
        if self.vector_backend is VectorBackend.QDRANT and not self.qdrant_url:
            msg = f"vector_backend is 'qdrant' but {ENV_PREFIX}QDRANT_URL is not set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _every_reachable_model_has_a_price(self) -> Settings:
        """A networked lane must be able to cost every model it can reach.

        Without a price the spend guard cannot account, and the safe reading of "unknown
        cost" is a refusal to start -- not an assumption of zero, which would leave the
        ceiling permanently uncrossed while real money was spent.

        **The embedding model is in this set.** It was not until the run budget was ruled to
        mean all spend rather than chat spend, and its absence was the whole of leak-inventory
        item 9: indexing and querying the corpus cost real money that no ceiling could see. An
        embedding model with no price is the same refusal as a chat model with no price, for
        the same reason.

        **So is every lane the router can choose**, which it was not until item 13. "Reachable"
        was read off the configured lane, so a hybrid deployment's sovereign model -- the one the
        policy gate exists to send restricted content to -- was outside the guard entirely. The
        guard was not wrong about what it checked; it was asking about the wrong set.
        """
        if not self.requires_network:
            return self
        reachable = {
            self.model_for(tier, lane=lane) for lane in self.routable_lanes for tier in Tier
        }
        if self.embedding_model:
            reachable.add(self.embedding_model)
        # The decider's model is priced like any other it can reach, and only when it can.
        if self.decider_backend is DeciderBackend.JEV:
            reachable.add(self.jev_model)
        unpriced = sorted(
            model for model in reachable if model not in self.model_prices_usd_per_million
        )
        if unpriced:
            msg = (
                f"no price configured for {', '.join(unpriced)}; the spend guard cannot "
                f"account for a model it cannot cost. Set {ENV_PREFIX}"
                "MODEL_PRICES_USD_PER_MILLION (see .env.example)"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _tracing_backend_has_its_destination(self) -> Settings:
        """A backend selected but not addressable would silently drop every span.

        Worse than tracing being off, because the operator believes it is on.
        """
        if self.tracing_backend is TracingBackend.OTLP and not self.otel_exporter_endpoint:
            msg = f"tracing_backend is 'otlp' but {ENV_PREFIX}OTEL_EXPORTER_ENDPOINT is not set"
            raise ValueError(msg)
        if self.tracing_backend is TracingBackend.LANGSMITH and not self.langsmith_api_key:
            msg = f"tracing_backend is 'langsmith' but {ENV_PREFIX}LANGSMITH_API_KEY is not set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _session_ceiling_is_not_below_the_run_ceiling(self) -> Settings:
        if self.max_session_spend_usd < self.max_spend_usd:
            msg = (
                f"{ENV_PREFIX}MAX_SESSION_SPEND_USD ({self.max_session_spend_usd}) is below "
                f"{ENV_PREFIX}MAX_SPEND_USD ({self.max_spend_usd}); the session ceiling bounds "
                "every run together, so a single run could never reach its own limit"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _budget_ceilings_are_ordered(self) -> Settings:
        if self.recursion_limit <= self.max_iterations:
            msg = (
                f"{ENV_PREFIX}RECURSION_LIMIT ({self.recursion_limit}) must exceed "
                f"{ENV_PREFIX}MAX_ITERATIONS ({self.max_iterations}); the recursion limit is a "
                "backstop behind the iteration budget, so the budget guard has to trip first"
            )
            raise ValueError(msg)
        return self

    # ---------------------------------------------------------------- derived helpers

    @property
    def requires_network(self) -> bool:
        """Whether the configured lane reaches outside the process."""
        return self.lane is not Lane.FAKE

    def max_tokens_for(self, call_class: CallClass) -> int:
        """Output ceiling for a class of call."""
        match call_class:
            case CallClass.ROUTING:
                return self.max_tokens_routing
            case CallClass.CLASSIFICATION:
                return self.max_tokens_classification
            case CallClass.RESEARCH:
                return self.max_tokens_research
            case CallClass.SYNTHESIS:
                return self.max_tokens_synthesis
            case CallClass.REPAIR:
                return self.max_tokens_repair

    def price_for(self, model: str) -> ModelPrice:
        """Price for a model identifier.

        Raises:
            ConfigurationError: if the model has no configured price. Validation makes this
                unreachable at startup for a networked lane, so reaching it means a model was
                selected at runtime that configuration never saw.
        """
        if self.lane is Lane.FAKE:
            return ModelPrice(input=0.0, output=0.0)
        try:
            return self.model_prices_usd_per_million[model]
        except KeyError:
            msg = f"no price configured for model {model!r}; refusing to guess at spend"
            raise ConfigurationError(msg) from None

    @property
    def routable_lanes(self) -> frozenset[Lane]:
        """Every lane a request can actually be served from in this deployment.

        Not every lane that is configured. The policy route narrows rather than widens -- see
        :func:`narrower_of` -- so a sovereign-default deployment cannot serve a request from the
        cloud lane even with a working key, and the cloud lane is therefore not routable there.

        This is what makes the price guard's question answerable. "Can this deployment reach a
        model it cannot cost?" was previously asked of the default lane alone, which on a hybrid
        deployment excluded the sovereign model the router is specifically there to send
        restricted content to.
        """
        if self.lane is Lane.FAKE:
            return frozenset({Lane.FAKE})
        lanes = {self.lane}
        if self.sovereign_base_url and CONTAINMENT[Lane.SOVEREIGN] <= CONTAINMENT[self.lane]:
            lanes.add(Lane.SOVEREIGN)
        return frozenset(lanes)

    @property
    def most_contained_lane(self) -> Lane:
        """The most contained lane this deployment can reach.

        Named for the rule rather than a caller, because it has more than one: classification
        runs on it, and retrieval embeds on it. It was ``classification_lane`` until the second
        caller made that name wrong.

        For classification it is the only answer available: classification runs before the
        router, so there is no routed lane to honour and the request has not yet been judged.
        Treating it as maximally sensitive until something has looked at it is the same
        fail-closed reading the router applies to an unclassified request.

        What it resolves to, and none of these needed a new setting:

        - **fake** deployment: fake. Nothing leaves the process, and the offline suite is
          untouched.
        - **hybrid** deployment: sovereign. The raw request no longer goes to a third party in
          order to decide whether it was allowed to.
        - **cloud-only** deployment: cloud. There is nowhere else to send it, so the egress
          remains and is documented rather than closed. Leak inventory item 14.

        **Retrieval embeds on it too** -- the corpus index and every research query -- which is
        leak inventory item 17 closed on hybrid deployments and left open on cloud-only ones, for
        the same reason as item 14. One rule for "the most contained lane", read by both, rather
        than a second answer that could drift from this one.

        Folded with :func:`narrower_of` rather than sorted, because that function is the one
        place the ordering is defined and commutativity is what makes the fold well defined
        whatever order the set iterates in.
        """
        return reduce(narrower_of, self.routable_lanes)

    def model_for(self, tier: Tier, *, lane: Lane | None = None) -> str:
        """Resolve the model identifier for a tier on a lane.

        Args:
            tier: Capable or cheap.
            lane: Which lane's identifier to resolve. Defaults to the configured lane.

        The lane argument is not a convenience. A model identifier belongs to the endpoint that
        serves it, so resolving it against the configured lane while the request travels to
        another one sends a name the receiving server has never heard of -- and records that
        name in the audit trail as the model that answered. That was the second half of leak
        inventory item 13, and fixing only the endpoint would have left it in place.

        Raises:
            ConfigurationError: if the lane has no model configured for that tier. Validation
                makes this unreachable for a well-formed configuration, so it exists to turn a
                future lane added without models into an obvious failure.
        """
        resolved = lane or self.lane
        match resolved:
            case Lane.CLOUD:
                chosen = (
                    self.cloud_capable_model if tier is Tier.CAPABLE else self.cloud_cheap_model
                )
            case Lane.SOVEREIGN:
                chosen = self.sovereign_model
            case Lane.FAKE:
                return f"fake-{tier.value}"
        if not chosen:
            # Names the resolved lane rather than the configured one. A message that said
            # "cloud" while the caller had asked for the sovereign identifier would send the
            # reader to the wrong half of their configuration.
            msg = f"no model configured for tier {tier.value!r} on lane {resolved.value!r}"
            raise ConfigurationError(msg)
        return chosen


def _recognised_variable_names() -> frozenset[str]:
    """Every environment variable name the settings model will actually read."""
    names: set[str] = set()
    for field_name, field in Settings.model_fields.items():
        names.add(f"{ENV_PREFIX}{field_name}".upper())
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            names.update(str(choice).upper() for choice in alias.choices)
        elif isinstance(alias, str):
            names.add(alias.upper())
    return frozenset(names)


def _declared_variable_names() -> dict[str, str]:
    """Prefixed variable names present in the environment or the local ``.env``.

    Maps each name to where it came from, so the error can point at the right file.
    """
    found: dict[str, str] = {
        name.upper(): "the environment"
        for name in os.environ
        if name.upper().startswith(ENV_PREFIX)
    }
    env_file = Settings.model_config.get("env_file")
    if isinstance(env_file, str | Path):
        path = Path(env_file)
        if path.is_file():
            for name in dotenv_values(path):
                if name and name.upper().startswith(ENV_PREFIX):
                    found.setdefault(name.upper(), str(path))
    return found


def _reject_unknown_variables() -> None:
    """Raise if any declared ``AGENTGATE_*`` variable matches no field.

    Raises:
        ValueError: so pydantic folds it into the surrounding ``ValidationError`` and it
            reaches the operator through the same formatting as every other config problem.
    """
    recognised = _recognised_variable_names()
    unknown = sorted(
        (name, source)
        for name, source in _declared_variable_names().items()
        if name not in recognised
    )
    if not unknown:
        return

    lines = []
    for name, source in unknown:
        suggestion = get_close_matches(name, recognised, n=1, cutoff=0.7)
        hint = f", did you mean {suggestion[0]}?" if suggestion else ""
        lines.append(f"{name} (set in {source}) is not a known setting{hint}")
    raise ValueError("; ".join(lines))


def _readable(error: ValidationError) -> str:
    """Render a pydantic validation failure as something an operator can act on."""
    count = error.error_count()
    lines = [f"agentgate cannot start: {count} configuration problem{'' if count == 1 else 's'}."]
    for detail in error.errors():
        # pydantic prefixes messages raised from validators; the operator does not need it.
        message = detail["msg"].removeprefix("Value error, ")
        if detail["loc"]:
            location = ".".join(str(part) for part in detail["loc"])
            lines.append(f"  - {ENV_PREFIX}{location.upper()}: {message}")
        else:
            lines.append(f"  - {message}")
    lines.append("")
    lines.append("See .env.example for every supported variable and its default.")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, validating them on the first call.

    Cached, so configuration is read once and every caller sees the same object. Tests that
    manipulate the environment must call ``get_settings.cache_clear()`` first.

    **Every entry point must call this as the first statement inside its handler**, so a
    broken environment stops the process at startup rather than at the first model call. It
    is deliberately not called at import time -- see
    ``docs/adr/0007-configuration-validated-at-startup.md``.

    Raises:
        ConfigurationError: with a message naming each variable at fault.
    """
    try:
        return Settings()
    except ValidationError as error:
        raise ConfigurationError(_readable(error)) from error
