"""The decider's configuration, and the combinations that contradict themselves.

A decider is an egress point with the authority to skip a human. Every setting here either says
where that egress goes or bounds when the authority applies, so a configuration that is wrong in
a way nothing notices until a request arrives is the failure to rule out -- at startup, with a
message naming the variable, the same treatment every other lane gets.

Four contradictions are refused:

- **A Jev backend with no key.** It would fail at the first assessment rather than at startup.
- **A decider on a deployment with no cloud lane.** The decider only runs when the routed lane is
  cloud, so here it can never run -- and a deployment that will not send data to OpenAI but will
  send it to TypeSafe contradicts itself.
- **Enforce mode with a missing or vacuous threshold.** Each of the three auto-approve conditions
  has to be able to fail, or enforcement is a rubber stamp.
- **An unpriced Jev model.** Refused by the same guard as every other reachable model.

And one that is the tracing validator's reasoning applied here: enforce mode with no backend is
enforcement the operator believes is on and is not.

Model identifiers here are invented except where the real one is the point: the alias test uses
the aliases TypeSafe actually documents, because refusing those specific names is what it pins.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentgate.config import DeciderBackend, DeciderMode, Lane, Settings

pytestmark = pytest.mark.usefixtures("isolated_env")

CLOUD_CAPABLE = "cloud-capable-test"
CLOUD_CHEAP = "cloud-cheap-test"
SOVEREIGN = "sovereign-test"
JEV = "jev-9.9.9"


def prices(*models: str) -> dict[str, dict[str, float]]:
    return {model: {"input": 0.10, "output": 0.40} for model in models}


KEY = "AGENTGATE_JEV_API_KEY"
"""The key is passed under its declared name, never as ``jev_api_key=``.

With a validation alias and ``case_sensitive=False``, a constructor keyword is matched against
the aliases, not the field name -- and ``extra="ignore"`` drops one that matches nothing, so
``jev_api_key="..."`` is silently discarded. ``openai_api_key=`` works elsewhere only because its
unprefixed alias happens to spell the field name. Adding the field name as an alias would make a
bare ``JEV_API_KEY`` readable from the shared namespace, which ``test_env_namespace.py`` forbids.
"""


def cloud(**overrides: object) -> Settings:
    """A cloud-only deployment with a Jev decider fully specified, overridable field by field.

    ``**{KEY: None}`` removes the key rather than setting it to ``None``.
    """
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "cloud_capable_model": CLOUD_CAPABLE,
        "cloud_cheap_model": CLOUD_CHEAP,
        "decider_backend": "jev",
        KEY: "not-required",
        "jev_model": JEV,
        "model_prices_usd_per_million": prices(CLOUD_CAPABLE, CLOUD_CHEAP, JEV),
    }
    fields.update(overrides)
    if fields.get(KEY, "") is None:
        del fields[KEY]
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


ENFORCED: dict[str, object] = {
    "decider_mode": "enforce",
    "auto_approve_min_probability": 0.9,
    "auto_approve_min_confidence": 0.8,
    "auto_approve_max_irreversibility": 0.1,
}


# ------------------------------------------------------------------------------- defaults


def test_the_default_is_no_decider_in_shadow_mode() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.decider_backend is DeciderBackend.NONE
    assert settings.decider_mode is DeciderMode.SHADOW
    assert settings.jev_api_key is None
    assert settings.jev_base_url == "https://api.typesafe.ai/v1"


def test_no_threshold_has_a_default() -> None:
    """The thresholds come from measured shadow-mode agreement, so none is chosen here.

    A default would be a number nobody measured, in force for everyone who did not override it.
    """
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.auto_approve_min_probability is None
    assert settings.auto_approve_min_confidence is None
    assert settings.auto_approve_max_irreversibility is None


def test_the_default_jev_model_is_an_exact_version() -> None:
    """Pinned, never an alias: an alias moves on a release and silently invalidates a threshold
    tuned against the version it used to point at."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.jev_model == "jev-1.13.0"


# ----------------------------------------------------------------------- the valid shapes


def test_a_cloud_only_deployment_can_run_a_jev_decider_in_shadow_mode() -> None:
    """The presence half. Every refusal below would pass against a validator that refused
    everything, so the configurations that should start are shown to start."""
    settings = cloud()

    assert settings.decider_backend is DeciderBackend.JEV
    assert settings.decider_mode is DeciderMode.SHADOW


def test_a_hybrid_deployment_can_run_a_jev_decider() -> None:
    settings = cloud(
        sovereign_base_url="http://127.0.0.1:9",
        sovereign_model=SOVEREIGN,
        model_prices_usd_per_million=prices(CLOUD_CAPABLE, CLOUD_CHEAP, SOVEREIGN, JEV),
    )

    assert settings.decider_backend is DeciderBackend.JEV
    assert Lane.CLOUD in settings.routable_lanes


def test_enforce_mode_starts_with_all_three_thresholds_set() -> None:
    settings = cloud(**ENFORCED)

    assert settings.decider_mode is DeciderMode.ENFORCE


# ------------------------------------------------------------------------------- the key


def test_a_jev_backend_with_no_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="AGENTGATE_JEV_API_KEY"):
        cloud(**{KEY: None})


@pytest.mark.parametrize("name", ["AGENTGATE_JEV_API_KEY", "TYPESAFE_API_KEY"])
def test_the_key_is_read_from_either_declared_name(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TYPESAFE_API_KEY`` is the name TypeSafe's own tooling uses, so it is accepted -- as a
    declared alias, listed in ``test_env_namespace.py``, not an undeclared read."""
    monkeypatch.setenv(name, "from-the-environment")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.jev_api_key is not None
    assert settings.jev_api_key.get_secret_value() == "from-the-environment"


def test_the_key_never_appears_in_the_settings_repr() -> None:
    settings = cloud(**{KEY: "jev-key-canary"})

    # Presence first. `extra="ignore"` drops an unknown keyword silently, so without this the
    # absence below passed before the field existed -- observed, not supposed.
    assert settings.jev_api_key is not None
    assert settings.jev_api_key.get_secret_value() == "jev-key-canary"
    assert "jev-key-canary" not in repr(settings)


# ------------------------------------------------------------------ no cloud lane to ride


@pytest.mark.parametrize("backend", ["jev", "llm"])
def test_a_decider_on_a_fake_lane_deployment_is_refused(backend: str) -> None:
    with pytest.raises(ValidationError, match="no cloud lane"):
        Settings(  # type: ignore[call-arg]
            _env_file=None, decider_backend=backend, **{KEY: "not-required"}
        )


@pytest.mark.parametrize("backend", ["jev", "llm"])
def test_a_decider_on_a_sovereign_default_deployment_is_refused(backend: str) -> None:
    """Even with a working OpenAI key. A sovereign-default deployment cannot widen to the cloud
    (``narrower_of``), so the decider could never run -- and configuring it says the operator
    will send data to TypeSafe from a deployment built not to send it to OpenAI."""
    with pytest.raises(ValidationError, match="no cloud lane"):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            lane="sovereign",
            sovereign_base_url="http://127.0.0.1:9",
            sovereign_model=SOVEREIGN,
            openai_api_key="not-required",
            decider_backend=backend,
            model_prices_usd_per_million=prices(SOVEREIGN, JEV),
            **{KEY: "not-required"},
        )


# -------------------------------------------------------------------------- the model pin


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "jev-1.13"])
def test_a_jev_model_that_is_not_an_exact_version_is_refused(alias: str) -> None:
    """``jev-latest`` and ``jev-preview`` are TypeSafe's documented aliases; ``jev-1.13`` names
    a line rather than a release. All three can change under a tuned threshold."""
    with pytest.raises(ValidationError, match="exact version"):
        cloud(jev_model=alias, model_prices_usd_per_million=prices(CLOUD_CAPABLE, CLOUD_CHEAP))


def test_an_unpriced_jev_model_is_refused() -> None:
    with pytest.raises(ValidationError, match=JEV):
        cloud(model_prices_usd_per_million=prices(CLOUD_CAPABLE, CLOUD_CHEAP))


def test_an_unpriced_jev_model_is_irrelevant_when_no_decider_is_configured() -> None:
    """The control for the test above: the Jev model is only reachable when the backend is Jev,
    so pricing it is not a precondition for a deployment that never calls it."""
    settings = cloud(
        decider_backend="none", model_prices_usd_per_million=prices(CLOUD_CAPABLE, CLOUD_CHEAP)
    )

    assert settings.decider_backend is DeciderBackend.NONE


# ---------------------------------------------------------------------------- enforce mode


@pytest.mark.parametrize(
    "missing",
    [
        "auto_approve_min_probability",
        "auto_approve_min_confidence",
        "auto_approve_max_irreversibility",
    ],
)
def test_enforce_mode_with_a_missing_threshold_is_refused(missing: str) -> None:
    with pytest.raises(ValidationError, match=missing.upper()):
        cloud(**{**ENFORCED, missing: None})


@pytest.mark.parametrize(
    ("field", "vacuous"),
    [
        # The route Choice has two options, so the winner is always at 0.5 or above: a floor at
        # or below 0.5 is satisfied by "route == auto_approve" alone.
        ("auto_approve_min_probability", 0.5),
        ("auto_approve_min_probability", 0.3),
        # Confidence is reported in [0, 1], so a floor of 0 is satisfied by every answer.
        ("auto_approve_min_confidence", 0.0),
        # A ceiling at or above 0.5 lets through a Noul that says "more likely irreversible".
        ("auto_approve_max_irreversibility", 0.5),
        ("auto_approve_max_irreversibility", 1.0),
    ],
)
def test_enforce_mode_with_a_threshold_that_cannot_fail_is_refused(
    field: str, vacuous: float
) -> None:
    with pytest.raises(ValidationError, match=field.upper()):
        cloud(**{**ENFORCED, field: vacuous})


@pytest.mark.parametrize(
    ("field", "just_inside"),
    [
        ("auto_approve_min_probability", 0.51),
        ("auto_approve_min_confidence", 0.01),
        ("auto_approve_max_irreversibility", 0.49),
    ],
)
def test_the_vacuity_bounds_are_exclusive(field: str, just_inside: float) -> None:
    """The control for the test above: the bound refuses exactly the values that cannot fail,
    and nothing a measurement might legitimately produce."""
    settings = cloud(**{**ENFORCED, field: just_inside})

    assert getattr(settings, field) == just_inside


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_threshold_outside_the_unit_interval_is_refused_in_any_mode(value: float) -> None:
    with pytest.raises(ValidationError):
        cloud(auto_approve_min_probability=value)


def test_shadow_mode_needs_no_threshold() -> None:
    """Shadow mode is how the thresholds get measured, so it cannot require them."""
    settings = cloud()

    assert settings.auto_approve_min_probability is None


def test_enforce_mode_with_no_decider_is_refused() -> None:
    """Enforcement the operator believes is on and is not -- the tracing validator's reasoning."""
    with pytest.raises(ValidationError, match="DECIDER_BACKEND"):
        cloud(decider_backend="none", **ENFORCED)
