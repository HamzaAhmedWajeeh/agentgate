"""Configuration is the only place behaviour varies, so it is worth testing hard.

Three properties matter more than the rest and are asserted directly:

1. An unconfigured process is offline and free. If this regresses, CI starts costing money.
2. A misconfigured process dies at import with a message naming the variable at fault.
3. Secrets never render. If this regresses, a key reaches a log.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentgate.config import (
    CheckpointerBackend,
    ConfigurationError,
    Lane,
    Settings,
    StoreBackend,
    Tier,
    VectorBackend,
    get_settings,
)

pytestmark = pytest.mark.usefixtures("isolated_env")


def build(**overrides: object) -> Settings:
    """Construct settings from explicit values only, ignoring any ``.env`` on disk."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- safe defaults


def test_unconfigured_process_is_offline_and_free() -> None:
    settings = build()

    assert settings.lane is Lane.FAKE
    assert settings.requires_network is False
    assert settings.checkpointer is CheckpointerBackend.MEMORY
    assert settings.store_backend is StoreBackend.MEMORY
    assert settings.vector_backend is VectorBackend.MEMORY


def test_default_temperature_is_deterministic() -> None:
    assert build().temperature == 0.0


def test_fake_lane_resolves_a_model_per_tier_without_configuration() -> None:
    settings = build()

    assert settings.model_for(Tier.CAPABLE) != settings.model_for(Tier.CHEAP)


# --------------------------------------------------------------------------- cloud lane


def test_cloud_lane_without_credentials_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        build(lane="cloud")

    message = str(caught.value)
    assert "OPENAI_API_KEY" in message
    assert "AGENTGATE_CLOUD_CAPABLE_MODEL" in message
    assert "AGENTGATE_CLOUD_CHEAP_MODEL" in message


def test_cloud_lane_reports_only_the_variables_actually_missing() -> None:
    with pytest.raises(ValidationError) as caught:
        build(lane="cloud", openai_api_key="sk-test", cloud_capable_model="a-capable-model")

    message = str(caught.value)
    assert "AGENTGATE_CLOUD_CHEAP_MODEL" in message
    assert "AGENTGATE_CLOUD_CAPABLE_MODEL" not in message


def test_cloud_lane_with_complete_configuration_resolves_both_tiers() -> None:
    settings = build(
        lane="cloud",
        openai_api_key="sk-test",
        cloud_capable_model="a-capable-model",
        cloud_cheap_model="a-cheap-model",
    )

    assert settings.requires_network is True
    assert settings.model_for(Tier.CAPABLE) == "a-capable-model"
    assert settings.model_for(Tier.CHEAP) == "a-cheap-model"


def test_bare_openai_api_key_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The conventional unprefixed name works, so an existing shell export is picked up."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-shell")

    assert build().openai_api_key is not None


# --------------------------------------------------------------------------- sovereign lane


def test_sovereign_lane_without_an_endpoint_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        build(lane="sovereign")

    message = str(caught.value)
    assert "AGENTGATE_SOVEREIGN_BASE_URL" in message
    assert "AGENTGATE_SOVEREIGN_MODEL" in message


def test_sovereign_lane_needs_no_api_key() -> None:
    """Self-hosted servers ignore the key, so requiring one would be theatre."""
    settings = build(
        lane="sovereign",
        sovereign_base_url="http://localhost:11434/v1",
        sovereign_model="a-local-model",
    )

    assert settings.model_for(Tier.CAPABLE) == "a-local-model"
    assert settings.model_for(Tier.CHEAP) == "a-local-model"


@pytest.mark.parametrize(
    "url",
    ["localhost:11434", "ftp://localhost/v1", "not a url", "//localhost/v1"],
)
def test_endpoints_that_are_not_http_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        build(lane="sovereign", sovereign_base_url=url, sovereign_model="m")


def test_trailing_slash_is_normalised_away() -> None:
    settings = build(
        lane="sovereign",
        sovereign_base_url="http://localhost:11434/v1/",
        sovereign_model="m",
    )

    assert settings.sovereign_base_url == "http://localhost:11434/v1"


# --------------------------------------------------------------------------- backends


def test_postgres_checkpointer_without_a_dsn_is_rejected() -> None:
    with pytest.raises(ValidationError, match="AGENTGATE_POSTGRES_DSN"):
        build(checkpointer="postgres")


def test_postgres_store_without_a_dsn_is_rejected() -> None:
    with pytest.raises(ValidationError, match="AGENTGATE_POSTGRES_DSN"):
        build(store_backend="postgres")


def test_qdrant_backend_without_a_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="AGENTGATE_QDRANT_URL"):
        build(vector_backend="qdrant")


def test_sqlite_checkpointer_needs_nothing_beyond_a_path() -> None:
    settings = build(checkpointer="sqlite")

    assert settings.checkpointer is CheckpointerBackend.SQLITE
    assert settings.sqlite_path.name.endswith(".db")


# --------------------------------------------------------------------------- budgets


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iterations", 0),
        ("max_iterations", -1),
        ("max_total_tokens", 0),
        ("max_spend_usd", 0),
        ("max_spend_usd", -0.01),
        ("retrieval_top_k", 0),
        ("request_timeout_seconds", 0),
        ("api_port", 0),
        ("api_port", 70000),
        ("temperature", -0.1),
        ("max_retries", 99),
    ],
)
def test_nonsense_numeric_settings_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        build(**{field: value})


def test_recursion_limit_must_sit_above_the_iteration_budget() -> None:
    """The budget guard is the policy; the recursion limit is only a backstop behind it.

    Inverting them means LangGraph aborts the run before the guard can finalise cleanly, which
    turns a governed stop into a crash.
    """
    with pytest.raises(ValidationError, match="backstop"):
        build(max_iterations=40, recursion_limit=10)


def test_equal_limits_are_rejected_because_the_guard_would_never_win() -> None:
    with pytest.raises(ValidationError):
        build(max_iterations=20, recursion_limit=20)


# --------------------------------------------------------------------------- typos


def test_unknown_prefixed_variable_is_a_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misspelled variable must fail loudly, not be silently ignored.

    Silent tolerance here is how a production deployment ends up running on defaults while its
    operator believes a budget is in force.
    """
    monkeypatch.setenv("AGENTGATE_MAX_ITERATION", "3")  # missing the plural

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["clowd", "", "CLOUD_LANE", "local"])
def test_unknown_lane_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        build(lane=value)


# --------------------------------------------------------------------------- secrets


def test_secrets_do_not_render_in_repr_or_str() -> None:
    settings = build(
        lane="cloud",
        openai_api_key="sk-do-not-leak-me",
        cloud_capable_model="a",
        cloud_cheap_model="b",
        postgres_dsn="postgresql://user:hunter2@db/agentgate",
    )

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert "sk-do-not-leak-me" not in rendered
        assert "hunter2" not in rendered


def test_the_secret_is_still_retrievable_deliberately() -> None:
    settings = build(
        lane="cloud",
        openai_api_key="sk-do-not-leak-me",
        cloud_capable_model="a",
        cloud_cheap_model="b",
    )

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-do-not-leak-me"


# --------------------------------------------------------------------------- immutability


def test_settings_cannot_be_mutated_mid_run() -> None:
    """Two nodes disagreeing about the budget is a class of bug worth designing out."""
    settings = build()

    with pytest.raises(ValidationError):
        settings.max_spend_usd = 999.0


# --------------------------------------------------------------------------- accessor


def test_get_settings_returns_the_same_object_every_time() -> None:
    assert get_settings() is get_settings()


def test_get_settings_reflects_the_environment_after_a_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert get_settings().max_iterations == 8

    monkeypatch.setenv("AGENTGATE_MAX_ITERATIONS", "3")
    get_settings.cache_clear()

    assert get_settings().max_iterations == 3


def test_get_settings_raises_an_operator_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTGATE_LANE", "cloud")
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError) as caught:
        get_settings()

    message = str(caught.value)
    assert "agentgate cannot start" in message
    assert "OPENAI_API_KEY" in message
    assert ".env.example" in message
    assert "Traceback" not in message


def test_operator_error_does_not_wrap_a_leaked_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTGATE_LANE", "cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-leak-me")
    monkeypatch.setenv("AGENTGATE_CLOUD_CAPABLE_MODEL", "a")
    monkeypatch.setenv("AGENTGATE_RECURSION_LIMIT", "1")
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError) as caught:
        get_settings()

    assert "sk-do-not-leak-me" not in str(caught.value)
