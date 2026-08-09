"""Configuration is the only place behaviour varies, so it is worth testing hard.

Three properties matter more than the rest and are asserted directly:

1. An unconfigured process is offline and free. If this regresses, CI starts costing money.
2. A misconfigured process dies at import with a message naming the variable at fault.
3. Secrets never render. If this regresses, a key reaches a log.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentgate.config import (
    ENV_PREFIX,
    CallClass,
    CheckpointerBackend,
    ConfigurationError,
    Lane,
    ModelPrice,
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


def priced(*models: str) -> dict[str, dict[str, float]]:
    """A price entry per model, so a networked lane passes the spend-guard precondition.

    The numbers are arbitrary; what matters is that a price exists. Identifiers are
    invented here on purpose -- no real model name belongs in a test.
    """
    return {model: {"input": 0.10, "output": 0.40} for model in models}


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
        model_prices_usd_per_million=priced("a-capable-model", "a-cheap-model"),
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
        model_prices_usd_per_million=priced("a-local-model"),
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
        model_prices_usd_per_million=priced("m"),
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


# --------------------------------------------------------------------------- cost


def test_output_ceilings_are_tight_where_the_answer_is_a_label() -> None:
    """A single global cap would have to be sized for synthesis.

    Every cheap call would then carry a synthesis-sized worst case, which is how a routing
    decision ends up costing as much as the deliverable.
    """
    settings = build()

    assert settings.max_tokens_for(CallClass.ROUTING) < settings.max_tokens_for(
        CallClass.CLASSIFICATION
    )
    assert settings.max_tokens_for(CallClass.CLASSIFICATION) < settings.max_tokens_for(
        CallClass.RESEARCH
    )
    assert settings.max_tokens_for(CallClass.RESEARCH) < settings.max_tokens_for(
        CallClass.SYNTHESIS
    )


def test_every_call_class_has_a_ceiling() -> None:
    """An unmapped class would fall through to a provider default, which is unbounded."""
    settings = build()

    for call_class in CallClass:
        assert settings.max_tokens_for(call_class) > 0


def test_a_networked_lane_without_prices_will_not_start() -> None:
    """An unpriced model looks free, so the ceiling is never crossed while money is spent.

    Refusing to start is the safe reading of "unknown cost".
    """
    with pytest.raises(ValidationError, match="no price configured"):
        build(
            lane="sovereign",
            sovereign_base_url="http://localhost:9/v1",
            sovereign_model="an-unpriced-model",
        )


def test_the_fake_lane_needs_no_prices_because_it_is_free() -> None:
    settings = build()

    price = settings.price_for(settings.model_for(Tier.CHEAP))

    assert price.cost_usd(1_000_000, 1_000_000) == 0.0


def test_price_is_charged_per_million_tokens_split_by_direction() -> None:
    price = ModelPrice(input=0.50, output=2.00)

    assert price.cost_usd(1_000_000, 0) == pytest.approx(0.50)
    assert price.cost_usd(0, 1_000_000) == pytest.approx(2.00)
    assert price.cost_usd(500_000, 250_000) == pytest.approx(0.25 + 0.50)


def test_pricing_an_unknown_model_refuses_rather_than_guessing() -> None:
    settings = build(
        lane="cloud",
        openai_api_key="sk-test",
        cloud_capable_model="a",
        cloud_cheap_model="b",
        model_prices_usd_per_million=priced("a", "b"),
    )

    with pytest.raises(ConfigurationError, match="refusing to guess"):
        settings.price_for("a-model-configuration-never-saw")


def test_session_ceiling_below_the_run_ceiling_is_rejected() -> None:
    """Otherwise a single run could never reach its own limit before the session one bit."""
    with pytest.raises(ValidationError, match="session ceiling"):
        build(max_spend_usd=10.0, max_session_spend_usd=1.0)


def test_session_ceiling_may_equal_the_run_ceiling() -> None:
    """A single-run session is a legitimate configuration, not an error."""
    assert build(max_spend_usd=1.0, max_session_spend_usd=1.0).max_session_spend_usd == 1.0


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


def test_another_tool_s_variables_in_a_shared_env_file_are_tolerated(
    tmp_path: Path,
) -> None:
    """`.env` belongs to the project, not to this application.

    Rejecting keys other tools keep there would make this config the owner of a file it
    merely reads -- and it really happened: an unprefixed LANGSMITH_* block from a developer's
    own `.env` stopped the process from starting.
    """
    env_file = tmp_path / "shared.env"
    env_file.write_text(
        "SOME_OTHER_TOOL_SETTING=1\nRAILS_ENV=production\nAGENTGATE_MAX_ITERATIONS=3\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.max_iterations == 3


def test_tolerance_does_not_extend_to_the_agentgate_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring a foreign key is courtesy. Ignoring a misspelled own key is a silent default."""
    monkeypatch.setenv("AGENTGATE_MAX_ITERATION", "3")

    with pytest.raises(ValidationError, match="AGENTGATE_MAX_ITERATIONS"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_the_langsmith_key_is_read_from_its_conventional_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared as an alias rather than left to happen by accident.

    Without the alias this field still picked the value up from a shared `.env`, and a
    credential being read by accident is worth making explicit.
    """
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-from-the-shell")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == "lsv2-from-the-shell"


# --------------------------------------------------------------------------- secrets


def test_secrets_do_not_render_in_repr_or_str() -> None:
    settings = build(
        lane="cloud",
        openai_api_key="sk-do-not-leak-me",
        cloud_capable_model="a",
        cloud_cheap_model="b",
        postgres_dsn="postgresql://user:hunter2@db/agentgate",
        model_prices_usd_per_million=priced("a", "b"),
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
        model_prices_usd_per_million=priced("a", "b"),
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


def test_importing_the_module_has_no_side_effects(tmp_path: Path) -> None:
    """Importing must not raise, however broken the environment is.

    Validation used to run at import, which made the import statement itself the thing that
    failed -- a landmine every future entry point had to know to step around, and one that
    already produced a wrong exit code once. A cold subprocess is the only way to observe
    this; by the time an in-process test runs, the module is already in sys.modules.
    """
    clean = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIX)}
    clean.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "-c", "import agentgate.config"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**clean, "AGENTGATE_LANE": "cloud"},  # cloud with no key: definitely invalid
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


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
