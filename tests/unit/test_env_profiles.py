"""The deployment examples in ``env/`` are configuration, and configuration is checked.

Each is held to the same rules as ``.env.example``: every variable it names is one the settings
model reads, no credential ships in it, and it describes a runnable deployment once the secrets the
operator supplies are present. And one rule of their own: **no profile variable**. Every setting is
stated; nothing sets others implicitly (ADR 0009).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentgate.config import (
    DeciderBackend,
    DeciderMode,
    EffectSinkBackend,
    Lane,
    Settings,
    TracingBackend,
    _recognised_variable_names,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

ENV = Path(__file__).resolve().parents[2] / "env"
PROFILES = sorted(ENV.glob("*.env"))
ASSIGNMENT = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
LIVE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\S*)", re.MULTILINE)


def test_both_deployment_examples_exist() -> None:
    assert {path.name for path in PROFILES} == {"hybrid.env", "sovereign.env"}


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_every_variable_named_is_one_the_settings_model_reads(path: Path) -> None:
    named = set(ASSIGNMENT.findall(path.read_text(encoding="utf-8")))

    assert named, "precondition: the file names variables"
    assert named - _recognised_variable_names() == set()


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_no_profile_variable_and_no_live_credential(path: Path) -> None:
    body = path.read_text(encoding="utf-8")

    assert "PROFILE" not in {name.split("_")[-1] for name in ASSIGNMENT.findall(body)}
    for name, value in LIVE.findall(body):
        if "KEY" in name or "SECRET" in name or "DSN" in name:
            pytest.fail(f"{path.name} assigns a live value to {name}")
        assert not value.startswith("sk-"), f"{name} looks like a real key"


def test_the_sovereign_example_is_runnable_and_sends_nothing_to_a_third_party() -> None:
    settings = Settings(_env_file=ENV / "sovereign.env")  # type: ignore[call-arg]

    assert settings.lane is Lane.SOVEREIGN
    assert Lane.CLOUD not in settings.routable_lanes
    assert settings.decider_backend is DeciderBackend.NONE
    assert settings.tracing_backend is TracingBackend.OTLP
    assert settings.effect_sink is EffectSinkBackend.OUTBOX
    assert settings.outbox_path == Path("data/outbox.jsonl")


def test_the_hybrid_example_is_runnable_once_its_secrets_are_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secrets are blank in the file on purpose. Supplied from the environment, as the file
    says to, it describes a valid hybrid deployment with Jev in shadow mode."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-required")
    monkeypatch.setenv("AGENTGATE_JEV_API_KEY", "not-required")

    settings = Settings(_env_file=ENV / "hybrid.env")  # type: ignore[call-arg]

    assert settings.lane is Lane.CLOUD
    assert settings.routable_lanes == frozenset({Lane.CLOUD, Lane.SOVEREIGN})
    assert settings.most_contained_lane is Lane.SOVEREIGN
    assert settings.decider_backend is DeciderBackend.JEV
    assert settings.decider_mode is DeciderMode.SHADOW
    assert settings.auto_approve_min_probability is None, "no threshold is chosen for anyone"
    assert settings.effect_sink is EffectSinkBackend.OUTBOX


def test_the_hybrid_example_refuses_to_start_without_its_decider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: the file alone is not enough, which is the point of
    leaving the secret out of it."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-required")

    with pytest.raises(ValueError, match="JEV_API_KEY"):
        Settings(_env_file=ENV / "hybrid.env")  # type: ignore[call-arg]
