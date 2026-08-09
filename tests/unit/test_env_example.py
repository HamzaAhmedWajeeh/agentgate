"""`.env.example` is documentation, and documentation rots.

It is the only reference an operator has for what this system can be told to do, so it is
checked against the settings model in both directions: nothing documented that does not exist,
nothing existing that is undocumented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentgate.config import ENV_PREFIX, Lane, Settings, _recognised_variable_names

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# Matches both a live assignment and a commented-out one, since optional variables are
# documented commented out and still count as documented.
ASSIGNMENT = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


@pytest.fixture(scope="module")
def documented_names() -> frozenset[str]:
    return frozenset(ASSIGNMENT.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file(), "the quickstart begins with copying this file"


def test_every_documented_variable_is_real(documented_names: frozenset[str]) -> None:
    """A variable in the example that the model does not read is a lie in the docs."""
    unknown = documented_names - _recognised_variable_names()

    assert not unknown, f"documented but not read by Settings: {sorted(unknown)}"


def test_every_setting_is_documented(documented_names: frozenset[str]) -> None:
    """A setting missing from the example is a knob no operator will ever find."""
    undocumented = {
        f"{ENV_PREFIX}{name}".upper()
        for name in Settings.model_fields
        if f"{ENV_PREFIX}{name}".upper() not in documented_names
    }
    # The cloud key is documented under its unprefixed alias, which is how it is usually set.
    undocumented.discard(f"{ENV_PREFIX}OPENAI_API_KEY")

    assert not undocumented, f"settings absent from .env.example: {sorted(undocumented)}"


def test_openai_key_is_documented_under_its_conventional_name(
    documented_names: frozenset[str],
) -> None:
    assert "OPENAI_API_KEY" in documented_names


@pytest.mark.usefixtures("isolated_env")
def test_copying_the_example_verbatim_produces_a_runnable_configuration() -> None:
    """`cp .env.example .env` must work with no edits and no API key.

    This is the first step of the quickstart. If it needs a key to get past import, the
    five-minute promise in the README is false.
    """
    settings = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]

    assert settings.lane is Lane.FAKE
    assert settings.requires_network is False


@pytest.mark.usefixtures("isolated_env")
def test_the_example_does_not_ship_a_credential() -> None:
    """A real key committed here would be public within a second of the next push."""
    body = ENV_EXAMPLE.read_text(encoding="utf-8")
    live_assignments = re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\S+)", body, re.MULTILINE)

    for name, value in live_assignments:
        if "KEY" in name or "DSN" in name or "SECRET" in name:
            pytest.fail(f"{name} is assigned a live value in .env.example")
        assert not value.startswith("sk-"), f"{name} looks like a real key"
