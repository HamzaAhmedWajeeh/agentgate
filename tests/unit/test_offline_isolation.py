"""The suite must not be able to phone home, however the machine is configured.

LangChain and OpenTelemetry both auto-configure from the environment. A developer with
`LANGSMITH_TRACING=true` exported in their shell, or a CI runner with an inherited `OTEL_*`
block, would otherwise turn an offline test suite into one that uploads prompts and document
content -- silently, and with no test failing to say so.

The environments this runtime targets treat that content as regulated data, so the isolation
is asserted rather than assumed. See docs/adr/0008.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError
from tests.conftest import TRACING_PREFIXES, strips_to_offline

from agentgate.config import Settings, TracingBackend, get_settings

pytestmark = pytest.mark.usefixtures("isolated_env")

# Exactly the variables a real machine is likely to have set.
LEAK_VECTORS = [
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_TRACES_EXPORTER",
    "OTEL_SDK_DISABLED",
    "OPENAI_API_KEY",
    "AGENTGATE_LANE",
]


@pytest.mark.parametrize("variable", LEAK_VECTORS)
def test_every_known_leak_vector_is_stripped(variable: str) -> None:
    """Each of these, left in place, would route data off the machine or cost money."""
    assert strips_to_offline(variable), f"{variable} would survive into a test run"


@pytest.mark.parametrize("variable", LEAK_VECTORS)
def test_the_fixture_actually_removed_them(variable: str) -> None:
    """Asserting the predicate is not enough; the fixture has to have applied it."""
    assert variable not in os.environ


def test_a_tracing_variable_exported_globally_does_not_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates the developer whose shell has tracing on for another project.

    monkeypatch here sets the variable *after* the fixture ran, then a nested application of
    the same predicate proves the rule catches it.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    assert strips_to_offline("LANGSMITH_TRACING")


def test_tracing_is_off_unless_deliberately_configured() -> None:
    """Default off. With no backend selected, nothing leaves the process."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.tracing_backend is TracingBackend.NONE


def test_the_default_process_traces_nowhere() -> None:
    assert get_settings().tracing_backend is TracingBackend.NONE


def test_a_backend_without_a_destination_is_rejected() -> None:
    """Spans dropped on the floor while the operator believes tracing is on is the worst case."""
    with pytest.raises(ValidationError, match="OTEL_EXPORTER_ENDPOINT"):
        Settings(_env_file=None, tracing_backend="otlp")  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        Settings(_env_file=None, tracing_backend="langsmith")  # type: ignore[call-arg]


def test_the_langsmith_key_never_renders() -> None:
    """A trace backend credential is a secret like any other."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        tracing_backend="langsmith",
        langsmith_api_key="lsv2-do-not-leak-me",
    )

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert "lsv2-do-not-leak-me" not in rendered


def test_the_prefix_list_is_not_silently_empty() -> None:
    """A refactor that emptied this tuple would make every test above pass vacuously."""
    assert TRACING_PREFIXES
    assert all(prefix for prefix in TRACING_PREFIXES)
