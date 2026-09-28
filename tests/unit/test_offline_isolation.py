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

# Every variable TypeSafe's official SDKs read on their own, from the Python SDK's
# `typesafe_sdk.constants` and the JavaScript SDK's `ENV`, both read 2026-09-28. The key is a
# credential for a cloud egress; the base URL and default model decide where a call goes and what
# answers it -- `TYPESAFE_DEFAULT_MODEL` defaults to the `jev-latest` alias this project refuses.
TYPESAFE_SDK_VARIABLES = (
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_DEFAULT_MODEL",
    "TYPESAFE_LOG_LEVEL",
)

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
    *TYPESAFE_SDK_VARIABLES,
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


class TestTypeSafeVariablesExportedBeforeIsolation:
    """The named case: a machine with TypeSafe configured, and the fixture shown to undo it.

    ``test_the_fixture_actually_removed_them`` passes on any machine that never set the variable,
    which is most of them, so on its own it proves nothing about these. Here every SDK variable is
    exported *before* ``isolated_env`` runs -- an autouse fixture runs first within its scope --
    and the test body sees what survived.
    """

    exported: tuple[str, ...] = ()

    @pytest.fixture(autouse=True)
    def typesafe_configured_on_this_machine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (*TYPESAFE_SDK_VARIABLES, "TYPESAFE_ANY_FUTURE_SETTING"):
            monkeypatch.setenv(name, "set-by-the-developer-shell")
        type(self).exported = (*TYPESAFE_SDK_VARIABLES, "TYPESAFE_ANY_FUTURE_SETTING")

    def test_no_typesafe_variable_survives_into_a_test(self) -> None:
        # Presence: the export ran, so the absence below is about variables that were there.
        assert self.exported, "the exporting fixture did not run, so this proves nothing"
        survivors = [name for name in self.exported if name in os.environ]
        assert survivors == [], f"survived isolated_env: {survivors}"

    def test_a_key_left_in_place_would_have_been_read(self) -> None:
        """Why it matters here and not only for the SDK: ``TYPESAFE_API_KEY`` is a declared alias
        of ``jev_api_key``, so a surviving key configures the decider in every test."""
        assert get_settings().jev_api_key is None
