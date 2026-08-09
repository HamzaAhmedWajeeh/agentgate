"""Shared fixtures.

The suite must pass offline with no API key set, on a machine that may well have a populated
``.env`` and a shell full of ``AGENTGATE_*`` exports. Anything that touches configuration
therefore runs inside :func:`isolated_env`, which strips both sources.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentgate.config import ENV_PREFIX, get_settings

# Variables that would send trace data out of the process if they leaked in from the
# developer's shell or a CI secret. LangChain and OpenTelemetry both auto-configure from the
# environment, so an exported LANGSMITH_TRACING is enough to make an otherwise offline test
# suite start uploading prompts. Stripped unconditionally; see docs/adr/0008.
TRACING_PREFIXES = ("LANGSMITH_", "LANGCHAIN_TRACING", "LANGCHAIN_ENDPOINT", "OTEL_")


def strips_to_offline(name: str) -> bool:
    """Whether an environment variable must not survive into a test."""
    return (
        name.startswith(ENV_PREFIX) or name == "OPENAI_API_KEY" or name.startswith(TRACING_PREFIXES)
    )


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run a test with no inherited configuration and no route off the machine.

    Clears every ``AGENTGATE_*`` variable, the bare ``OPENAI_API_KEY`` alias, and every
    tracing variable, then moves to an empty directory so the developer's own ``.env`` cannot
    be discovered. The settings cache is cleared on both sides: a stale cached object would
    otherwise leak a previous test's environment into this one, and this test's environment
    into the next.
    """
    for key in list(os.environ):
        if strips_to_offline(key):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
