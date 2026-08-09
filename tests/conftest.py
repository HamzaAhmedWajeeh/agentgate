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


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run a test with no inherited configuration.

    Clears every ``AGENTGATE_*`` variable and the bare ``OPENAI_API_KEY`` alias, then moves to
    an empty directory so the developer's own ``.env`` cannot be discovered. The settings cache
    is cleared on both sides: a stale cached object would otherwise leak a previous test's
    environment into this one, and this test's environment into the next.
    """
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX) or key == "OPENAI_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
