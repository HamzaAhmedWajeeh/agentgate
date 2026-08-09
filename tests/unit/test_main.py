"""`python -m agentgate` is the diagnostic reached for when a deployment misbehaves.

It has to be trustworthy under exactly the conditions where trust is scarce: a broken
environment, and one holding a live credential.
"""

from __future__ import annotations

import json

import pytest

from agentgate.__main__ import EXIT_BAD_CONFIG, EXIT_OK, main

pytestmark = pytest.mark.usefixtures("isolated_env")


def test_valid_configuration_prints_json_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([])

    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["lane"] == "fake"
    assert payload["checkpointer"] == "memory"


def test_output_is_machine_readable_on_stdout_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostics on stdout would break `python -m agentgate | jq`."""
    main([])

    captured = capsys.readouterr()
    assert captured.err == ""
    json.loads(captured.out)


def test_broken_configuration_exits_nonzero_with_the_reason_on_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTGATE_LANE", "cloud")

    code = main([])

    assert code == EXIT_BAD_CONFIG
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "agentgate cannot start" in captured.err
    assert "OPENAI_API_KEY" in captured.err


def test_secrets_are_masked_in_the_dump(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """This output gets pasted into issues and chat threads. It must be safe to paste."""
    monkeypatch.setenv("AGENTGATE_LANE", "cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-leak-me")
    monkeypatch.setenv("AGENTGATE_CLOUD_CAPABLE_MODEL", "a")
    monkeypatch.setenv("AGENTGATE_CLOUD_CHEAP_MODEL", "b")
    monkeypatch.setenv("AGENTGATE_POSTGRES_DSN", "postgresql://u:hunter2@db/agentgate")

    code = main([])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "sk-do-not-leak-me" not in out
    assert "hunter2" not in out
    assert json.loads(out)["cloud_capable_model"] == "a"


def test_arguments_are_rejected_rather_than_ignored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--verbose"])

    assert code == EXIT_BAD_CONFIG
    assert "usage" in capsys.readouterr().err
