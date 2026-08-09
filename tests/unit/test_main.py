"""`python -m agentgate` is the diagnostic reached for when a deployment misbehaves.

It has to be trustworthy under exactly the conditions where trust is scarce: a broken
environment, and one holding a live credential.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentgate.__main__ import EXIT_BAD_CONFIG, EXIT_OK, main

pytestmark = pytest.mark.usefixtures("isolated_env")


def run_as_a_real_process(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Invoke the entry point the way an operator or a container does.

    In-process tests import ``agentgate.config`` once and reuse it, so the import-time
    validation never runs a second time and its failure mode goes unobserved. A subprocess
    gets a cold interpreter, which is the only way to see what actually happens at startup.
    """
    clean = {k: v for k, v in os.environ.items() if not k.startswith("AGENTGATE_")}
    clean.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "agentgate"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**clean, **env},
        timeout=60,
        check=False,
    )


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


# ------------------------------------------------------------------- cold-start behaviour
#
# Everything above runs in an interpreter that has already imported agentgate.config. These
# do not. The difference is not academic: it is the difference between exit 2 with a readable
# message and exit 1 with a traceback, which is what a container actually did.


def test_cold_start_with_valid_configuration_exits_zero(tmp_path: Path) -> None:
    result = run_as_a_real_process(tmp_path)

    assert result.returncode == EXIT_OK, result.stderr
    assert json.loads(result.stdout)["lane"] == "fake"


def test_cold_start_with_broken_configuration_exits_two(tmp_path: Path) -> None:
    result = run_as_a_real_process(tmp_path, AGENTGATE_LANE="cloud")

    assert result.returncode == EXIT_BAD_CONFIG
    assert result.stdout == ""


def test_cold_start_failure_shows_no_traceback(tmp_path: Path) -> None:
    """An operator should see what to fix, not the internals of pydantic."""
    result = run_as_a_real_process(tmp_path, AGENTGATE_LANE="cloud")

    assert "Traceback" not in result.stderr
    assert "pydantic" not in result.stderr
    assert "agentgate cannot start" in result.stderr
    assert "OPENAI_API_KEY" in result.stderr


def test_cold_start_rejects_a_typo_with_a_suggestion(tmp_path: Path) -> None:
    result = run_as_a_real_process(tmp_path, AGENTGATE_MAX_ITERATION="3")

    assert result.returncode == EXIT_BAD_CONFIG
    assert "Traceback" not in result.stderr
    assert "AGENTGATE_MAX_ITERATIONS" in result.stderr
