"""The command line, exercised the way it is used: one process per command.

**The durability claim is the reason this file spawns subprocesses.** `run` pauses at the
approval gate and exits; `approve` is a separate invocation that picks the run up from the
checkpoint. A test where both halves share a Python session proves the graph can be resumed
from an object it still holds, which is not the claim and not what a demo does. So every
command here runs in its own interpreter, and the only thing carried between them is the
checkpoint on disk.

The guard against that passing for the wrong reason is
``test_the_same_flow_fails_on_an_ephemeral_checkpointer``: point the identical commands at the
in-memory checkpointer and `approve` must fail to find the run. If it does not, the sqlite test
is not proving what it says.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpus"
THREAD = "cli-test-thread"


def environment(tmp_path: Path, checkpointer: str = "sqlite") -> dict[str, str]:
    """A configuration that reaches nothing and writes only into the test's directory."""
    return {
        "AGENTGATE_LANE": "fake",
        "AGENTGATE_CHECKPOINTER": checkpointer,
        "AGENTGATE_SQLITE_PATH": str(tmp_path / "state.db"),
        "AGENTGATE_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        "AGENTGATE_CORPUS_PATH": str(CORPUS),
    }


def cli(
    tmp_path: Path, *args: str, checkpointer: str = "sqlite"
) -> subprocess.CompletedProcess[str]:
    """One command, one interpreter. Never reused between calls.

    The parent environment is inherited rather than replaced -- a stripped one breaks winsock
    on Windows before Python finishes importing -- and the AGENTGATE_* settings are overlaid on
    top. `cwd` is the test's own directory so the developer's real `.env` is never read: this
    suite must not behave differently on a machine that happens to have a key configured.
    """
    env = {**os.environ, **environment(tmp_path, checkpointer)}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "agentgate.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )


def start(tmp_path: Path, checkpointer: str = "sqlite") -> subprocess.CompletedProcess[str]:
    return cli(
        tmp_path,
        "run",
        "Draft a refund note",
        "-q",
        "refund escalation",
        "--thread",
        THREAD,
        checkpointer=checkpointer,
    )


# ------------------------------------------------------------------ the interface


def test_a_run_pauses_at_the_gate_and_shows_a_readable_packet(tmp_path: Path) -> None:
    """Not a JSON dump. A person has to be able to act on this without parsing it."""
    result = start(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "APPROVAL REQUIRED" in result.stdout
    assert "Draft a refund note" in result.stdout
    assert "findings" in result.stdout
    assert "{" not in result.stdout, "the review packet is being dumped as a structure"


def test_the_thread_id_and_the_next_commands_are_on_screen(tmp_path: Path) -> None:
    """Hunting for the thread id in scrollback is a failure of the interface."""
    result = start(tmp_path)

    assert f"agentgate approve {THREAD}" in result.stdout
    assert f"agentgate reject {THREAD}" in result.stdout


def test_progress_is_streamed_node_by_node(tmp_path: Path) -> None:
    """A demo that shows nothing until it finishes has hidden the part worth seeing."""
    result = start(tmp_path)

    for node in ("classify", "supervisor", "researcher", "research_branch", "drafter"):
        assert f"> {node}" in result.stdout, f"{node} never appeared in the progress output"


def test_every_line_survives_a_cp1252_console(tmp_path: Path) -> None:
    """The demo is given from a Windows terminal, where `typer.echo` raises rather than
    degrades on anything outside the code page. The first version printed a box-drawing rule
    and crashed on its own first line."""
    result = start(tmp_path)

    result.stdout.encode("cp1252")  # raises UnicodeEncodeError if anything is out of range
    assert "UnicodeEncodeError" not in result.stderr


# ------------------------------------------------------------------ the durability claim


def test_approving_from_a_fresh_process_resumes_the_same_run(tmp_path: Path) -> None:
    """The claim, tested as it is used.

    Two interpreters. Nothing shared but the checkpoint on disk. If the graph could only be
    resumed from an object still held in memory, this would fail -- and a same-session test
    would not notice.
    """
    started = start(tmp_path)
    assert "APPROVAL REQUIRED" in started.stdout

    approved = cli(tmp_path, "approve", THREAD)

    assert approved.returncode == 0, approved.stderr
    assert "FINISHED" in approved.stdout
    assert "decision  approved" in approved.stdout
    assert "> finalise" in approved.stdout, "the resumed run did not reach the end of the graph"


def test_rejecting_from_a_fresh_process_returns_to_the_gate_with_a_revision(
    tmp_path: Path,
) -> None:
    started = start(tmp_path)
    assert "revision  0" in started.stdout

    rejected = cli(tmp_path, "reject", THREAD, "--feedback", "cite the retention schedule")

    assert rejected.returncode == 0, rejected.stderr
    assert "APPROVAL REQUIRED" in rejected.stdout
    assert "revision  1" in rejected.stdout


def test_a_third_process_finishes_what_the_first_two_started(tmp_path: Path) -> None:
    """Run, reject, approve -- three interpreters, one run. This is the live demo."""
    start(tmp_path)
    cli(tmp_path, "reject", THREAD, "--feedback", "more detail")
    approved = cli(tmp_path, "approve", THREAD)

    assert "FINISHED" in approved.stdout
    assert "revisions 1" in approved.stdout


def test_the_same_flow_fails_on_an_ephemeral_checkpointer(tmp_path: Path) -> None:
    """The guard. Without it, the tests above could pass for the wrong reason.

    Point the identical commands at the in-memory checkpointer and the second process must not
    find the run -- because there is nothing on disk for it to find. If this ever passes, the
    durability tests are proving something weaker than they claim.
    """
    started = start(tmp_path, checkpointer="memory")
    assert "warning" in started.stdout, "the ephemeral checkpointer was not called out"

    approved = cli(tmp_path, "approve", THREAD, checkpointer="memory")

    assert approved.returncode != 0
    assert "No run found" in approved.stdout


# ------------------------------------------------------------------ what is not here


def test_help_says_time_travel_is_not_available(tmp_path: Path) -> None:
    """`history` and `fork` are absent rather than stubbed, and absence without explanation
    reads as an oversight. A stub that printed "not implemented" would be worse: a command that
    exists and does nothing."""
    result = cli(tmp_path, "--help")

    assert "history" in result.stdout
    assert "not built" in result.stdout or "Not available yet" in result.stdout
    assert "fork" in result.stdout


def test_the_absent_commands_really_are_absent(tmp_path: Path) -> None:
    for command in ("history", "fork"):
        assert cli(tmp_path, command).returncode != 0, f"{command} exists and should not"
