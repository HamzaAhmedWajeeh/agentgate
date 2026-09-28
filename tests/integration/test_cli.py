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

**The last section points the CLI at a networked lane, which nothing here had ever done.** Every
test above runs on the fake lane, where no lane can be unavailable and no provider can fail -- so
the combination of the real client, a real socket and the command line was untested, and two
things were hiding in it. The CLI streams (``stream_mode=["updates", "messages"]``), so every
model call it makes is a streamed one, and the stub server answered a streamed request with an
ordinary JSON body until this file asked it to. And an ``AgentgateError`` raised mid-run printed
its message followed by eighty-eight lines of traceback and exited 1, under a docstring promising
it was "reported, not traced".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpus"
THREAD = "cli-test-thread"

CLOUD_CAPABLE = "cloud-capable-stub"
CLOUD_CHEAP = "cloud-cheap-stub"
SOVEREIGN_MODEL = "sovereign-stub"

RESTRICTED_REQUEST = (
    "Draft a refund letter for Jane Doe, account 4929-1123-8876, "
    "NHS number 485 777 3456, who was overcharged 240 GBP."
)

RESTRICTED_VERDICT = {
    "sensitivity": "restricted",
    "complexity": "simple",
    "contains_pii": True,
    "reason": "names an individual and an account number",
}


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


# ------------------------------------------------ a networked lane, through the command line


def networked_environment(tmp_path: Path, cloud: StubServer, **extra: str) -> dict[str, str]:
    """A cloud-default deployment pointed at a loopback stub.

    ``AGENTGATE_*`` is stripped from the inherited environment rather than merely overlaid, and
    the bare ``OPENAI_API_KEY`` alias with it. Every test above can overlay safely because it
    sets the variables it cares about; this one cares about a variable being **absent** --
    a developer with `AGENTGATE_SOVEREIGN_BASE_URL` exported would turn the cloud-only case into
    a hybrid one and the refusal under test would never happen.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("AGENTGATE_")}
    env.pop("OPENAI_API_KEY", None)
    env.update(
        {
            "AGENTGATE_LANE": "cloud",
            "AGENTGATE_OPENAI_API_KEY": "not-required",
            "AGENTGATE_OPENAI_BASE_URL": cloud.base_url,
            "AGENTGATE_CLOUD_CAPABLE_MODEL": CLOUD_CAPABLE,
            "AGENTGATE_CLOUD_CHEAP_MODEL": CLOUD_CHEAP,
            "AGENTGATE_MODEL_PRICES_USD_PER_MILLION": json.dumps(
                {
                    CLOUD_CAPABLE: {"input": 1.0, "output": 4.0},
                    CLOUD_CHEAP: {"input": 0.1, "output": 0.4},
                    SOVEREIGN_MODEL: {"input": 0.0, "output": 0.0},
                }
            ),
            "AGENTGATE_CHECKPOINTER": "sqlite",
            "AGENTGATE_SQLITE_PATH": str(tmp_path / "state.db"),
            "AGENTGATE_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
            "AGENTGATE_CORPUS_PATH": str(CORPUS),
            **extra,
        }
    )
    return env


def networked_cli(
    tmp_path: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "agentgate.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )


@pytest.fixture
def cloud_stub() -> Iterator[StubServer]:
    """Stands in for the third party, and classifies restricted so the policy gate has to act."""
    with running_stub(StubBehaviour(reply=RESTRICTED_VERDICT)) as server:
        yield server


@pytest.fixture
def sovereign_stub() -> Iterator[StubServer]:
    with running_stub(StubBehaviour(reply=RESTRICTED_VERDICT)) as server:
        yield server


def test_a_deployment_with_one_lane_refuses_a_restricted_request_readably(
    tmp_path: Path, cloud_stub: StubServer
) -> None:
    """The failure the policy fix introduced, seen the way an operator sees it.

    A cloud-only deployment has nowhere to serve a request the router sends somewhere stricter,
    and the only alternative to stopping is the lane policy just ruled out. So it stops -- and
    the whole question for a command line is whether that reads as a decision or as a crash.

    All three assertions matter and none of them held when this was written. The exit code was 1
    rather than the 2 the handler claimed, because `typer.Exit` raised outside the click
    invocation is just an exception. The message was there, followed by eighty-eight lines of
    traceback. And nothing had ever checked, because no CLI test could make the CLI raise.
    """
    env = networked_environment(tmp_path, cloud_stub)

    result = networked_cli(tmp_path, env, "run", RESTRICTED_REQUEST, "--thread", "refusal")

    assert result.returncode == 2, (
        f"expected the documented exit code for an agentgate error, got {result.returncode}"
    )
    assert "sovereign" in result.stderr, (
        "the message must name the lane that is missing, or the operator cannot act on it"
    )
    assert "Traceback" not in result.stderr, (
        "an agentgate error is reported, not traced -- that is what `main` promises"
    )
    assert "LaneUnavailableError" not in result.stderr, (
        "a class name is an implementation detail leaking into an operator-facing message"
    )
    assert len(result.stderr.splitlines()) <= 6, (
        f"{len(result.stderr.splitlines())} lines of stderr for one refusal; the traceback is back"
    )


def test_the_refusal_happens_before_the_draft_is_ever_requested(
    tmp_path: Path, cloud_stub: StubServer
) -> None:
    """Refusing late would mean refusing after sending the content somewhere.

    The point of stopping at the lane node is that nothing downstream runs. Read off the request
    log rather than the output: the classifier's call is there, and no synthesis call follows it.
    """
    env = networked_environment(tmp_path, cloud_stub)

    networked_cli(tmp_path, env, "run", RESTRICTED_REQUEST, "--thread", "refusal-early")

    models = [str(body.get("model", "")) for body in cloud_stub.behaviour.requests_seen]
    assert models, "nothing reached the stub, so this test is asserting about an empty log"
    assert CLOUD_CHEAP in models, "the classification call should have happened"
    assert CLOUD_CAPABLE not in models, (
        "the drafter was asked for a synthesis after policy had already refused the request"
    )


def test_a_hybrid_deployment_runs_and_reports_the_lane_it_actually_used(
    tmp_path: Path, cloud_stub: StubServer, sovereign_stub: StubServer
) -> None:
    """The non-vacuity partner, and the fix to a line that was quietly wrong.

    The two tests above pass just as well against a CLI that cannot reach a networked lane at
    all, so one case has to get all the way through one. It also pins the header: the CLI printed
    `lane  cloud` flat, which is what an operator reads while a restricted request is classified
    and drafted on their own endpoint -- the interface repeating item 13 back at them. It now says
    `default` there, and reports the lanes actually used from the audit trail.
    """
    env = networked_environment(
        tmp_path,
        cloud_stub,
        AGENTGATE_SOVEREIGN_BASE_URL=sovereign_stub.base_url,
        AGENTGATE_SOVEREIGN_MODEL=SOVEREIGN_MODEL,
    )

    result = networked_cli(tmp_path, env, "run", RESTRICTED_REQUEST, "--thread", "hybrid")

    assert result.returncode == 0, result.stderr
    assert "default   cloud lane" in result.stdout, (
        "the configured lane must be labelled as a default, not stated as the lane in use"
    )
    assert "classified on sovereign" in result.stdout, (
        "the run reported nothing about where the call went, which is the only fact that matters"
    )
    assert sovereign_stub.behaviour.requests_seen, "the sovereign endpoint was never called"


def test_every_streamed_call_asks_for_its_usage_block(
    tmp_path: Path, cloud_stub: StubServer, sovereign_stub: StubServer
) -> None:
    """Leak inventory item 18, closed, and this is the test that used to assert it was open.

    A streamed OpenAI response carries no token counts unless the caller sets
    ``stream_options.include_usage``, and the CLI is the only surface that streams. While chat
    spend went to no ledger the absent number was latent; with every model call accounted, the
    ledger refuses a call that reported nothing -- so an unfixed item 18 would stop every networked
    CLI run at its first model call. ``stream_usage=True`` on every OpenAI-compatible client is
    the fix, and the run completing is the proof it took.

    The non-vacuity guard stays: ``all()`` over an empty list is ``True``, and on a restricted
    request the cloud stub receives no streamed calls at all.
    """
    env = networked_environment(
        tmp_path,
        cloud_stub,
        AGENTGATE_SOVEREIGN_BASE_URL=sovereign_stub.base_url,
        AGENTGATE_SOVEREIGN_MODEL=SOVEREIGN_MODEL,
    )

    result = networked_cli(tmp_path, env, "run", RESTRICTED_REQUEST, "--thread", "usage")

    assert result.returncode == 0, result.stderr
    streamed = sovereign_stub.behaviour.streamed_requests
    assert streamed, (
        "no request arrived with stream=true, so either the CLI stopped streaming or this test "
        "is reading an empty list"
    )
    assert sovereign_stub.behaviour.usage_requested_on_every_stream(), (
        "a streamed call went out without stream_options.include_usage, so its cost is unmeasured "
        "and the ledger will refuse it -- ADR 0004 item 18 has reopened"
    )


def test_the_endpoint_does_return_usage_when_a_stream_asks_for_it() -> None:
    """The control for the test above, and it is not optional.

    "No usage came back" has two possible causes: the client did not ask, or the server cannot
    send. Only the first is a finding about this system; the second would be a finding about the
    test double, and asserting the absence without separating them is how a leak-inventory row
    gets written about the wrong thing.

    So this asks the endpoint directly, over real HTTP, with ``stream_options.include_usage`` set,
    and requires the usage chunk to arrive. `urllib` rather than a client library: the point is to
    speak the protocol without the layer whose defaults are under suspicion.
    """
    payload = json.dumps(
        {
            "model": "control-stub",
            "messages": [{"role": "user", "content": "anything"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()

    with running_stub(StubBehaviour(reply={"answer": "ok"})) as stub:
        request = urllib.request.Request(  # noqa: S310 - loopback, literal http scheme
            f"{stub.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            body = response.read().decode()

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert chunks, "the endpoint streamed nothing, so this control is reading an empty list"

    with_usage = [chunk for chunk in chunks if "usage" in chunk]
    assert with_usage, (
        "the endpoint sent no usage chunk even when asked, so the absence recorded in the test "
        "above is a limitation of this double rather than a fact about the client"
    )
    assert with_usage[-1]["usage"]["prompt_tokens"] > 0
    assert with_usage[-1]["choices"] == [], (
        "usage arrived attached to a choice; OpenAI sends it on a final chunk with none, and a "
        "client reading it off the last choice would find nothing"
    )


# ---------------------------------------------------------------------- the run ledger


def test_the_report_says_what_the_run_spent(tmp_path: Path) -> None:
    """Every model call is now accounted, so the operator is told the total, not left to infer
    it from a bill."""
    result = start(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "spent     " in result.stdout
    assert " tokens" in result.stdout


def test_a_crossed_ceiling_stops_the_run_readably(tmp_path: Path) -> None:
    """The ceiling trips at the first model call, and the command line says so: exit 2, a message
    naming which ceiling and by how much, and no traceback -- the same contract as every other
    refusal, now reachable by spending too much."""
    env = {**os.environ, **environment(tmp_path), "AGENTGATE_MAX_TOTAL_TOKENS": "1"}

    result = subprocess.run(
        [sys.executable, "-m", "agentgate.cli", "run", "Draft a refund note", "--thread", "cap"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "over the ceiling of 1" in result.stderr
    assert "Traceback" not in result.stderr
    assert "drafter" not in result.stdout, "nothing ran after the call that crossed it"


def spent_tokens(output: str) -> int:
    line = next(line for line in output.splitlines() if line.strip().startswith("spent"))
    return int(line.split()[1])


def test_approving_in_a_new_process_reports_the_whole_runs_spend(tmp_path: Path) -> None:
    """The ledger object dies with `run`'s process. `approve` is another process, and the only
    thing carried between them is the checkpoint -- so the figure it reports can only include
    what `run` spent if the new ledger was started from the spend written to state. A ledger
    started from zero would report less than `run` already had."""
    first = start(tmp_path)
    assert first.returncode == 0, first.stderr

    second = cli(tmp_path, "approve", THREAD)
    assert second.returncode == 0, second.stderr

    assert spent_tokens(second.stdout) >= spent_tokens(first.stdout) > 0


# ------------------------------------------------------------------ actions past the gate


PROPOSING_REPLY = {
    # Read by the classifier as a public verdict, and by the drafter as its final message.
    "sensitivity": "public",
    "complexity": "simple",
    "contains_pii": False,
    "reason": "a published refund policy question",
    "draft": "We will refund the overcharge.",
    "proposed_actions": [
        {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 240.0}}
    ],
}


def test_the_packet_shows_the_action_and_approve_performs_exactly_it(tmp_path: Path) -> None:
    """ADR 0004 item 24, through the command line. `run` shows the proposed refund on the
    packet; `approve`, in a new process, carries back the hash of that packet -- the one stored
    with the pause -- and the refund lands in the outbox once."""
    reply = StubBehaviour(reply=PROPOSING_REPLY, supports_native_structured_output=True)
    with running_stub(reply) as cloud:
        env = networked_environment(
            tmp_path, cloud, AGENTGATE_OUTBOX_PATH=str(tmp_path / "outbox.jsonl")
        )

        shown = networked_cli(
            tmp_path, env, "run", "Refund Jane.", "-q", "refund escalation", "--thread", "acts"
        )
        assert shown.returncode == 0, shown.stderr
        assert "issue_refund(account='4929', amount_units=240.0)" in shown.stdout
        assert not (tmp_path / "outbox.jsonl").exists(), "nothing performed before approval"

        approved = networked_cli(tmp_path, env, "approve", "acts")
        assert approved.returncode == 0, approved.stderr

    lines = (tmp_path / "outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool"] for line in lines] == ["issue_refund"]
