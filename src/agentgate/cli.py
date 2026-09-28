"""The command line: the surface this system is demonstrated from.

Treated as a user interface, not a test harness. The difference shows up in three places, and
each is a decision rather than a detail:

*The review packet is formatted for a person.* A human gate whose output is a JSON dump asks
someone to approve a deliverable they have to parse first, which is not informed approval and
is barely a gate. What gets printed is the draft, how much evidence stood behind it, and
whether any of that evidence is missing.

*Progress is streamed.* ``stream_mode=["updates", "messages"]`` means the graph is visibly
working -- classify, lane, research fanning out, drafting -- rather than a pause followed by an
answer. A demo of a governed runtime that shows nothing until it finishes has hidden the part
worth seeing.

*The thread id is the first thing printed and the last thing printed.* Every subsequent command
takes it, so hunting for it in scrollback is a failure of the interface.

**Durability is the claim, so the commands do not share a process.** ``run`` pauses at the gate
and exits. ``approve`` is a separate invocation that picks the run up from the checkpoint. That
only works against a checkpointer that outlives the process, which is why this warns when it is
pointed at the in-memory one.

**``history`` and ``fork`` are not here yet.** Time travel is Phase 6 and is not built; the
commands are absent rather than stubbed, and ``--help`` says so. A stub that printed "not
implemented" would be a command that exists and does nothing, which is worse than one that
does not exist.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Final

import typer

from agentgate.config import CheckpointerBackend, Settings, get_settings
from agentgate.errors import AgentgateError
from agentgate.graph.build import build_graph, checkpointer_for, resume_config, run_config
from agentgate.graph.state import initial_state
from agentgate.guardrails.run_ledger import ledger_of
from agentgate.guardrails.spend import SpendLedger

app = typer.Typer(
    name="agentgate",
    help=(
        "A gated agent runtime.\n\n"
        "Run a request, watch it work, and approve or reject the draft before anything "
        "irreversible happens.\n\n"
        "Not available yet: `history` and `fork`. Time travel over past checkpoints is Phase 6 "
        "and is not built, so those commands are absent rather than stubbed."
    ),
    add_completion=False,
    no_args_is_help=True,
)

QUESTION_OPTION = typer.Option([], "--question", "-q", help="A research sub-question. Repeatable.")

RULE = "-" * 72
"""ASCII, like everything else this module prints.

The Windows console defaults to cp1252, which cannot encode a box-drawing character or an
em-dash -- and `typer.echo` does not degrade, it raises `UnicodeEncodeError`. The first version
of this used both and crashed on the first line of output, on the machine the demo is given
from. A demo surface is only as portable as its narrowest terminal."""


def _echo(text: str = "") -> None:
    typer.echo(text)


def _warn_if_ephemeral(settings: Settings) -> None:
    """Say so when the run cannot survive this process.

    The in-memory checkpointer is the right default for tests and the wrong one for a demo:
    ``run`` would pause, exit, and take the run with it, and ``approve`` would report a thread
    that does not exist. That failure looks like a bug in the gate rather than a configuration
    choice, so it is called out before it happens rather than diagnosed afterwards.
    """
    if settings.checkpointer is CheckpointerBackend.MEMORY:
        _echo(
            typer.style(
                "  warning: AGENTGATE_CHECKPOINTER=memory. This run will not survive the "
                "process,\n           so `approve` will not find it. Use sqlite or postgres to "
                "resume across commands.",
                fg=typer.colors.YELLOW,
            )
        )
        _echo()


def _render_packet(packet: dict[str, Any], thread_id: str) -> None:
    """The review, for a person deciding whether to let something irreversible happen."""
    research = packet.get("research", {})
    complete = packet.get("answer_complete", True)

    _echo()
    _echo(typer.style("  APPROVAL REQUIRED", fg=typer.colors.YELLOW, bold=True))
    _echo(f"  {RULE}")
    _echo(f"  request   {packet.get('request', '')}")
    _echo(f"  revision  {packet.get('revision', 0)}")

    evidence = f"{packet.get('findings', 0)} findings"
    if not complete:
        missing = int(research.get("failed", 0)) + int(research.get("silent", 0))
        evidence += typer.style(
            f"  --  {missing} of {research.get('dispatched', 0)} research branches did not report",
            fg=typer.colors.RED,
        )
    _echo(f"  evidence  {evidence}")
    _echo(f"  {RULE}")
    _echo()
    for line in str(packet.get("draft", "")).splitlines() or [""]:
        _echo(f"    {line}")
    _echo()

    # The actions are shown one by one, exactly as they will be performed if approved. Approving
    # approves these and only these: `approve` carries back the hash of this packet.
    actions = list(packet.get("proposed_actions", []))
    if actions:
        _echo(typer.style("  ACTIONS -- performed only if you approve", fg=typer.colors.YELLOW))
        for number, action in enumerate(actions, start=1):
            arguments = ", ".join(
                f"{name}={value!r}" for name, value in dict(action.get("arguments", {})).items()
            )
            _echo(f"    {number}. {action.get('tool', '?')}({arguments})")
        _echo("    recorded to the outbox; nothing real is performed")
        _echo()
    else:
        _echo("  ACTIONS   none proposed -- approving releases the draft only")
        _echo()

    if not complete:
        _echo(
            typer.style(
                "  This draft was written from incomplete research. Approving it approves a "
                "partial answer.",
                fg=typer.colors.RED,
            )
        )
        _echo()

    _echo(f"  {RULE}")
    _echo(f"  thread    {typer.style(thread_id, bold=True)}")
    _echo(f"    approve:  agentgate approve {thread_id}")
    _echo(f'    reject:   agentgate reject {thread_id} --feedback "what to change"')
    _echo()


def _stream(
    graph: Any, payload: Any, config: dict[str, Any]
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Run the graph, printing each node as it completes, and return the final state.

    ``updates`` is what makes the progress line possible: it yields one entry per node as that
    node finishes, keyed by node name. ``messages`` is streamed alongside it so token output
    can be surfaced when there is a surface for it; the CLI prints node progress rather than
    tokens, because a governed runtime's interesting behaviour is which nodes ran, not the
    prose coming out of the last one.
    """
    for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "messages"]):
        if mode != "updates" or not isinstance(chunk, dict):
            continue
        for node in chunk:
            if node == "__interrupt__":
                continue
            # ASCII, not a bullet character. The Windows console default code page mangles
            # anything outside it, and a demo whose progress line renders as replacement
            # characters is a demo with a bug in the first thing anyone sees.
            _echo(f"  {typer.style('>', fg=typer.colors.GREEN)} {node}")

    # Interrupts live on the snapshot, not in `values`. Reading them from state returns nothing
    # and a paused run reports itself as having stopped without finalising -- which is what the
    # first version of this did.
    snapshot = graph.get_state(config)
    return dict(snapshot.values), tuple(snapshot.interrupts)


def _resume_with(thread_id: str, verdict: dict[str, Any], *, approving: bool = False) -> None:
    """Pick a paused run up from its checkpoint and hand it a decision."""
    from langgraph.types import Command  # noqa: PLC0415 - keeps `--help` fast

    settings = get_settings()

    with checkpointer_for(settings) as checkpointer:
        graph = build_graph(settings, checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.created_at:
            _echo(
                typer.style(
                    f"  No run found for thread {thread_id}. If the run was started under "
                    "AGENTGATE_CHECKPOINTER=memory it did not outlive that process.",
                    fg=typer.colors.RED,
                )
            )
            raise typer.Exit(code=1)

        # An approval carries the hash of the packet that was shown -- the one stored with the
        # pause, not one recomputed now -- so it cannot authorise actions that changed since.
        if approving and snapshot.interrupts:
            shown = dict(snapshot.interrupts[0].value).get("proposals_digest")
            verdict = {**verdict, "approved_digest": shown}

        # Resumed with a ledger that starts from what the run already spent, read off the
        # checkpoint -- this is a different process from the one that started the run.
        config = resume_config(graph, settings, thread_id)
        state, interrupts = _stream(graph, Command(resume=verdict), config)

    _report(state, thread_id, interrupts, ledger_of(config))


# Events that represent a model call, and therefore an endpoint something was sent to. Read
# from the trail rather than from configuration, because the difference between those two is the
# whole of leak inventory item 13.
LANE_BEARING_EVENTS: Final = {"classified": "classified", "drafted": "drafted"}


def _lanes_used(state: dict[str, Any]) -> str:
    """Which lanes model calls actually went to, in the order they happened.

    Empty when the trail carries neither event, which is the correct answer for a run that got
    nowhere -- rather than a reassuring line about the configured lane.
    """
    seen: dict[str, str] = {}
    for event in state.get("audit_trail", []):
        decided = str(event.get("decided", ""))
        lane = event.get("lane")
        if decided in LANE_BEARING_EVENTS and lane:
            seen[LANE_BEARING_EVENTS[decided]] = str(lane)
    return ", ".join(f"{what} on {lane}" for what, lane in seen.items())


def _spent(ledger: SpendLedger) -> str:
    """The whole run's spend so far, including what earlier invocations of it spent."""
    return f"{ledger.total_tokens} tokens, ${ledger.total_usd:.4f}"


def _report(
    state: dict[str, Any],
    thread_id: str,
    interrupts: tuple[Any, ...] = (),
    ledger: SpendLedger | None = None,
) -> None:
    """Print whatever the run arrived at: another review, or an ending."""
    if interrupts:
        _render_packet(dict(interrupts[0].value), thread_id)
        if ledger is not None:
            _echo(f"  spent     {_spent(ledger)}")
            _echo()
        return

    _echo()
    if state.get("finalised"):
        complete = state.get("answer_complete", True)
        colour = typer.colors.GREEN if complete else typer.colors.YELLOW
        _echo(typer.style("  FINISHED", fg=colour, bold=True))
        if not complete:
            _echo(
                typer.style(
                    "  The answer is incomplete: some research branches did not report.",
                    fg=typer.colors.YELLOW,
                )
            )
        _echo(f"  {RULE}")
        for line in str(state.get("draft", "")).splitlines() or [""]:
            _echo(f"    {line}")
        _echo(f"  {RULE}")
        _echo(f"  decision  {state.get('decision', 'pending')}")
        _echo(f"  revisions {state.get('revisions', 0)}")
        _echo(f"  events    {len(state.get('audit_trail', []))} audit events")
        if lanes := _lanes_used(state):
            _echo(f"  lanes     {lanes}")
    else:
        _echo(typer.style("  STOPPED without finalising.", fg=typer.colors.RED))
    if ledger is not None:
        _echo(f"  spent     {_spent(ledger)}")
    _echo(f"  thread    {thread_id}")
    _echo()


@app.command()
def run(
    request: str = typer.Argument(..., help="What you want the system to do."),
    # B008 is silenced rather than worked around: calling typer.Option in the default is how
    # Typer declares an option, and the rule exists for mutable defaults in ordinary functions.
    question: list[str] = QUESTION_OPTION,
    thread: str = typer.Option("", "--thread", help="Thread id to use. Generated if omitted."),
) -> None:
    """Start a run. It pauses at the approval gate and exits; approve or reject separately."""
    settings = get_settings()
    thread_id = thread or str(uuid.uuid4())

    _echo()
    _echo(f"  thread    {typer.style(thread_id, bold=True)}")
    # Labelled as the default, because it is not necessarily where anything will go. This line
    # said "lane  cloud" flat, which on a hybrid deployment is what the operator reads while a
    # restricted request is classified and drafted on their own endpoint -- the interface
    # repeating leak 13's confusion back at them. Where the calls actually went is printed at
    # the end, read off the audit trail.
    _echo(f"  default   {settings.lane.value} lane")
    _echo()
    _warn_if_ephemeral(settings)

    # One ledger for this run, reaching every model call it makes. A crossed ceiling raises out
    # of the stream and is reported by `main` like any other refusal: exit 2, no traceback.
    config = run_config(settings, thread_id)
    state = initial_state(request, thread_id)
    state["sub_questions"] = list(question)

    with checkpointer_for(settings) as checkpointer:
        graph = build_graph(settings, checkpointer)
        final, interrupts = _stream(graph, state, config)

    _report(final, thread_id, interrupts, ledger_of(config))


@app.command()
def resume(thread: str = typer.Argument(..., help="Thread id from `run`.")) -> None:
    """Show what a paused run is waiting on, without deciding anything."""
    settings = get_settings()
    config = {"configurable": {"thread_id": thread}}

    with checkpointer_for(settings) as checkpointer:
        graph = build_graph(settings, checkpointer)
        snapshot = graph.get_state(config)

    if not snapshot.created_at:
        _echo(typer.style(f"  No run found for thread {thread}.", fg=typer.colors.RED))
        raise typer.Exit(code=1)

    if not snapshot.interrupts:
        _report(dict(snapshot.values), thread)
        return

    _render_packet(dict(snapshot.interrupts[0].value), thread)


@app.command()
def approve(thread: str = typer.Argument(..., help="Thread id from `run`.")) -> None:
    """Approve the draft and the actions shown with it, and only those."""
    _resume_with(thread, {"decision": "approved"}, approving=True)


@app.command()
def reject(
    thread: str = typer.Argument(..., help="Thread id from `run`."),
    feedback: str = typer.Option(
        ..., "--feedback", "-f", help="What to change. Goes to the drafter."
    ),
) -> None:
    """Reject the draft and send it back for revision with feedback."""
    _resume_with(thread, {"decision": "rejected", "feedback": feedback})


def main() -> None:
    """Entry point. An agentgate error is reported, not traced.

    That sentence was here for a phase while both happened. The handler printed the message and
    then raised ``typer.Exit``, which is a click exception and only means anything *inside* the
    click invocation -- by the time ``app()`` has raised, there is nothing left to honour it. So
    it escaped as an ordinary exception: the readable message, then eighty-eight lines of
    traceback after it, and an exit code of 1 rather than the 2 the line claimed.

    Nothing caught it because nothing had ever made the CLI raise. Every command test runs on the
    fake lane, where no lane is unavailable and no provider can fail. The case that exercises
    this -- a restricted request on a deployment with one lane -- only became reachable when the
    policy gate started being enforced.
    """
    try:
        app()
    except AgentgateError as error:
        typer.echo(typer.style(f"\n  {error}\n", fg=typer.colors.RED), err=True)
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
