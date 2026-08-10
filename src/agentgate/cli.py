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

import uuid
from typing import Any

import typer

from agentgate.config import CheckpointerBackend, Settings, get_settings
from agentgate.errors import AgentgateError
from agentgate.graph.build import build_graph, checkpointer_for
from agentgate.graph.state import initial_state

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


def _resume_with(thread_id: str, verdict: dict[str, Any]) -> None:
    """Pick a paused run up from its checkpoint and hand it a decision."""
    from langgraph.types import Command  # noqa: PLC0415 - keeps `--help` fast

    settings = get_settings()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }

    with checkpointer_for(settings) as checkpointer:
        graph = build_graph(settings, checkpointer)
        snapshot = graph.get_state(config)
        if not snapshot.created_at:
            _echo(
                typer.style(
                    f"  No run found for thread {thread_id}. If the run was started under "
                    "AGENTGATE_CHECKPOINTER=memory it did not outlive that process.",
                    fg=typer.colors.RED,
                )
            )
            raise typer.Exit(code=1)

        state, interrupts = _stream(graph, Command(resume=verdict), config)

    _report(state, thread_id, interrupts)


def _report(state: dict[str, Any], thread_id: str, interrupts: tuple[Any, ...] = ()) -> None:
    """Print whatever the run arrived at: another review, or an ending."""
    if interrupts:
        _render_packet(dict(interrupts[0].value), thread_id)
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
    else:
        _echo(typer.style("  STOPPED without finalising.", fg=typer.colors.RED))
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
    _echo(f"  lane      {settings.lane.value}")
    _echo()
    _warn_if_ephemeral(settings)

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }
    state = initial_state(request, thread_id)
    state["sub_questions"] = list(question)

    with checkpointer_for(settings) as checkpointer:
        graph = build_graph(settings, checkpointer)
        final, interrupts = _stream(graph, state, config)

    _report(final, thread_id, interrupts)


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
    """Approve the draft. Everything past the gate becomes reachable."""
    _resume_with(thread, {"decision": "approved"})


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
    """Entry point. Configuration errors are reported, not traced."""
    try:
        app()
    except AgentgateError as error:
        typer.echo(typer.style(f"\n  {error}\n", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=2) from error


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
