"""Leak inventory item 24: the human gate approves a draft, and no action exists to approve.

The thesis is that a human gate approves anything irreversible. What a run actually reaches past
the gate is ``execute``, which records ``irreversible_effects: []`` -- because nothing in the
system ever proposes an effect. The executor's allowlist (``issue_refund``, ``send_customer_email``)
is declared and held by no agent; there is no state channel a proposal could travel in; and
``execute`` reads nothing that could name one.

Asserted as the current truth, in the same way as items 14, 17 and 18 were, so that closing it is a
visible inversion of these tests rather than an improvement nobody can date. Every absence here is
paired with a presence: the run does reach ``execute`` on an approval, so "nothing irreversible
happened" is a statement about a run that got there and not about one that stopped short.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from langgraph.types import Command

from agentgate.config import Settings
from agentgate.graph.build import build_checkpointer, build_graph, run_config
from agentgate.graph.nodes.execute import execute
from agentgate.graph.state import AgentState, Decision, initial_state
from agentgate.tools.registry import ALLOWLISTS, IRREVERSIBLE, Agent

pytestmark = pytest.mark.usefixtures("isolated_env")

SOURCE = Path(__file__).resolve().parents[2] / "src" / "agentgate"


def approved_run() -> dict[str, Any]:
    """A fake-lane run driven through the gate on an approval, to the end."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    graph = build_graph(settings, build_checkpointer(settings))
    config = run_config(settings, str(uuid.uuid4()))
    state = initial_state("Refund the customer the 240 GBP they were overcharged.", "item-24")
    state["sub_questions"] = ["refund escalation"]
    graph.invoke(state, config)
    return dict(graph.invoke(Command(resume={"decision": "approved"}), config))


def executed_event(result: dict[str, Any]) -> dict[str, Any]:
    events = [e for e in result["audit_trail"] if e["decided"] == "executed"]
    assert len(events) == 1, "precondition: the run passed the gate and reached execute once"
    return events[0]


def test_an_approved_run_reaches_execute_and_nothing_irreversible_happens() -> None:
    """The headline claim, read off a real run. The request asks for a refund in so many words;
    the gate approves; ``execute`` runs -- and records that it did nothing irreversible, because
    nothing ever proposed that it should."""
    result = approved_run()

    assert result["decision"] == Decision.APPROVED.value, "precondition: a human approved"
    assert executed_event(result)["detail"]["irreversible_effects"] == []
    assert "proposed_actions" not in result, "no proposal travelled to the gate"


def test_there_is_no_state_channel_a_proposal_could_travel_in() -> None:
    """Not just empty on this run: absent from the schema, so no node could write one."""
    channels = set(get_type_hints(AgentState, include_extras=False))

    assert channels, "precondition: the schema was read"
    assert "draft" in channels, "precondition: the thing the gate does approve is a channel"
    assert not {name for name in channels if "action" in name or "proposal" in name}


def test_execute_ignores_a_proposal_even_when_one_is_handed_to_it() -> None:
    """``execute`` reads nothing that could name an effect. Given an approved state carrying a
    fabricated proposal, it still records none -- so no upstream change alone could make it act."""
    state = initial_state("x", "item-24")
    state["decision"] = Decision.APPROVED.value
    state["proposed_actions"] = [  # type: ignore[typeddict-unknown-key]
        {"tool": "issue_refund", "arguments": {"account": "4929-1123-8876", "amount": 240}}
    ]
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    update = execute(state, settings)

    assert update["audit_trail"][0]["decided"] == "executed", "precondition: it ran"
    assert update["audit_trail"][0]["detail"]["irreversible_effects"] == []


def test_the_executor_allowlist_is_declared_and_held_by_no_agent() -> None:
    """The two irreversible tools exist, and are allowlisted to an executor that nothing in
    ``src/`` constructs or binds -- the privilege separation is half built."""
    assert ALLOWLISTS[Agent.EXECUTOR] == IRREVERSIBLE, "precondition: the executor's tools"
    assert IRREVERSIBLE, "precondition: there are irreversible tools to hold"

    holders = sorted(
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*.py")
        if path.name != "registry.py" and "Agent.EXECUTOR" in path.read_text(encoding="utf-8")
    )
    assert holders == [], f"something now holds the executor's tools: {holders}"
