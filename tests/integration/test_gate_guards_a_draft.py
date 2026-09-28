"""Leak inventory item 24, closed: the gate now guards the action, not only a draft.

These were written as assertions of the old truth -- an approved run reached ``execute`` and
recorded ``irreversible_effects: []``, no state channel could carry a proposal, ``execute`` ignored
one handed to it, and nothing held the executor's tools. Closing the item inverted each of them,
which is the point of pinning an open row as a passing test: the change is visible, named, and
dated, rather than an improvement nobody noticed.

Every effect is read off the outbox on disk. The outbox is the only effect sink, and configuration
refuses any other, so what these tests prove happened is a record -- nothing real.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from langgraph.types import Command

from agentgate.config import CallClass, Settings
from agentgate.effects.proposals import digest_of
from agentgate.graph.build import build_checkpointer, build_graph, run_config
from agentgate.graph.nodes.execute import execute
from agentgate.graph.state import AgentState, Decision, Proposal, initial_state
from agentgate.models.fake import FakeChatModel, scripted_json
from agentgate.tools.registry import ALLOWLISTS, IRREVERSIBLE, Agent

pytestmark = pytest.mark.usefixtures("isolated_env")

SOURCE = Path(__file__).resolve().parents[2] / "src" / "agentgate"
CORPUS = Path(__file__).resolve().parents[2] / "corpus"
REFUND = {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 240.0}}
VERDICT = scripted_json(
    {"sensitivity": "internal", "complexity": "simple", "contains_pii": False, "reason": "t"}
)


def settings_for(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, corpus_path=CORPUS, outbox_path=tmp_path / "outbox.jsonl"
    )


def outbox(settings: Settings) -> list[dict[str, Any]]:
    if not settings.outbox_path.exists():
        return []
    return [json.loads(line) for line in settings.outbox_path.read_text("utf-8").splitlines()]


def factory(_s: Settings, _t: object, call_class: CallClass, **_k: object) -> Any:
    if call_class is CallClass.SYNTHESIS:
        reply = scripted_json({"draft": "Refunding the overcharge.", "proposed_actions": [REFUND]})
        return FakeChatModel(responses=[reply])
    return FakeChatModel(responses=[VERDICT])


def test_an_approved_run_performs_the_action_that_was_proposed_and_shown(tmp_path: Path) -> None:
    """Was: an approved run reached execute and nothing irreversible happened. Now the request
    that asks for a refund produces one, the human is shown it, and approving performs it."""
    settings = settings_for(tmp_path)
    graph = build_graph(settings, build_checkpointer(settings), model_factory=factory)
    config = run_config(settings, str(uuid.uuid4()))
    state = initial_state("Refund the customer the 240 GBP they were overcharged.", "item-24")
    state["sub_questions"] = ["refund escalation"]
    graph.invoke(state, config)
    shown = dict(graph.get_state(config).interrupts[0].value)

    result = graph.invoke(
        Command(resume={"decision": "approved", "approved_digest": shown["proposals_digest"]}),
        config,
    )

    assert result["decision"] == Decision.APPROVED.value, "precondition: a human approved"
    executed = [e for e in result["audit_trail"] if e["decided"] == "executed"]
    assert len(executed) == 1
    assert [e["tool"] for e in executed[0]["detail"]["irreversible_effects"]] == ["issue_refund"]
    assert [(e["tool"], e["arguments"]) for e in outbox(settings)] == [
        (REFUND["tool"], REFUND["arguments"])
    ]


def test_proposals_travel_in_their_own_state_channel() -> None:
    """Was: no state channel a proposal could travel in."""
    channels = set(get_type_hints(AgentState, include_extras=False))

    assert "draft" in channels, "precondition: the schema was read"
    assert "proposed_actions" in channels
    assert "approved_digest" in channels


def test_execute_performs_an_approved_proposal_handed_to_it(tmp_path: Path) -> None:
    """Was: execute ignored a proposal even when one was handed to it."""
    settings = settings_for(tmp_path)
    proposal = Proposal.model_validate(REFUND)
    state = initial_state("x", "item-24")
    state["decision"] = Decision.APPROVED.value
    state["proposed_actions"] = [proposal.as_channel()]
    state["approved_digest"] = digest_of([proposal])

    update = execute(state, settings, run_config(settings, "item-24"))

    effects = update["audit_trail"][0]["detail"]["irreversible_effects"]
    assert [e["tool"] for e in effects] == ["issue_refund"]
    assert [e["key"] for e in outbox(settings)] == [effects[0]["key"]]


def test_the_executor_allowlist_is_held_by_the_executor() -> None:
    """Was: declared and held by no agent. Now exactly two files use it: the screen, which reads
    it to decide what may be *proposed*, and `execute`, which checks it before running anything.
    The drafter -- the thing that proposes -- is not one of them."""
    assert ALLOWLISTS[Agent.EXECUTOR] == IRREVERSIBLE, "precondition: the executor's tools"

    holders = sorted(
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*.py")
        if path.name != "registry.py" and "Agent.EXECUTOR" in path.read_text(encoding="utf-8")
    )
    assert holders == ["effects/proposals.py", "graph/nodes/execute.py"]
    assert "graph/nodes/drafter.py" not in holders
