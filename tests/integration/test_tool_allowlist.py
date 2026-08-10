"""The drafter cannot reach an irreversible tool.

Not *does not*. The distinction is the whole test file, so it is worth being explicit about
what would satisfy each claim:

**Does not** is satisfied by the drafter never being given the tool and the model never asking
for it. That holds right up until something makes the model ask -- a prompt injection in
retrieved content, a hallucinated tool name, a refactor that binds one list and enforces
another. It is a statement about what has happened so far.

**Cannot** requires that the call be refused when it *is* made. So every test here that matters
scripts the model to demand the tool explicitly and then asserts nothing ran.

**No part of this is prompt-enforced, and that claim is itself tested.** The system prompt says
nothing about tools it may not use; ``test_the_instruction_does_not_mention_the_allowlist``
pins that, because a prompt-enforced allowlist is not an allowlist and the failure mode of
believing otherwise is that it works in every test and fails against an adversary.

Enforcement is two layers, and they are separately checked:

1. **Binding.** Only allowlisted tools are given to the model. This is why the ordinary case
   never arises.
2. **Authorisation.** ``AllowlistMiddleware.wrap_tool_call`` sits between the model's request
   and the executor. Every call goes through it, there is no path around it, and the check
   happens *before* the handler is invoked -- so a denied tool does not run and get discarded,
   it never runs.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agentgate.config import CallClass, Settings
from agentgate.graph.nodes.drafter import INSTRUCTION, draft
from agentgate.graph.state import AgentState, Finding
from agentgate.models.fake import FakeChatModel
from agentgate.tools.allowlist import AllowlistMiddleware
from agentgate.tools.registry import (
    ALLOWLISTS,
    IRREVERSIBLE,
    TOOLS,
    Agent,
    is_allowed,
    tools_for,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

TRIPPED: list[str] = []


@tool
def tripwire(reason: str) -> str:
    """A tool that records having been executed. Never on anyone's allowlist."""
    TRIPPED.append(reason)
    return "executed"


@pytest.fixture(autouse=True)
def _reset_tripwire() -> None:
    TRIPPED.clear()


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {"lane": "fake", "corpus_path": CORPUS}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def state_with_findings() -> AgentState:
    return {
        "request": "Draft a response about the refund timeline.",
        "correlation_id": str(uuid.uuid4()),
        "findings": [
            Finding(question="refunds", content="Thirty days.", source="a.md#Window").as_channel()
        ],
        "dispatched": 1,
        "research_outcomes": [],
    }


def demanding(tool_name: str, args: dict[str, Any]) -> AIMessage:
    """A model reply that insists on calling a named tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": "call_demanded", "type": "tool_call"}],
    )


# ------------------------------------------------------------------ the static claim


def test_the_drafter_holds_no_irreversible_tool() -> None:
    """The permission table itself. Necessary, and on its own worth very little."""
    assert ALLOWLISTS[Agent.DRAFTER] & IRREVERSIBLE == frozenset()


def test_every_irreversible_tool_is_held_by_exactly_one_agent() -> None:
    """A tool that changes the world should have one holder, and it should be the one behind
    the approval gate. Two holders means two paths to the same effect, and the second one is
    the one nobody reviews."""
    holders = {
        name: [agent for agent, allowed in ALLOWLISTS.items() if name in allowed]
        for name in IRREVERSIBLE
    }

    assert all(agents == [Agent.EXECUTOR] for agents in holders.values()), holders


def test_every_declared_tool_exists_and_every_allowlisted_name_is_declared() -> None:
    """An allowlist entry naming a tool that does not exist grants nothing and looks like it
    grants something."""
    assert set(TOOLS) >= IRREVERSIBLE
    for agent, allowed in ALLOWLISTS.items():
        assert allowed <= set(TOOLS), f"{agent} is allowed a tool that does not exist"


# ------------------------------------------------------------------ layer 1: binding


def test_only_allowlisted_tools_are_bound_to_the_drafters_model() -> None:
    """The reason the ordinary case never arises."""
    settings = settings_with()
    model = FakeChatModel(responses=["A draft."])

    draft(state_with_findings(), settings, model_factory=lambda *_a, **_k: model)

    assert model.bound_tools == sorted(ALLOWLISTS[Agent.DRAFTER])
    assert not set(model.bound_tools) & IRREVERSIBLE


def test_tools_for_returns_the_allowlist_and_nothing_else() -> None:
    assert {t.name for t in tools_for(Agent.DRAFTER)} == set(ALLOWLISTS[Agent.DRAFTER])


# ------------------------------------------------------------------ layer 2: authorisation


def test_a_demanded_irreversible_tool_does_not_execute() -> None:
    """The claim that matters, made the only way it can be made.

    The model is scripted to call ``issue_refund`` outright. If the allowlist were only the
    binding, this is the moment it would not help: the request exists regardless of what was
    offered. The handler is never reached, which is checkable here because ``issue_refund``
    raises if it ever is.
    """
    settings = settings_with()
    model = FakeChatModel(
        responses=[
            demanding("issue_refund", {"account": "A-1", "amount_units": 500.0}),
            "Fine, here is the draft without it.",
        ]
    )

    result = draft(state_with_findings(), settings, model_factory=lambda *_a, **_k: model)

    denials = [e for e in result["audit_trail"] if e["decided"] == "tool_denied"]
    assert len(denials) == 1
    assert denials[0]["detail"]["tool"] == "issue_refund"
    assert result["draft"] == "Fine, here is the draft without it."


def test_the_denial_happens_before_the_handler_not_after() -> None:
    """Order, checked rather than assumed.

    A guard that runs the tool and then decides whether it should have is not a guard. The
    tripwire records execution; asserting it stayed empty is asserting that authorisation
    preceded the effect.
    """
    guard = AllowlistMiddleware(Agent.DRAFTER, "corr")
    request = type("Request", (), {"tool_call": {"name": "tripwire", "args": {}, "id": "x"}})()

    def handler(_request: object) -> object:
        return tripwire.invoke({"reason": "the handler was reached"})

    message = guard.wrap_tool_call(request, handler)

    assert TRIPPED == [], "the handler ran; the check is happening after the effect"
    assert message.status == "error"
    assert guard.denied == ["tripwire"]


def test_an_unrecognisable_request_shape_is_denied_rather_than_allowed() -> None:
    """The authorisation input is read defensively, and this pins which way it fails.

    A request shape the parser does not understand must not resolve to a name that happens to
    be on the allowlist. It resolves to the empty string, which is on nobody's list.
    """
    guard = AllowlistMiddleware(Agent.DRAFTER, "corr")

    message = guard.wrap_tool_call(object(), lambda _r: tripwire.invoke({"reason": "reached"}))

    assert TRIPPED == []
    assert message.status == "error"


def test_an_allowed_tool_still_runs() -> None:
    """Separating the scopes must not disarm the thing. A guard that denies everything passes
    every test above and is useless."""
    guard = AllowlistMiddleware(Agent.DRAFTER, "corr")
    request = type("Request", (), {"tool_call": {"name": "lookup_policy", "args": {}, "id": "y"}})()

    result = guard.wrap_tool_call(request, lambda _r: "handler ran")

    assert result == "handler ran"
    assert guard.denied == []
    assert guard.events == []


def test_is_allowed_is_the_whole_decision() -> None:
    assert is_allowed(Agent.DRAFTER, "lookup_policy") is True
    assert is_allowed(Agent.DRAFTER, "issue_refund") is False
    assert is_allowed(Agent.DRAFTER, "a_tool_nobody_declared") is False


# ------------------------------------------------------------------ not the prompt


def test_the_instruction_does_not_mention_the_allowlist() -> None:
    """If a prompt were doing any of the work, this file would have to say so.

    It is not. The drafter's system prompt is about how to write, and names no tool and no
    prohibition. Asking a model not to do something is a request; this is the test that keeps
    the request from being mistaken for the control.
    """
    lowered = INSTRUCTION.lower()

    assert not any(name in lowered for name in TOOLS), "the prompt names a tool"
    for word in ("allowlist", "not allowed", "must not use", "forbidden", "permitted"):
        assert word not in lowered, f"the prompt is arguing with the model about '{word}'"


# ------------------------------------------------------------------ failure into state


def test_a_tool_that_raises_is_summarised_back_rather_than_raised() -> None:
    """A tool exception would abort the agent mid-draft. The model should see what went wrong
    and carry on, and the trail should record that it happened."""
    guard = AllowlistMiddleware(Agent.DRAFTER, "corr")
    request = type("Request", (), {"tool_call": {"name": "lookup_policy", "args": {}, "id": "z"}})()

    def explodes(_request: object) -> object:
        msg = "corpus index unavailable"
        raise RuntimeError(msg)

    message = guard.wrap_tool_call(request, explodes)

    assert message.status == "error"
    assert "corpus index unavailable" in message.content
    failures = [e for e in guard.events if e["decided"] == "tool_failed"]
    assert len(failures) == 1
    assert failures[0]["detail"]["tool"] == "lookup_policy"


def test_the_drafter_records_which_tools_it_was_denied() -> None:
    """A denial the run does not carry forward is a decision nobody can review."""
    settings = settings_with()
    model = FakeChatModel(
        responses=[
            demanding("send_customer_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            "Draft without sending anything.",
        ]
    )

    result = draft(state_with_findings(), settings, model_factory=lambda *_a, **_k: model)

    drafted = next(e for e in result["audit_trail"] if e["decided"] == "drafted")
    assert drafted["detail"]["tools_denied"] == ["send_customer_email"]
    assert drafted["detail"]["tools_available"] == sorted(ALLOWLISTS[Agent.DRAFTER])


def test_the_drafter_is_told_when_it_is_drafting_from_partial_research() -> None:
    """A model shown a partial evidence set with no indication that it is partial writes
    around the gaps instead of naming them."""
    settings = settings_with()
    model = FakeChatModel(responses=["A draft."])
    state = state_with_findings()
    state["dispatched"] = 3  # three went out, no outcomes came back

    result = draft(state, settings, model_factory=lambda *_a, **_k: model)

    brief = str(model.calls[0][-1].content)
    assert "did not report" in brief
    drafted = next(e for e in result["audit_trail"] if e["decided"] == "drafted")
    assert drafted["detail"]["drafted_from_partial_research"] is True


def test_the_drafter_runs_on_the_synthesis_call_class() -> None:
    """The one place a capable-tier, synthesis-sized budget is correct. Pinned so a future
    change cannot quietly make every draft a routing-sized reply."""
    seen: list[CallClass] = []

    def factory(_settings: Settings, _tier: object, call_class: CallClass, **_k: object) -> Any:
        seen.append(call_class)
        return FakeChatModel(responses=["A draft."])

    draft(state_with_findings(), settings_with(), model_factory=factory)

    assert seen == [CallClass.SYNTHESIS]
