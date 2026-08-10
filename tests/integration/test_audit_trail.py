"""The audit trail: readable by a stranger, and complete by discovery rather than by list.

Two properties, and they are the two the thesis actually rests on.

**Readable with no imports from agentgate.** The claim is that every gate decision is auditable,
and a record only the producing system can decode does not support that — it supports the weaker
claim that the system remembers what it did. So the reading test imports `json` and `pathlib`
and nothing else, deliberately, and asserts on field names and value shapes a stranger would
have to rely on.

**Every gate wrote an event, enumerated from the code.** A hand-maintained list of gates is a
list someone forgets to update, and the failure mode is a gate that decides silently — the one
thing this system claims cannot happen. So the gates are discovered, and the discovery is
mutation-checked rather than trusted.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentgate.audit.writer import REQUIRED_FIELDS
from agentgate.config import CallClass, Settings
from agentgate.graph.build import build_graph
from agentgate.graph.nodes.lanes import LANE_NODES
from agentgate.graph.state import initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpus"
NODE_SOURCES = sorted((REPO / "src" / "agentgate" / "graph").rglob("*.py"))
DECLARES_NODE = re.compile(r'^NODE\s*=\s*"([^"]+)"', re.M)

# Nodes that exist in the topology and decide nothing. `budget_guard` is a pass-through whose
# only job is to be the place the budget edge hangs off; the decision lives in
# `route_by_budget`, which is a conditional edge and cannot write state. A node that records
# nothing because it decides nothing is not a silent gate.
DECIDES_NOTHING = frozenset({"budget_guard"})

MINIMUM_GATES = 10


def settings_with(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "lane": "fake",
        "corpus_path": CORPUS,
        "audit_log_path": tmp_path / "audit.jsonl",
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def verdict(sensitivity: str, complexity: str = "involved") -> str:
    return scripted_json(
        {
            "sensitivity": sensitivity,
            "complexity": complexity,
            "contains_pii": sensitivity == "restricted",
            "reason": "a reason",
        }
    )


def factory_for(classification: str) -> Any:
    def factory(_s: Settings, _t: object, call_class: CallClass, **_k: object) -> FakeChatModel:
        if call_class is CallClass.SYNTHESIS:
            return FakeChatModel(responses=["A draft citing nothing in particular."])
        return FakeChatModel(responses=[classification])

    return factory


def complete_run(
    settings: Settings, sensitivity: str, complexity: str = "involved"
) -> dict[str, Any]:
    """Drive one run from START to finalise, approving at the gate, and return final state."""
    graph = build_graph(
        settings, InMemorySaver(), model_factory=factory_for(verdict(sensitivity, complexity))
    )
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }
    state = initial_state(f"A {sensitivity} request about refunds.", str(uuid.uuid4()))
    state["sub_questions"] = ["refund escalation"]

    result = graph.invoke(state, config)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume={"decision": "approved"}), config)
    return dict(result)


def every_lane_exercised(settings: Settings) -> None:
    """Three runs, one per policy destination.

    One run cannot cover the lane nodes: the policy gate picks exactly one, which is the point
    of the policy gate. Coverage of a branching graph needs a run per branch.
    """
    complete_run(settings, "public", "simple")
    complete_run(settings, "internal", "involved")
    complete_run(settings, "restricted", "involved")


def discovered_gates() -> set[str]:
    """The gates, enumerated from the code rather than from a list kept by hand.

    Two sources, because the codebase names nodes in two ways:

    1. A module-level ``NODE = "name"`` constant. Every node module declares one.
    2. The keys of ``LANE_NODES``, which are generated -- the lane nodes take their name from
       the dict rather than from a constant.

    **This is the limitation, stated rather than papered over.** The scan finds nodes that
    declare their name as a literal at module level. A node that computed its name, or built it
    from a variable, or declared it inside a function, would be invisible here — and a gate
    invisible to the enumeration is a gate this test cannot notice going silent. The lane nodes
    are exactly that case, which is why they are enumerated separately and explicitly; a future
    dynamically-named node would need the same treatment, and nothing automatic will remind
    anyone. The honest summary: this catches a gate added the ordinary way, which is how gates
    are added, and it does not catch one added unusually.
    """
    declared = {
        name for source in NODE_SOURCES for name in DECLARES_NODE.findall(source.read_text("utf-8"))
    }
    return (declared | set(LANE_NODES)) - DECIDES_NOTHING


def read_trail_with_stdlib_only(path: Path) -> list[dict[str, Any]]:
    """Read the trail the way a stranger would: json and pathlib, nothing else."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------------ 1. readable by a stranger


def test_the_trail_reads_with_the_standard_library_alone(tmp_path: Path) -> None:
    """No imports from agentgate anywhere in what this asserts on.

    Field names, value shapes, and the one-object-per-line structure are all a reader gets, so
    they are all this checks.
    """
    settings = settings_with(tmp_path)
    complete_run(settings, "internal")

    events = read_trail_with_stdlib_only(settings.audit_log_path)

    assert events, "the run wrote no trail"
    for event in events:
        assert isinstance(event, dict)
        assert {"at", "node", "decided", "correlation_id", "input_digest"} <= set(event)
        assert isinstance(event["decided"], str) and event["decided"]
        assert isinstance(event["node"], str) and event["node"]
        assert isinstance(event["detail"], dict)


def test_every_timestamp_carries_a_zone(tmp_path: Path) -> None:
    """A trail spanning hosts in different zones, compared by naive timestamps, is worse than
    no timestamps: it looks orderable and is not."""
    from datetime import datetime  # noqa: PLC0415 - stdlib only, and the point is it is enough

    settings = settings_with(tmp_path)
    complete_run(settings, "internal")

    for event in read_trail_with_stdlib_only(settings.audit_log_path):
        parsed = datetime.fromisoformat(event["at"])
        assert parsed.tzinfo is not None, f"{event['node']} wrote a naive timestamp"


def test_the_request_appears_only_as_a_hash(tmp_path: Path) -> None:
    """The trail has to be readable by people not cleared to read what it describes. A log that
    copies the regulated data it audits is one nobody can hand to a reviewer."""
    settings = settings_with(tmp_path)
    restricted_text = "account 4471 belonging to a named individual"
    graph = build_graph(settings, InMemorySaver(), model_factory=factory_for(verdict("restricted")))
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }
    state = initial_state(restricted_text, str(uuid.uuid4()))
    state["sub_questions"] = ["refund escalation"]
    graph.invoke(state, config)
    graph.invoke(Command(resume={"decision": "approved"}), config)

    raw = settings.audit_log_path.read_text(encoding="utf-8")

    assert restricted_text not in raw
    assert "4471" not in raw
    assert all(
        len(event["input_digest"]) == 16
        for event in read_trail_with_stdlib_only(settings.audit_log_path)
    )


def test_the_trail_is_append_only_across_runs(tmp_path: Path) -> None:
    """A second run adds lines and rewrites none. Append-only is the property being claimed,
    and a writer that could rewrite would make it a convention."""
    settings = settings_with(tmp_path)

    complete_run(settings, "internal")
    first = settings.audit_log_path.read_text(encoding="utf-8")
    complete_run(settings, "public", "simple")
    second = settings.audit_log_path.read_text(encoding="utf-8")

    assert second.startswith(first), "the earlier run's lines changed"
    assert len(second) > len(first)


def test_one_object_per_line_survives_a_truncated_file(tmp_path: Path) -> None:
    """The reason for JSON lines rather than a JSON array. A file cut short loses its last
    record; an array loses all of them."""
    settings = settings_with(tmp_path)
    complete_run(settings, "internal")

    whole = settings.audit_log_path.read_text(encoding="utf-8").splitlines()
    truncated = "\n".join(whole[:-1]) + "\n" + whole[-1][: len(whole[-1]) // 2]
    (tmp_path / "cut.jsonl").write_text(truncated, encoding="utf-8")

    readable = []
    for line in (tmp_path / "cut.jsonl").read_text(encoding="utf-8").splitlines():
        # A stranger reading this file skips what it cannot parse and keeps the rest, which is
        # the property being demonstrated -- so the suppression is the assertion.
        with contextlib.suppress(json.JSONDecodeError):
            readable.append(json.loads(line))

    assert len(readable) == len(whole) - 1


# ------------------------------------------------------------------ 2. coverage by discovery


def test_the_gate_enumeration_is_not_empty(tmp_path: Path) -> None:
    """Guards the test below, which asserts over the discovered set.

    A discovery test asserting over an empty set is the failure mode it exists to prevent, and
    this build has produced that shape more than once. If the scan stopped matching, the
    coverage check would pass by looking at nothing.
    """
    gates = discovered_gates()

    assert len(gates) >= MINIMUM_GATES, f"only discovered {sorted(gates)}; the scan drifted"
    assert "approval_gate" in gates, "the human gate is not in the enumeration"
    assert set(LANE_NODES) <= gates, "the dynamically-named lane nodes were not picked up"
    assert "budget_guard" not in gates, "a node that decides nothing is not a gate"


def test_every_discovered_gate_wrote_an_event(tmp_path: Path) -> None:
    """The property: no gate decides silently.

    Enumerated from the code, so a gate added later without an audit event fails here rather
    than being noticed by whoever eventually reads the trail and wonders what happened.
    """
    settings = settings_with(tmp_path)
    every_lane_exercised(settings)

    recorded = {event["node"] for event in read_trail_with_stdlib_only(settings.audit_log_path)}
    silent = sorted(discovered_gates() - recorded)

    assert not silent, (
        f"these gates decided without writing to the audit trail: {silent}. Either they need an "
        "audit event, or -- if they genuinely decide nothing -- they belong in DECIDES_NOTHING "
        "with a comment saying why."
    )


def test_every_recorded_event_carries_the_fields_the_writer_requires(tmp_path: Path) -> None:
    """The writer refuses an unattributable event. This confirms the real events satisfy it,
    rather than the refusal only being exercised by a constructed failure."""
    settings = settings_with(tmp_path)
    every_lane_exercised(settings)

    for event in read_trail_with_stdlib_only(settings.audit_log_path):
        assert all(event.get(field) for field in REQUIRED_FIELDS), event


def test_the_file_and_the_state_channel_hold_the_same_events(tmp_path: Path) -> None:
    """Two records of the same run must not drift apart.

    `finalise` is the one node that writes the file directly as well as returning its event to
    state, which makes it the one place the two can diverge -- and the mutation check found
    exactly that: dropping finalise's event from its return left the file complete and the
    state channel short, and every other test still passed. A caller reading state would have
    seen a run with no ending.
    """
    settings = settings_with(tmp_path)

    final = complete_run(settings, "internal")

    on_disk = read_trail_with_stdlib_only(settings.audit_log_path)
    in_state = final["audit_trail"]

    assert [event["decided"] for event in on_disk] == [event["decided"] for event in in_state]
    assert [event["node"] for event in on_disk] == [event["node"] for event in in_state]
