"""The decider in front of the approval gate (B5/B6), asserted on the wire and in the outbox.

The decider is asked once, by the assess node, before the gate; the gate reads what it stored. It
is sent structured facts -- routed lane, finding count, denied tools, provenance result, proposed
actions -- and never the draft. It can approve in a human's place only in enforce mode, on a
cloud-routed request, once the deterministic preconditions have been checked in code, and when every
threshold holds. Every other outcome, including every failure, is a human.

Graph-level cases run a cloud-only deployment against the OpenAI-compatible stub, with the Jev stub
or a scripted fake decider. The individual conditions are exercised on ``auto_approval`` directly,
one case per condition, because a graph run per condition would test the plumbing six times over.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tests.doubles.openai_compatible import StubBehaviour, StubServer, running_stub
from tests.doubles.typesafe_systemone import JevBehaviour, JevStub, running_jev_stub

from agentgate.audit.events import Decided
from agentgate.config import Settings
from agentgate.decider.assessment import (
    AUTO_APPROVE,
    HUMAN_REVIEW,
    IRREVERSIBILITY,
    ROUTE,
    Assessment,
)
from agentgate.decider.fake import FakeDecider
from agentgate.effects.proposals import digest_of
from agentgate.graph.build import build_graph, resume_config, run_config
from agentgate.graph.nodes.approval import auto_approval
from agentgate.graph.nodes.assess import assess
from agentgate.graph.state import Proposal, initial_state
from agentgate.guardrails.run_ledger import ledger_of

pytestmark = pytest.mark.usefixtures("isolated_env")

CAPABLE = "cloud-capable-stub"
CHEAP = "cloud-cheap-stub"
SOVEREIGN = "sovereign-stub"
JEV = "jev-9.9.9"
CORPUS = Path(__file__).resolve().parents[2] / "corpus"
DRAFT_CANARY = "DRAFT-CANARY-7731 Jane Doe account 4929-1123-8876"
REFUND = {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 240.0}}

PUBLIC_REPLY = {
    # Read by the classifier as its verdict, and by the drafter as its final message.
    "sensitivity": "public",
    "complexity": "simple",
    "contains_pii": False,
    "reason": "t",
    "draft": f"Refunding the overcharge. {DRAFT_CANARY}",
    "proposed_actions": [REFUND],
}
RESTRICTED_VERDICT = {
    "sensitivity": "restricted",
    "complexity": "simple",
    "contains_pii": True,
    "reason": "t",
}

CONFIDENT_JEV = {
    "choice_probabilities": {ROUTE: {AUTO_APPROVE: 0.95, HUMAN_REVIEW: 0.05}},
    "choice_confidence": {ROUTE: 0.9},
    "noul": {IRREVERSIBILITY: 0.05},
}
ENFORCED: dict[str, object] = {
    "decider_mode": "enforce",
    "auto_approve_min_probability": 0.9,
    "auto_approve_min_confidence": 0.8,
    "auto_approve_max_irreversibility": 0.1,
}


def confident(**overrides: Any) -> Assessment:
    fields: dict[str, Any] = {
        "backend": "fake",
        "model": "fake-decider",
        "route": AUTO_APPROVE,
        "route_probabilities": {AUTO_APPROVE: 0.95, HUMAN_REVIEW: 0.05},
        "route_confidence": 0.9,
        "irreversibility": 0.05,
    }
    fields.update(overrides)
    return Assessment(**fields)


@pytest.fixture
def cloud() -> Iterator[StubServer]:
    behaviour = StubBehaviour(reply=PUBLIC_REPLY, supports_native_structured_output=True)
    with running_stub(behaviour) as server:
        yield server


@pytest.fixture
def jev() -> Iterator[JevStub]:
    with running_jev_stub(JevBehaviour(**CONFIDENT_JEV)) as server:  # type: ignore[arg-type]
        yield server


def settings_for(tmp_path: Path, cloud: StubServer, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "lane": "cloud",
        "openai_api_key": "not-required",
        "openai_base_url": cloud.base_url,
        "cloud_capable_model": CAPABLE,
        "cloud_cheap_model": CHEAP,
        "corpus_path": CORPUS,
        "outbox_path": tmp_path / "outbox.jsonl",
        "decider_backend": "fake",
        "model_prices_usd_per_million": {
            CAPABLE: {"input": 1.0, "output": 4.0},
            CHEAP: {"input": 0.1, "output": 0.4},
            SOVEREIGN: {"input": 0.0, "output": 0.0},
            JEV: {"input": 0.042, "output": 0.0},
        },
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


def jev_settings(tmp_path: Path, cloud: StubServer, jev: JevStub, **overrides: object) -> Settings:
    return settings_for(
        tmp_path,
        cloud,
        decider_backend="jev",
        jev_model=JEV,
        jev_base_url=jev.base_url,
        request_timeout_seconds=2.0,
        AGENTGATE_JEV_API_KEY="not-required",  # ADR 0004 item 21
        **overrides,
    )


class Run:
    def __init__(self, settings: Settings, decider: FakeDecider | None = None) -> None:
        self.settings = settings
        options: dict[str, Any] = {}
        if decider is not None:
            options["decider_factory"] = lambda _s, _l: decider
        self.graph = build_graph(settings, InMemorySaver(), **options)
        self.thread = str(uuid.uuid4())
        self.config = run_config(settings, self.thread)

    def start(self) -> dict[str, Any]:
        state = initial_state("Refund Jane the overcharge.", self.thread)
        state["sub_questions"] = ["refund escalation"]
        return dict(self.graph.invoke(state, self.config))

    def paused(self) -> bool:
        return bool(self.graph.get_state(self.config).interrupts)

    def approve(self) -> dict[str, Any]:
        shown = dict(self.graph.get_state(self.config).interrupts[0].value)["proposals_digest"]
        config = resume_config(self.graph, self.settings, self.thread)
        return dict(
            self.graph.invoke(
                Command(resume={"decision": "approved", "approved_digest": shown}), config
            )
        )


def events(result: dict[str, Any], kind: Decided) -> list[dict[str, Any]]:
    return [e for e in result.get("audit_trail", []) if e["decided"] == kind.value]


def outbox(settings: Settings) -> list[dict[str, Any]]:
    if not settings.outbox_path.exists():
        return []
    return [json.loads(line) for line in settings.outbox_path.read_text("utf-8").splitlines()]


# ------------------------------------------------------------ what the decider is shown


def test_the_decider_is_sent_structured_facts_and_never_the_draft(
    tmp_path: Path, cloud: StubServer, jev: JevStub
) -> None:
    run = Run(jev_settings(tmp_path, cloud, jev))

    run.start()

    assert jev.behaviour.request_count == 1, "precondition: the decider was asked, once"
    state = jev.behaviour.requests_seen[0]["state"]
    assert set(state) == {
        "routed_lane",
        "finding_count",
        "denied_tools",
        "provenance_check_passed",
        "proposed_actions",
    }
    assert state["proposed_actions"] == [REFUND], "the actions it judges are the real proposals"
    assert state["routed_lane"] == "cloud"
    body = json.dumps(jev.behaviour.requests_seen[0])
    assert "DRAFT-CANARY-7731" not in body, "the draft text reached the decider"
    assert "4929-1123-8876" not in body
    draft = dict(run.graph.get_state(run.config).values)["draft"]
    assert "DRAFT-CANARY-7731" in draft, "precondition: the canary really was in the draft"


def test_a_restricted_request_never_reaches_the_decider(tmp_path: Path, jev: JevStub) -> None:
    """Hybrid deployment, restricted request: routed to the sovereign lane, so the decider -- a
    cloud egress -- is not asked, and the run goes to a human."""
    with (
        running_stub(StubBehaviour(reply=RESTRICTED_VERDICT)) as cloud,
        running_stub(StubBehaviour(reply=RESTRICTED_VERDICT)) as sovereign,
    ):
        settings = jev_settings(
            tmp_path, cloud, jev, sovereign_base_url=sovereign.base_url, sovereign_model=SOVEREIGN
        )
        run = Run(settings)

        run.start()

        assert sovereign.behaviour.requests_seen, "precondition: the sovereign lane did the work"
        assert jev.behaviour.request_count == 0, "the decider was asked about a restricted request"
        assert run.paused(), "and a human decides"
        assessed = events(dict(run.graph.get_state(run.config).values), Decided.ASSESSED)
        assert assessed[-1]["detail"]["called"] is False
        assert "sovereign" in assessed[-1]["detail"]["reason"]


# ------------------------------------------------------------------ shadow and enforce


def test_shadow_mode_never_skips_the_human_even_for_a_confident_approval(
    tmp_path: Path, cloud: StubServer
) -> None:
    run = Run(settings_for(tmp_path, cloud), FakeDecider(confident()))

    run.start()

    assert run.paused(), "shadow mode acted on the verdict"
    assert outbox(run.settings) == []


def test_in_shadow_mode_the_trail_holds_the_verdict_beside_the_humans_decision(
    tmp_path: Path, cloud: StubServer
) -> None:
    """What makes agreement measurable later: one event, both views."""
    run = Run(settings_for(tmp_path, cloud), FakeDecider(confident()))
    run.start()

    result = run.approve()

    approved = events(result, Decided.APPROVED)[-1]["detail"]
    assert approved["approved_by"] == "human"
    assert approved["assessment"]["route"] == AUTO_APPROVE
    assert approved["assessment"]["route_confidence"] == 0.9
    assert "shadow" in approved["decider_declined_because"]


def test_enforce_mode_approves_in_the_humans_place_when_every_condition_holds(
    tmp_path: Path, cloud: StubServer
) -> None:
    run = Run(settings_for(tmp_path, cloud, **ENFORCED), FakeDecider(confident()))

    result = run.start()

    assert not run.paused(), "every condition held and a human was still asked"
    approved = events(result, Decided.APPROVED)[-1]["detail"]
    assert approved["approved_by"] == "decider"
    assert [e["tool"] for e in outbox(run.settings)] == ["issue_refund"]


@pytest.mark.parametrize(
    ("status", "timeout"),
    [([500, 500], False), ([422], False), ([], True)],
)
def test_every_decider_failure_resolves_to_a_human(
    tmp_path: Path, cloud: StubServer, jev: JevStub, status: list[int], timeout: bool
) -> None:
    jev.behaviour.fail_with = list(status)
    if timeout:
        jev.behaviour.delay_seconds = 3.0
    run = Run(jev_settings(tmp_path, cloud, jev, **ENFORCED))

    run.start()

    assert jev.behaviour.request_count >= 1, "precondition: the decider was asked"
    assert run.paused(), "a failed assessment auto-approved"
    assert outbox(run.settings) == []


@pytest.mark.parametrize(
    "broken", ["malformed", ("route", "confidence"), ("irreversibility", "noul")]
)
def test_an_answer_the_decider_cannot_read_resolves_to_a_human(
    tmp_path: Path, cloud: StubServer, jev: JevStub, broken: object
) -> None:
    if broken == "malformed":
        jev.behaviour.malformed = True
    else:
        jev.behaviour.omit_answer_field = broken  # type: ignore[assignment]
    run = Run(jev_settings(tmp_path, cloud, jev, **ENFORCED))

    run.start()

    assert jev.behaviour.request_count == 1
    assert run.paused()


def test_resuming_after_the_pause_does_not_ask_the_decider_again(
    tmp_path: Path, cloud: StubServer, jev: JevStub
) -> None:
    """The gate re-executes from its top on resume; the decider is not in the gate."""
    run = Run(jev_settings(tmp_path, cloud, jev))
    run.start()
    assert jev.behaviour.request_count == 1

    run.approve()

    assert jev.behaviour.request_count == 1


def test_the_deciders_spend_is_in_the_run_ledger_against_its_model(
    tmp_path: Path, cloud: StubServer, jev: JevStub
) -> None:
    run = Run(jev_settings(tmp_path, cloud, jev))

    run.start()

    assert JEV in ledger_of(run.config).usage_by_model


# ------------------------------------------------ the conditions, one at a time


def gate_state(**overrides: Any) -> dict[str, Any]:
    """A state as the gate sees it after a clean draft and a confident, current assessment."""
    proposal = Proposal.model_validate(REFUND)
    record = {
        "called": True,
        "mode": "enforce",
        "lane": "cloud",
        "proposals_digest": digest_of([proposal]),
        **confident().as_channel(),
    }
    state: dict[str, Any] = {
        **initial_state("x", "run-x"),
        "lane": "cloud_capable",
        "proposed_actions": [proposal.as_channel()],
        "assessment": record,
        "audit_trail": [
            {"decided": "drafted", "detail": {"tools_denied": [], "citations_clean": True}}
        ],
    }
    for key, value in overrides.items():
        if key.startswith("record_"):
            state["assessment"] = {**state["assessment"], key[len("record_") :]: value}
        else:
            state[key] = value
    return state


def enforced() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane="cloud",
        openai_api_key="not-required",
        cloud_capable_model="m",
        cloud_cheap_model="m",
        decider_backend="fake",
        model_prices_usd_per_million={"m": {"input": 1.0, "output": 1.0}},
        **ENFORCED,
    )


def test_the_control_every_condition_holding_approves() -> None:
    assert auto_approval(gate_state(), enforced()) == (True, "every condition held")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"record_route": HUMAN_REVIEW}, "route is"),
        (
            {"record_route_probabilities": {AUTO_APPROVE: 0.85, HUMAN_REVIEW: 0.15}},
            "route probability",
        ),
        ({"record_route_confidence": 0.5}, "route confidence"),
        ({"record_route_confidence": None}, "route confidence"),
        ({"record_irreversibility": 0.3}, "irreversibility"),
        ({"record_irreversibility": None}, "irreversibility"),
        ({"record_failure": "HTTP 529"}, "decider failed"),
        ({"record_lane": "sovereign"}, "cloud-routed"),
        ({"record_called": False, "record_reason": "no decider is configured"}, "no verdict"),
        ({"assessment": None}, "no verdict"),
        ({"record_proposals_digest": "stale"}, "changed after they were assessed"),
    ],
)
def test_enforce_mode_asks_a_human_when_any_one_condition_fails(
    overrides: dict[str, Any], reason: str
) -> None:
    approved, why = auto_approval(gate_state(**overrides), enforced())  # type: ignore[arg-type]

    assert approved is False
    assert reason in why


@pytest.mark.parametrize(
    "drafted",
    [
        {"tools_denied": [], "citations_clean": False},
        {"tools_denied": ["issue_refund"], "citations_clean": True},
    ],
)
def test_a_failed_deterministic_precondition_is_checked_before_the_verdict(
    drafted: dict[str, Any],
) -> None:
    """Even with a confident, current, passing verdict: the fact decides, in code, first -- and
    the reason given is the precondition, not anything about the verdict."""
    state = gate_state(audit_trail=[{"decided": "drafted", "detail": drafted}])

    approved, why = auto_approval(state, enforced())  # type: ignore[arg-type]

    assert approved is False
    assert why.startswith("deterministic precondition failed")


def test_with_no_drafting_record_the_preconditions_fail_closed() -> None:
    approved, why = auto_approval(gate_state(audit_trail=[]), enforced())  # type: ignore[arg-type]

    assert approved is False
    assert "deterministic precondition" in why


def test_shadow_mode_declines_after_the_preconditions() -> None:
    shadow = enforced().model_copy(update={"decider_mode": "shadow"})

    approved, why = auto_approval(gate_state(), shadow)  # type: ignore[arg-type]

    assert approved is False
    assert "shadow" in why


def test_a_skipped_assessment_overwrites_a_stale_verdict(tmp_path: Path) -> None:
    """A verdict about an earlier draft must never be read as one about this draft. When the
    decider is not asked -- here, no backend -- the stored assessment is replaced, not kept."""
    settings = Settings(_env_file=None, outbox_path=tmp_path / "o.jsonl")  # type: ignore[call-arg]
    stale = gate_state()
    assert stale["assessment"]["route"] == AUTO_APPROVE, "precondition: a confident old verdict"

    update = assess(stale, settings, run_config(settings, "run-x"))  # type: ignore[arg-type]

    assert update["assessment"]["called"] is False
    assert "route" not in update["assessment"]
