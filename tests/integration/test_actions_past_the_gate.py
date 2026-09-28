"""Actions past the gate: proposed by the drafter, approved as shown, performed exactly once.

The drafter's final message is a JSON object -- the draft and the proposed actions -- parsed by
this code, strictly, on the same lane-aware path classification uses. It can propose; it cannot
perform. The executor holds the ``EXECUTOR`` allowlist and runs the approved proposals through the
effect sink, whose only implementation is an append-only outbox: nothing real can happen, and
configuration refuses to be told otherwise.

Every assertion about an effect reads the outbox on disk -- the side effect itself -- and never a
field the graph wrote about itself.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from agentgate.config import CallClass, Settings
from agentgate.effects.proposals import digest_of
from agentgate.effects.sink import OutboxSink
from agentgate.graph.build import build_checkpointer, build_graph, resume_config, run_config
from agentgate.graph.nodes.execute import UnapprovedExecutionError, execute
from agentgate.graph.state import Proposal, initial_state
from agentgate.guardrails.run_ledger import LedgerMissingError
from agentgate.guardrails.spend import SpendCeilingExceededError
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

VERDICT = scripted_json(
    {"sensitivity": "internal", "complexity": "simple", "contains_pii": False, "reason": "t"}
)
REFUND = {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 240.0}}
EMAIL = {
    "tool": "send_customer_email",
    "arguments": {"to": "jane@example.test", "subject": "Your refund", "body": "It is done."},
}


def drafted(*proposals: object, draft: str = "We will refund the overcharge.") -> str:
    return scripted_json({"draft": draft, "proposed_actions": list(proposals)})


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "lane": "fake",
        "corpus_path": CORPUS,
        "outbox_path": tmp_path / "outbox.jsonl",
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


def factory_drafting(reply: str) -> Any:
    def factory(_s: Settings, _t: object, call_class: CallClass, **_k: object) -> Any:
        if call_class is CallClass.SYNTHESIS:
            return FakeChatModel(responses=[reply])
        return FakeChatModel(responses=[VERDICT])

    return factory


def outbox(settings: Settings) -> list[dict[str, Any]]:
    path = settings.outbox_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Run:
    def __init__(self, settings: Settings, reply: str, **graph_options: Any) -> None:
        self.settings = settings
        self.factory = factory_drafting(reply)
        self.checkpointer = build_checkpointer(settings)
        self.graph = build_graph(
            settings, self.checkpointer, model_factory=self.factory, **graph_options
        )
        self.thread = str(uuid.uuid4())
        self.config = run_config(settings, self.thread)

    def start(self) -> dict[str, Any]:
        state = initial_state("Refund Jane the 240 she was overcharged.", self.thread)
        state["sub_questions"] = ["refund escalation"]
        return dict(self.graph.invoke(state, self.config))

    def packet(self) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.config)
        assert snapshot.interrupts, "precondition: paused at the approval gate"
        return dict(snapshot.interrupts[0].value)

    def approve_as_shown(self) -> dict[str, Any]:
        """What a human does: approve the proposals they were shown, by the hash of the packet."""
        shown = self.packet()["proposals_digest"]
        return self.resume({"decision": "approved", "approved_digest": shown})

    def resume(self, verdict: dict[str, Any]) -> dict[str, Any]:
        config = resume_config(self.graph, self.settings, self.thread)
        return dict(self.graph.invoke(Command(resume=verdict), config))


def events(result: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in result.get("audit_trail", []) if event["decided"] == kind]


# ---------------------------------------------------------------- proposed, shown, performed


def test_an_approved_proposal_is_performed_once_and_lands_in_the_outbox(tmp_path: Path) -> None:
    """Item 24, inverted end to end. The drafter proposes a refund; the human is shown it and
    approves it; the executor performs it -- and the effect is in the outbox, not just the trail."""
    settings = settings_for(tmp_path)
    run = Run(settings, drafted(REFUND))

    run.start()
    shown = run.packet()
    assert shown["proposed_actions"] == [REFUND], "the human is shown the proposal itself"
    assert outbox(settings) == [], "nothing is performed before the gate"

    result = run.approve_as_shown()

    effects = outbox(settings)
    assert [(e["tool"], e["arguments"]) for e in effects] == [(REFUND["tool"], REFUND["arguments"])]
    executed = events(result, "executed")[0]["detail"]["irreversible_effects"]
    assert [e["key"] for e in executed] == [effects[0]["key"]]
    assert events(result, "approved")[0]["detail"]["proposals_digest"] == shown["proposals_digest"]


def test_a_drafter_reply_that_is_not_the_json_asked_for_proposes_nothing(tmp_path: Path) -> None:
    """Fails closed: no proposals, the drop recorded, and the reply still shown as the draft."""
    settings = settings_for(tmp_path)
    run = Run(settings, "Here is the refund letter, and please do refund her.")

    run.start()
    packet = run.packet()

    assert packet["proposed_actions"] == []
    assert packet["draft"] == "Here is the refund letter, and please do refund her."
    dropped = events(dict(run.graph.get_state(run.config).values), "proposals_dropped")
    assert dropped, "the parse failure is recorded, not silent"
    assert "JSON" in dropped[0]["detail"]["reason"]


def test_on_a_native_lane_a_reply_wrapped_in_prose_proposes_nothing(tmp_path: Path) -> None:
    """Strict where the lane is recorded as native -- the fake is -- as classification is: JSON
    and nothing else. Prose around it is the repair path's business, not something to accept
    silently on a lane that claims not to need it."""
    settings = settings_for(tmp_path)
    wrapped = scripted_json(
        {"draft": "We will refund it.", "proposed_actions": [REFUND]}, wrapped_in_prose=True
    )
    run = Run(settings, wrapped)

    run.start()

    assert run.packet()["proposed_actions"] == []
    assert events(dict(run.graph.get_state(run.config).values), "proposals_dropped")


def test_an_invalid_proposal_is_dropped_before_the_gate_and_audited(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    bad = {"tool": "lookup_policy", "arguments": {"topic": "refund"}}
    run = Run(settings, drafted(REFUND, bad))

    run.start()

    assert run.packet()["proposed_actions"] == [REFUND], "the human never sees the invalid one"
    dropped = events(dict(run.graph.get_state(run.config).values), "proposals_dropped")
    assert dropped[0]["detail"]["dropped"][0]["proposal"] == bad


def test_a_rejection_clears_the_proposals_with_the_draft(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    run = Run(settings, drafted(REFUND))
    run.start()

    run.resume({"decision": "rejected", "feedback": "not yet"})

    assert outbox(settings) == []
    state = dict(run.graph.get_state(run.config).values)
    assert "decision" in state, "precondition: the rejection was applied"
    rejected = [e for e in state["audit_trail"] if e["decided"] == "rejected"]
    assert rejected, "precondition: a rejection was recorded"
    # The revision re-drafts and proposes again; what matters is that the rejected proposal was
    # cleared at the moment of rejection, which the checkpoint history shows.
    history = [dict(snap.values) for snap in run.graph.get_state_history(run.config)]
    cleared = [h for h in history if h.get("decision") == "rejected" and h.get("draft") == ""]
    assert cleared and all(h.get("proposed_actions") == [] for h in cleared)


# --------------------------------------------------------------- approving what was shown


def test_approving_without_the_hash_of_what_was_shown_performs_nothing(tmp_path: Path) -> None:
    """An approval has to say which proposals it approves. One that does not is refused."""
    settings = settings_for(tmp_path)
    run = Run(settings, drafted(REFUND))
    run.start()

    result = run.resume({"decision": "approved"})

    assert outbox(settings) == []
    assert events(result, "approval_refused"), "refused, and recorded as a refusal"


def test_a_proposal_changed_between_the_pause_and_the_resume_is_refused(tmp_path: Path) -> None:
    """The human approves the packet they were shown. If the proposals in state are changed
    after the pause -- here, the refund's amount -- the approval does not transfer to them."""
    settings = settings_for(tmp_path)
    run = Run(settings, drafted(REFUND))
    run.start()
    shown = run.packet()["proposals_digest"]

    # Written as the drafter, so the run goes back through the gate with the changed proposal in
    # state. (`update_state` with no node named ends the run instead, and would test nothing.)
    tampered = {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 24000.0}}
    run.graph.update_state(run.config, {"proposed_actions": [tampered]}, as_node="drafter")
    run.graph.invoke(None, resume_config(run.graph, settings, run.thread))
    assert run.packet()["proposals_digest"] != shown, "precondition: state now differs"

    result = run.resume({"decision": "approved", "approved_digest": shown})

    assert outbox(settings) == [], "neither the shown refund nor the changed one was performed"
    assert events(result, "approval_refused")


def test_the_executor_refuses_proposals_that_do_not_match_the_approval(tmp_path: Path) -> None:
    """The second check, in ``execute`` itself, for a route to it that skipped the gate's."""
    settings = settings_for(tmp_path)
    state = initial_state("x", "run-x")
    state["decision"] = "approved"
    state["proposed_actions"] = [REFUND]
    state["approved_digest"] = "not-the-digest"

    with pytest.raises(UnapprovedExecutionError):
        execute(state, settings, run_config(settings, "run-x"))
    assert outbox(settings) == []


# --------------------------------------------------------------------- exactly once


class CrashAfterWriting(OutboxSink):
    """Writes the effect, then dies -- after the side effect, before the checkpoint."""

    def record(self, key: str, proposal: Proposal) -> dict[str, Any]:
        effect = super().record(key, proposal)
        msg = "simulated crash after the effect was written"
        raise RuntimeError(msg)
        return effect  # pragma: no cover


def test_an_effect_written_before_a_crash_is_not_written_again_on_resume(tmp_path: Path) -> None:
    """The case that charges someone twice. The effect reaches the outbox, the process dies
    before the checkpoint records it, and a new process resumes from that checkpoint -- which
    re-runs ``execute``. The key it computes is the one already in the outbox, so it is skipped."""
    settings = settings_for(tmp_path)
    crashing = Run(
        settings,
        drafted(REFUND, EMAIL),
        effect_sink_factory=lambda s: CrashAfterWriting(s.outbox_path),
    )
    crashing.start()
    shown = crashing.packet()["proposals_digest"]

    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.resume({"decision": "approved", "approved_digest": shown})
    assert len(outbox(settings)) == 1, "precondition: the first effect was written, then it died"

    # As a new process would: a fresh graph and a fresh sink, over the same durable checkpoint.
    # (The checkpoint's durability across real processes is the CLI tests' job.)
    recovered = build_graph(settings, crashing.checkpointer, model_factory=crashing.factory)
    result = dict(recovered.invoke(None, resume_config(recovered, settings, crashing.thread)))

    keys = [effect["key"] for effect in outbox(settings)]
    assert len(keys) == 2, "both effects, once each"
    assert len(set(keys)) == 2, "no key appears twice"
    assert [e["tool"] for e in outbox(settings)] == ["issue_refund", "send_customer_email"]
    assert events(result, "executed")


# ----------------------------------------------------------------------- the run ledger


def test_execute_screens_again_and_refuses_a_tool_that_is_not_the_executors(
    tmp_path: Path,
) -> None:
    """Even approved, even with a hash that matches: `execute` checks its own allowlist, for the
    same reason it checks the decision -- a route to it that skipped the screen is still refused."""
    settings = settings_for(tmp_path)
    smuggled = Proposal(tool="lookup_policy", arguments={"topic": "refund"})
    state = initial_state("x", "run-x")
    state["decision"] = "approved"
    state["proposed_actions"] = [smuggled.as_channel()]
    state["approved_digest"] = digest_of([smuggled])

    with pytest.raises(UnapprovedExecutionError, match="not the executor's"):
        execute(state, settings, run_config(settings, "run-x"))
    assert outbox(settings) == []


def test_execute_refuses_to_run_without_the_runs_ledger(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    state = initial_state("x", "run-x")
    state["decision"] = "approved"

    with pytest.raises(LedgerMissingError):
        execute(state, settings, {"configurable": {"thread_id": "run-x"}})


def test_execute_performs_nothing_on_a_run_already_over_its_ceiling(tmp_path: Path) -> None:
    """Charged to the run ledger like every other call site: a run that is over budget does not
    get to perform its effects on the way out."""
    settings = settings_for(tmp_path, max_total_tokens=10)
    proposal = Proposal.model_validate(REFUND)
    state = initial_state("x", "run-x")
    state["decision"] = "approved"
    state["proposed_actions"] = [proposal.as_channel()]
    state["approved_digest"] = digest_of([proposal])
    config = run_config(
        settings, "run-x", spent={"fake-capable": {"input_tokens": 50, "output_tokens": 0}}
    )

    with pytest.raises(SpendCeilingExceededError):
        execute(state, settings, config)
    assert outbox(settings) == []


# ---------------------------------------------------------------------- old checkpoints


def test_a_checkpoint_from_before_proposals_approves_and_performs_nothing(tmp_path: Path) -> None:
    """ADR 0011. State written before the channel existed has no proposals and no digest; it
    approves the draft, as it always did, and performs nothing."""
    settings = settings_for(tmp_path)
    state = initial_state("x", "run-old")
    state["decision"] = "approved"

    update = execute(state, settings, run_config(settings, "run-old"))

    assert update["audit_trail"][0]["detail"]["irreversible_effects"] == []
    assert outbox(settings) == []
