"""The pieces behind an action past the gate: screening, the approval hash, the key, the sink."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentgate.config import EffectSinkBackend, Settings
from agentgate.effects.proposals import digest_of, effect_key, screen_proposals
from agentgate.effects.sink import OutboxSink, build_effect_sink
from agentgate.graph.state import Proposal, proposals_of

pytestmark = pytest.mark.usefixtures("isolated_env")

REFUND = {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 240.0}}
EMAIL = {
    "tool": "send_customer_email",
    "arguments": {"to": "jane@example.test", "subject": "Refund", "body": "Done."},
}


# ------------------------------------------------------------------------------ screening


def test_valid_proposals_for_executor_tools_are_kept() -> None:
    screened = screen_proposals([REFUND, EMAIL])

    assert [p.tool for p in screened.valid] == ["issue_refund", "send_customer_email"]
    assert screened.dropped == []


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ({"tool": "lookup_policy", "arguments": {"topic": "refund"}}, "executor"),
        ({"tool": "wire_transfer", "arguments": {}}, "executor"),
        ({"tool": "issue_refund", "arguments": {"account": "4929"}}, "arguments"),
        (
            {"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": -5}},
            "arguments",
        ),
        ({"tool": "issue_refund"}, "arguments"),
        ("issue a refund please", "shape"),
    ],
)
def test_an_invalid_proposal_is_dropped_with_its_reason(raw: object, reason: str) -> None:
    """A proposal the executor could not run never reaches the human: a tool outside the
    executor's allowlist, arguments its schema rejects, or something that is not a proposal."""
    screened = screen_proposals([REFUND, raw])

    assert [p.tool for p in screened.valid] == ["issue_refund"], "the valid one survives"
    assert len(screened.dropped) == 1
    assert reason in screened.dropped[0]["reason"]


def test_something_that_is_not_a_list_drops_everything() -> None:
    screened = screen_proposals({"tool": "issue_refund"})

    assert screened.valid == []
    assert len(screened.dropped) == 1


# ------------------------------------------------------------------------ the approval hash


def test_the_digest_changes_when_any_argument_changes() -> None:
    original = screen_proposals([REFUND]).valid
    mutated = screen_proposals(
        [{"tool": "issue_refund", "arguments": {"account": "4929", "amount_units": 2400.0}}]
    ).valid

    assert digest_of(original) != digest_of(mutated)


def test_the_digest_is_order_sensitive_and_stable() -> None:
    forward = screen_proposals([REFUND, EMAIL]).valid
    backward = screen_proposals([EMAIL, REFUND]).valid

    assert digest_of(forward) == digest_of(screen_proposals([REFUND, EMAIL]).valid)
    assert digest_of(forward) != digest_of(backward)


def test_no_proposals_has_a_digest_too() -> None:
    """So an approval of a draft with no actions is still an approval of exactly that."""
    assert digest_of([]) == digest_of([])
    assert digest_of([]) != digest_of(screen_proposals([REFUND]).valid)


# --------------------------------------------------------------------- the idempotency key


def test_the_key_is_the_run_the_position_and_the_arguments() -> None:
    refund = Proposal.model_validate(REFUND)

    assert effect_key("run-1", 0, refund) == effect_key("run-1", 0, refund)
    assert effect_key("run-1", 0, refund) != effect_key("run-2", 0, refund)
    assert effect_key("run-1", 0, refund) != effect_key("run-1", 1, refund)
    changed = Proposal(tool="issue_refund", arguments={"account": "4929", "amount_units": 1.0})
    assert effect_key("run-1", 0, refund) != effect_key("run-1", 0, changed)


# ------------------------------------------------------------------------------- the sink


def test_the_outbox_appends_one_line_per_effect(tmp_path: Path) -> None:
    sink = OutboxSink(tmp_path / "outbox.jsonl")
    refund = Proposal.model_validate(REFUND)

    effect = sink.record("k-1", refund)

    lines = (tmp_path / "outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["k-1"]
    assert effect["tool"] == "issue_refund"
    assert effect["simulated"] is True, "the only sink performs nothing real, and says so"


def test_recording_the_same_key_twice_writes_it_once(tmp_path: Path) -> None:
    """The property the crash-and-resume case depends on. A second writer -- a resumed process
    with a fresh sink -- sees the key the first one wrote and does not write it again."""
    refund = Proposal.model_validate(REFUND)
    OutboxSink(tmp_path / "outbox.jsonl").record("k-1", refund)

    again = OutboxSink(tmp_path / "outbox.jsonl").record("k-1", refund)

    lines = (tmp_path / "outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert again["key"] == "k-1"


def test_configuration_builds_the_outbox_at_its_configured_path(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, outbox_path=tmp_path / "o.jsonl")  # type: ignore[call-arg]

    sink = build_effect_sink(settings)

    assert isinstance(sink, OutboxSink)
    assert sink.path == tmp_path / "o.jsonl"
    assert settings.effect_sink is EffectSinkBackend.OUTBOX


@pytest.mark.parametrize("name", ["stripe", "smtp", "none", ""])
def test_configuration_refuses_any_sink_but_the_outbox(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "This performs nothing real" is enforced at startup rather than documented: no other sink
    exists, so naming one is refused -- the way an unimplemented vector backend is -- rather than
    quietly falling back to the outbox."""
    monkeypatch.setenv("AGENTGATE_EFFECT_SINK", name)

    with pytest.raises(ValidationError, match="no effect sink but 'outbox' exists"):
        Settings(_env_file=None)  # type: ignore[call-arg]


# ----------------------------------------------------------------------- old checkpoints


def test_state_from_before_proposals_existed_reads_as_none() -> None:
    """ADR 0011: a checkpoint written before the channel existed is normal, not an error."""
    assert proposals_of({}) == []  # type: ignore[arg-type]
    assert proposals_of({"proposed_actions": [{"garbage": 1}, REFUND]}) == [  # type: ignore[typeddict-item]
        Proposal.model_validate(REFUND)
    ]
