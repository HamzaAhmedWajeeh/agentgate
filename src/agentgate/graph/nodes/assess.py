"""The assess node: the decider's view of a draft, taken once, before the approval gate.

**Before the gate, never inside it.** The gate re-executes from its top on every resume, so a
decider called there would be called again each time -- a second billed request that could disagree
with the first. Here it is called once per draft, and the gate reads the stored verdict.

**Structured facts only, never the draft.** The decider is sent the routed lane, the finding count,
the tools that were denied, whether the provenance check passed, and the proposed actions -- facts
the drafter did not author as prose. The draft is model output built from retrieved content, and
TypeSafe's own documentation says Jev does not treat state as hostile, so draft text would be a
channel through which the corpus could address the decider. Leaving it out narrows that surface; it
does not close it -- a proposal's arguments are still model output. See docs/adr/0012.

**A cloud egress, so only on a cloud-routed request.** A request the policy gate sent anywhere more
contained never reaches the decider, and always goes to a human. The call is charged to the run
ledger like every other.

**Every outcome is recorded, including not calling.** A skipped assessment overwrites any earlier
one, so a verdict about a draft that has since been rejected and rewritten can never be read as a
verdict about this one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import DeciderBackend, Lane, Settings, narrower_of
from agentgate.decider.assessment import Decider
from agentgate.decider.build import build_decider
from agentgate.effects.proposals import digest_of
from agentgate.graph.state import AgentState, findings_of, lane_of, proposals_of
from agentgate.guardrails.run_ledger import ledger_of
from agentgate.guardrails.spend import SpendLedger

NODE = "assess"

DeciderFactory = Callable[[Settings, SpendLedger], Decider | None]


def decision_facts(state: AgentState, settings: Settings) -> dict[str, Any]:
    """The structured state the decider is shown, and the gate checks its preconditions against.

    ``denied_tools`` and ``provenance_check_passed`` are read off the drafter's own audit event for
    the current draft. If that event is missing, they read as a denial and a failure -- the gate's
    deterministic preconditions then fail closed rather than pass on missing evidence.
    """
    drafted = next(
        (
            event
            for event in reversed(state.get("audit_trail", []))
            if event.get("decided") == Decided.DRAFTED.value
        ),
        None,
    )
    detail = dict(drafted.get("detail", {})) if drafted else {}
    effective = narrower_of(lane_of(state), settings.lane)
    return {
        "routed_lane": effective.value,
        "finding_count": len(findings_of(state)),
        "denied_tools": sorted(detail.get("tools_denied", ["(no drafting record)"])),
        "provenance_check_passed": bool(detail.get("citations_clean", False)),
        "proposed_actions": [proposal.as_channel() for proposal in proposals_of(state)],
    }


def assess(
    state: AgentState,
    settings: Settings,
    config: RunnableConfig,
    decider_factory: DeciderFactory = build_decider,
) -> AgentState:
    facts = decision_facts(state, settings)
    effective = Lane(facts["routed_lane"])
    record: dict[str, Any] = {
        "mode": settings.decider_mode.value,
        "lane": effective.value,
        "proposals_digest": digest_of(proposals_of(state)),
        "thresholds": {
            "min_probability": settings.auto_approve_min_probability,
            "min_confidence": settings.auto_approve_min_confidence,
            "max_irreversibility": settings.auto_approve_max_irreversibility,
        },
    }

    if settings.decider_backend is DeciderBackend.NONE:
        record |= {"called": False, "reason": "no decider is configured"}
    elif effective is not Lane.CLOUD:
        record |= {
            "called": False,
            "reason": f"routed to {effective.value}; the decider is a cloud egress",
        }
    else:
        decider = decider_factory(settings, ledger_of(config))
        if decider is None:  # pragma: no cover - backend NONE is handled above
            record |= {"called": False, "reason": "no decider was built"}
        else:
            record |= {"called": True, **decider.assess(facts).as_channel()}

    return {
        "assessment": record,
        "audit_trail": [
            audit_event(
                node=NODE,
                decided=Decided.ASSESSED,
                correlation_id=state.get("correlation_id", ""),
                input_digest=digest(json.dumps(facts, sort_keys=True)),
                model=record.get("model"),
                lane=effective.value,
                detail=record,
            )
        ],
    }
