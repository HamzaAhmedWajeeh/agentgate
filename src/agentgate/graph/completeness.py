"""Whether a run's research actually finished.

Its own module because two nodes need the answer and neither should own it. The drafter needs
it to say so in the deliverable; ``finalise`` needs it to record it. If either one computed it
privately they could disagree, and the disagreement would surface as a document that reads as
whole sitting next to an audit event that says it is not.

The distinction being tracked is between two losses that look identical from the fan-in:

*A branch that failed and said so.* It appears in ``research_outcomes`` with ``ok=False``.
Attributable to a question, visible, recoverable.

*A branch that never reported at all.* Nothing in the outcomes can show this -- you have to
compare against ``dispatched``, which is what was sent. This is the failure mode a fan-out has
and a sequential loop does not: a loop cannot skip an iteration without raising, and a
super-step can lose a branch and merge cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgate.graph.state import AgentState, outcomes_of


@dataclass(frozen=True)
class ResearchGaps:
    """What is missing, counted by how it went missing."""

    dispatched: int
    reported: int
    failed: int
    silent: int
    failed_questions: list[str]

    @property
    def complete(self) -> bool:
        """Whether every dispatched branch reported success."""
        return self.failed == 0 and self.silent == 0

    def as_detail(self) -> dict[str, int | list[str]]:
        """The audit-event form. Plain types, because state is checkpointed."""
        return {
            "dispatched": self.dispatched,
            "reported": self.reported,
            "failed": self.failed,
            "silent": self.silent,
            "failed_questions": self.failed_questions,
        }


def research_gaps(state: AgentState) -> ResearchGaps:
    """Count what the fan-out lost, if anything."""
    dispatched = state.get("dispatched", 0)
    outcomes = outcomes_of(state)
    failed = [outcome for outcome in outcomes if not outcome.ok]

    return ResearchGaps(
        dispatched=dispatched,
        reported=len(outcomes),
        failed=len(failed),
        silent=max(0, dispatched - len(outcomes)),
        failed_questions=[outcome.question for outcome in failed],
    )
