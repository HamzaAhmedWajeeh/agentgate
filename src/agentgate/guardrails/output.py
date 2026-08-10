"""Output checks on the draft, before a human is asked to approve it.

**Only one check here, and it is deliberate.** The tempting output guardrails are heuristics:
does the draft look like it leaked personal data, does it look like it obeyed an instruction
hidden in a retrieved document, does it overstate its confidence. Each of those is a pattern
match that is right most of the time, and a guardrail that is right most of the time in a
system whose whole argument is "verified, not asserted" is worse than no guardrail — it moves
the claim from *we do not check this* to *we check this*, and the second is false in exactly
the cases that matter.

What can be checked exactly is **provenance**. Every source the draft cites either came back
from research or it did not, and that is a set comparison rather than a judgement. A citation
naming a document the run never retrieved is a fabrication, and in a compliance domain a
fabricated citation is worse than a missing one: it is a claim a reader can look up, fail to
find, and reasonably conclude was invented.

What this does not do, stated plainly because the gap is the interesting part: it cannot see an
*uncited* fabrication. A draft that states something false without attributing it passes here.
Provenance checking bounds where a claim says it came from, not whether the claim is true, and
nothing in this repository claims the latter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentgate.graph.state import AgentState, findings_of

# Sources are written by the retrieval subgraph as `file.md#Heading`. The draft is asked to
# cite them verbatim, so this looks for that exact shape rather than trying to parse prose.
CITATION = re.compile(r"\b([\w.-]+\.md#[\w \-']+?)(?=[,.;:)\]]|\s*$|\n)", re.M)


@dataclass(frozen=True)
class ProvenanceReport:
    """Which citations in a draft are backed by a finding, and which are not."""

    cited: frozenset[str] = field(default_factory=frozenset)
    retrieved: frozenset[str] = field(default_factory=frozenset)
    fabricated: frozenset[str] = field(default_factory=frozenset)

    @property
    def clean(self) -> bool:
        """Whether every citation in the draft names something research actually returned."""
        return not self.fabricated

    def as_detail(self) -> dict[str, int | list[str]]:
        """The audit-event form. Plain types, because state is checkpointed (ADR 0011)."""
        return {
            "cited": len(self.cited),
            "retrieved": len(self.retrieved),
            "fabricated": sorted(self.fabricated),
        }


def check_provenance(state: AgentState) -> ProvenanceReport:
    """Compare what the draft cites against what research actually retrieved.

    A draft with no citations is reported as clean rather than as suspicious. Citing nothing is
    a different failure from citing something invented, and this check is not the one that
    notices an unsupported draft — ``research_gaps`` already tells a reviewer how much evidence
    there was, and the review packet carries it.
    """
    draft = state.get("draft", "")
    retrieved = frozenset(finding.source for finding in findings_of(state) if finding.source)
    cited = frozenset(match.group(1).strip() for match in CITATION.finditer(draft))

    return ProvenanceReport(
        cited=cited,
        retrieved=retrieved,
        fabricated=frozenset(citation for citation in cited if citation not in retrieved),
    )
