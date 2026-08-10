"""Measure what one complete run actually consumes, so the ceilings have a basis.

A ceiling nobody ever approaches is decoration. It sits in the config file looking like a
control while the real failure -- a loop, a runaway fan-out -- sails under it. The only way to
set one honestly is to measure a real run and put the budget a deliberate multiple above it.

This runs the graph on the fake lane, which reports honest ``usage_metadata``, and prints the
figures to paste into ``.env.example``. Costs nothing and touches no network.

It is a script for the same reason `probe_capabilities.py` is: it produces a fact. The fact
then gets recorded by a person, and the recorded value is what the system enforces.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from typing import Final

from langchain_core.messages import AIMessage

from agentgate.config import CallClass, Lane, Settings, Tier
from agentgate.graph.build import build_checkpointer, build_graph
from agentgate.graph.state import initial_state
from agentgate.models.fake import FakeChatModel, scripted_json

EXIT_OK: Final = 0

# How far above a measured run the ceiling sits. Ten leaves room for a request several times
# heavier than the sample without leaving room for a runaway.
HEADROOM: Final = 10

# Representative requests rather than one. A single sample measures that sample; the ceiling
# has to cover the spread, so the largest run is what it is derived from.
SAMPLES: Final = [
    ("a short public request", "Summarise our refund policy in two sentences.", []),
    (
        "an involved internal request",
        (
            "Compare our refund policy against the three most common customer complaints "
            "from last quarter and identify where the policy and complaints disagree."
        ),
        [
            "refund policy eligibility window",
            "most common customer complaint categories",
            "complaints root cause linkage to incidents",
        ],
    ),
    (
        "a restricted request",
        (
            "Draft a response to the regulator about the incident affecting account 4471, "
            "including the remediation timeline and the affected customer count."
        ),
        [
            "regulator notification deadline for personal data",
            "incident severity levels and declaration",
            "post-incident review contents",
            "data retention access logging",
        ],
    ),
    # The ceiling has to cover the widest thing the graph can legally do, and that is not any
    # particular request -- it is a request whose sub-questions saturate the fan-out. Anything
    # narrower measures a typical run and leaves the ceiling below the worst legal one.
    (
        "a request at the fan-out limit",
        (
            "Produce a full compliance review covering refunds, complaints, incidents, "
            "retention, and access logging, with the gaps between them called out."
        ),
        [
            "refund policy escalation and second approver",
            "complaint acknowledgement and substantive response deadlines",
            "incident severity one regulator notification",
            "transaction and authentication log retention periods",
            "access logging of restricted records",
            "backup retention and erasure limits",
            "prorated partial refunds",
            "post-incident review timeline",
        ],
    ),
]

DRAFT = (
    "The refund window and the complaint response deadline disagree on timing; the gap is "
    "documented below with the sections it comes from."
)

CLASSIFICATIONS = {
    "a short public request": {
        "sensitivity": "public",
        "complexity": "simple",
        "contains_pii": False,
        "reason": "general policy question",
    },
    "an involved internal request": {
        "sensitivity": "internal",
        "complexity": "involved",
        "contains_pii": False,
        "reason": "internal analysis across sources",
    },
    "a restricted request": {
        "sensitivity": "restricted",
        "complexity": "involved",
        "contains_pii": True,
        "reason": "identifies an account and a regulator matter",
    },
    "a request at the fan-out limit": {
        "sensitivity": "internal",
        "complexity": "involved",
        "contains_pii": False,
        "reason": "broad review across every policy area",
    },
}


@dataclass
class Measured:
    label: str
    input_tokens: int
    output_tokens: int
    calls: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def measure(settings: Settings, label: str, request: str, sub_questions: list[str]) -> Measured:
    """Run the graph once and total the usage every model call reported."""
    seen: list[AIMessage] = []
    verdict = scripted_json(CLASSIFICATIONS[label])

    def recording_factory(
        _settings: Settings,
        _tier: Tier,
        call_class: CallClass,
        *,
        lane: Lane | None = None,  # noqa: ARG001 - part of the ModelFactory signature
    ) -> FakeChatModel:
        class Recording(FakeChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
                result = super()._generate(messages, stop, run_manager, **kwargs)
                seen.append(result.generations[0].message)
                return result

        # The drafter is the expensive call and it is the one whose prompt grows with the
        # fan-out: every finding from every branch is in it. Scripting a short reply keeps the
        # *output* honest-but-small while the input carries the real weight, which is the
        # right way round -- output length is a configured ceiling, input length is not.
        if call_class is CallClass.SYNTHESIS:
            return Recording(responses=[DRAFT])
        return Recording(responses=[verdict])

    graph = build_graph(settings, build_checkpointer(settings), model_factory=recording_factory)
    state = initial_state(request, str(uuid.uuid4()))
    state["sub_questions"] = sub_questions
    graph.invoke(
        state,
        {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": settings.recursion_limit,
        },
    )

    usage = [message.usage_metadata for message in seen if message.usage_metadata]
    return Measured(
        label=label,
        input_tokens=sum(int(u["input_tokens"]) for u in usage),
        output_tokens=sum(int(u["output_tokens"]) for u in usage),
        calls=len(usage),
    )


def main(argv: list[str] | None = None) -> int:
    """Measure the samples and print the ceilings they imply."""
    if argv:
        print(f"usage: python scripts/measure_run.py  (got {argv})", file=sys.stderr)
        return 1

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    results = [
        measure(settings, label, request, questions) for label, request, questions in SAMPLES
    ]

    print("\n  Measured on the fake lane. No network, no cost.\n")
    print(f"  {'sample':<32} {'calls':>6} {'in':>8} {'out':>8} {'total':>8}")
    for r in results:
        print(f"  {r.label:<32} {r.calls:>6} {r.input_tokens:>8} {r.output_tokens:>8} {r.total:>8}")

    heaviest = max(results, key=lambda r: r.total)
    ceiling = heaviest.total * HEADROOM

    print(f"\n  heaviest run: {heaviest.total} tokens ({heaviest.label})")
    print(f"  ceiling at {HEADROOM}x: {ceiling} tokens\n")
    print("  Paste into .env.example, replacing the UNMEASURED PLACEHOLDER comment:\n")
    print(f"    AGENTGATE_MAX_TOTAL_TOKENS={ceiling}")
    print(
        f"\n  Dollar ceilings follow from that budget priced at your table. With the cheap tier "
        f"at $I/$O per million:\n"
        f"    max_spend_usd ~= ({heaviest.input_tokens * HEADROOM} * I + "
        f"{heaviest.output_tokens * HEADROOM} * O) / 1e6\n"
    )
    print(
        f"  The heaviest sample saturates the fan-out at AGENTGATE_MAX_FAN_OUT="
        f"{settings.max_fan_out}. That is deliberate: the ceiling has to cover the widest run\n"
        "  the graph will legally perform, not a typical one. Re-derive after any change to\n"
        "  the fan-out limit, the retrieval top-k, or the number of model calls per run --\n"
        "  each of those moves the worst case, and this number is only as good as its date.\n"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
