"""Discover what a lane can actually do, and emit a matrix entry recording it.

This is a script, not a test, and the distinction is the point. A test that asserts nothing
because it does not yet know the answer is a script wearing a test costume: it runs in CI, it
is always green, and it proves nothing. Facts come from here. Tests enforce the facts that were
recorded.

The workflow:

  1. Run this against a configured lane. It costs a few small calls.
  2. Read what it observed.
  3. Paste the emitted entry into ``CAPABILITY_MATRIX`` in ``agentgate/models/registry.py``.
  4. The live test for that capability then *enforces* the row: it exercises the behaviour and
     fails if reality and the record disagree.

Step 4 is what makes step 3 safe. A recorded observation that nothing checks is just a comment
with punctuation, and it goes stale the first time a provider changes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, Field

from agentgate.config import CallClass, Lane, Settings, Tier, get_settings
from agentgate.errors import ConfigurationError
from agentgate.models.registry import Capability, build_model
from agentgate.models.structured import invoke_with_repair

EXIT_OK: Final = 0
EXIT_BAD_CONFIG: Final = 2
EXIT_PROBE_FAILED: Final = 3


class ProbeSchema(BaseModel):
    """Deliberately small. The probe measures whether structured output works, not whether the
    model is clever."""

    sensitivity: str
    confidence: float = Field(ge=0.0, le=1.0)


PROBE_PROMPT = "Classify the sensitivity of: 'the office coffee machine is broken'."


@dataclass(frozen=True)
class ProbeResult:
    """What was observed, and enough context to record it honestly."""

    lane: Lane
    capability: Capability
    supported: bool
    detail: str


def probe_native_structured_output(settings: Settings) -> ProbeResult:
    """Ask the lane for a schema-valid object and see whether it can produce one natively."""
    model = build_model(settings, Tier.CHEAP, CallClass.CLASSIFICATION)

    try:
        result = model.with_structured_output(ProbeSchema).invoke(PROBE_PROMPT)
    except Exception as error:  # any failure means the same thing here
        detail = f"native path raised {type(error).__name__}: {str(error)[:120]}"
        # Confirm the repair loop can still get an answer, so the row is actionable rather
        # than merely negative.
        repaired = invoke_with_repair(model, ProbeSchema, PROBE_PROMPT)
        detail += f"; validate-and-repair produced {repaired.sensitivity!r}"
        return ProbeResult(settings.lane, Capability.NATIVE_STRUCTURED_OUTPUT, False, detail)

    return ProbeResult(
        settings.lane,
        Capability.NATIVE_STRUCTURED_OUTPUT,
        True,
        f"native path returned a valid {type(result).__name__} without post-processing",
    )


def render_entry(result: ProbeResult, model_id: str) -> str:
    """A paste-ready ``CAPABILITY_MATRIX`` entry.

    Emitted rather than written automatically. Editing the matrix is a claim about the world,
    and a claim should pass through a person.
    """
    today = datetime.now(UTC).date().isoformat()
    lane = f"Lane.{result.lane.name}"
    capability = f"Capability.{result.capability.name}"
    return "\n".join(
        [
            f"    ({lane}, {capability}): Observation(",
            f"        supported={result.supported},",
            "        provenance=Provenance.LIVE_PROBE,",
            f"        recorded_on=date({today[:4]}, {int(today[5:7])}, {int(today[8:10])}),",
            (
                f'        note="Probed against {model_id} on {today} via '
                'scripts/probe_capabilities.py. "'
            ),
            f'        "{result.detail}",',
            "    ),",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Probe the configured lane and print a matrix entry for what it found."""
    if argv:
        print(f"usage: python scripts/probe_capabilities.py  (got {argv})", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    if not settings.requires_network:
        print(
            f"Lane is '{settings.lane.value}', whose behaviour this repository defines rather "
            "than observes. Probing it would record our own implementation back at us. "
            "Set AGENTGATE_LANE=cloud (or sovereign) to probe something real.",
            file=sys.stderr,
        )
        return EXIT_BAD_CONFIG

    model_id = settings.model_for(Tier.CHEAP)
    print(f"\n  Probing lane '{settings.lane.value}' via {model_id}.")
    print("  This makes a small number of billed calls.\n")

    try:
        result = probe_native_structured_output(settings)
    except Exception as error:  # a failed probe is a result to report, not a crash
        print(f"  Probe failed outright: {type(error).__name__}: {error}", file=sys.stderr)
        print("  Nothing recorded. A failed probe is not evidence of absence.", file=sys.stderr)
        return EXIT_PROBE_FAILED

    print(f"  OBSERVED  {result.capability.value} = {result.supported}")
    print(f"            {result.detail}\n")
    print("  Paste into CAPABILITY_MATRIX in src/agentgate/models/registry.py:\n")
    print(render_entry(result, model_id))
    print("\n  Then run the live suite, which enforces the row you just recorded.\n")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
