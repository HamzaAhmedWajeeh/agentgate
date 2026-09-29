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

from agentgate.config import CallClass, DeciderBackend, Lane, Settings, Tier, get_settings
from agentgate.decider.assessment import ROUTE
from agentgate.decider.capabilities import DeciderCapability
from agentgate.decider.jev import JevDecider
from agentgate.errors import ConfigurationError
from agentgate.guardrails.spend import Ceilings, MissingUsageError, SpendLedger, usage_of
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


@dataclass(frozen=True)
class DeciderProbeResult:
    """What one call to a decider's endpoint showed about one capability."""

    capability: DeciderCapability
    supported: bool
    detail: str


# Invented and inert: the probe measures what a response carries, not how good the answer is.
PROBE_STATE: Final = {
    "routed_lane": "cloud",
    "finding_count": 1,
    "denied_tools": [],
    "provenance_check_passed": True,
    "proposed_actions": [],
}


def probe_jev(settings: Settings) -> list[DeciderProbeResult]:
    """Make one real call and read the three capabilities off the raw response.

    One call, not one per capability: every one of them is a property of the same response, and
    each call is billed. The call is accounted in a ledger like any other.

    Raises:
        RuntimeError: if the endpoint did not answer 200.
        TypeError: if it answered with something other than a JSON object. A failed probe is not
            evidence that a capability is absent, so nothing is reported either way.
    """
    ledger = SpendLedger(settings, Ceilings.for_run(settings))
    decider = JevDecider(settings, ledger)
    response = decider.post(decider.request_body(PROBE_STATE))
    if response.status_code != 200:  # noqa: PLR2004 - HTTP OK
        msg = f"the endpoint answered HTTP {response.status_code}"
        raise RuntimeError(msg)
    payload = response.json()
    if not isinstance(payload, dict):
        msg = "the endpoint answered with something other than a JSON object"
        raise TypeError(msg)

    try:
        usage = usage_of(payload.get("usage"))
        usage_detail = f"usage block reported {usage.input_tokens} in, {usage.output_tokens} out"
        reports_usage = True
    except MissingUsageError as error:
        usage_detail, reports_usage = str(error)[:160], False

    route = (payload.get("answers") or {}).get(ROUTE) or {}
    reports_confidence = "confidence" in route
    answered_by = payload.get("model")

    return [
        DeciderProbeResult(DeciderCapability.REPORTS_USAGE, reports_usage, usage_detail),
        DeciderProbeResult(
            DeciderCapability.REPORTS_CONFIDENCE,
            reports_confidence,
            f"route Choice answer carried confidence={route.get('confidence')!r}",
        ),
        DeciderProbeResult(
            DeciderCapability.ECHOES_VERSIONED_MODEL,
            answered_by == settings.jev_model,
            f"asked for {settings.jev_model!r}, response named {answered_by!r}",
        ),
    ]


def render_decider_entry(result: DeciderProbeResult, model_id: str) -> str:
    """A paste-ready ``DECIDER_CAPABILITY_MATRIX`` entry, for a person to review and paste."""
    today = datetime.now(UTC).date().isoformat()
    return "\n".join(
        [
            f"    (DeciderBackend.JEV, DeciderCapability.{result.capability.name}): Observation(",
            f"        supported={result.supported},",
            "        provenance=Provenance.LIVE_PROBE,",
            f"        recorded_on=date({today[:4]}, {int(today[5:7])}, {int(today[8:10])}),",
            (
                f'        note="Probed against {model_id} on {today} via '
                'scripts/probe_capabilities.py jev. "'
            ),
            f'        "{result.detail}",',
            "    ),",
        ]
    )


def main_jev(settings: Settings) -> int:
    """Probe the configured Jev endpoint and print decider matrix entries for what it showed."""
    official = Settings.model_fields["jev_base_url"].default
    if settings.decider_backend is not DeciderBackend.JEV:
        print("AGENTGATE_DECIDER_BACKEND is not 'jev'; nothing to probe.", file=sys.stderr)
        return EXIT_BAD_CONFIG
    if settings.jev_base_url != official:
        print(
            f"AGENTGATE_JEV_BASE_URL is {settings.jev_base_url!r}, not the official API "
            f"({official}). A probe records LIVE_PROBE rows, which mean TypeSafe itself was "
            "observed; recording a stub or a proxy under that name is the thing this refuses.",
            file=sys.stderr,
        )
        return EXIT_BAD_CONFIG

    print(f"\n  Probing {settings.jev_base_url}/systemone as {settings.jev_model}.")
    print("  This makes one small billed call.\n")
    try:
        results = probe_jev(settings)
    except Exception as error:  # a failed probe is a result to report, not a crash
        print(f"  Probe failed outright: {type(error).__name__}: {error}", file=sys.stderr)
        print("  Nothing recorded. A failed probe is not evidence of absence.", file=sys.stderr)
        return EXIT_PROBE_FAILED

    for result in results:
        print(f"  OBSERVED  {result.capability.value} = {result.supported}")
        print(f"            {result.detail}")
    print("\n  Paste into DECIDER_CAPABILITY_MATRIX in src/agentgate/decider/capabilities.py:\n")
    for result in results:
        print(render_decider_entry(result, settings.jev_model))
    return EXIT_OK


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
    """Probe the configured lane -- or, with ``jev``, the decider -- and print matrix entries."""
    if argv and argv != ["jev"]:
        print(f"usage: python scripts/probe_capabilities.py [jev]  (got {argv})", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    if argv == ["jev"]:
        return main_jev(settings)

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
