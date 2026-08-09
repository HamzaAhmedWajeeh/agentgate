"""Gatekeeper for `make test-live`.

The live suite is the only thing in this repository that spends money. It gets three
protections rather than one, because each covers a failure the others do not:

*An estimate, shown before anything runs.* You cannot consent to a cost you were not told.

*A confirmation prompt.* Typing the wrong make target should not be able to bill you.

*A ceiling enforced against actual spend.* This is the one that matters. An estimate is a
guess made before the run; if the run turns out to cost several times more, an advisory
estimate has told you nothing and stopped nothing. The suite aborts when actual spend exceeds
the estimate by more than a configurable factor.

Actual is printed against estimated at the end, so the estimator's accuracy is visible and can
be corrected rather than quietly drifting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

from agentgate.config import Settings, Tier, get_settings
from agentgate.errors import ConfigurationError

EXIT_OK: Final = 0
EXIT_DECLINED: Final = 1
EXIT_BAD_CONFIG: Final = 2
EXIT_OVERSPENT: Final = 3

# Rough shape of the live suite: five cases, each a handful of small calls. Deliberately a
# constant in one place rather than scattered guesses -- when the suite changes, this changes
# with it, and the printed comparison at the end says whether it is still right.
ESTIMATED_CALLS: Final = 15
ESTIMATED_INPUT_TOKENS_PER_CALL: Final = 400
ESTIMATED_OUTPUT_TOKENS_PER_CALL: Final = 150

LEDGER_PATH: Final = Path("data/live-spend.json")
CONFIRMATION = "yes"


def estimate_usd(settings: Settings) -> float:
    """What the suite should cost, priced at the configured rates for the cheap tier."""
    price = settings.price_for(settings.model_for(Tier.CHEAP))
    return price.cost_usd(
        ESTIMATED_CALLS * ESTIMATED_INPUT_TOKENS_PER_CALL,
        ESTIMATED_CALLS * ESTIMATED_OUTPUT_TOKENS_PER_CALL,
    )


def unpriced_message() -> str:
    """Refuse rather than run unbounded when the price table is empty.

    A zero estimate means no cost model, and no cost model means the abort threshold is also
    zero -- the run would be unbounded while appearing to be guarded.
    """
    return (
        "Estimated cost is zero, which means the price table is unfilled. A run with no cost "
        "model cannot be bounded, and an unbounded live run is exactly what this guard exists "
        "to prevent. Populate AGENTGATE_MODEL_PRICES_USD_PER_MILLION first (see `make models`)."
    )


def render_plan(settings: Settings, estimate: float, tolerance: float) -> str:
    return "\n".join(
        [
            "",
            "  Live test run -- this spends real money.",
            "",
            f"    lane            {settings.lane.value}",
            f"    cheap tier      {settings.model_for(Tier.CHEAP)}",
            f"    capable tier    {settings.model_for(Tier.CAPABLE)}",
            f"    calls (est.)    {ESTIMATED_CALLS}",
            f"    cost  (est.)    ${estimate:.4f}",
            "",
            (
                f"    Hard abort if actual exceeds ${estimate * tolerance:.4f} "
                f"({tolerance:g}x the estimate)."
            ),
            f"    Run ceiling     ${settings.max_spend_usd:.4f}",
            f"    Session ceiling ${settings.max_session_spend_usd:.4f}",
            "",
            "  The estimate is a guess made before the run. The abort is not.",
            "",
        ]
    )


def confirm() -> bool:
    """Ask before spending. Anything other than an explicit yes declines."""
    if not sys.stdin.isatty():
        print(
            "Refusing to run without an interactive confirmation. "
            "This target spends money and must not run unattended.",
            file=sys.stderr,
        )
        return False
    answer = input(f"  Type '{CONFIRMATION}' to proceed: ").strip().lower()
    return answer == CONFIRMATION


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    """Estimate, confirm, run, then enforce against actual spend.

    Several distinct refusals, each returning its own code: bad configuration, a lane
    that reaches nothing, an unpriced model, a declined prompt, and an overspend. A
    gatekeeper with one exit would have to conflate reasons it should keep apart.
    """
    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    if not settings.requires_network:
        print(
            f"Lane is '{settings.lane.value}', which reaches nothing. "
            "Set AGENTGATE_LANE=cloud to run the live suite.",
            file=sys.stderr,
        )
        return EXIT_BAD_CONFIG

    try:
        estimate = estimate_usd(settings)
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    if estimate <= 0:
        print(unpriced_message(), file=sys.stderr)
        return EXIT_BAD_CONFIG

    tolerance = settings.live_spend_tolerance
    print(render_plan(settings, estimate, tolerance))

    if not confirm():
        print("  Declined. Nothing was spent.")
        return EXIT_DECLINED

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    environment = {
        **os.environ,
        "AGENTGATE_LIVE_SPEND_LEDGER": str(LEDGER_PATH),
        "AGENTGATE_LIVE_SPEND_ABORT_USD": str(estimate * tolerance),
    }
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-m", "live", "-v", *(argv or [])],
        env=environment,
        check=False,
    )

    actual = read_actual_spend()
    print(render_outcome(estimate, actual, tolerance))

    if actual > estimate * tolerance:
        print(
            f"  ABORT THRESHOLD EXCEEDED: ${actual:.4f} against a limit of "
            f"${estimate * tolerance:.4f}.",
            file=sys.stderr,
        )
        return EXIT_OVERSPENT
    return completed.returncode


def read_actual_spend() -> float:
    """Total spend recorded by the live suite, or 0.0 if it recorded nothing."""
    if not LEDGER_PATH.exists():
        return 0.0
    try:
        return float(json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["total_usd"])
    except (ValueError, KeyError):
        return 0.0


def render_outcome(estimate: float, actual: float, tolerance: float) -> str:
    ratio = (actual / estimate) if estimate else 0.0
    return "\n".join(
        [
            "",
            "  Spend",
            f"    estimated   ${estimate:.4f}",
            f"    actual      ${actual:.4f}",
            f"    ratio       {ratio:.2f}x  (abort above {tolerance:g}x)",
            "",
            (
                "  If the ratio is consistently far from 1.0, correct the constants at the "
                "top of scripts/run_live.py rather than widening the tolerance."
            ),
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
