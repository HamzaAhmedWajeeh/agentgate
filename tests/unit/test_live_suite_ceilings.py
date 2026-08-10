"""The live suite's ceilings, held to the basis they were derived from.

Two different failures are pinned here, and both were found by running something rather than
by reading it.

*A ceiling that outlives its basis.* ``AGENTGATE_MAX_LIVE_SUITE_TOKENS`` is the estimate at the
top of ``scripts/run_live.py`` times the tolerance. Written down once and never checked, the
two drift: the suite grows, the estimate is corrected, and the ceiling goes on describing a
suite that no longer exists. Recomputing it here turns that drift into a build failure.

*A gatekeeper whose own environment stops the thing it guards.* ``run_live.py`` sets
``AGENTGATE_*`` variables on the pytest subprocess it launches. The unknown-variable guard
rejects any ``AGENTGATE_*`` name that is not a declared setting -- so those variables made the
suite fail at startup, every time, and nothing noticed, because the suite is deselected by
default and had never actually been run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from scripts import run_live

from agentgate.config import Settings, _recognised_variable_names

pytestmark = pytest.mark.usefixtures("isolated_env")

# The variables the gatekeeper puts into the subprocess environment, read out of the script
# rather than restated here. A list kept by hand would agree with itself forever.
INJECTED_VARIABLES = frozenset(
    re.findall(r'"(AGENTGATE_[A-Z0-9_]+)":', Path(run_live.__file__).read_text(encoding="utf-8"))
)


# --------------------------------------------------------------------------- the basis


def test_the_suite_token_ceiling_is_still_the_estimate_times_the_tolerance() -> None:
    """Recomputed from the script's constants, so the ceiling cannot outlive its own basis."""
    settings = Settings(_env_file=None, lane="fake")  # type: ignore[call-arg]

    expected = int(run_live.ESTIMATED_TOKENS * settings.live_spend_tolerance)

    assert settings.max_live_suite_tokens == expected, (
        f"the suite ceiling is {settings.max_live_suite_tokens} but its stated basis -- "
        f"{run_live.ESTIMATED_CALLS} calls at {run_live.ESTIMATED_INPUT_TOKENS_PER_CALL} in and "
        f"{run_live.ESTIMATED_OUTPUT_TOKENS_PER_CALL} out, times a tolerance of "
        f"{settings.live_spend_tolerance:g} -- now comes to {expected}. Re-derive the ceiling "
        "in .env.example and in the field default; do not edit this assertion."
    )


def test_the_estimated_token_total_follows_from_the_per_call_constants() -> None:
    """The derived constant, derived. Pinned because the ceiling above rests on it."""
    assert run_live.ESTIMATED_TOKENS == run_live.ESTIMATED_CALLS * (
        run_live.ESTIMATED_INPUT_TOKENS_PER_CALL + run_live.ESTIMATED_OUTPUT_TOKENS_PER_CALL
    )


# --------------------------------------------------------------------------- the environment


def test_the_gatekeeper_injects_something_at_all() -> None:
    """Guards the test below.

    That test compares a set read out of the script against the declared settings. If the
    pattern ever stopped matching, the comparison would hold vacuously and report success
    while checking nothing -- which is the precise failure the suite exists to catch.
    """
    assert INJECTED_VARIABLES, "no AGENTGATE_* injection found in run_live.py; the check is blind"


def test_every_variable_the_gatekeeper_injects_is_a_declared_setting() -> None:
    """Found by running it. `make test-live` aborted before its first case.

    The gatekeeper set two AGENTGATE_* variables the settings model had never heard of, and the
    unknown-variable guard did exactly its job on them.
    """
    undeclared = INJECTED_VARIABLES - _recognised_variable_names()

    assert not undeclared, (
        f"run_live.py sets {sorted(undeclared)} on the suite's environment, and the "
        "unknown-variable guard rejects unrecognised AGENTGATE_* names. The suite cannot start."
    )


def test_the_suite_starts_under_the_environment_the_gatekeeper_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The names matching is not the claim. Constructing settings under them is.

    Values land where the suite reads them: the ledger path the fixture writes to on teardown,
    and the abort figure that tightens the suite ceiling mid-flight.
    """
    monkeypatch.setenv("AGENTGATE_LANE", "fake")
    monkeypatch.setenv("AGENTGATE_LIVE_SPEND_LEDGER", "data/live-spend.json")
    monkeypatch.setenv("AGENTGATE_LIVE_SPEND_ABORT_USD", "0.0075")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.live_spend_ledger == Path("data/live-spend.json")
    assert settings.live_spend_abort_usd == pytest.approx(0.0075)
