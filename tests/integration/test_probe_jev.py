"""The Jev probe: what it observes, and what it refuses to record.

The probe produces facts for the decider capability matrix; tests enforce the facts that were
recorded. So these tests do not claim anything about TypeSafe -- they check that the probe reads
the three things it says it reads, off a real HTTP response, and that it will not stamp
``LIVE_PROBE`` on an observation of this repository's own stub.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from scripts import probe_capabilities
from tests.doubles.typesafe_systemone import JevBehaviour, JevStub, running_jev_stub

from agentgate.config import Settings
from agentgate.decider.capabilities import DeciderCapability

pytestmark = pytest.mark.usefixtures("isolated_env")

PINNED = "jev-9.9.9"


@pytest.fixture
def stub() -> Iterator[JevStub]:
    with running_jev_stub(JevBehaviour()) as server:
        yield server


def settings_for(base_url: str) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        lane="cloud",
        openai_api_key="not-required",
        cloud_capable_model="m",
        cloud_cheap_model="m",
        decider_backend="jev",
        jev_model=PINNED,
        jev_base_url=base_url,
        model_prices_usd_per_million={
            "m": {"input": 1.0, "output": 1.0},
            PINNED: {"input": 0.042, "output": 0.0},
        },
        AGENTGATE_JEV_API_KEY="not-required",  # ADR 0004 item 21
    )


def observed(stub: JevStub) -> dict[DeciderCapability, bool]:
    results = probe_capabilities.probe_jev(settings_for(stub.base_url))
    return {result.capability: result.supported for result in results}


def test_the_probe_reads_all_three_capabilities_off_a_real_response(stub: JevStub) -> None:
    assert observed(stub) == dict.fromkeys(DeciderCapability, True)
    assert stub.behaviour.request_count == 1, "one small call, not one per capability"


def test_the_probe_sees_an_absent_usage_block(stub: JevStub) -> None:
    stub.behaviour.omit_usage = True

    assert observed(stub)[DeciderCapability.REPORTS_USAGE] is False


def test_the_probe_sees_a_model_that_did_not_echo_the_pin(stub: JevStub) -> None:
    stub.behaviour.answer_model = "jev-9.9.10"

    assert observed(stub)[DeciderCapability.ECHOES_VERSIONED_MODEL] is False


def test_the_probe_sees_a_choice_with_no_confidence(stub: JevStub) -> None:
    stub.behaviour.omit_answer_field = ("route", "confidence")

    assert observed(stub)[DeciderCapability.REPORTS_CONFIDENCE] is False


def test_the_probe_will_not_record_a_stub_as_a_live_observation(
    stub: JevStub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pointed anywhere but the official API, the probe would record this repository's stub back
    at it under ``LIVE_PROBE`` -- the one provenance that is supposed to mean "TypeSafe said so"."""
    for name, value in {
        "AGENTGATE_LANE": "cloud",
        "OPENAI_API_KEY": "not-required",
        "AGENTGATE_CLOUD_CAPABLE_MODEL": "m",
        "AGENTGATE_CLOUD_CHEAP_MODEL": "m",
        "AGENTGATE_DECIDER_BACKEND": "jev",
        "AGENTGATE_JEV_API_KEY": "not-required",
        "AGENTGATE_JEV_MODEL": PINNED,
        "AGENTGATE_JEV_BASE_URL": stub.base_url,
        "AGENTGATE_MODEL_PRICES_USD_PER_MILLION": (
            '{"m":{"input":1,"output":1},"jev-9.9.9":{"input":0.042,"output":0}}'
        ),
    }.items():
        monkeypatch.setenv(name, value)

    code = probe_capabilities.main(["jev"])

    assert code == probe_capabilities.EXIT_BAD_CONFIG
    assert stub.behaviour.request_count == 0, "refused before spending anything"
    assert "official" in capsys.readouterr().err
