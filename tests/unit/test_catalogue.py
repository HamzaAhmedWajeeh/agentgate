"""`make models` is the first thing run against a new key, so it must not mislead.

The specific failure to avoid: emitting anything that looks like a price. The endpoint returns
no cost information, and a number that appears here would be inferred from a model's name --
which is guesswork wearing the costume of data.
"""

from __future__ import annotations

import json

import pytest
from tests.doubles.openai_compatible import running_stub

from agentgate.models.catalogue import (
    EXIT_NO_KEY,
    EXIT_OK,
    EXIT_REQUEST_FAILED,
    fetch_models,
    main,
    price_line,
    render,
)

pytestmark = pytest.mark.usefixtures("isolated_env")

RECORDS = [
    {"id": "second-model", "owned_by": "system"},
    {"id": "first-model", "owned_by": "openai"},
]


# --------------------------------------------------------------------------- the price line


def test_every_emitted_price_is_zero() -> None:
    """A plausible-looking wrong number would start happily and mis-account every run."""
    assignment = price_line(["one", "two"])
    table = json.loads(assignment.split("=", 1)[1])

    assert table == {
        "one": {"input": 0.0, "output": 0.0},
        "two": {"input": 0.0, "output": 0.0},
    }


def test_the_line_is_the_variable_the_operator_must_set() -> None:
    assert price_line(["one"]).startswith("AGENTGATE_MODEL_PRICES_USD_PER_MILLION=")


def test_the_line_parses_back_into_configuration() -> None:
    """Paste-ready means it round-trips, not that it looks about right."""
    value = price_line(["one"]).split("=", 1)[1]

    assert json.loads(value)["one"]["input"] == 0.0


# --------------------------------------------------------------------------- the listing


def test_the_listing_states_that_pricing_is_unavailable() -> None:
    """Silence here would read as "no cheap option found" rather than "not knowable"."""
    output = render(RECORDS)

    assert "Pricing is not available from this API" in output
    assert "not ordered by" in output


def test_no_price_is_inferred_from_any_name() -> None:
    """`mini`, `nano`, and `turbo` are marketing, not a cost model."""
    output = render([{"id": "something-nano", "owned_by": "system"}])
    table = json.loads(output.rsplit("=", 1)[1])

    assert table["something-nano"] == {"input": 0.0, "output": 0.0}


def test_models_are_listed_in_a_stable_order() -> None:
    """Ordering by cost is impossible here, so it is alphabetical and predictable."""
    output = render(sorted(RECORDS, key=lambda record: record["id"]))

    assert output.index("first-model") < output.index("second-model")


def test_an_empty_listing_says_so_rather_than_emitting_an_empty_table() -> None:
    assert "no models" in render([]).lower()


def test_the_operator_is_told_to_point_both_tiers_at_the_cheap_model() -> None:
    output = render(RECORDS)

    assert "BOTH" in output
    assert "deliberate edit" in output


def test_the_operator_is_told_to_record_the_date_checked() -> None:
    """A price with no vintage is the thing this whole design refuses to ship."""
    assert "date you checked" in render(RECORDS)


def test_filtering_narrows_the_line_but_not_the_listing() -> None:
    """A key reaching a hundred models still needs a pasteable line."""
    output = render(RECORDS, only=["first-model"])
    table = json.loads(output.rsplit("=", 1)[1])

    assert set(table) == {"first-model"}
    assert "second-model" in output  # still listed as reachable


# --------------------------------------------------------------------------- transport


def test_models_are_fetched_over_real_http() -> None:
    with running_stub() as stub:
        records = fetch_models("not-required", stub.base_url, timeout=10.0)

    assert [record["id"] for record in records] == ["stub"]


def test_a_missing_key_exits_without_a_request(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])

    assert code == EXIT_NO_KEY
    assert "No API key" in capsys.readouterr().err


def test_an_unreachable_identifier_is_rejected_rather_than_silently_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Emitting a price line for a model the key cannot reach would be a trap."""
    with running_stub() as stub:
        monkeypatch.setenv("OPENAI_API_KEY", "not-required")
        monkeypatch.setenv("AGENTGATE_OPENAI_BASE_URL", stub.base_url)

        code = main(["a-model-this-key-cannot-reach"])

    assert code == EXIT_REQUEST_FAILED
    assert "Not reachable" in capsys.readouterr().err


def test_a_reachable_identifier_produces_a_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with running_stub() as stub:
        monkeypatch.setenv("OPENAI_API_KEY", "not-required")
        monkeypatch.setenv("AGENTGATE_OPENAI_BASE_URL", stub.base_url)

        code = main(["stub"])

    assert code == EXIT_OK
    assert '"stub":{"input":0.0,"output":0.0}' in capsys.readouterr().out
