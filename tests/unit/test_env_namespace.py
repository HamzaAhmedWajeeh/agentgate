"""What this application is allowed to read out of a shared environment.

`.env` is a project file. Other tools keep their keys there, and this application reads it
rather than owning it. Two obligations follow, and neither is obvious enough to leave to
review:

*It must tolerate keys it does not recognise* -- rejecting a foreign key would make it the
owner of a file it merely reads.

*It must not consume a name it never declared.* The bug that prompted this was a LangSmith
credential being read out of a shared `.env` by a field nobody had aliased to it. The value was
correct and the outcome was benign, but the read was invisible: nothing in the code said that
name was an input. A credential arriving from an undeclared source is the failure here, not a
wrong value.

So the guard below is not "the declared aliases work". It is "nothing undeclared is read".

See docs/adr/0009-env-is-a-shared-namespace.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import AliasChoices, ValidationError

from agentgate.config import ENV_PREFIX, Settings

pytestmark = pytest.mark.usefixtures("isolated_env")

# A value no field could legitimately hold. For typed fields it fails validation, so an
# accidental read surfaces as an exception; for string fields it survives, so an accidental
# read surfaces as an equality failure. Between them, every field is covered.
SENTINEL = "undeclared-read-sentinel-9d41c0"

# The complete set of unprefixed environment names this application is permitted to read,
# and the field each one feeds. Adding to this is a deliberate act: it means accepting an
# input from a namespace shared with every other tool on the machine.
PERMITTED_UNPREFIXED_READS: dict[str, set[str]] = {
    "openai_api_key": {"OPENAI_API_KEY"},
    "langsmith_api_key": {"LANGSMITH_API_KEY"},
    "langsmith_project": {"LANGSMITH_PROJECT"},
}


def declared_unprefixed_reads() -> dict[str, set[str]]:
    """Unprefixed names the settings model actually declares, read back off the model."""
    declared: dict[str, set[str]] = {}
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if not isinstance(alias, AliasChoices):
            continue
        unprefixed = {
            str(choice).upper()
            for choice in alias.choices
            if not str(choice).upper().startswith(ENV_PREFIX)
        }
        if unprefixed:
            declared[name] = unprefixed
    return declared


# --------------------------------------------------------------- nothing undeclared is read


@pytest.mark.parametrize("field_name", sorted(Settings.model_fields))
def test_no_field_reads_an_undeclared_unprefixed_name(
    field_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting a bare, unprefixed name must not reach any field that has not claimed it.

    Runs for every field in the model, so a new setting cannot quietly start consuming a
    generic name like ``PROJECT`` or ``API_HOST`` from the ambient environment.
    """
    if field_name in PERMITTED_UNPREFIXED_READS:
        pytest.skip(f"{field_name} declares an unprefixed alias on purpose")

    monkeypatch.setenv(field_name.upper(), SENTINEL)

    try:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
    except ValidationError as error:  # pragma: no cover - only on regression
        pytest.fail(
            f"{field_name} read the unprefixed name {field_name.upper()} and tried to "
            f"validate it: {error}"
        )

    assert getattr(settings, field_name) != SENTINEL, (
        f"{field_name} consumed {field_name.upper()} from the shared namespace without declaring it"
    )


def test_an_unrelated_tool_s_key_does_not_reach_any_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The concrete shape of the original bug, with a name from a different ecosystem."""
    monkeypatch.setenv("LANGSMITH_TRACING_V2", SENTINEL)
    monkeypatch.setenv("RAILS_ENV", SENTINEL)
    monkeypatch.setenv("DATABASE_URL", SENTINEL)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    rendered = settings.model_dump_json()
    assert SENTINEL not in rendered


# --------------------------------------------------------------- the permitted set is explicit


def test_the_model_declares_exactly_the_permitted_unprefixed_reads() -> None:
    """Adding an unprefixed alias must be a visible decision, not a quiet one.

    If this fails, someone widened what the application consumes from a namespace it shares
    with everything else on the machine. That may be correct -- but it should be argued for
    here and in the ADR, not discovered later.
    """
    assert declared_unprefixed_reads() == PERMITTED_UNPREFIXED_READS


@pytest.mark.parametrize(
    ("variable", "field_name"),
    [
        ("OPENAI_API_KEY", "openai_api_key"),
        ("LANGSMITH_API_KEY", "langsmith_api_key"),
        ("LANGSMITH_PROJECT", "langsmith_project"),
    ],
)
def test_each_permitted_read_actually_works(
    variable: str, field_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared alias that does not function is a different kind of lie."""
    monkeypatch.setenv(variable, SENTINEL)

    value = getattr(Settings(_env_file=None), field_name)  # type: ignore[call-arg]
    resolved = value.get_secret_value() if hasattr(value, "get_secret_value") else value

    assert resolved == SENTINEL


def test_every_permitted_read_is_a_conventional_third_party_name() -> None:
    """The justification for reading an unprefixed name is that the ecosystem already uses it.

    A name invented here has no such excuse and belongs under the prefix.
    """
    conventional_prefixes = ("OPENAI_", "LANGSMITH_")

    for names in PERMITTED_UNPREFIXED_READS.values():
        for name in names:
            assert name.startswith(conventional_prefixes), (
                f"{name} is not an established third-party variable; prefix it instead"
            )


# --------------------------------------------------------------- foreign keys are tolerated


def test_foreign_keys_in_a_shared_env_file_do_not_prevent_startup(tmp_path: Path) -> None:
    """The other half of the contract: read the file, do not police it."""
    env_file = tmp_path / "shared.env"
    env_file.write_text(
        "STRIPE_SECRET_KEY=sk_test_not_ours\n"
        "NEXT_PUBLIC_URL=http://localhost:3000\n"
        "AGENTGATE_MAX_ITERATIONS=5\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.max_iterations == 5


def test_policing_still_applies_inside_the_owned_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tolerance is for other people's names. Our own misspellings are still errors."""
    monkeypatch.setenv("AGENTGATE_MAX_ITERATION", "5")

    with pytest.raises(ValidationError, match="AGENTGATE_MAX_ITERATIONS"):
        Settings(_env_file=None)  # type: ignore[call-arg]
