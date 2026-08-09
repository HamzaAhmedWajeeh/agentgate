# 9. `.env` is a shared namespace this application reads, not one it owns

Date: 2026-08-09

Status: Accepted

Amends the configuration model established in
[ADR 0007](0007-configuration-validated-at-startup.md).

## Context

The settings model was configured with `extra="forbid"`, on the reasoning that an unrecognised
variable should be a loud error rather than a silent no-op. That reasoning is sound, and the
guard it produced — a typo like `AGENTGATE_MAX_ITERATION` failing at startup with a suggested
spelling — is one of the more useful things in this configuration.

But `extra="forbid"` was applied to the wrong scope. It rejected *any* key in `.env` that did
not correspond to a field, and `.env` is not this application's file. It is the project's file.
A developer working on this repository plausibly keeps a LangSmith key, a database URL, and
whatever else their tooling reads in the same place.

That is not hypothetical. Startup failed on a real `.env` containing an unprefixed `LANGSMITH_*`
block:

```
agentgate cannot start: 1 configuration problem.
  - AGENTGATE_LANGSMITH_TRACING_V2: Extra inputs are not permitted
```

The variable had nothing to do with this application. Refusing to run because someone else's
tool has a key in a shared file is the application claiming ownership of something it merely
reads.

Investigating that failure surfaced a second, more interesting problem. Two *other* keys in the
same block — `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` — were not rejected. They were
**consumed**. pydantic-settings matched them to the `langsmith_api_key` and `langsmith_project`
fields despite neither declaring an alias for the unprefixed form.

The values were correct and the effect was benign, since tracing defaults to off. That is not
the point. A credential was flowing into the application from a source that nothing in the code
declared as an input. Reading the configuration module would not have told you that name was
read. The failure is the invisibility, not the value.

## Decision

**Tolerate foreign keys.** `extra="ignore"`. A key this application does not recognise is
somebody else's business.

**Keep policing the owned prefix.** Typo protection moves entirely to
`_reject_unrecognised_variables`, which scans the environment and `.env` for `AGENTGATE_*`
names matching no field and reports the closest match. This is strictly stronger than
`extra="forbid"` ever was: as ADR 0007's investigation established, pydantic-settings builds
its environment source per field, so an unknown prefixed variable was being *silently dropped*
rather than caught. The guard catches what `extra` could not.

**Declare every unprefixed read.** A field that reads a conventional third-party name does so
through an explicit `AliasChoices`, so the read appears in the source:

```python
langsmith_api_key: SecretStr | None = Field(
    default=None,
    validation_alias=AliasChoices(f"{ENV_PREFIX}LANGSMITH_API_KEY", "LANGSMITH_API_KEY"),
)
```

The permitted set is currently `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`.
Each is justified by the same argument: the surrounding ecosystem already uses that exact name,
so requiring a prefixed duplicate would be friction with no security benefit. A name invented
by this project has no such excuse and lives under the prefix.

**Assert the absence, not the presence.** `tests/unit/test_env_namespace.py` parametrises over
*every* field in the settings model, sets the bare unprefixed form of its name, and asserts the
field did not consume it. Fields with a declared alias are skipped by name. A separate test
reads the aliases back off the model and asserts they equal the permitted set exactly, so
widening what this application consumes from a shared namespace fails the build until someone
updates the list deliberately.

The guard is deliberately framed as "nothing undeclared is read" rather than "the declared
aliases work". The bug was a read from an undeclared source; a test that only exercised the two
aliases would have passed throughout.

## Consequences

The application starts on a realistic developer machine, and stops being a nuisance to whatever
else lives in the project's `.env`.

Typo protection is unchanged in strength and improved in scope: it now covers exactly the
namespace this application owns, and nothing else.

Every credential this process can receive from an unprefixed name is enumerated in one place,
visible in the field definition and pinned by a test. Adding another is possible and sometimes
right — it just cannot happen by accident.

The cost is a small loss of strictness on programmatic construction: `Settings(typo=1)` is now
ignored rather than rejected. That is a real regression, judged acceptable because the risk
surface is the environment rather than in-process construction, and the environment is covered
more thoroughly than before. Nothing in the codebase constructs `Settings` with literal keyword
arguments outside tests.

The parametrised guard also has a blind spot worth naming: it detects a field consuming the
bare form of *its own name*. A field that consumed some unrelated third-party name — a
`project` field silently reading `RAILS_ENV` — would not be caught by the parametrised case,
only by the narrower test that asserts specific foreign keys never appear in a dump. That is a
weaker net, and it is the best available without enumerating the whole environment.

## Alternatives rejected

**Keep `extra="forbid"` and require every foreign key to be prefixed.** Preserves maximum
strictness. Rejected because it imposes this application's naming on a shared file, which is
not a trade a library-shaped component gets to make. It also would not have found the second
bug: the accidental reads were never *extra*, so forbidding extras never looked at them.

**Give the application its own configuration file instead of sharing `.env`.** Removes the
ambiguity entirely. Rejected because `.env` is the convention every reader expects, the
quickstart is `cp .env.example .env`, and a bespoke file would be one more thing to explain for
a problem that an alias declaration solves.

**Drop the unprefixed aliases and require `AGENTGATE_OPENAI_API_KEY`.** Perfectly consistent,
and the namespace question disappears. Rejected because `OPENAI_API_KEY` is already exported in
the shell of almost everyone who would run this, and refusing to read the name the entire
ecosystem uses buys tidiness at the cost of a confusing first five minutes.
