# 11. A checkpoint is a durable format, so state channels hold data

Date: 2026-08-10

Status: Accepted

## Context

Resuming a run started printing this, once per custom type in state:

```
Deserializing unregistered type agentgate.graph.state.Finding from checkpoint.
This will be blocked in a future version.
```

Six types: `Finding`, `Classification`, `ResearchOutcome`, and the `Decision`, `Sensitivity`
and `Complexity` enums. The framework offers two ways to make it stop — register the types with
the serialiser, or stop putting them in channels — and the choice looks like a matter of taste
until you ask what a checkpoint actually is.

**A checkpoint is a persistence format, not an in-process value.** It outlives the process. It
outlives the deploy that wrote it. Under Postgres it outlives the container. Anything crossing
that boundary is a wire format with a schema, whatever the language makes it look like.

## Decision

**State channels hold JSON-serialisable data only.** No pydantic models, no enums, no custom
classes. `Finding`, `Classification` and `ResearchOutcome` remain, as parse-and-serialise
helpers at node boundaries: nodes parse on the way in and write plain data on the way out, the
discipline an API edge uses. Enum values are stored as their strings.

Two arguments decided it, and the second is the one that matters.

**Registration keeps the coupling it silences.** A checkpoint written before a field is renamed
still has to be readable after it. With registered custom types, decoding names a class whose
definition has moved, and that is a hard failure — a run that cannot resume because the code
went on without it. Plain data with tolerant reads gives the opposite: unknown fields ignored,
missing fields defaulted, one unreadable entry costing that entry rather than the run.

**This repository's own thesis settles it.** The audit trail is worth something because it can
be read by someone who is not running this code. If the persisted form needs our class
definitions to decode, the record is legible only to the system that produced it — which is the
opposite of auditable. A format that requires the producer to read is a memo, not a record.

## Consequences

Reads go through `classification_of`, `findings_of`, `outcomes_of` and `decision_of`, which
parse and tolerate an older shape. `decision_of` fails closed: an unrecognised decision string
reads as `PENDING`, never `APPROVED`.

Three checks, in `tests/integration/test_checkpoint_serialisation.py`. Every channel of a real
completed run is walked and asserted to be JSON data that survives the serde round trip. A
state in an older shape — missing `source`, `detail`, `reason`, `revisions` — still parses,
with defaults. And the notice itself is asserted never to appear.

**The pytest configuration this was meant to use does not exist, and finding that out took two
wrong attempts worth recording.**

The plan was `filterwarnings = ["error:Deserializing unregistered type"]`. It was written, and
it enforced nothing: the notice is not a Python warning.
`warnings.catch_warnings(record=True)` around a full run and resume catches zero warnings while
the line still appears. A rule that looks like enforcement and enforces nothing is the exact
defect the leak inventory exists for, so it was removed rather than left in looking reassuring.

The second attempt read `capsys`, on the assumption it was printed. Also wrong — it is a
`logging` record from `langgraph.checkpoint.serde.jsonplus`. That was found only because the
guard test, the one that deliberately puts a custom type in a channel and asserts the notice
*appears*, failed. Without that guard the absent-string assertion would have passed forever
against an empty capture.

`LANGGRAPH_STRICT_MSGPACK=true` is not the answer either. It does not raise; it blocks the value
and continues, so under strict mode a custom type in a channel is silently dropped on resume
rather than failing loudly.

So the enforcement is ours: `caplog` against that named logger, plus the JSON-data walk, which
depends on no framework behaviour at all.

## Alternatives rejected

**Register the types with the serialiser.** One line, silences the notice today. Rejected
because it keeps the coupling that is the actual problem, and converts a future schema change
from a tolerant read into a failed resume.

**`LANGGRAPH_STRICT_MSGPACK=true` and leave the types in place.** Rejected on measurement: it
blocks rather than raises, which turns a visible notice into a silent data loss.

**Typed channels with custom serde hooks.** Keeps the in-process typing and makes the stored
form explicit. Rejected as the wrong trade for a repository meant to be read in an afternoon:
it adds a serialisation layer to maintain in exchange for typing that the boundary helpers
already provide at the only place it is used.
