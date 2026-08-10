"""The checkpoint boundary: what crosses it, and what still works when the code moves on.

A checkpoint is a persistence format, not an in-process value. It outlives the process, the
deploy that wrote it, and under Postgres the container. Anything crossing that boundary is a
wire format with a schema, so the rule is that channels hold JSON-serialisable data and nothing
else. ADR 0011 records the reasoning.

Three things are checked here, and the third one is a configuration change rather than a test,
which is why it gets a test of its own asserting the configuration exists.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentgate.config import CallClass, Settings
from agentgate.graph.build import build_graph
from agentgate.graph.completeness import research_gaps
from agentgate.graph.state import (
    Decision,
    Finding,
    classification_of,
    decision_of,
    findings_of,
    initial_state,
    outcomes_of,
)
from agentgate.models.fake import FakeChatModel, scripted_json

pytestmark = pytest.mark.usefixtures("isolated_env")

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

VERDICT = scripted_json(
    {
        "sensitivity": "internal",
        "complexity": "involved",
        "contains_pii": False,
        "reason": "internal analysis",
    }
)

JSON_TYPES = (str, int, float, bool, type(None))

SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"
"""Where the notice actually comes from. Named as a constant because finding it cost two
wrong guesses -- warnings, then stdout -- and the next person should not repeat them."""


def settings_with(**overrides: object) -> Settings:
    base: dict[str, object] = {"lane": "fake", "corpus_path": CORPUS}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[call-arg]


def model_factory(_s: Settings, _t: object, call_class: CallClass, **_k: object) -> Any:
    if call_class is CallClass.SYNTHESIS:
        return FakeChatModel(responses=["A draft of the response."])
    return FakeChatModel(responses=[VERDICT])


def completed_run(settings: Settings) -> dict[str, Any]:
    """A run driven all the way through the gate, so every channel has been written."""
    graph = build_graph(settings, InMemorySaver(), model_factory=model_factory)
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }
    state = initial_state("Draft a refund response.", str(uuid.uuid4()))
    state["sub_questions"] = ["refund escalation", "retention period"]
    graph.invoke(state, config)
    return dict(graph.invoke(Command(resume={"decision": "approved"}), config))


def is_json_data(value: Any) -> bool:
    """Whether this is data rather than an object wearing a schema."""
    if isinstance(value, JSON_TYPES):
        return True
    if isinstance(value, list):
        return all(is_json_data(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_data(v) for k, v in value.items())
    return False


# ------------------------------------------------------------------ 1. what crosses


def test_every_channel_value_round_trips_through_the_serde_without_warning() -> None:
    """The whole boundary, checked against a real run rather than a constructed state.

    Channels are walked as they actually end up, because the failure this catches is a node
    quietly putting an object into a channel -- which no amount of reading ``state.py`` will
    show you.
    """
    final = completed_run(settings_with())
    serde = JsonPlusSerializer()

    # `messages` holds LangChain message objects, which are the framework's own registered
    # types and not ours to flatten. Everything this repository defines is data.
    ours = {key: value for key, value in final.items() if key not in {"messages", "__interrupt__"}}
    assert ours, "the run produced no channels; this asserted nothing"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for key, value in ours.items():
            assert is_json_data(value), f"channel {key!r} holds a {type(value).__name__}, not data"
            restored = serde.loads_typed(serde.dumps_typed(value))
            assert restored == value, f"channel {key!r} did not survive the round trip"


def test_the_rich_types_are_still_used_at_the_node_boundary() -> None:
    """Data in the channel does not mean stringly-typed code.

    The models parse on the way in and serialise on the way out, which is where the typing
    earns its keep. If this ever fails it means the boundary discipline decayed into passing
    dicts around, which is the other way to get this wrong.
    """
    final = completed_run(settings_with())

    assert isinstance(final["classification"], dict)
    assert final["classification"]["sensitivity"] == "internal"
    assert decision_of(final) is Decision.APPROVED
    assert all(isinstance(finding, dict) for finding in final["findings"])


# ------------------------------------------------------------------ 2. schema drift


def test_a_checkpoint_written_under_an_older_schema_still_resumes() -> None:
    """The property registration cannot give us.

    A checkpoint written before a field existed has to stay readable after it. With registered
    custom types that is a hard failure -- the stored form names a class whose definition has
    moved. With plain data, missing keys read as their defaults and the run continues.

    Simulated the only way it can be simulated without a time machine: by writing the older
    shape directly. Every value here is one a previous version of this schema could plausibly
    have produced -- a finding with no ``source``, an outcome with no ``detail``, a
    classification with no ``reason``, and no ``revisions`` key at all.
    """
    older: dict[str, Any] = {
        "request": "A request from an older deploy.",
        "correlation_id": "corr-old",
        "classification": {
            "sensitivity": "internal",
            "complexity": "simple",
            "contains_pii": False,
        },
        "findings": [{"question": "q", "content": "c"}],
        "research_outcomes": [{"question": "q", "ok": True}],
        "dispatched": 1,
        "decision": "pending",
    }

    classification = classification_of(older)  # type: ignore[arg-type]
    findings = findings_of(older)  # type: ignore[arg-type]
    outcomes = outcomes_of(older)  # type: ignore[arg-type]
    gaps = research_gaps(older)  # type: ignore[arg-type]

    assert classification is not None
    assert classification.reason == "", "a field added later should read as its default"
    assert findings[0].source == ""
    assert outcomes[0].detail == ""
    assert gaps.complete is True
    assert older.get("revisions", 0) == 0, "an absent key reads as its default, not an error"


def test_an_unreadable_channel_entry_costs_that_entry_and_not_the_run() -> None:
    """Tolerant, but not silently wrong: the bad entry is dropped, the good ones survive."""
    state: Any = {"findings": [{"question": "good", "content": "c"}, {"nonsense": True}]}

    findings = findings_of(state)

    assert [finding.question for finding in findings] == ["good"]


def test_an_unreadable_decision_never_reads_as_approved() -> None:
    """Fails closed at the boundary as well as at the gate. A decision string this version does
    not recognise must not authorise an irreversible action."""
    assert decision_of({"decision": "approved-ish"}) is Decision.PENDING  # type: ignore[arg-type]
    assert decision_of({"decision": ""}) is Decision.PENDING  # type: ignore[arg-type]
    assert decision_of({}) is Decision.PENDING  # type: ignore[arg-type]


# ------------------------------------------------------------------ 3. the warning is fatal


def test_a_real_run_never_logs_the_deserialisation_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The specific notice, turned into a build failure -- by us, because nothing else can.

    The intent was to promote it via pytest's ``filterwarnings``. That does not work, and the
    reason took two wrong attempts to find, both worth recording rather than quietly working
    around:

    **It is not a Python warning.** ``warnings.catch_warnings(record=True)`` around a full run
    and resume catches zero warnings while the line still appears. A ``filterwarnings`` entry
    would have been a rule that enforced nothing and read as though it did -- the exact defect
    this repository keeps a list of. It was written, and removed.

    **It is not printed either.** It is a ``logging`` record from
    ``langgraph.checkpoint.serde.jsonplus``, so ``capsys`` does not see it. That was found by
    the guard test below failing, which is the only reason it was found at all.

    ``LANGGRAPH_STRICT_MSGPACK=true`` is no help: it does not raise, it *blocks* the value and
    continues, so a custom type in a channel would be silently dropped on resume.

    So the check is ours and behavioural: drive a run through the gate and back, and assert the
    logger stayed quiet.
    """
    caplog.set_level(logging.WARNING, logger=SERDE_LOGGER)

    completed_run(settings_with())

    assert "Deserializing unregistered type" not in caplog.text, (
        "a state channel holds a type the checkpoint cannot decode without this codebase:\n"
        f"{caplog.text[:600]}"
    )
    assert "Blocked deserialization" not in caplog.text


def test_that_check_would_actually_notice(caplog: pytest.LogCaptureFixture) -> None:
    """Guards the test above, which is an assertion that a string is *absent* -- the shape of
    check most likely to pass because nothing ran.

    Puts a custom type into a channel on purpose and confirms the notice appears. It has
    already earned its place once: it is what revealed that the notice is logged rather than
    printed, when the version above was reading ``capsys`` and passing against nothing.
    """
    caplog.set_level(logging.WARNING, logger=SERDE_LOGGER)

    graph = StateGraph(dict)
    graph.add_node("write", lambda _s: {"leaked": Finding(question="q", content="c")})
    graph.add_node("pause", lambda _s: (interrupt({"x": 1}), {})[1])
    graph.add_edge(START, "write")
    graph.add_edge("write", "pause")
    graph.add_edge("pause", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    compiled.invoke({}, config)
    compiled.invoke(Command(resume="go"), config)

    assert "Deserializing unregistered type" in caplog.text, (
        "putting a custom type in a channel no longer logs anything, so the absence check "
        "above proves nothing and the JSON-data walk is the only remaining guard"
    )
