"""A stand-in for TypeSafe's ``POST /v1/systemone``, over real HTTP.

Shaped from the official API reference at docs.typesafe.ai, read 2026-09-28: a request carries
``state``, ``model`` and a map of typed ``questions``; a response carries the versioned ``model``
that answered, one typed answer per question id, and a ``usage`` block with ``input_tokens`` and
``output_tokens``. Choice answers carry ``choice``, ``probabilities`` and ``confidence``; Noul
answers carry only ``noul`` -- the docs say explicitly that a Noul has no confidence.

It exists for the same reason the OpenAI-compatible stub does: an assertion about this egress has
to read what reached a socket, not what a client object was configured with. So every request is
logged with its **path**, its **headers** and its **body**, and a test reads the log.

**Every way it can fail is a knob**, because the decider's whole contract is that each of them
resolves to a human: the statuses the docs name (``422`` validation, ``429`` rate limit, ``529``
overloaded), a generic ``500``, a body that is not JSON, a body missing a field, a response with
no ``usage`` block, a response naming a model other than the one requested, and a reply slower
than the client's timeout.

**What it does not know.** The docs name the error statuses and say each carries "a JSON body
describing what went wrong", but do not document that body's shape. The bodies here are
placeholders, and the client under test reads only the status. A test that depended on the error
body would be pinning a guess.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

READY_TIMEOUT_SECONDS: Final = 15.0

# Documented statuses. The bodies are placeholders; see the module docstring.
ERROR_BODIES: Final[dict[int, dict[str, Any]]] = {
    401: {"error": "stub: missing or invalid API key"},
    422: {"error": "stub: request body failed validation", "field": "questions"},
    429: {"error": "stub: rate limit exceeded"},
    500: {"error": "stub: internal error"},
    529: {"error": "stub: overloaded"},
}


@dataclass
class JevBehaviour:
    """Knobs for making the stub answer, or fail, in specific and reproducible ways.

    Attributes:
        choice_probabilities: Per question id, the probabilities a Choice answer reports. The
            chosen option is the highest.
        choice_confidence: Per question id, the ``confidence`` a Choice answer reports. Reported
            as given, never derived: the decider must read the field, not recompute it.
        noul: Per question id, the probability a Noul answer reports.
        answer_model: The ``model`` the response names. ``None`` echoes the requested model,
            which is what the real API does for a versioned ID.
        fail_with: Statuses to return, one per request, before answering normally.
        retry_after: Sent as ``retry-after`` on a 429 or 529, when set.
        delay_seconds: How long to wait before answering. Drives the client's timeout.
        omit_usage: Leave the ``usage`` block out entirely.
        omit_answer_field: A ``(question id, field)`` to delete from an otherwise valid answer.
        malformed: Answer 200 with a body that is not JSON.
        usage: What the ``usage`` block reports.
    """

    choice_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    choice_confidence: dict[str, float] = field(default_factory=dict)
    noul: dict[str, float] = field(default_factory=dict)
    answer_model: str | None = None
    fail_with: list[int] = field(default_factory=list)
    retry_after: str | None = None
    delay_seconds: float = 0.0
    omit_usage: bool = False
    omit_answer_field: tuple[str, str] | None = None
    malformed: bool = False
    usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 296, "output_tokens": 20}
    )

    requests_seen: list[dict[str, Any]] = field(default_factory=list)
    paths_seen: list[str] = field(default_factory=list)
    headers_seen: list[dict[str, str]] = field(default_factory=list)

    @property
    def request_count(self) -> int:
        return len(self.requests_seen)


def _answer(question_id: str, question: dict[str, Any], behaviour: JevBehaviour) -> dict[str, Any]:
    kind = question.get("type")
    if kind == "noul":
        return {"type": "noul", "noul": behaviour.noul.get(question_id, 0.5)}
    if kind == "choice":
        options = list(question.get("criteria", {}))
        probabilities = behaviour.choice_probabilities.get(
            question_id, {option: 1.0 / len(options) for option in options}
        )
        return {
            "type": "choice",
            "choice": max(probabilities, key=lambda option: probabilities[option]),
            "probabilities": probabilities,
            "confidence": behaviour.choice_confidence.get(question_id, 0.0),
        }
    msg = f"the stub does not answer question type {kind!r}"
    raise ValueError(msg)


def build_app(behaviour: JevBehaviour) -> FastAPI:
    app = FastAPI()

    @app.post("/{path:path}")
    async def systemone(path: str, request: Request) -> Response:
        body = await request.json()
        behaviour.requests_seen.append(body)
        behaviour.paths_seen.append(request.url.path)
        behaviour.headers_seen.append({k.lower(): v for k, v in request.headers.items()})

        if behaviour.delay_seconds:
            await asyncio.sleep(behaviour.delay_seconds)

        if behaviour.fail_with:
            status = behaviour.fail_with.pop(0)
            headers = {"retry-after": behaviour.retry_after} if behaviour.retry_after else None
            return JSONResponse(ERROR_BODIES.get(status, {}), status_code=status, headers=headers)

        if request.url.path != "/v1/systemone":
            return JSONResponse({"error": "stub: not found"}, status_code=404)

        if behaviour.malformed:
            return PlainTextResponse("<html>this is not json</html>", status_code=200)

        answers = {
            question_id: _answer(question_id, question, behaviour)
            for question_id, question in body.get("questions", {}).items()
        }
        if behaviour.omit_answer_field is not None:
            question_id, name = behaviour.omit_answer_field
            answers.get(question_id, {}).pop(name, None)

        payload: dict[str, Any] = {
            "model": behaviour.answer_model or body.get("model"),
            "answers": answers,
        }
        if not behaviour.omit_usage:
            payload["usage"] = dict(behaviour.usage)
        return Response(json.dumps(payload), media_type="application/json")

    return app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class JevStub:
    """A running stub, addressable by URL. Real socket, real client, as with the chat stub."""

    def __init__(self, behaviour: JevBehaviour) -> None:
        self.behaviour = behaviour
        self._port = _free_port()
        config = uvicorn.Config(
            build_app(behaviour),
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def base_url(self) -> str:
        """The value to hand to ``AGENTGATE_JEV_BASE_URL``, shaped like the official default."""
        return f"http://127.0.0.1:{self._port}/v1"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._server.started:
                return
            time.sleep(0.02)
        msg = f"jev stub did not start within {READY_TIMEOUT_SECONDS}s"
        raise RuntimeError(msg)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=READY_TIMEOUT_SECONDS)


@contextlib.contextmanager
def running_jev_stub(behaviour: JevBehaviour | None = None) -> Iterator[JevStub]:
    server = JevStub(behaviour or JevBehaviour())
    server.start()
    try:
        yield server
    finally:
        server.stop()
