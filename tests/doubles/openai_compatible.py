"""A deliberately imperfect OpenAI-compatible server.

This stands in for the sovereign lane: a self-hosted endpoint you control, speaking the OpenAI
chat-completions dialect over HTTP. It exists to prove the plumbing end to end -- that a lane
selected by config, routed by ``base_url``, reaches a real socket and comes back through the
same abstraction as the cloud lane.

**It does not support native structured output, and that is the point.** Asked for a JSON
object via ``response_format``, it ignores the request and answers the way a small local model
habitually does: the right JSON, wrapped in conversational prose and a code fence. This is the
exact place the provider abstraction leaks, so the validate-and-repair fallback is exercised
against a server that really behaves this way rather than against a mock asserting that we
think it might.

**It speaks SSE when asked to stream, and that was added because nothing had ever asked.** The
CLI runs the graph with ``stream_mode=["updates", "messages"]``, which makes every model call a
streamed one -- and this stub answered a streamed request with an ordinary JSON body, so the
OpenAI client raised a bare ``AssertionError`` from inside its stream-state machine. Every CLI
test ran on the fake lane, so the combination of *the real client* and *a networked lane* had
never been exercised through the command line at all. The streaming path here exists to close
that gap rather than to be clever.

**It serves embeddings, on a separate log.** That was added with leak inventory item 17, and the
combined :attr:`StubBehaviour.every_request` view exists because of the trap it created: an
absence assertion that reads only the chat log passes perfectly while restricted content leaves
through the embedding endpoint. Two logs so a test can tell chat egress from retrieval egress;
one combined view so "did anything reach this endpoint" stays answerable.

``stream_options.include_usage`` is recorded per request rather than assumed. A streamed OpenAI
response carries no usage block unless the caller asks for one, which makes it the same shape as
leak inventory item 11: the number the budget depends on is absent by default and nothing
notices. :func:`usage_requested_on_every_stream` is what a test asks.

Everything it does is deterministic. Given the same request it returns the same body, so a
failure here is a real failure and never a flake.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Final

import tiktoken
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Matches the fake lane's estimator so token figures are comparable across doubles.
CHARS_PER_TOKEN: Final = 4
READY_TIMEOUT_SECONDS: Final = 15.0

EMBEDDING_DIMENSIONS: Final = 64
"""Small, and nothing depends on the number being realistic.

A provider's real dimensionality is irrelevant to what this endpoint is for: the tests pointing
at it assert on *which request bodies arrived*, never on retrieval quality. What does matter is
that it is fixed, because :class:`DenseIndex` zips a query vector against document vectors with
``strict=True`` -- a varying width would fail as a length mismatch rather than as a bad ranking.
"""


@dataclass
class StubBehaviour:
    """Knobs for making the stub misbehave in specific, reproducible ways.

    Attributes:
        reply: The payload the model should "decide on". Returned as prose-wrapped JSON when
            the caller asked for structured output, and as plain text otherwise.
        fail_first_n: Reject this many requests with a 500 before answering normally. Drives
            the retry chain against real HTTP errors rather than synthetic exceptions.
        status_for_failures: Status code used for those rejections. 429 exercises rate-limit
            handling; 500 exercises a generic server fault.
        reject: Asked about every chat request, after ``fail_first_n`` has had its say. Return
            ``True`` to reject it. ``fail_first_n`` can only express "the first few", which is
            the wrong shape for an outage confined to one tier or starting part-way through a
            run -- and those are exactly the conditions a fallback exists for. A predicate that
            reads the request body can say "the capable tier is down and the cheap one is up",
            which is what testing a *fallback* rather than a *retry* requires.
        supports_native_structured_output: Left as ``False`` for the sovereign stand-in. Set
            ``True`` only to demonstrate the contrast between lanes.
    """

    reply: dict[str, Any] = field(default_factory=lambda: {"answer": "stub"})
    fail_first_n: int = 0
    status_for_failures: int = 500
    reject: Callable[[dict[str, Any]], bool] | None = None
    supports_native_structured_output: bool = False

    requests_seen: list[dict[str, Any]] = field(default_factory=list)
    embedding_requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def request_count(self) -> int:
        return len(self.requests_seen)

    def embedding_texts(self) -> list[str]:
        """Every text this endpoint was asked to embed, decoded off the wire.

        What a canary assertion about retrieval has to read. See
        :func:`decode_embedding_input` for why the raw body is not enough.
        """
        return [text for body in self.embedding_requests for text in decode_embedding_input(body)]

    @property
    def every_request(self) -> list[dict[str, Any]]:
        """Every body this endpoint received, chat and embedding alike.

        The canary assertions read this rather than :attr:`requests_seen`, because "did restricted
        content reach this endpoint" is a question about traffic and not about one route. Keeping
        the logs separate and only ever checking the chat one is exactly how an embedding leak
        hides behind a passing absence assertion -- which is what would have happened here, since
        the wire tests for item 13 were written before this endpoint existed.
        """
        return [*self.requests_seen, *self.embedding_requests]

    @property
    def streamed_requests(self) -> list[dict[str, Any]]:
        return [body for body in self.requests_seen if body.get("stream")]

    def usage_requested_on_every_stream(self) -> bool:
        """Whether every streamed request asked for a usage block.

        A streamed OpenAI response omits usage unless ``stream_options.include_usage`` is set, so
        a caller that forgets it gets no token counts and no error. Exposed as a question a test
        can ask rather than left for someone to notice when a ceiling never trips.
        """
        return all(
            bool((body.get("stream_options") or {}).get("include_usage"))
            for body in self.streamed_requests
        )

    def reset(self) -> None:
        self.requests_seen.clear()
        self.embedding_requests.clear()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def wrap_in_prose(payload: dict[str, Any]) -> str:
    """Return JSON the way a small local model returns it: correct, but not alone."""
    body = json.dumps(payload, sort_keys=True)
    return (
        "Certainly! Based on what you've described, here is the JSON:\n\n"
        f"```json\n{body}\n```\n\n"
        "I hope this helps. Let me know if you'd like me to adjust anything."
    )


def build_app(behaviour: StubBehaviour) -> FastAPI:
    """Build the ASGI app. One behaviour object per app, mutated by the test that owns it."""
    app = FastAPI(title="openai-compatible-stub", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "stub", "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        behaviour.embedding_requests.append(body)

        raw = body.get("input")
        texts = [raw] if isinstance(raw, str) else [str(item) for item in (raw or [])]
        prompt_tokens = sum(estimate_tokens(text) for text in texts)
        return JSONResponse(
            content={
                "object": "list",
                "model": body.get("model", "stub"),
                "data": [
                    {"object": "embedding", "index": index, "embedding": _embedding_for(text)}
                    for index, text in enumerate(texts)
                ],
                # Reported, unlike the field `OpenAIEmbeddings` discards on the way back up --
                # leak inventory item 11. The endpoint is not where that goes missing.
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        behaviour.requests_seen.append(body)

        rejected = behaviour.request_count <= behaviour.fail_first_n or (
            behaviour.reject is not None and behaviour.reject(body)
        )
        if rejected:
            return JSONResponse(
                status_code=behaviour.status_for_failures,
                content={
                    "error": {
                        "message": f"stub failing request {behaviour.request_count} on purpose",
                        "type": "server_error",
                    }
                },
            )

        asked_for_json = "response_format" in body or "tools" in body
        # The leak. A native implementation would honour response_format and return a bare
        # object. This returns the object wrapped in prose, exactly as the endpoints this
        # lane targets do.
        content = (
            wrap_in_prose(behaviour.reply)
            if asked_for_json and not behaviour.supports_native_structured_output
            else json.dumps(behaviour.reply, sort_keys=True)
        )

        prompt_text = "".join(
            str(message.get("content", "")) for message in body.get("messages", [])
        )
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(content)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        if body.get("stream"):
            return StreamingResponse(
                _sse(
                    content,
                    model=str(body.get("model", "stub")),
                    usage=usage
                    if (body.get("stream_options") or {}).get("include_usage")
                    else None,
                ),
                media_type="text/event-stream",
            )

        return JSONResponse(
            content={
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    return app


def _chunk(model: str, **choice: Any) -> str:
    """One ``chat.completion.chunk``, framed as an SSE data line."""
    payload = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": None, **choice}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse(content: str, *, model: str, usage: dict[str, int] | None) -> Iterator[str]:
    """The same answer, delivered as the wire protocol the client expects when streaming.

    Deliberately more than one content chunk. A single-chunk stream is indistinguishable from a
    non-streamed reply as far as the client's assembly code is concerned, so it would not
    exercise the thing that makes streaming different.
    """
    yield _chunk(model, delta={"role": "assistant", "content": ""})
    midpoint = max(1, len(content) // 2)
    for piece in (content[:midpoint], content[midpoint:]):
        yield _chunk(model, delta={"content": piece})
    yield _chunk(model, delta={}, finish_reason="stop")
    if usage is not None:
        # Shaped the way OpenAI sends it: a final chunk with no choices at all, carrying only
        # the usage block. A client that reads usage off the last *choice* chunk finds nothing.
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
            + "\n\n"
        )
    yield "data: [DONE]\n\n"


EMBEDDING_ENCODING: Final = "cl100k_base"
"""The tokeniser `langchain-openai` used for an unrecognised embedding model, observed.

Not chosen: read off a real request. The first batch this endpoint received decoded to
`"# Complaints handling\n\n"`, which is the first chunk of the committed corpus, so the encoding
is confirmed rather than assumed. It is a client-library choice and an upgrade can change it,
which is why :func:`decode_embedding_input` raises on a sequence it cannot decode instead of
returning an empty string that every canary assertion would then pass against.
"""


def decode_embedding_input(body: dict[str, Any]) -> list[str]:
    """The text an embedding request carried, decoded from whatever form it arrived in.

    **This exists because a substring canary over an embedding request body matches nothing.**
    `OpenAIEmbeddings` tokenises before sending, so the wire carries ``"input": [[2, 68538, ...]]``
    rather than the string -- and an absence assertion written against the raw body would pass
    forever while restricted content left the building. Found by writing exactly that assertion
    and watching it fail for the right reason. Leak inventory item 17.

    Raises:
        TypeError: if the input is a shape this cannot decode. Loudly, because the alternative
            is returning nothing and making every caller's assertion vacuous -- which is the
            failure mode being guarded against in the first place.
    """
    raw = body.get("input")
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        msg = f"embedding request had no decodable input: {type(raw).__name__}"
        raise TypeError(msg)

    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    decoded: list[str] = []
    for item in raw:
        if isinstance(item, str):
            decoded.append(item)
        elif isinstance(item, list) and all(isinstance(token, int) for token in item):
            decoded.append(encoding.decode(item))
        else:
            msg = f"cannot decode embedding input element of type {type(item).__name__}"
            raise TypeError(msg)
    return decoded


def _embedding_for(text: str) -> list[float]:
    """A deterministic unit vector for a string.

    ``blake2b`` rather than ``hash()``, for the reason recorded in
    ``agentgate.retrieval.embeddings``: the built-in is salted per process, so an index built in
    one interpreter and queried from another would silently rank on nothing. That trap applies to
    a double exactly as much as to the real thing.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=EMBEDDING_DIMENSIONS).digest()
    values = [byte / 255.0 for byte in digest]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _free_port() -> int:
    """Reserve an ephemeral port, so parallel test runs do not collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class StubServer:
    """A running stub, addressable by URL.

    Runs uvicorn on a background thread rather than in-process via ASGI transport, because the
    point is to prove that a configured ``base_url`` reaches a real socket through the real
    client library.
    """

    def __init__(self, behaviour: StubBehaviour) -> None:
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
        """The value to hand to ``AGENTGATE_SOVEREIGN_BASE_URL``."""
        return f"http://127.0.0.1:{self._port}/v1"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._server.started:
                return
            time.sleep(0.02)
        msg = f"stub server did not start within {READY_TIMEOUT_SECONDS}s"
        raise RuntimeError(msg)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=READY_TIMEOUT_SECONDS)


@contextlib.contextmanager
def running_stub(behaviour: StubBehaviour | None = None) -> Iterator[StubServer]:
    """Run a stub for the duration of a block, and shut it down afterwards."""
    server = StubServer(behaviour or StubBehaviour())
    server.start()
    try:
        yield server
    finally:
        server.stop()
