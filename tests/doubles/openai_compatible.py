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

Everything it does is deterministic. Given the same request it returns the same body, so a
failure here is a real failure and never a flake.
"""

from __future__ import annotations

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
from fastapi.responses import JSONResponse

# Matches the fake lane's estimator so token figures are comparable across doubles.
CHARS_PER_TOKEN: Final = 4
READY_TIMEOUT_SECONDS: Final = 15.0


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
        supports_native_structured_output: Left as ``False`` for the sovereign stand-in. Set
            ``True`` only to demonstrate the contrast between lanes.
    """

    reply: dict[str, Any] = field(default_factory=lambda: {"answer": "stub"})
    fail_first_n: int = 0
    status_for_failures: int = 500
    supports_native_structured_output: bool = False

    requests_seen: list[dict[str, Any]] = field(default_factory=list)

    @property
    def request_count(self) -> int:
        return len(self.requests_seen)

    def reset(self) -> None:
        self.requests_seen.clear()


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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        behaviour.requests_seen.append(body)

        if behaviour.request_count <= behaviour.fail_first_n:
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
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


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
