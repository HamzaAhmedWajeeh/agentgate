"""TypeSafe's Jev, over plain HTTP. See docs/adr/0012.

**No SDK.** One documented endpoint, one JSON body in and one out, and ``httpx`` is already a
declared dependency. The SDK would add a dependency to save perhaps thirty lines, and it reads
four ``TYPESAFE_*`` variables on its own -- including a default model of ``jev-latest``, the alias
this project refuses -- which is exactly the undeclared-read problem ADR 0009 exists to prevent.

The request is ``POST {jev_base_url}/systemone`` with ``state``, the pinned ``model`` and the two
questions in :mod:`agentgate.decider.assessment`. The base URL carries the version, as the
official default ``https://api.typesafe.ai/v1`` does, so the path appended is ``/systemone`` and
never ``/v1/systemone`` -- which would double it.

**Every outcome is an assessment.** The four things that must be true before an answer counts:

1. The HTTP exchange succeeded, within the timeout, after at most one retry on 429 or 5xx.
2. The response named the pinned model. An alias that moved answers as a new version, and
   thresholds tuned on shadow data from the old one say nothing about it.
3. The response carried a complete ``usage`` block, which is accounted before anything else. The
   block is documented as required and has not yet been seen on a real wire (B1); if it is
   absent the call cost money nobody can count, so the answer is discarded.
4. Both answers carried every field they are documented to carry.

Anything else is ``Assessment.failed`` with the reason -- which the gate reads as "ask a human".
The one thing that escapes is :class:`SpendCeilingExceededError`: a budget is not a decider's to
overrule, so crossing one aborts the run exactly as it does for every other call.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Final

import httpx

from agentgate.config import DeciderBackend, Settings
from agentgate.decider.assessment import (
    IRREVERSIBILITY,
    QUESTIONS,
    ROUTE,
    ROUTE_OPTIONS,
    Assessment,
)
from agentgate.guardrails.spend import MissingUsageError, SpendLedger, usage_of

RETRYABLE: Final = frozenset({429}) | frozenset(range(500, 600))
"""Rate limited, or the service's own fault -- including the documented 529 overloaded.

Not 401 or 422: those are this system's fault, and a second identical request gets the same
answer. Not a timeout either: the first request may have been processed and billed, and a
retry doubles the wait in front of a human who is going to be asked anyway."""

MAX_RETRY_WAIT_SECONDS: Final = 5.0
"""The longest a ``retry-after`` is honoured for. The decider sits in front of a human who will
be asked if it gives up, so waiting longer than this buys nothing a human could not provide."""

DEFAULT_RETRY_WAIT_SECONDS: Final = 1.0


class _Failed(Exception):  # noqa: N818 - control flow inside this module, never raised out of it
    """An answer that does not count, with the reason recorded on the assessment."""

    def __init__(self, reason: str, *, model: str | None = None) -> None:
        super().__init__(reason)
        self.model = model


class JevDecider:
    """Assess a state by asking Jev the route and irreversibility questions.

    Args:
        settings: Supplies the endpoint, key, pinned model and timeout.
        ledger: Where the call's usage is recorded. Required rather than optional, so a decider
            cannot be constructed that spends without accounting.
        sleep: How to wait before a retry. Injected so tests assert the wait instead of enduring
            it.
    """

    backend = DeciderBackend.JEV

    def __init__(
        self,
        settings: Settings,
        ledger: SpendLedger,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.sleep = sleep

    @property
    def endpoint(self) -> str:
        """``{base}/systemone``. The base is stripped of trailing slashes by configuration and
        again here, so neither ``/v1/`` nor a hand-built ``Settings`` can produce ``//``."""
        return f"{self.settings.jev_base_url.rstrip('/')}/systemone"

    def request_body(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {"state": dict(state), "model": self.settings.jev_model, "questions": QUESTIONS}

    def post(self, body: Mapping[str, Any]) -> httpx.Response:
        """Send one request, retrying at most once on a retryable status.

        Raises:
            httpx.HTTPError: on a transport failure or timeout. Not retried; see ``RETRYABLE``.
        """
        key = self.settings.jev_api_key
        headers = {"Authorization": f"Bearer {key.get_secret_value() if key else ''}"}
        timeout = self.settings.request_timeout_seconds
        with httpx.Client(timeout=timeout) as client:
            response = client.post(self.endpoint, json=body, headers=headers)
            if response.status_code in RETRYABLE:
                self.sleep(_retry_wait(response))
                response = client.post(self.endpoint, json=body, headers=headers)
        return response

    def assess(self, state: Mapping[str, Any]) -> Assessment:
        try:
            return self._assess(state)
        except _Failed as failure:
            return Assessment.failed(self.backend, str(failure), model=failure.model)

    def _assess(self, state: Mapping[str, Any]) -> Assessment:
        pinned = self.settings.jev_model
        try:
            response = self.post(self.request_body(state))
        except httpx.TimeoutException:
            msg = f"timed out after {self.settings.request_timeout_seconds}s"
            raise _Failed(msg) from None
        except httpx.HTTPError as error:
            msg = f"transport error: {type(error).__name__}"
            raise _Failed(msg) from None

        if response.status_code != httpx.codes.OK:
            msg = f"HTTP {response.status_code}"
            raise _Failed(msg)
        try:
            payload = response.json()
        except ValueError:
            msg = "response body is not JSON"
            raise _Failed(msg) from None
        if not isinstance(payload, dict):
            msg = "response body is not a JSON object"
            raise _Failed(msg)

        answered_by = _as_str(payload.get("model"))

        # Accounted before the answer is judged: an answer discarded below was still paid for.
        # Recorded against the pinned model, whose price is configured -- a moved alias may name
        # a version that has none, and the spend is real either way. A ceiling crossing raises.
        try:
            usage = usage_of(payload.get("usage"))
        except MissingUsageError as error:
            raise _Failed(str(error), model=answered_by) from None
        self.ledger.record_usage(pinned, usage)
        self.ledger.check()

        if answered_by != pinned:
            msg = (
                f"answered by {answered_by!r}, not the pinned {pinned!r}; thresholds tuned on "
                "the pinned version say nothing about another"
            )
            raise _Failed(msg, model=answered_by)

        try:
            return _read_answers(payload.get("answers"), pinned)
        except _Failed as failure:
            raise _Failed(str(failure), model=pinned) from None


def _retry_wait(response: httpx.Response) -> float:
    header = response.headers.get("retry-after")
    try:
        wait = float(header) if header is not None else DEFAULT_RETRY_WAIT_SECONDS
    except ValueError:  # an HTTP-date, which this does not parse; wait the default instead
        wait = DEFAULT_RETRY_WAIT_SECONDS
    return max(0.0, min(wait, MAX_RETRY_WAIT_SECONDS))


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _probability(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        msg = f"{where} is {value!r}, not a probability"
        raise _Failed(msg)
    return float(value)


def _answer(answers: object, question_id: str, kind: str) -> dict[str, Any]:
    if not isinstance(answers, dict):
        msg = "response has no answers map"
        raise _Failed(msg)
    answer = answers.get(question_id)
    if not isinstance(answer, dict) or answer.get("type") != kind:
        msg = f"no {kind} answer for {question_id!r}"
        raise _Failed(msg)
    return answer


def _read_answers(answers: object, model: str) -> Assessment:
    route = _answer(answers, ROUTE, "choice")
    choice = route.get("choice")
    if choice not in ROUTE_OPTIONS:
        msg = f"{ROUTE}.choice is {choice!r}, not one of {ROUTE_OPTIONS}"
        raise _Failed(msg)
    probabilities = route.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(ROUTE_OPTIONS):
        msg = f"{ROUTE}.probabilities is {probabilities!r}, not one entry per option"
        raise _Failed(msg)
    if "confidence" not in route:
        msg = f"{ROUTE}.confidence is missing"
        raise _Failed(msg)

    irreversibility = _answer(answers, IRREVERSIBILITY, "noul")
    if "noul" not in irreversibility:
        msg = f"{IRREVERSIBILITY}.noul is missing"
        raise _Failed(msg)

    return Assessment(
        backend=DeciderBackend.JEV.value,
        model=model,
        route=choice,
        route_probabilities={
            option: _probability(probabilities[option], f"{ROUTE}.probabilities.{option}")
            for option in ROUTE_OPTIONS
        },
        route_confidence=_probability(route["confidence"], f"{ROUTE}.confidence"),
        irreversibility=_probability(irreversibility["noul"], f"{IRREVERSIBILITY}.noul"),
    )
