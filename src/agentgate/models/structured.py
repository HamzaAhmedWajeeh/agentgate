"""Structured output, with a fallback for lanes that cannot do it natively.

The cloud lane constrains decoding and hands back an object. A self-hosted endpoint generally
cannot: asked for JSON it returns the right object wrapped in a code fence and an apology.
Both are reachable through one interface here, because a node that classifies a request should
not know or care which lane it landed on.

Two strategies:

*Native.* Delegate to the provider's own structured-output support and trust the result.

*Validate and repair.* Ask for JSON in the prompt, extract it from whatever prose came back,
validate it against the schema, and on failure ask again with the validation error quoted.
This is strictly more expensive -- extra tokens on every retry -- so it is used only where the
capability matrix records that native support is absent, never as a blanket default.

Which strategy a lane gets is a recorded observation, not an assumption. See
``agentgate.models.registry``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from agentgate.errors import AgentgateError

DEFAULT_REPAIR_ATTEMPTS: Final = 2

# A fenced block, with or without a language tag. Non-greedy so the first block wins.
FENCED_BLOCK = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(AgentgateError):
    """A model could not be made to produce output matching the schema.

    Carries the last raw text so the failure is diagnosable without re-running the call.
    """

    def __init__(self, message: str, *, raw: str, attempts: int) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


def extract_json_object(text: str) -> Any:
    """Pull a JSON value out of text that may be wrapped in prose, a code fence, or both.

    Tries the cheapest interpretation first and widens only on failure, so well-behaved output
    costs nothing extra and badly-behaved output is still recoverable.

    Raises:
        ValueError: if no parseable JSON value can be found.
    """
    candidates = [text.strip()]
    candidates.extend(match.strip() for match in FENCED_BLOCK.findall(text))
    balanced = _first_balanced_object(text)
    if balanced is not None:
        candidates.append(balanced)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    msg = "no JSON object found in the response"
    raise ValueError(msg)


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` span, ignoring braces inside strings.

    A naive search for the first ``{`` and last ``}`` breaks the moment a model writes prose
    containing a brace, which they do constantly.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _schema_instruction(schema: type[BaseModel]) -> str:
    """The system message used when a lane cannot be told to emit JSON structurally."""
    return (
        "Respond with a single JSON object and nothing else. No prose, no explanation, no "
        "code fence. The object must match this JSON Schema exactly:\n"
        f"{json.dumps(schema.model_json_schema(), sort_keys=True)}"
    )


def _as_messages(prompt: str | list[BaseMessage]) -> list[BaseMessage]:
    return [HumanMessage(prompt)] if isinstance(prompt, str) else list(prompt)


def invoke_with_repair[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    prompt: str | list[BaseMessage],
    *,
    max_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
) -> SchemaT:
    """Coax a schema-valid object out of a model that cannot produce one natively.

    Args:
        model: The chat model to call.
        schema: The pydantic model the result must satisfy.
        prompt: The request, as text or a message list.
        max_attempts: Total attempts, including the first. Each repair costs another call, so
            this is deliberately small; a model that cannot comply in three tries will not
            comply in ten, and the budget is finite.

    Returns:
        A validated instance of ``schema``.

    Raises:
        StructuredOutputError: when every attempt fails validation.
    """
    conversation: list[BaseMessage] = [
        SystemMessage(_schema_instruction(schema)),
        *_as_messages(prompt),
    ]

    raw = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        reply = model.invoke(conversation)
        raw = str(reply.content)

        try:
            return schema.model_validate(extract_json_object(raw))
        except (ValueError, ValidationError) as error:
            last_error = str(error)
            if attempt == max_attempts:
                break
            # Quote the model's own output back at it alongside the specific complaint.
            # Repeating the bare instruction produces the same wrong answer again.
            conversation.extend(
                [
                    AIMessage(raw),
                    HumanMessage(
                        "That did not validate against the schema.\n"
                        f"Error: {last_error}\n"
                        "Return only the corrected JSON object. No prose, no code fence."
                    ),
                ]
            )

    msg = (
        f"{schema.__name__} could not be parsed from the model output after "
        f"{max_attempts} attempt(s): {last_error}"
    )
    raise StructuredOutputError(msg, raw=raw, attempts=max_attempts)


def invoke_structured[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    prompt: str | list[BaseMessage],
    *,
    native: bool,
    max_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
) -> SchemaT:
    """Get a schema-valid object, by whichever route this lane supports.

    Args:
        native: Whether this lane is *observed* to support native structured output. Supplied
            by the capability matrix rather than guessed, because guessing wrong in the
            optimistic direction produces a confusing provider error, and guessing wrong in
            the pessimistic direction quietly doubles the token cost of every call.

    Raises:
        StructuredOutputError: when the output cannot be made to satisfy the schema.
    """
    if not native:
        return invoke_with_repair(model, schema, prompt, max_attempts=max_attempts)

    try:
        result = model.with_structured_output(schema).invoke(_as_messages(prompt))
    except (NotImplementedError, ValueError, ValidationError):
        # A lane recorded as native that turns out not to be. Fall through rather than fail:
        # the matrix is an observation and observations go stale when a provider changes.
        return invoke_with_repair(model, schema, prompt, max_attempts=max_attempts)
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)
