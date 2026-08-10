"""The middleware that makes the allowlist an allowlist.

``wrap_tool_call`` runs between the model's tool call and the executor that would run it. Every
call goes through it and there is no path around it, which is the property being relied on:
enforcement here does not depend on what the model was told, on which tools were bound, or on
any node remembering to check.

That distinction is the point. Not binding a tool means the model will not ordinarily ask for
it. It does not mean the model *cannot* ask -- a prompt injection in retrieved content, a
hallucinated name, a future refactor that binds one list and enforces another, and the request
exists. Whether it then *runs* is decided here.

Tool failures are handled in the same place, for the same reason: an exception out of a tool
would abort the agent, and the useful behaviour is for the model to see what went wrong and
carry on. Both denial and failure come back as a ``ToolMessage`` the model can read, and both
leave an audit event behind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.tools.registry import Agent, is_allowed


class AllowlistMiddleware(AgentMiddleware[Any, Any]):
    """Refuse tool calls this agent may not make, and contain the ones that fail.

    Args:
        agent: Whose allowlist applies.
        correlation_id: Ties the events this raises to the rest of the run.

    Events are collected on the instance rather than written to graph state, because middleware
    runs inside the compiled agent and does not share the parent's channels. The node that owns
    the agent drains them afterwards -- see ``graph/nodes/drafter.py``.
    """

    def __init__(self, agent: Agent, correlation_id: str = "") -> None:
        super().__init__()
        self.agent = agent
        self.correlation_id = correlation_id
        self.events: list[dict[str, Any]] = []
        self.denied: list[str] = []

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Authorise, then run, then contain.

        The order is the whole design. Authorisation happens before ``handler`` is called, so a
        denied tool is not executed and then discarded -- it never runs at all. There is no
        window in which an irreversible effect has happened and the decision to allow it has
        not.
        """
        name = _tool_name(request)

        if not is_allowed(self.agent, name):
            self.denied.append(name)
            self.events.append(
                audit_event(
                    node=f"{self.agent.value}:tools",
                    decided=Decided.TOOL_DENIED,
                    correlation_id=self.correlation_id,
                    input_digest=digest(name),
                    detail={
                        "tool": name,
                        "agent": self.agent.value,
                        "reason": "not in this agent's allowlist",
                    },
                )
            )
            return ToolMessage(
                content=(
                    f"Denied: '{name}' is not available to the {self.agent.value}. "
                    "Continue without it."
                ),
                tool_call_id=_tool_call_id(request),
                name=name,
                status="error",
            )

        try:
            return handler(request)
        except Exception as error:  # a tool that raises must not take the agent down
            self.events.append(
                audit_event(
                    node=f"{self.agent.value}:tools",
                    decided=Decided.TOOL_FAILED,
                    correlation_id=self.correlation_id,
                    input_digest=digest(name),
                    detail={
                        "tool": name,
                        "agent": self.agent.value,
                        "error": f"{type(error).__name__}: {str(error)[:200]}",
                    },
                )
            )
            return ToolMessage(
                content=f"'{name}' failed: {type(error).__name__}: {str(error)[:200]}",
                tool_call_id=_tool_call_id(request),
                name=name,
                status="error",
            )


def _tool_name(request: Any) -> str:
    """The tool being asked for, however the request happens to be shaped.

    Read defensively because this is the authorisation input. A request shape that this does
    not understand must not resolve to something that happens to be on the allowlist, so the
    fallback is the empty string, which is on nobody's list.
    """
    call = getattr(request, "tool_call", None)
    if isinstance(call, dict):
        return str(call.get("name", ""))
    return str(getattr(request, "name", "") or "")


def _tool_call_id(request: Any) -> str:
    call = getattr(request, "tool_call", None)
    if isinstance(call, dict):
        return str(call.get("id", ""))
    return str(getattr(request, "id", "") or "")
