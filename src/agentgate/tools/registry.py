"""Tool definitions and the per-agent allowlists.

Two facts about a tool matter here, and they are independent:

**Is it irreversible?** Whether running it changes something outside this process that cannot
be undone by not running it again. Reading a policy is reversible; refunding money is not. This
is a property of the tool.

**Which agents may call it?** A property of the *pairing*, not of the tool. The drafter is
allowed to read; it is not allowed to pay anyone. Recording that as a set per agent rather than
as a flag per tool means adding a third worker cannot silently inherit a permission.

The allowlists are the enforcement surface, not documentation of intent. ``AllowlistMiddleware``
sits in ``wrap_tool_call``, between the model's request and the executor, and there is no path
around it: a call whose name is not in the caller's set does not reach a handler. The tools are
separately never bound to a model that may not call them, so the ordinary case is that the
model cannot form the request at all -- but "cannot form it" is not the same claim as "cannot
run it", and the second one is the one that survives a prompt injection.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class Agent(StrEnum):
    """A worker that may hold tools.

    A closed set so an allowlist cannot be looked up for an agent nobody declared, which would
    otherwise return an empty set and read as "denied everything" -- a safe-looking answer to
    a question that should have been an error.
    """

    DRAFTER = "drafter"
    EXECUTOR = "executor"


# --------------------------------------------------------------------------- reversible


class PolicyLookup(BaseModel):
    """Arguments for :func:`lookup_policy`."""

    topic: str = Field(description="The policy area to look up, in a few words.")


@tool(args_schema=PolicyLookup)
def lookup_policy(topic: str) -> str:
    """Look up which corpus document covers a policy area.

    Read-only. Returns a pointer rather than content: the retrieval subgraph is what fetches
    text, and a tool that duplicated it would give the drafter a second, unaudited route to
    the corpus.
    """
    areas = {
        "refund": "refund-policy.md",
        "incident": "incident-response.md",
        "retention": "data-retention.md",
        "complaint": "complaints-handling.md",
    }
    for keyword, document in areas.items():
        if keyword in topic.lower():
            return f"{document} covers '{topic}'."
    return f"No committed document covers '{topic}'."


# --------------------------------------------------------------------------- irreversible


class RefundRequest(BaseModel):
    """Arguments for :func:`issue_refund`."""

    account: str = Field(description="Account reference the refund is credited to.")
    amount_units: float = Field(gt=0, description="Amount, in the fictional corpus's units.")


@tool(args_schema=RefundRequest)
def issue_refund(account: str, amount_units: float) -> str:
    """Credit a refund to an account. Irreversible: money moves.

    Not wired to anything. It exists so the allowlist has something real to exclude and so the
    approval gate in Phase 5 has something real to gate. A tool that is declared irreversible
    and quietly does nothing would make every test of the gate vacuous, so this raises rather
    than returning a plausible string.
    """
    msg = (
        f"issue_refund({account}, {amount_units}) reached its handler. This tool is "
        "irreversible and is not wired to anything before the approval gate exists (Phase 5). "
        "Reaching this line means an allowlist or a gate did not hold."
    )
    raise NotImplementedError(msg)


class CustomerEmail(BaseModel):
    """Arguments for :func:`send_customer_email`."""

    to: str = Field(description="Recipient address.")
    subject: str = Field(description="Subject line.")
    body: str = Field(description="Message body.")


@tool(args_schema=CustomerEmail)
def send_customer_email(to: str, subject: str, body: str) -> str:
    """Send a message to a customer. Irreversible: it cannot be unsent.

    Not wired, for the same reason as :func:`issue_refund`.
    """
    msg = (
        f"send_customer_email({to}) reached its handler. This tool is irreversible and is not "
        "wired to anything before the approval gate exists (Phase 5). Reaching this line "
        "means an allowlist or a gate did not hold."
    )
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------- the registry

TOOLS: Final[dict[str, BaseTool]] = {
    "lookup_policy": lookup_policy,
    "issue_refund": issue_refund,
    "send_customer_email": send_customer_email,
}

IRREVERSIBLE: Final[frozenset[str]] = frozenset({"issue_refund", "send_customer_email"})
"""Tools whose effects outlive the run.

A property of the tool. Declared here rather than inferred from a naming convention, because
a convention is something a future tool can fail to follow without anything noticing.
"""

ALLOWLISTS: Final[dict[Agent, frozenset[str]]] = {
    Agent.DRAFTER: frozenset({"lookup_policy"}),
    # Phase 5. The executor exists in this table so the shape of the eventual permission is
    # visible now, and so the test asserting that only one agent holds irreversible tools has
    # something to assert against rather than an empty table.
    Agent.EXECUTOR: frozenset({"issue_refund", "send_customer_email"}),
}


def tools_for(agent: Agent) -> list[BaseTool]:
    """The tools this agent may hold, for binding to its model.

    Binding is the first of the two layers: a model that was never given a tool will not
    ordinarily produce a call for it. The second layer is the middleware, which is what makes
    this an allowlist rather than a list of suggestions.
    """
    return [TOOLS[name] for name in sorted(ALLOWLISTS[agent])]


def is_allowed(agent: Agent, tool_name: str) -> bool:
    """Whether this agent may execute this tool. The whole authorisation decision."""
    return tool_name in ALLOWLISTS[agent]
