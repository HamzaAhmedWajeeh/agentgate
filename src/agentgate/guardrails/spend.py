"""Cumulative token and cost accounting.

Every model reply carries ``usage_metadata``. This adds it up and prices it, so a ceiling can
be enforced against what was actually consumed rather than against an estimate made before the
run started.

Two properties matter more than the arithmetic:

*A reply with no usage is an error, not a zero.* A provider that stops reporting usage would
otherwise make every run look free and no ceiling would ever be reached -- the same silent
disarming that an unpriced model would cause, arriving by a different route.

*Ceilings are checked, not observed.* :meth:`SpendLedger.check` raises. A ledger that merely
recorded a number and left enforcement to a caller who might forget is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from agentgate.config import Settings
from agentgate.errors import AgentgateError


class SpendCeilingExceededError(AgentgateError):
    """A configured spend or token ceiling was crossed.

    Carries the figures so the abort can be recorded and explained rather than merely raised.
    """

    def __init__(self, message: str, *, spent_usd: float, ceiling_usd: float, scope: str) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        self.scope = scope


class MissingUsageError(AgentgateError):
    """A model reply arrived without usage metadata, so it could not be accounted for."""


@dataclass
class Usage:
    """Tokens consumed, split by direction because they are priced differently."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def usage_of(reply: AIMessage) -> Usage:
    """Extract usage from a reply.

    Raises:
        MissingUsageError: if the reply reports nothing. Treated as a failure rather than as
            zero, because a silent zero disarms every ceiling built on top of it.
    """
    metadata = reply.usage_metadata
    if not metadata:
        msg = (
            "model reply carried no usage_metadata, so its cost cannot be accounted for; "
            "refusing to treat an unmeasured call as free"
        )
        raise MissingUsageError(msg)
    return Usage(
        input_tokens=int(metadata.get("input_tokens", 0)),
        output_tokens=int(metadata.get("output_tokens", 0)),
    )


@dataclass
class SpendLedger:
    """Running total for one run, optionally rolling up into a session total.

    Args:
        settings: Supplies the prices and the ceilings.
        session: A parent ledger accumulating across runs. A loop of individually cheap runs
            is invisible to any per-run ceiling, so the session total is not optional
            bookkeeping -- it is the only thing that bounds that shape of failure.
    """

    settings: Settings
    session: SpendLedger | None = None

    usage_by_model: dict[str, Usage] = field(default_factory=dict)
    calls: int = 0

    def record(self, model: str, reply: AIMessage) -> Usage:
        """Account for one model call, and roll it up into the session if there is one."""
        usage = usage_of(reply)
        self.usage_by_model[model] = self.usage_by_model.get(model, Usage()) + usage
        self.calls += 1
        if self.session is not None:
            self.session.record(model, reply)
        return usage

    @property
    def total_tokens(self) -> int:
        return sum(usage.total_tokens for usage in self.usage_by_model.values())

    @property
    def total_usd(self) -> float:
        """Cost so far, priced per model.

        Raises:
            ConfigurationError: via ``Settings.price_for`` if a model was used that
                configuration never priced.
        """
        return sum(
            self.settings.price_for(model).cost_usd(usage.input_tokens, usage.output_tokens)
            for model, usage in self.usage_by_model.items()
        )

    def check(self) -> None:
        """Raise if any ceiling has been crossed.

        Raises:
            SpendCeilingExceededError: naming which ceiling and by how much.
        """
        if self.total_tokens > self.settings.max_total_tokens:
            msg = (
                f"run consumed {self.total_tokens} tokens, over the ceiling of "
                f"{self.settings.max_total_tokens}"
            )
            raise SpendCeilingExceededError(
                msg,
                spent_usd=self.total_usd,
                ceiling_usd=self.settings.max_spend_usd,
                scope="run_tokens",
            )

        spent = self.total_usd
        if spent > self.settings.max_spend_usd:
            msg = f"run spent ${spent:.4f}, over the ceiling of ${self.settings.max_spend_usd:.4f}"
            raise SpendCeilingExceededError(
                msg,
                spent_usd=spent,
                ceiling_usd=self.settings.max_spend_usd,
                scope="run",
            )

        if self.session is not None:
            session_spent = self.session.total_usd
            if session_spent > self.settings.max_session_spend_usd:
                msg = (
                    f"session spent ${session_spent:.4f}, over the ceiling of "
                    f"${self.settings.max_session_spend_usd:.4f}"
                )
                raise SpendCeilingExceededError(
                    msg,
                    spent_usd=session_spent,
                    ceiling_usd=self.settings.max_session_spend_usd,
                    scope="session",
                )

    def summary(self) -> str:
        """One line per model plus a total, for printing at the end of a run."""
        lines = []
        for model, usage in sorted(self.usage_by_model.items()):
            cost = self.settings.price_for(model).cost_usd(usage.input_tokens, usage.output_tokens)
            lines.append(
                f"  {model:<32} {usage.input_tokens:>8} in  "
                f"{usage.output_tokens:>8} out  ${cost:.6f}"
            )
        lines.append(
            f"  {'TOTAL':<32} {self.total_tokens:>8} tokens          ${self.total_usd:.6f}"
        )
        return "\n".join(lines)
