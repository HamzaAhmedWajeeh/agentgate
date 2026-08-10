"""Cumulative token and cost accounting.

Every model reply carries ``usage_metadata``. This adds it up and prices it, so a ceiling can
be enforced against what was actually consumed rather than against an estimate made before the
run started.

Three properties matter more than the arithmetic:

*A reply with no usage is an error, not a zero.* A provider that stops reporting usage would
otherwise make every run look free and no ceiling would ever be reached -- the same silent
disarming that an unpriced model would cause, arriving by a different route.

*Ceilings are checked, not observed.* :meth:`SpendLedger.check` raises. A ledger that merely
recorded a number and left enforcement to a caller who might forget is decoration.

*A ledger says which ceilings it enforces.* :class:`Ceilings` is a required argument, not a
default read off ``Settings``. Reading the run ceilings implicitly made every ledger a run
ledger, including the live suite's -- which is not a run, and tripped on being a suite rather
than on anything being wrong.
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


@dataclass(frozen=True)
class Ceilings:
    """The limits one ledger enforces, and the name a breach is reported under.

    Named at construction rather than read off ``Settings`` inside :meth:`SpendLedger.check`,
    because two different things get accounted here and they are not interchangeable:

    *A run* is one request through the graph. Its ceiling exists to catch a runaway inside that
    request -- a loop, a fan-out that will not converge.

    *The live suite* is several independent cases sharing one book so that the total is visible
    and the gatekeeper can read it. Nothing about it is a run. Measured against a per-run
    ceiling it trips on being a suite, which reports a budget failure where there is none and
    teaches whoever sees it to raise the run ceiling -- weakening the guard that was working.
    """

    scope: str
    """What is being accounted. Appears in the error, suffixed ``_tokens`` for a token breach,
    so an abort says which of the two ceilings stopped it."""

    max_total_tokens: int
    max_spend_usd: float

    @classmethod
    def for_run(cls, settings: Settings) -> Ceilings:
        """One request through the graph."""
        return cls("run", settings.max_total_tokens, settings.max_spend_usd)

    @classmethod
    def for_live_suite(cls, settings: Settings) -> Ceilings:
        """One complete live suite, bounded on its own basis.

        Tightened to ``live_spend_abort_usd`` when the gatekeeper set it. That figure is the
        estimate the operator was shown times the tolerance -- the number they actually
        consented to, and the smaller one whenever the gatekeeper is what launched this.
        """
        spend = settings.max_live_suite_spend_usd
        if settings.live_spend_abort_usd is not None:
            spend = min(spend, settings.live_spend_abort_usd)
        return cls("live_suite", settings.max_live_suite_tokens, spend)


@dataclass
class SpendLedger:
    """Running total for one scope of accounting, optionally rolling up into a session total.

    Args:
        settings: Supplies the prices.
        ceilings: What this ledger is accounting and the limits that apply to it. Required:
            a ledger that inferred its own ceilings is how the live suite came to be measured
            against a per-run budget.
        session: A parent ledger accumulating across runs. A loop of individually cheap runs
            is invisible to any per-run ceiling, so the session total is not optional
            bookkeeping -- it is the only thing that bounds that shape of failure.
    """

    settings: Settings
    ceilings: Ceilings
    session: SpendLedger | None = None

    usage_by_model: dict[str, Usage] = field(default_factory=dict)
    calls: int = 0

    def record(self, model: str, reply: AIMessage) -> Usage:
        """Account for one model call, and roll it up into the session if there is one."""
        return self.record_usage(model, usage_of(reply))

    def record_usage(self, model: str, usage: Usage) -> Usage:
        """Account for usage that did not arrive on a chat reply.

        Embeddings are the reason this exists: the API reports usage but not as
        ``usage_metadata`` on an ``AIMessage``, and the alternative to a second entry point was
        fabricating a message to carry it. Recorded against the model rather than against a
        category, so the summary says `text-embedding-3-small` and not `embeddings`.
        """
        self.usage_by_model[model] = self.usage_by_model.get(model, Usage()) + usage
        self.calls += 1
        if self.session is not None:
            self.session.record_usage(model, usage)
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
        limits = self.ceilings

        if self.total_tokens > limits.max_total_tokens:
            msg = (
                f"{limits.scope} consumed {self.total_tokens} tokens, over the ceiling of "
                f"{limits.max_total_tokens}"
            )
            raise SpendCeilingExceededError(
                msg,
                spent_usd=self.total_usd,
                ceiling_usd=limits.max_spend_usd,
                scope=f"{limits.scope}_tokens",
            )

        spent = self.total_usd
        if spent > limits.max_spend_usd:
            msg = (
                f"{limits.scope} spent ${spent:.4f}, over the ceiling of "
                f"${limits.max_spend_usd:.4f}"
            )
            raise SpendCeilingExceededError(
                msg,
                spent_usd=spent,
                ceiling_usd=limits.max_spend_usd,
                scope=limits.scope,
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
