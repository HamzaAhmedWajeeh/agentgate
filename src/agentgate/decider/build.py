"""Construct the configured decider.

Its own abstraction, not a lane behind ``build_model``: a decider is asked typed questions and
answers with probabilities, and giving it a chat-model interface would invite exactly the free
text it cannot produce. Configuration decides which backend is *available*; whether a given
request reaches it is the policy router's decision, made per request (B5).
"""

from __future__ import annotations

from agentgate.config import DeciderBackend, Settings
from agentgate.decider.assessment import Decider
from agentgate.decider.fake import FakeDecider
from agentgate.decider.jev import JevDecider
from agentgate.decider.llm import LlmDecider
from agentgate.guardrails.spend import SpendLedger


def build_decider(settings: Settings, ledger: SpendLedger) -> Decider | None:
    """The decider this configuration asks for, or ``None`` when there is none.

    ``ledger`` is required, so no decider can be built that spends without accounting.
    """
    match settings.decider_backend:
        case DeciderBackend.NONE:
            return None
        case DeciderBackend.JEV:
            return JevDecider(settings, ledger)
        case DeciderBackend.LLM:
            return LlmDecider(settings, ledger)
        case DeciderBackend.FAKE:
            return FakeDecider()
