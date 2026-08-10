"""Classify a request's sensitivity, so the policy gate has something to route on.

Runs on the cheap tier by definition: it produces a handful of tokens and its output is a
label, not a deliverable. Running classification on a capable model would mean paying synthesis
prices to decide where to send the synthesis.

The output is structured and validated. Everything downstream routes on ``sensitivity``, so a
malformed value would not be a parsing inconvenience -- it would be a policy decision made by
accident.
"""

from __future__ import annotations

from agentgate.audit.events import Decided, audit_event, digest
from agentgate.config import CallClass, Settings, Tier
from agentgate.graph.state import AgentState, Classification, Complexity, Sensitivity
from agentgate.models.registry import Capability, ModelFactory, build_model, supports
from agentgate.models.structured import StructuredOutputError, invoke_structured

NODE = "classify"

INSTRUCTION = """You classify requests before they are routed to a language model.

Judge only what the request itself reveals. Do not speculate about what answering it might
involve.

sensitivity:
  public      nothing confidential; could appear on a public website
  internal    ordinary business content, not for outside the organisation
  restricted  personal data, financial or legal specifics, credentials, or anything
              identifying a named individual or client

complexity:
  simple      answerable directly
  involved    needs research across several sub-questions

contains_pii: true if any personal data appears in the request.
reason: one short sentence.
"""


def classify(
    state: AgentState, settings: Settings, model_factory: ModelFactory = build_model
) -> AgentState:
    """Classify the request and record the decision.

    A failure to classify is not fatal and is not silently ignored either. The request is
    treated as restricted, which routes it to the sovereign lane, and the audit trail records
    that the classification failed rather than that the content was judged restricted. Those
    are different facts and a reviewer needs to be able to tell them apart.
    """
    request = state.get("request", "")
    correlation_id = state.get("correlation_id", "")
    model_id = settings.model_for(Tier.CHEAP)
    input_digest = digest(request)

    model = model_factory(settings, Tier.CHEAP, CallClass.CLASSIFICATION)
    native = supports(settings.lane, Capability.NATIVE_STRUCTURED_OUTPUT)

    try:
        classification = invoke_structured(
            model,
            Classification,
            f"{INSTRUCTION}\n\nRequest:\n{request}",
            native=native,
        )
        failed_reason = None
    except StructuredOutputError as error:
        # Fail closed. An unclassifiable request is treated as the most restrictive thing it
        # could be, because the alternative is letting a parse failure decide that content may
        # leave the boundary.
        classification = Classification(
            sensitivity=Sensitivity.RESTRICTED,
            complexity=Complexity.INVOLVED,
            contains_pii=True,
            reason="classification failed; treated as restricted",
        )
        failed_reason = str(error)[:200]

    return {
        "classification": classification.as_channel(),
        "audit_trail": [
            audit_event(
                node=NODE,
                decided=Decided.CLASSIFIED,
                correlation_id=correlation_id,
                input_digest=input_digest,
                model=model_id,
                lane=settings.lane.value,
                detail={
                    "sensitivity": classification.sensitivity.value,
                    "complexity": classification.complexity.value,
                    "contains_pii": classification.contains_pii,
                    # Present only when the classifier could not produce a verdict, so a
                    # reviewer can distinguish "judged restricted" from "failed, so assumed
                    # restricted".
                    "classification_failed": failed_reason,
                },
            )
        ],
    }
