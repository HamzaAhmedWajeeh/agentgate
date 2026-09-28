"""Classify a request's sensitivity, so the policy gate has something to route on.

Runs on the cheap tier by definition: it produces a handful of tokens and its output is a
label, not a deliverable. Running classification on a capable model would mean paying synthesis
prices to decide where to send the synthesis.

**It runs on the most contained lane the deployment can reach, not the configured default.**
This node is upstream of the policy gate, which is the whole difficulty: the request has to go
somewhere to be judged, and nothing has judged it yet. Sending it to the default lane means a
cloud-default deployment shows a third party the raw request in order to decide whether it was
allowed to see it. On a hybrid deployment that is now the sovereign lane; on a cloud-only
deployment there is nowhere else, and the egress is recorded as leak inventory item 14 rather
than papered over.

Classification *quality* on a self-hosted lane is not claimed anywhere. Ollama and vLLM have
never been run against, so what a weaker classifier does to routing is unmeasured. The
direction is at least safe: a classifier that cannot produce a verdict fails closed to
`restricted`, so the cost of a bad one is the cloud lane going unused rather than restricted
content escaping.

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
    input_digest = digest(request)

    # Classification runs before the router, so there is no routed lane to honour -- and the
    # request has not been judged yet, which is exactly why it cannot be sent wherever the
    # deployment happens to default to. The most contained lane available is the only defensible
    # destination. See Settings.most_contained_lane, and item 14 for what this does not close.
    lane = settings.most_contained_lane
    model_id = settings.model_for(Tier.CHEAP, lane=lane)

    model = model_factory(settings, Tier.CHEAP, CallClass.CLASSIFICATION, lane=lane)
    # Asked of the lane the call is actually made on. Looking this up on the configured lane
    # would take the native path against an endpoint recorded as not having it, which is not a
    # silent failure -- it is a measured 2 calls and 733 prompt tokens where 1 and 537 would do,
    # every classification, for as long as nobody checked.
    native = supports(lane, Capability.NATIVE_STRUCTURED_OUTPUT)

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
                lane=lane.value,
                detail={
                    "sensitivity": classification.sensitivity.value,
                    "complexity": classification.complexity.value,
                    "contains_pii": classification.contains_pii,
                    # Which lane the request was shown to in order to be judged, and whether
                    # that was the deployment's default. A reader auditing an egress needs the
                    # first; a reader auditing a *configuration* needs to see they differ.
                    "classified_on_lane": lane.value,
                    "deployment_default_lane": settings.lane.value,
                    "native_structured_output": native,
                    # Present only when the classifier could not produce a verdict, so a
                    # reviewer can distinguish "judged restricted" from "failed, so assumed
                    # restricted".
                    "classification_failed": failed_reason,
                },
            )
        ],
    }
