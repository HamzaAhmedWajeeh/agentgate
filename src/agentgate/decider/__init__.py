"""The decider: an assessment of a draft, made before the approval gate. See docs/adr/0012.

It can only ever produce "auto-approve" or "ask a human", never a rejection, and every failure
resolves to "ask a human". Budgets, caps and spend stay deterministic code; the decider decides
nothing numeric.

Nothing is re-exported here. Construction lives in :mod:`agentgate.decider.build` and is imported
from there, so the concept map's "called from nowhere" check -- which reads every file in
``src/`` -- sees exactly who constructs a decider, and a re-export does not count as a caller.
"""
