"""agentgate: a gated agent runtime for regulated environments.

Every agent action passes a gate. A policy gate decides which model tier may see the
data. Budget gates cap iterations, tokens, and spend. A human gate approves anything
irreversible. All three write to an append-only audit trail.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentgate")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a raw checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
