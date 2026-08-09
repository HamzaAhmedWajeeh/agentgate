"""Exception types.

Separate from the modules that raise them so an entry point can catch a failure that happens
during import. ``agentgate.config`` validates at import time, which means importing it is
itself the operation that can fail -- and you cannot catch that with an exception class you
imported from the module that just blew up.
"""

from __future__ import annotations


class AgentgateError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigurationError(AgentgateError):
    """The environment does not describe a runnable configuration.

    Raised instead of letting :class:`pydantic.ValidationError` surface, because the operator
    reading this message is holding a ``.env`` file, not a stack trace.
    """
