"""``python -m agentgate`` -- print the configuration this process would actually run under.

The first question asked of a misbehaving deployment is "what does it think its settings are",
and the answer should not require attaching a debugger to a container. This resolves settings
exactly as the application does, redacts every secret, and exits non-zero if the environment
does not describe a runnable system -- which makes it usable as a container start-up probe.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Final

from agentgate.config import ConfigurationError, get_settings

EXIT_OK: Final = 0
EXIT_BAD_CONFIG: Final = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Render the resolved configuration as JSON on stdout.

    Returns:
        ``0`` when the configuration is valid, ``2`` when it is not. Errors go to stderr so
        stdout stays machine-readable and can be piped into ``jq`` without filtering.
    """
    if argv:
        print(f"usage: python -m agentgate  (no arguments; got {list(argv)})", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG

    # mode="json" renders every SecretStr as a mask rather than its value.
    print(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
