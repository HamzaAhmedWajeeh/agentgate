"""Smoke test: the package imports and reports a version.

This exists so the toolchain is exercised from the first commit onward. It is replaced in
substance by the node and guardrail suites, not removed -- an import failure should surface
as one obvious red test rather than fifty confusing ones.
"""

import agentgate


def test_package_exposes_a_version() -> None:
    assert isinstance(agentgate.__version__, str)
    assert agentgate.__version__


def test_version_resolves_from_installed_metadata() -> None:
    """A src layout only imports once the distribution is installed.

    The fallback in ``agentgate.__init__`` covers a raw checkout, so asserting the real
    version proves the environment is the installed one rather than a stray sys.path entry.
    """
    assert agentgate.__version__ != "0.0.0+unknown"
