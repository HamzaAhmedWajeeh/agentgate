# 7. Validate configuration at startup, not at import

Date: 2026-08-09

Status: Accepted

Supersedes the import-time validation introduced in `feat(config): add settings with fail-fast
validation` (0946370).

## Context

The requirement is that a broken environment stops the process immediately, with a message
naming the variable at fault, rather than surfacing thirty seconds into a graph run as a
provider stack trace. That requirement is not in dispute; only the mechanism is.

The first implementation ran validation as a side effect of importing `agentgate.config`:

```python
def _fail_fast() -> None:
    get_settings()

_fail_fast()
```

This works, in the narrow sense that a bad environment does stop the process. It also has a
consequence that only became visible when the code was run rather than read.

Because validation happened during import, **the import statement was the operation that
failed**. An entry point written the obvious way —

```python
from agentgate.config import get_settings

def main() -> int:
    try:
        settings = get_settings()
    except ConfigurationError as error:
        ...
```

— cannot work. The exception is raised while the module is being imported, long before `main`
exists, let alone its `try` block. The handler is unreachable.

This was not theoretical. `python -m agentgate` exited 1 with a raw pydantic traceback where
its tests asserted a clean exit 2 and a readable message. The unit tests did not catch it: by
the time they ran, `agentgate.config` was already in `sys.modules` from an earlier test, so
import-time validation never ran a second time and the failure path was never exercised. A
container run caught it. The workaround at the time was to move the import inside the function
and pull `ConfigurationError` out into `agentgate.errors` so it could be imported safely.

That workaround is sound but it is a tax. Every future entry point — the FastAPI app, the Typer
CLI, the demo script, any test helper — has to know that this one module must not be imported
at module scope, and must repeat the function-scope import and the separated exception class.
A convention that must be remembered at every call site, whose failure mode is a wrong exit
code and an ugly traceback, is a landmine rather than a guarantee.

There is a second cost. A module that can raise on import is hostile to tooling: documentation
builders, IDE completion, static analysers, and `import agentgate.config` in a REPL all trip
over it. The first implementation had already grown a `if "sphinx" in sys.modules: return`
escape hatch, which is a reliable sign that the design is fighting its environment.

## Decision

Importing `agentgate.config` has no side effects and cannot raise.

Validation happens on an explicit `get_settings()` call, which every entry point makes as the
first statement inside its handler, where a `ConfigurationError` can be caught and rendered:

```python
from agentgate.config import get_settings
from agentgate.errors import ConfigurationError

def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = get_settings()          # first statement in the handler
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_CONFIG
```

`get_settings()` remains `lru_cache`d, so configuration is still read and validated exactly
once per process, and every caller still sees the same frozen object.

`agentgate/errors.py` stays. The separation is worth having on its own merits — the exception
hierarchy does not belong to the module that happens to raise from it first — even though the
specific problem that forced it is now gone.

## Consequences

The guarantee the brief asked for is fully preserved. A process with a broken environment still
dies in its first second, still names every variable at fault, and still exits 2. What changed
is *where* that happens: in a handler that can render it, rather than in an import that cannot.

Entry points can be written the obvious way. There is no convention to remember beyond "call
`get_settings()` first", which is enforced by the thing every entry point already needs to do.

The failure is now observable in ordinary tests as well as cold-start ones, because it happens
at a call rather than at an import.

The cost is that the guarantee is now a convention rather than a mechanism. A future entry
point that forgets to call `get_settings()` early will validate late, at whatever point it
first touches configuration. This is judged acceptable: the failure mode of forgetting is a
late error, whereas the failure mode of the previous design was a wrong exit code and an
unreadable traceback in the entry point that *did* remember. `test_importing_the_module_has_no_side_effects`
pins the property in a cold subprocess so it cannot silently regress.

## Alternatives rejected

**Keep import-time validation, keep the function-scope-import workaround.** Preserves the
literal wording of the original brief. Rejected because it makes every future entry point pay
for a guarantee it does not get — the import still fails first — and because the escape hatch
for documentation tooling shows the design does not fit how Python modules are used.

**Validate at import but swallow the error, exposing a `settings.is_valid` flag.** Import stays
safe. Rejected because it turns a loud failure into a quiet one that every caller must
remember to check, which is the exact failure mode the unknown-variable guard exists to
prevent elsewhere in this module.

**A `configure()` function that entry points call explicitly, separate from `get_settings()`.**
Makes the startup step explicit and greppable. Rejected as redundant: `get_settings()` already
validates on first call, and a second function that must also be called adds a way to get it
wrong without removing any.
