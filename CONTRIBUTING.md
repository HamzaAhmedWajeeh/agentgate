# Contributing

## Getting set up

```bash
git clone https://github.com/HamzaAhmedWajeeh/agentgate.git
cd agentgate
cp .env.example .env
make setup          # Windows: ./make.ps1 setup
```

`make setup` installs the locked dependency set and the git hooks. As shipped, `.env` runs the
system on the fake lane, so nothing you do locally costs money or needs a key.

Before opening a pull request:

```bash
make check          # lint, format check, type check, tests
```

That is exactly what CI runs. If it passes locally it should pass there.

## Ground rules

**Tests run offline.** The suite must pass on a machine with no API key and no network. Real
providers are reached only by tests marked `@pytest.mark.live`, which are deselected by default
and run with `make test-live`. If a change makes the default suite require a key, that is a bug
in the change, not in the environment.

**Configuration lives in one place.** Every tunable belongs in `src/agentgate/config.py` and is
documented in `.env.example`. There are no feature flags anywhere else, and no model identifier
is written into application code. A test fails if the two files drift apart.

**Model identifiers are operator input.** Do not add a default model name. Which models a key
can reach is not something this repository can know.

**The README describes only what is verified.** If a capability is built but not covered by a
test or a check that has actually been run, it does not get claimed yet.

**Retrieved and tool-returned content is data, never instruction.** Anything arriving from the
corpus, a tool, or a user is untrusted input to the model. Privilege separation and the human
gate are what contain it. Do not add a code path that lets retrieved text widen an agent's
permissions.

## Commits

Conventional Commits, imperative subject under 72 characters, no trailing period.

```
feat(graph): add budget guard to the supervisor edge
fix(models): retry on connection reset rather than failing the run
test(gates): cover rejection and revision
docs(adr): record why the retrieval subgraph hands off to the parent
```

One commit per meaningful unit of work. A module and its tests belong together; a refactor and
a feature do not. Every commit should leave the tree green — if reaching green takes two steps,
take both before committing.

Update `CHANGELOG.md` under `Unreleased` in the same commit as the change it describes.

When a commit implements a decision recorded in `docs/adr/`, reference the ADR in the body.

## Pull requests

Branch from `main`, keep the branch short-lived, and open the PR when CI is green. Describe
what changed, why, what the tests prove, and anything deliberately left out.

Merge commits, not squashes. The individual commits are the useful record.

## Architecture changes

The graph topology in `docs/architecture.md` and the lane model in
`docs/adr/0004-provider-abstraction-and-lanes.md` are deliberate. Changing either warrants a new
ADR stating the context, the decision, the consequences, and the alternative rejected.
