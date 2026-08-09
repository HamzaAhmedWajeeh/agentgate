# agentgate

> Every agent action passes a gate. A policy gate decides which model tier may see the data.
> Budget gates cap iterations, tokens, and spend. A human gate approves anything irreversible.
> All three write to an append-only audit trail.

An agent runtime for environments where an autonomous action has consequences: the model tier a
request reaches is a policy decision, the number of steps it may take is a budget decision, and
anything irreversible is a human decision. `agentgate` makes all three explicit, enforces them in
the graph rather than in a prompt, and records every one of them.

Built on LangGraph and LangChain. The demo domain is a compliance-aware research and drafting
workflow.

## Status

Under construction. This README documents only what is verified by a test or by a check that has
actually been run, so it grows as the system does. Nothing here is aspirational.

## License

MIT. See [LICENSE](LICENSE).
