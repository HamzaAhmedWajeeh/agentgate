# Corpus

Synthetic documents describing **Northwind Mutual**, an organisation that does not exist. Every
policy, figure, and procedure here was invented for this repository. No real customer, client,
employer, or institution is described, and nothing in this directory originates from one.

It is committed rather than generated so that retrieval tests can assert against known content.
A failing retrieval test then means retrieval broke, not that a fixture drifted.

The documents are deliberately written the way real internal policy is written: short sections
under headings, each stating a rule and the reason for it. `agentgate.retrieval.corpus` chunks
on those headings, so the structure here is load-bearing rather than decorative.

To index it: `python scripts/seed_corpus.py`.
