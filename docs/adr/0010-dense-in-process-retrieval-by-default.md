# 10. Dense in-process retrieval by default, Qdrant declared but not built

Date: 2026-08-10

Status: Accepted

## Context

The demo domain is compliance-aware research over an internal document store, so retrieval is
in the critical path. Two decisions had to be made before any of it could be written: what
indexes the corpus, and whether search is dense or hybrid.

Both have an obvious answer that is wrong for this repository. The obvious answer is a real
vector service with hybrid search, because that is what a production system would use. The cost
is paid by every reader: a service dependency in the default path means the test suite needs
Docker, CI needs a container, and the ninety-second walkthrough begins with `docker compose up`.

The corpus is four synthetic documents, twenty chunks. Nothing about that size needs a service.

## Decision

**Dense, exhaustive, in-process is the default path.** The index is a list of L2-normalised
vectors and the search is a dot product over all of them. Exhaustive rather than approximate:
at this size an approximate index trades exactness for a speedup too small to measure, and it
makes a retrieval test's expected output depend on index-construction parameters.

**The similarity search is written rather than imported.** LangChain's `InMemoryVectorStore`
requires numpy, which is not a dependency of this project. Cosine similarity over unit vectors
is a dot product — six lines — and six lines is cheaper than a dependency in the path of every
retrieval, in the image, and in the dependency audit. If the corpus ever grows to where this
matters, the answer is Qdrant, not numpy.

**Embeddings are chosen by lane, like models.** The cloud lane uses the configured embedding
model. Every other lane uses `HashingEmbeddings`: term frequency, hashed into 4,096 buckets,
L2 normalised. No network, no cost, deterministic across processes — which is what lets the
whole offline suite exercise real retrieval rather than a stub returning whatever the test
wanted.

**Qdrant is declared in configuration and not implemented.** `AGENTGATE_VECTOR_BACKEND=qdrant`
raises. It does not fall back to the in-memory index, because a backend that silently degrades
to a different one is how a deployment comes to believe it has hybrid search.

## Consequences

The quickstart needs no service and `make test` needs no key, which is the property the whole
offline posture rests on.

**Search is dense only, and hybrid search is therefore not claimed anywhere.** Dense retrieval
misses exact-term matches that BM25 would catch — an identifier, a policy number, a section
reference. That is a real limitation of the default path and it is stated in the README rather
than hedged.

**The offline embedder has no notion of synonymy.** A query matches a document when they share
vocabulary. That is enough to prove chunking, indexing, top-k, the subgraph handoff, and
fan-in, which is what the offline suite is for. Retrieval *quality* is a live-lane question and
no claim about it rests on the fake lane.

Two things about the offline embedder were found by running it, not by reasoning about it, and
both are recorded where the constant is defined:

- **256 dimensions put 70% of the corpus vocabulary into a shared bucket.** The symptom was
  retrieval that looked plausible and ranked wrongly: a query about log retention returned a
  section on incident severity above the retention schedule. Dropping stopwords did not help,
  because the problem was never the stopwords. Measured at 256/1024/4096/16384 and set to
  4096, where the curve flattens against a pure-Python dot product. A test pins the collision
  rate, so a corpus that outgrows the space fails the build instead of degrading every ranking.
- **`hash()` is salted per process.** An embedder built on it is deterministic within one
  interpreter — so a same-process test passes — and produces a different vector space in the
  next one, which would make an index built by `seed_corpus.py` disagree with every query the
  application later makes, presenting as poor relevance rather than as an error. `blake2b` is
  used instead, and the determinism test crosses a subprocess boundary because the same-process
  version passes either way.

## Alternatives rejected

**Qdrant in the default path.** Gives hybrid search and a realistic production story. Rejected
because it puts a service between a reader and a green test run, for a corpus that fits in
memory several thousand times over. It stays available as an optional Compose profile, and the
day the corpus justifies it, this ADR gets superseded rather than amended.

**A text-splitter library for chunking.** Rejected on a principle worth stating: chunking
decides what the model is allowed to see, so it is application logic, not plumbing to delegate.
Splitting on Markdown headings uses the author's own statement about where one idea ends;
fixed-width splitting cuts mid-sentence and is tuned by a number nobody can justify.

**Inverse document frequency instead of a stopword list.** IDF is the better technique and it
needs corpus statistics, which would make the embedder *fitted* — carrying state learned from
one corpus and silently wrong against another. The `Embeddings` interface has nowhere honest to
put that, and a stateful embedder that looks stateless is a worse problem than an imperfect
ranking on a lane whose purpose is to be free and deterministic.

**numpy for the dot product.** Rejected above. Worth naming separately because it is the kind
of dependency that arrives without discussion: it would have been added for one line of
arithmetic over twenty short vectors.
