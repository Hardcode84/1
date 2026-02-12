# Importance & Novelty Scoring

Principled approaches to ranking text chunks by importance, grounded in information theory and statistics.

## Core duality

**Importance and novelty are orthogonal axes:**
- A chunk can be novel but unimportant (off-topic tangent).
- A chunk can be important but not novel (restating known facts).
- We want chunks that are both: central to the content AND new relative to prior knowledge.

## Novelty methods

### Information-theoretic surprise
- Shannon self-information: `I(x) = -log P(x)`.
- The less probable an observation, the more information it carries.
- Foundation of IDF: rare terms = high information content.

### Embedding-based density estimation
- `novelty = 1 - max(cos_sim(embed(x), embed(known_i)))`.
- Nearest-neighbor approximation of probability density in embedding space.
- Low-density regions (far from known points) = high surprise.
- Pros: works with our existing embedding infrastructure.
- Cons: conflates "novel" with "off-topic" without a centrality correction.

### Bayesian surprise
- KL divergence between prior and posterior: `D_KL(P_posterior || P_prior)`.
- Measures how much an observation changes beliefs.
- In practice: how much does adding a chunk shift the centroid or distribution of the knowledge base?

### Compression-based
- Normalized Compression Distance (NCD): approximates Kolmogorov complexity.
- A chunk is novel if it can't be compressed given existing context.
- `NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))` where C = compressed size.

## Importance / centrality methods

### TextRank / LexRank
- Build similarity graph over chunks (edges = cosine similarity above threshold).
- Run PageRank. High-rank nodes = central/representative chunks.
- Captures "what is this collection of text mainly about?"

### MMR (Maximal Marginal Relevance)
- Iterative selection: `argmax [lambda * sim(chunk, query) - (1 - lambda) * max(sim(chunk, selected))]`.
- Balances relevance to a query with diversity from already-selected items.
- Good for building non-redundant summaries.

### TF-IDF / BM25
- Term frequency x inverse document frequency.
- Terms frequent in this document but rare globally are characteristic.
- BM25 adds saturation (diminishing returns for repeated terms) and length normalization.
- Can be applied at chunk level: chunks with high characteristic-term density are more important.

### Graph centrality variants
- Degree centrality: how many other chunks is this similar to?
- Betweenness centrality: does this chunk bridge different topics?
- Eigenvector centrality: is this chunk similar to other important chunks? (= PageRank)

## Practical considerations

- Embedding similarity is the cheapest and most aligned with our existing infrastructure.
- TextRank adds one matrix operation (similarity matrix + PageRank iteration).
- BM25 is already implemented in the memory store (FTS5).
- LLM-based scoring is most accurate but expensive (API call per chunk).
- Start simple (embedding novelty), add centrality if results are noisy.
