# Dartboard RAG: Balancing Relevance and Diversity in Every Retrieval

---

## Introduction

In any dense or semantically redundant corpus, standard top-k retrieval returns near-duplicate results. If your document repeats similar concepts in multiple sections (which most real-world documents do), the top-3 chunks by cosine similarity will often be slight paraphrases of each other — giving the LLM three copies of the same context with no additional information.

**Dartboard RAG** addresses this through a principled algorithm that simultaneously optimizes for:
1. **Relevance** to the query
2. **Diversity** among selected chunks — no two selected chunks should contain redundant information

The algorithm is grounded in information theory: each selected chunk should provide **new information** not already present in the selected set. Like throwing darts — the first dart can land anywhere high-scoring; subsequent darts should land in *different* high-scoring regions, not cluster where you've already thrown.

---

## The Redundancy Problem, Quantified

Consider retrieving k=3 chunks for the query "causes of climate change":

**Standard top-3 result** (dense corpus, many overlapping sections):
1. "Greenhouse gases trap solar radiation, causing the Earth's temperature to rise..." (similarity: 0.91)
2. "The primary driver of climate change is the accumulation of greenhouse gases..." (similarity: 0.89)
3. "Human-produced greenhouse gas emissions are the main cause of global warming..." (similarity: 0.87)

All three chunks say essentially the same thing — the LLM gets no additional information from chunks 2 and 3 beyond what chunk 1 provided. **Token budget wasted; retrieval quality low.**

**Dartboard top-3 result**:
1. "Greenhouse gases trap solar radiation, causing temperature rise..." (most relevant)
2. "Land use changes, deforestation, and agricultural practices contribute..." (diverse: different cause)
3. "Volcanic eruptions and solar variability are natural factors, though..." (diverse: natural causes)

Three non-redundant perspectives; the LLM has genuinely richer context.

---

## The Algorithm: Log-Normal Probability + Greedy Selection

### Distance Space (not Similarity)

Dartboard works in *distance* space (how different two vectors are), not *similarity* space (how close). This is because diversity requires maximizing inter-document distance — which is natural in distance space.

```python
# Convert from cosine similarity to cosine distance
query_distances = 1.0 - np.dot(query_norm, cand_norm.T)        # (1, N)
document_distances = 1.0 - np.dot(cand_norm, cand_norm.T)      # (N, N)
```

- `query_distances[i]`: how far candidate i is from the query (lower = more relevant)
- `document_distances[i,j]`: how far candidate i is from candidate j (higher = more diverse)

### Log-Normal Probability Conversion

Raw distances are converted to log-probabilities using a Gaussian kernel:

```python
def lognorm(dist: np.ndarray, sigma: float) -> np.ndarray:
    """
    Convert distances to log-normal probabilities.
    
    log P(dist) = -log(σ) - 0.5 log(2π) - dist² / (2σ²)
    
    This is just the log of a Gaussian density function.
    
    Key intuition:
    - Small distance → high probability (close = likely relevant/diverse)
    - sigma (bandwidth): smaller σ = sharper peaks = more sensitive to distance differences
      larger σ = flatter distribution = treats similar distances as more equal
    
    Returning log-probabilities (not probabilities) allows numerically stable
    summation via logsumexp — avoids float underflow with tiny probabilities.
    """
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)
```

The `sigma` parameter controls selectivity:
- **Small sigma (0.05)**: Only very close neighbors get high scores — tight selection
- **Large sigma (0.3)**: Distances are smoothed out — selection is less discriminating

### The Greedy Dartboard Selection

```python
def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int,
    relevance_weight: float = 1.0,
    diversity_weight: float = 1.0,
    sigma: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """
    Greedy algorithm for jointly optimizing relevance + diversity.
    
    Step 1: Convert distances to log-probabilities
    Step 2: First pick = most relevant (ignores diversity — no comparison yet)
    Step 3: For each subsequent pick:
        a. For each candidate, compute:
           - diversity = max log-prob distance from ANY already-selected document
           - relevance = log-prob distance from query
           - combined = diversity_weight × diversity + relevance_weight × relevance
        b. Reduce across dimensions via logsumexp (numerically stable)
        c. Pick candidate with highest combined score
        d. Update tracking: max_distances updated with new selection
    """
    query_probs = lognorm(query_distances, sigma)   # relevance probabilities
    doc_probs = lognorm(document_distances, sigma)  # diversities as log-probs
    
    # Pick most relevant first
    most_relevant_idx = np.argmax(query_probs)
    selected_indices = [int(most_relevant_idx)]
    max_distances = doc_probs[most_relevant_idx].copy()  # diversity from first pick
    
    while len(selected_indices) < num_results:
        # For each candidate: max diversity from any selected doc + relevance
        updated_distances = np.maximum(max_distances, doc_probs)  # (N, N)
        combined = (
            updated_distances * diversity_weight
            + query_probs[np.newaxis, :] * relevance_weight
        )
        
        # Aggregate using logsumexp: numerically stable log(sum(exp(combined)))
        normalized = logsumexp(combined, axis=1)  # (N,)
        
        # Mask already-selected
        for idx in selected_indices:
            normalized[idx] = -np.inf
        
        # Pick best
        best_idx = int(np.argmax(normalized))
        max_distances = updated_distances[best_idx]
        selected_indices.append(best_idx)
    
    return [documents[i] for i in selected_indices], [...]
```

### logsumexp: The Numerical Stability Trick

```python
def logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute log(sum(exp(a))) stably: log(Σ exp(aᵢ)) = max(a) + log(Σ exp(aᵢ - max(a)))
    
    Why: exp(very_negative_number) underflows to 0 in float32.
    Factoring out the max prevents underflow.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(a_max) + np.log(np.sum(np.exp(a - a_max), axis=axis))
```

---

## Relevance vs. Diversity Weighting

The `relevance_weight` and `diversity_weight` parameters control the tradeoff:

```python
DartboardRAG(
    file_path="document.pdf",
    relevance_weight=1.0,   # weight on query-relevance score
    diversity_weight=1.0,   # weight on inter-document diversity
    k=5,
    sigma=0.1,
    oversampling=3,         # fetch k × 3 candidates before dartboard selection
)
```

**Tuning guide:**

| Configuration | When to Use | Effect |
|--------------|-------------|--------|
| `relevance=1.0, diversity=1.0` | Default — balanced | Each subsequent pick is a compromise |
| `relevance=2.0, diversity=1.0` | Dense corpus, high-precision needed | Stays near query — less exploration |
| `relevance=1.0, diversity=2.0` | Highly redundant corpus | Aggressively diverse, may sacrifice relevance |
| `relevance=3.0, diversity=0.0` | Essentially standard top-k | No diversity consideration |

---

## The Built-In Comparison Tool

`DartboardRAG` provides a `compare()` method that demonstrates the diversity benefit:

```python
rag = DartboardRAG(file_path="report.pdf", duplicate_factor=5)

rag.compare("What are the main causes of climate change?")

# Output:
# STANDARD TOP-5 (may have duplicates):
#   1. "Greenhouse gases trap radiation..." 
#   2. "Greenhouse gases trap radiation..." ← DUPLICATE
#   3. "Greenhouse gases trap radiation..." ← DUPLICATE
#   4. "Greenhouse gases trap radiation..." ← DUPLICATE
#   5. "The accumulation of CO2..."
#   Unique results: standard=2/5
#
# DARTBOARD TOP-5 (relevance + diversity):
#   1. "Greenhouse gases trap radiation..." [score=1.847]
#   2. "Deforestation reduces carbon absorption..." [score=1.203]
#   3. "Industrial emissions from manufacturing..." [score=0.941]
#   4. "Agricultural methane from livestock..." [score=0.887]
#   5. "Natural factors: volcanic activity and solar..." [score=0.762]
#   Unique results: dartboard=5/5
```

The `duplicate_factor=5` parameter replicates chunks to simulate a dense corpus — exactly the scenario where dartboard's diversity benefit is most visible.

---

## Dartboard vs. MMR (from Adaptive Retrieval)

Both Dartboard and MMR (Maximal Marginal Relevance) solve the diversity problem, but with different mathematical formulations:

| Dimension | Dartboard | MMR |
|-----------|-----------|-----|
| Scoring | Log-normal probabilities | Linear combination |
| Stability | Numerically stable (logsumexp) | Can suffer from float precision issues |
| Theoretical basis | Information gain (paper) | Marginal relevance (Carbonell & Goldstein 1998) |
| Sigma parameter | Controls distance sensitivity | λ parameter for relevance/diversity tradeoff |
| Oversampling | Yes (fetch 3× candidates) | Yes (fetch M, select k) |

MMR is simpler and well-established. Dartboard's log-normal formulation is better grounded mathematically and produces sharper diversity selection in practice.

---

## When to Use Dartboard RAG

Dartboard RAG adds the most value for documents with repetitive or overlapping sections — policy documents, long reports that recap key data in multiple places, and corpora indexed from multiple similar sources such as news articles covering the same event from different outlets. When standard top-k consistently returns near-duplicate results, Dartboard's diversity selection directly addresses the symptom.

For corpora with high natural diversity, the algorithm adds overhead — the log-normal scoring and greedy selection — with no observable benefit, since the initial top-k candidates are already distinct. When k=1, diversity is meaningless. And for queries so specific that there's genuinely only one relevant chunk in the corpus, Dartboard's diversity mechanism introduces no improvement and slight additional cost.

---

## Summary

Dartboard RAG wraps an otherwise standard FAISS retrieval in a principled diversity-promoting selection algorithm grounded in information theory. By converting cosine distances to log-normal probabilities and greedily selecting chunks that maximize the joint score of query-relevance and inter-chunk diversity, it guarantees that each retrieved chunk adds genuinely new information to the context window.

The practical result: for any corpus with overlapping or redundant content, the LLM receives richer, more diverse context — and the wasted token budget of near-duplicate chunks is recycled into coverage of different document regions. A direct application of "diminishing returns" thinking to retrieval: the second chunk should tell the LLM something the first chunk didn't.
