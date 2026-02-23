# Reranking: A Second Opinion on What's Actually Relevant

> **Technique:** Reranking (Two-Stage Retrieval)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`, `sentence-transformers`

---

## Introduction

First-stage retrieval is fast but imperfect. Vector search retrieves chunks based on embedding similarity — a useful proxy for relevance, but not the same thing. Similarity in embedding space doesn't always equal relevance to the query. A chunk might be topically related without containing the answer. Another chunk might be stylistically different but contain the exact information needed.

**Reranking** adds a second stage that re-examines the initial candidates with a more powerful (and slower) relevance model. Think of it as a specialist reviewing the work of a generalist: the first stage casts a wide net quickly; the second stage applies deeper judgment to determine which catches were actually valuable.

This two-stage architecture — broad retrieval followed by precise reranking — is the foundation of most enterprise-grade search systems.

---

## The Two-Stage Architecture

```
User Query
    ↓
Stage 1: Vector Search
  → Fast: milliseconds for millions of chunks
  → Broad: k_initial = 10-20 candidates retrieved
  → Metric: Cosine similarity (approximate relevance)
    ↓
Stage 2: Reranking
  → Slow: seconds per candidate
  → Precise: "Is this candidate truly relevant to this query?"
  → Metric: Cross-attention relevance score (precise relevance)
    ↓
Top-k_final reranked results (e.g., k_final=3)
    ↓
LLM Answer Generation
```

The key insight: it's computationally infeasible to run the expensive reranker on every chunk in the corpus. But running it on 10-20 pre-screened candidates is perfectly practical.

---

## Two Reranking Strategies

The implementation supports two reranking methods, each with different trade-offs:

### Strategy 1: LLM Reranker

Uses GPT-4o-mini to score each candidate on a 0-10 relevance scale.

```python
class LLMReranker:
    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        scored = []
        for chunk in chunks:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Score the relevance of the document to the query on a scale of 0-10. "
                        "10 = perfectly answers the query. 0 = completely irrelevant. "
                        'Return JSON: {"score": <int>}'
                    )
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocument: {chunk}"
                }
            ]
            result = self.llm.chat_json(messages)
            score = float(result.get("score", 0))
            scored.append((chunk, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]
```

**Pros:**
- No additional model to download or run locally
- Can understand complex relevance criteria
- Flexible — you can customize the scoring criteria in the prompt

**Cons:**
- 1 API call per candidate (10-20 calls per query)
- Higher latency (seconds of added delay)
- API cost scales with candidates

### Strategy 2: Cross-Encoder Reranker (local model)

Uses a locally-run cross-encoder model from `sentence-transformers`, such as `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)  # one forward pass, all candidates
        
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
```

**Pros:**
- All candidates scored in a **single forward pass** (batched efficiency)
- Low latency after model loads (milliseconds per query)
- No API cost after initial download
- Cross-encoders are highly accurate for passage ranking tasks

**Cons:**
- Model download required (~90MB for MiniLM variants)
- Requires local compute
- Fixed model capability (can't be prompted for custom behavior)

---

## Why Cross-Encoders Outperform Bi-Encoders for Reranking

The standard embedding models (bi-encoders) encode queries and documents *independently* — the query and document never "see" each other during encoding. Relevance is computed after-the-fact via cosine similarity.

Cross-encoders process the query and document *together* through the same attention mechanism. The model can directly attend to which tokens in the document are most relevant to which tokens in the query. This joint attention is far more accurate for relevance assessment but requires a separate pass for every (query, document) pair — making it too slow for first-stage retrieval but ideal for reranking a small candidate set.

```
Bi-encoder:   embed(query) → q_vec   embed(doc) → d_vec   cosine(q_vec, d_vec)
                                                            ↑ indirect comparison

Cross-encoder: cross_encode(query, doc) → relevance_score
                                          ↑ direct joint attention
```

---

## Practical Example

**Query**: "What is the impact of sleep deprivation on cognitive performance?"

**Initial retrieval** (cosine similarity, top-5):
1. "Sleep disorders affect millions of Americans annually." (score: 0.82)  
2. "Cognitive performance declines with age and poor nutrition." (score: 0.79)  
3. "Sleep deprivation reduces reaction time by 23% in standardized tests." (score: 0.78)  
4. "Memory consolidation occurs primarily during REM sleep." (score: 0.77)  
5. "Caffeine can temporarily mask sleep deprivation effects." (score: 0.75)  

**After reranking** (by LLM/cross-encoder, top-3):
1. "Sleep deprivation reduces reaction time by 23% in standardized tests." ← most relevant
2. "Cognitive performance declines with age and poor nutrition." ← partial relevance
3. "Memory consolidation occurs primarily during REM sleep." ← relevant (indirect)

Chunks 1 and 5 from the initial retrieval, while topically adjacent, don't actually answer the question about *impact on cognitive performance*. Reranking demotes them correctly.

---

## Configuration

```python
class RerankingRAG:
    def __init__(self,
                 file_path: str,
                 k_initial: int = 10,    # broad first-stage retrieval
                 k_final: int = 3,       # after reranking
                 reranker_type: str = "llm",  # or "cross_encoder"
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
```

**Choosing k_initial**: A good rule of thumb is `k_initial = k_final × 3` to `k_final × 5`. Retrieving 10-20 candidates with 3 final results gives the reranker enough to work with while keeping reranking time manageable.

**Choosing reranker type:**

| Scenario | Recommended |
|----------|-------------|
| No local GPU/CPU budget | `llm` |
| Latency-sensitive production | `cross_encoder` |
| Custom relevance criteria | `llm` (with custom prompt) |
| High query volume | `cross_encoder` (no API cost) |

---

## Reranking + Other Techniques

Reranking is *composable* with other RAG improvements:

- **Reranking + Fusion Retrieval**: First fuse dense + BM25 scores for top-20, then rerank the 20 results to get top-3. The reranker benefits from the diverse candidate set.
- **Reranking + Contextual Compression**: Rerank first (for ordering), then compress each reranked chunk (for noise reduction).
- **Reranking + Context Window**: Rerank the small precise chunks, then expand winners to their neighbors before answer generation.

---

## Performance Impact

Reranking consistently improves metrics across RAG evaluation benchmarks:

| Metric | Without Reranking | With Reranking |
|--------|--------------------|----------------|
| Answer correctness | Baseline | +8-15% |
| Context precision | Baseline | +10-20% |
| Added latency | 0ms | 500ms–2s |
| API cost increase | 0 | +$0.001–0.01/query |

The latency increase depends heavily on `k_initial` and reranker type. Cross-encoder adds ~100-300ms for 10-20 candidates. LLM reranker adds 1-3 seconds.

---

## When to Use Reranking

**Best for:**
- Applications where answer quality is paramount and small latency increases are acceptable
- High-stakes domains (legal, medical, financial) where wrong retrieval is costly
- Corpora with many near-duplicate or topically adjacent chunks
- Queries with precise information needs

**Skip when:**
- Sub-100ms response times are required
- Corpus is small and well-structured enough that first-stage retrieval is already precise
- API costs are tightly constrained

---

## Summary

Reranking transforms RAG from single-stage retrieval into a two-stage precision pipeline. By applying a more powerful relevance model to a small set of pre-screened candidates, it corrects the inevitable imprecisions of embedding-based similarity search.

Whether you choose an LLM reranker (flexible, no setup) or a cross-encoder (fast, cost-efficient), the improvement to answer quality is consistent and measurable. For production RAG systems serving real users, reranking is one of the highest-ROI optimizations in the toolkit — delivering meaningful quality improvements at acceptable latency costs.
