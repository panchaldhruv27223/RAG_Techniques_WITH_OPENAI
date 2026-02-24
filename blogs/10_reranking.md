# Reranking: Precision as a Second Act

## Introduction

Vector similarity search is fast but imprecise. FAISS can scan millions of vectors in milliseconds using a dot product computation — but cosine similarity is a blunt instrument. Two texts with a cosine similarity of 0.82 might both be about the same general topic, but one might actually answer the query while the other is only thematically adjacent. FAISS has no way to distinguish them with a single embedding comparison.

**Reranking** introduces a second, more powerful evaluation stage after initial retrieval. The pipeline becomes:

1. **Stage 1 — Recall-focused retrieval**: FAISS retrieves a large candidate set (top-20, top-50) using fast vector search. Recall is prioritized — the goal is to ensure the relevant document is *somewhere* in the candidate set.
2. **Stage 2 — Precision-focused reranking**: A more powerful model evaluates each candidate against the specific query and reorders them. The goal is to ensure the relevant document is at the *top* of the final list.

This two-stage architecture — **ANN retrieval + learned reranker** — is used by all major search engines and retrieval systems at production scale. Google, Bing, Elasticsearch, and Cohere all implement this pattern.

---

## Why Single-Stage Retrieval Has a Precision Ceiling

### The Embedding Representation Problem

When you embed a text, you create a single vector that represents *everything* the text is about. This aggregation is both the strength and weakness of dense retrieval.

Consider query: "What are the side effects of aspirin when taken with ibuprofen?"

This query has multiple facets: aspirin, ibuprofen, side effects, drug interaction, co-administration. One 1536-dim vector must represent all of them simultaneously.

Now consider two chunks:
- Chunk A: "Aspirin and NSAIDs like ibuprofen can cause GI bleeding when co-administered. Patients should be warned of increased risk of gastrointestinal adverse events including ulceration."
- Chunk B: "Aspirin is widely used for cardiovascular protection. Common aspirin side effects include tinnitus, GI upset, and bleeding. Ibuprofen reduces the cardioprotective effects of aspirin."

Both chunks are semantically relevant (contain aspirin + ibuprofen + side effects). Their embeddings are close to the query embedding. But Chunk A *directly* answers "what are the side effects when taken together," while Chunk B answers a different sub-question. A dot-product computation can't discriminate between these.

A **cross-encoder reranker** reads the full query-document pair as a unit and outputs a relevance score that accounts for the *specific relationship* between that query and that document — not just their proximity in embedding space.

---

## Two Reranking Strategies

This implementation provides two reranker strategies:

### Strategy 1: LLM-Based Reranking

```python
class LLMReranker:
    """
    Uses a chat LLM (e.g. gpt-4o-mini) as the reranker.
    
    Sends query + document as a prompt and asks for a relevance score.
    More flexible but more expensive per document than a cross-encoder.
    """
    
    def score_relevance(self, query: str, document: str) -> float:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise relevance scorer for information retrieval. "
                    "Score how relevant the document is to answering the query.\n\n"
                    "Return JSON: {\"relevance_score\": <float>}\n"
                    "Score meaning:\n"
                    "  1.0 = Document directly and completely answers the query\n"
                    "  0.8 = Document substantially addresses the query\n"
                    "  0.6 = Document partially addresses the query\n"
                    "  0.4 = Document is tangentially related\n"
                    "  0.2 = Document is on the same topic but doesn't address the query\n"
                    "  0.0 = Document is irrelevant to the query"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Document (first 2000 chars):\n{document[:2000]}\n\n"
                    "Score this document's relevance to the query:"
                )
            }
        ]
        result = self.llm.chat_json(messages)
        score = result.get("relevance_score", 0.5)
        return max(0.0, min(1.0, float(score)))
```

**When to use the LLM reranker:**
- No GPU available for running local cross-encoder models
- Queries require complex NLU (multi-hop reasoning, implication, negation)
- Domain-specific scoring criteria need to be expressed in natural language
- Integration simplicity matters more than per-query cost

**Cost**: For k=20 initial candidates, the LLM reranker makes 20 API calls. At `gpt-4o-mini` pricing (~$0.00015/1K input tokens), with ~500 tokens per call, that's ~$0.0015 per query. Acceptable for most enterprise use cases.

### Strategy 2: Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Uses a local cross-encoder model to score query-document pairs.
    
    Cross-encoders jointly encode query + document, allowing full
    bidirectional attention across the pair. This gives much better
    nuanced relevance scoring than separate bi-encoder embeddings.
    
    No GPU required for inference on short documents (<512 tokens).
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # ms-marco models are trained specifically for passage relevance scoring
        # Available sizes: L-2 (tiny), L-4 (small), L-6 (default), L-12 (large)
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        # Cross-encoder takes (query, document) pairs simultaneously
        pairs = [(query, doc[:512]) for doc in documents]  # truncate to model limit
        
        # Single forward pass scores all pairs
        scores = self.model.predict(pairs)
        
        # Return (document, score) sorted by score descending
        scored = list(zip(documents, scores))
        return sorted(scored, key=lambda x: x[1], reverse=True)
```

**Cross-encoders vs. bi-encoders (the key distinction):**

| | Bi-encoder (FAISS) | Cross-encoder (reranker) |
|-|--------------------|--------------------------|
| Encoding | Query and document encoded *separately* | Query and document encoded *jointly* |
| Index time | Documents pre-encoded, stored | Not needed |
| Query time | One query encoding + dot products | One forward pass per document |
| Attention | No cross-attention between Q and D | Full bidirectional Q↔D attention |
| Speed | Sub-millisecond for millions of docs | ~50ms per doc (CPU), ~2ms (GPU) |
| Precision | Good | **Excellent** |
| Scalability | To hundreds of millions | Scales to hundreds of candidates |

The bi-encoder cannot let the query "attend" to specific parts of the document — each is encoded in isolation. The cross-encoder can. When the model reads "aspirin and ibuprofen drug interaction" it can attend to "interaction" in the query while reading "co-administered" in the document — understanding the connection. This is the source of cross-encoders' superior precision.

**Recommended model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` is the standard production choice — excellent quality, fast CPU inference (~50ms per candidate), open source, no API key needed. It was trained on the MS-MARCO passage reranking dataset with ~500K annotated query-passage pairs.

---

## The Full Two-Stage RAG Pipeline

```python
class RerankingRAG:
    def query(
        self,
        question: str,
        initial_k: int = 20,    # Stage 1: over-retrieve
        final_k: int = 3,       # Stage 2: top-k after reranking
        reranker_type: str = "cross_encoder"  # "llm" or "cross_encoder"
    ) -> Tuple[str, List[str]]:
        
        # Stage 1: Fast vector retrieval — maximize recall, ignore precision
        question_embedding = self.embedder.embed_text(question)
        initial_results = self.vector_store.search(question_embedding, k=initial_k)
        initial_chunks = [r.document.content for r in initial_results]
        
        print(f"Stage 1: Retrieved {len(initial_chunks)} candidates via FAISS")
        
        # Stage 2: Precision reranking — score each candidate for query relevance
        if reranker_type == "cross_encoder":
            reranked = self.cross_encoder_reranker.rerank(question, initial_chunks)
        else:
            reranked = self.llm_reranker.rerank(question, initial_chunks)
        
        # Take only top final_k after reranking
        top_chunks = [chunk for chunk, score in reranked[:final_k]]
        top_scores = [score for chunk, score in reranked[:final_k]]
        
        print(f"Stage 2: Reranked to top-{final_k}")
        print(f"Top reranking scores: {[f'{s:.3f}' for s in top_scores]}")
        
        # Generate answer from precisely selected context
        answer = self._generate_answer(question, top_chunks)
        
        return answer, top_chunks
```

---

## The Precision-Recall Trade-off in Stage 1

How large should `initial_k` be? This is the central configuration question.

**Too small** (e.g., initial_k=5): If the truly relevant document is at rank 6, the reranker never sees it. Precision improves within the candidate set, but recall was already capped by Stage 1.

**Too large** (e.g., initial_k=100): The reranker scores 100 documents, but for LLM reranking this means 100 API calls per query — expensive. For cross-encoder reranking, 100 documents × 50ms/doc = 5 seconds of reranking latency.

**Rule of thumb for initial_k:**
```
initial_k = max(20, final_k × 5)
```

For `final_k=3`: Use `initial_k=20` (5× ratio)  
For `final_k=5`: Use `initial_k=25` (5× ratio)  
For `final_k=10`: Use `initial_k=50` (5× ratio)

The 5× ratio gives a comfortable recall buffer while keeping reranking compute manageable.

---

## Worked Example: Where Reranking Changes the Ranking

**Query**: "What is the maximum safe daily dose of acetaminophen?"

**Stage 1 FAISS results** (ordered by cosine similarity):

| Rank | Chunk summary | FAISS sim |
|------|---------------|-----------|
| 1 | General acetaminophen overview, uses, mechanism | 0.83 |
| 2 | **Maximum 4g/day for healthy adults, 2g for liver patients** | 0.81 |
| 3 | Acetaminophen overdose symptoms and treatment | 0.79 |
| 4 | Comparison of acetaminophen vs NSAIDs for pain | 0.76 |
| 5 | Pediatric dosing guidelines for acetaminophen | 0.74 |

The most relevant chunk (Rank 2 — the one containing "maximum 4g/day") is at position 2, *not* position 1. The LLM would use the Rank 1 chunk first, which gives a general overview but no dosing limit.

**Stage 2 Cross-Encoder Reranking:**

| Rank | Chunk summary | Reranker score |
|------|---------------|---------------|
| 1 | **Maximum 4g/day for healthy adults, 2g for liver patients** | 0.94 |
| 2 | General acetaminophen overview | 0.61 |
| 3 | Pediatric dosing guidelines | 0.58 |
| 4 | Comparison vs NSAIDs | 0.40 |
| 5 | Overdose symptoms | 0.35 |

The cross-encoder correctly promotes Rank 2→1 because it reads "maximum safe daily dose" in the query and "maximum 4g/day" in the chunk — the specific connection that FAISS's dot product couldn't capture.

---

## Cost and Latency Analysis

| Configuration | Latency | Cost per query |
|--------------|---------|---------------|
| FAISS only (k=3) | 600ms | $0.0001 |
| FAISS (k=20) + LLM rerank → 3 | 7-15s | $0.003 |
| FAISS (k=20) + Cross-encoder → 3 (CPU) | 1.5-2.5s | $0.0001 (local model) |
| FAISS (k=20) + Cross-encoder → 3 (GPU) | 650ms | $0.0001 (local model) |

Cross-encoder reranking on CPU is the sweet spot for most deployments:
- 2-3× better precision than FAISS-only (documented on MS-MARCO benchmarks)
- Adds only ~1-2 seconds latency
- Zero API cost (model runs locally)
- No data leaves your infrastructure

---

## Advanced: Score Combination

Instead of discarding FAISS scores after reranking, combine them:

```python
# Weighted combination of FAISS similarity + reranking score
FAISS_WEIGHT = 0.3
RERANK_WEIGHT = 0.7

final_scores = []
for (chunk, rerank_score), (_, faiss_score) in zip(reranked_docs, faiss_results):
    combined = FAISS_WEIGHT * faiss_score + RERANK_WEIGHT * rerank_score
    final_scores.append((chunk, combined))

final_scores.sort(key=lambda x: x[1], reverse=True)
```

The FAISS score captures broad semantic alignment; the reranker score captures specific relevance. Combining them retains both signals and sometimes outperforms reranker-only ranking.

---

## When to Use Reranking

Reranking is worth adding whenever answer quality is the primary success metric and the system can absorb 1–2 extra seconds of latency. The cross-encoder reranker in particular is an exceptional value proposition: it runs locally (no API cost), adds only modest latency on CPU, and produces measurably better precision than FAISS-alone. If your initial retrieval consistently surfaces the right topic but the most directly relevant chunk isn't landing at rank 1, reranking is the fix.

It's especially valuable in domains that demand precise, exact answers — medical dosing, legal citations, technical specifications — where the difference between the best and second-best retrieved chunk can meaningfully change the generated answer. In these contexts, the cost of a wrong or imprecise answer is high, and the reranker's ability to read query-document pairs with full attention is exactly what's needed.

Reraking is harder to justify when sub-second responses are a hard requirement and no GPU is available, or when your initial FAISS precision is already very high (P@3 above 0.85) and there's little room to improve. For very small corpora — under a hundred chunks — FAISS already returns precise results and the reranking overhead adds cost without meaningful gain.

---

## Summary

Reranking transforms RAG from a single-pass retrieval into a carefully staged precision pipeline. Stage 1 — fast, recall-focused vector search — ensures that the relevant document is in the candidate set. Stage 2 — precise, computationally intense cross-encoder scoring — ensures that the relevant document rises to the top of the final list.

The key insight is that FAISS and cross-encoders excel at different tasks: FAISS at scale, cross-encoders at nuanced precision. Combining them — using FAISS to reduce the candidate space from millions to hundreds, then cross-encoder to reduce from hundreds to three — gives you both scale and precision that neither achieves alone.

For production RAG systems where answer quality is the primary success metric, two-stage reranking is one of the highest-value architectural decisions you can make.
