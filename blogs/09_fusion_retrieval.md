# Fusion Retrieval: When Keyword Search and Semantic Search Become Allies

> **Technique:** Fusion Retrieval (Hybrid Search)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`, `rank_bm25`, `numpy`

---

## Introduction

Dense vector search (semantic retrieval) revolutionized information retrieval. Suddenly, "dog" could match "canine", and "cardiac arrest" could surface documents about "heart attack". But in gaining semantic understanding, dense retrieval gave up something older and more reliable: **exact keyword matching**.

Type a rare product code, a person's full name, or a specific technical term into a pure dense retrieval system, and it may retrieve documents that are "semantically similar" but don't actually contain your search term. BM25 — the gold standard sparse retrieval algorithm — never makes this mistake. It finds exact and near-exact keyword matches with high recall.

**Fusion Retrieval** (also called Hybrid Search) combines both worlds: the conceptual understanding of dense retrieval with the precision of BM25 keyword search. The result is a retrieval system that handles both "what documents are *about* this topic?" and "which documents *contain* this exact term?" simultaneously.

---

## Dense vs. Sparse: A Tale of Two Strengths

### Dense Retrieval (Vector Search)
- **Strength**: Semantic understanding, synonym matching, conceptual lookup
- **Weakness**: May miss documents with exact rare terms; can retrieve thematically-related but irrelevant docs
- **Example**: "ML model training" → finds documents about "neural network optimization"

### Sparse Retrieval (BM25)
- **Strength**: Exact keyword matching, rare term handling, technical precision
- **Weakness**: Misses synonyms, vocabulary mismatch, no conceptual understanding
- **Example**: "ML model training" → only finds documents containing those exact words

### Hybrid (Fusion)
Combines both signals for a result that neither alone could achieve. A query like "BERT fine-tuning on medical NER datasets" benefits from:
- BM25 to guarantee "BERT", "NER", "medical" are present in results
- Dense search to include semantically relevant papers that may use "named entity recognition" instead of "NER"

---

## How Fusion Retrieval Works

### The Alpha Parameter

The fusion is controlled by a single parameter `alpha ∈ [0, 1]`:

```
final_score = alpha × normalized_vector_score + (1 - alpha) × normalized_bm25_score
```

| `alpha` | Behavior |
|---------|---------|
| `1.0` | Pure dense retrieval (identical to Simple RAG) |
| `0.0` | Pure BM25 retrieval |
| `0.5` | Equal weight to both |
| `0.7` | Favors semantic search (good default for general Q&A) |
| `0.3` | Favors keyword search (good for technical/code search) |

### Score Normalization

Before combining, both score sets must be normalized to the same scale [0, 1]:

```python
def normalize_scores(scores: List[float]) -> List[float]:
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]
```

Without normalization, BM25 scores (which can be large floats like 15.7) would dominate cosine similarity scores (which are bounded between -1 and 1). Min-max normalization puts both on equal footing before the alpha-weighted combination.

---

## Code Deep Dive

### Initializing Both Retrievers

```python
class FusionRetriever:
    def __init__(self, embedding_model="text-embedding-3-small",
                 chunk_size=1000, chunk_overlap=200, 
                 k=3, alpha=0.5):
        self.alpha = alpha
        self.k = k
        
        # Dense retrieval components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        
        # Sparse retrieval: BM25 operates on tokenized documents
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_chunks: List[str] = []  # original text for result lookup
```

### Index Building

Both indices must be built from the same chunks:

```python
def index_document(self, text: str) -> int:
    chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
    self.bm25_chunks = chunks
    
    # Build FAISS index (dense)
    documents = [Document(content=c, metadata={"chunk_index": i})
                 for i, c in enumerate(chunks)]
    documents = self.embedder.embed_documents(documents)
    self.vector_store.add_documents(documents)
    
    # Build BM25 index (sparse — tokenize by whitespace)
    tokenized = [chunk.lower().split() for chunk in chunks]
    self.bm25 = BM25Okapi(tokenized)
    
    return len(chunks)
```

BM25 tokenization is deliberately simple (lowercase + split). More sophisticated tokenization (stemming, stop word removal) can improve BM25 performance further but adds complexity.

### The Fusion Retrieval Step

```python
def retrieve(self, query: str) -> List[str]:
    # ── Dense retrieval (search all, not just top-k) ──
    query_embedding = self.embedder.embed_text(query)
    dense_results = self.vector_store.search(query_embedding, k=len(self.bm25_chunks))
    
    # Map chunk index → dense score
    dense_scores = {r.document.metadata["chunk_index"]: r.score 
                   for r in dense_results}
    
    # ── Sparse retrieval (BM25) ──
    bm25_scores_raw = self.bm25.get_scores(query.lower().split())
    
    # ── Normalize both score sets ──
    dense_values = [dense_scores.get(i, 0.0) for i in range(len(self.bm25_chunks))]
    dense_norm = normalize_scores(dense_values)
    bm25_norm = normalize_scores(bm25_scores_raw.tolist())
    
    # ── Combine with alpha weighting ──
    combined = []
    for i in range(len(self.bm25_chunks)):
        fused_score = self.alpha * dense_norm[i] + (1 - self.alpha) * bm25_norm[i]
        combined.append((i, fused_score))
    
    # ── Sort and return top-k ──
    combined.sort(key=lambda x: x[1], reverse=True)
    return [self.bm25_chunks[i] for i, _ in combined[:self.k]]
```

Key implementation detail: dense retrieval is done for **all** chunks (not just top-k), so we have a complete score map to combine with BM25.

---

## BM25: How It Works

BM25 (Best Matching 25) is a probabilistic ranking function. For a query with terms `t1, t2, ..., tn` and document `D`:

```
BM25(D, Q) = Σ IDF(ti) × (tf(ti,D) × (k1+1)) / (tf(ti,D) + k1 × (1 - b + b × |D|/avgdl))
```

Where:
- **IDF**: Inverse Document Frequency — rare terms get higher weight
- **tf**: Term Frequency in document
- **k1**: Term frequency saturation parameter (default ~1.5)
- **b**: Length normalization (default ~0.75)
- **|D|/avgdl**: Document length relative to corpus average

In plain English: BM25 scores highly documents that contain the query terms many times (but with diminishing returns), especially if those terms are rare across the corpus, and adjusts for document length.

---

## Choosing Alpha: A Practical Guide

Alpha tuning depends on query patterns and corpus characteristics:

| Query Type | Recommended Alpha | Reason |
|-----------|------------------|--------|
| General Q&A ("How does X work?") | 0.7 | Semantic understanding dominates |
| Technical lookup ("Error code 0x8007000E") | 0.2 | Exact term matching critical |
| Named entities ("Elon Musk SpaceX 2024") | 0.4 | Name matching critical; context helpful |
| Conceptual ("implications of climate policy") | 0.8 | Pure semantic; no exact terms needed |
| Code search ("numpy vectorized operations") | 0.3 | Keyword precision matters |

**A/B testing**: If you have evaluation queries, measure Recall@k or MRR for alpha values [0.2, 0.4, 0.5, 0.6, 0.8] and pick the best-performing value for your corpus.

---

## Reciprocal Rank Fusion (RRF): An Alternative Approach

Rather than score normalization + weighted average, some implementations use **Reciprocal Rank Fusion**:

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
```

For each chunk, take its rank in the dense results and its rank in BM25 results, compute the RRF score for each, and sum them. RRF is threshold-free (no alpha tuning) and robust to outlier scores. It's worth considering as an alternative if you want to avoid hyperparameter tuning.

---

## When to Use Fusion Retrieval

**Best for:**
- Enterprise search over technical documentation, code, or product catalogs
- Queries that mix conceptual terms (semantic) with exact identifiers (keyword)
- Corpora with high technical vocabulary where synonyms are less common
- Applications where missing an exact-match document is unacceptable

**Less critical when:**
- Corpus uses highly consistent, standardized vocabulary
- All queries are purely conceptual (definitions, explanations)
- Simplicity is valued over marginal quality gains

---

## Summary

Fusion Retrieval is one of the most impactful and well-understood improvements to the Simple RAG baseline. By combining dense embeddings with BM25 through a weighted score fusion, you get a retrieval system that:

- **Never misses exact keyword matches** (thanks to BM25)
- **Always finds conceptually relevant content** (thanks to dense retrieval)
- **Adapts to query type** through the alpha parameter

In evaluation after evaluation, hybrid search outperforms either pure method by significant margins (5-15% in recall and precision). For production systems serving diverse user populations with varied query styles, this should be a standard component.
