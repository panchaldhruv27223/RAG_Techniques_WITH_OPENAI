# Feedback Loop RAG: A Retrieval System That Gets Smarter with Use

---

## Introduction

Every RAG system deployment surfaces the same problem: user queries cluster predictably. A company's internal knowledge base might receive 80% of its queries from a handful of recurring question patterns. A medical information system might field similar symptom-management questions thousands of times. Each query is answered from scratch — the retrieval strategy doesn't improve from seeing the same question answered well before.

**Feedback Loop RAG** turns this repeated usage into a learning opportunity. It records which chunks are retrieved for each query, solicits (or infers) a usefulness signal for each result, and uses this accumulated signal to adjust retrieval behavior over time. Chunks that consistently help users get good answers are scored higher in future retrievals. Chunks that are consistently unhelpful are down-ranked.

The result is a retrieval system with *institutional memory* — one where quality compounds over time rather than remaining static.

---

## The Core Feedback Mechanism

### What Gets Tracked

For every query-chunk pair, the system records:

```python
@dataclass
class FeedbackRecord:
    query: str                   # original user query
    query_hash: str              # MD5 hash for lookup
    chunk_id: int                # which chunk was retrieved
    chunk_content: str           # chunk text (for verification)
    was_used: bool               # was this chunk in the final top-k?
    relevance_score: float       # LLM's assessed relevance (0-1)
    timestamp: str               # when this feedback was recorded
    session_id: str              # groups queries from the same session
```

### The Relevance Adjuster

The `LLMRelevanceAdjuster` automatically evaluates chunk-query relevance after retrieval and stores it as feedback — no explicit user rating required:

```python
class LLMRelevanceAdjuster:
    """
    Automatically generates relevance feedback using the LLM.
    
    In a production system, this could be replaced by:
    - Explicit user thumbs up/down ratings
    - Implicit signals (time spent reading, follow-up questions)
    - Click-through rates on retrieved chunks
    
    The auto-evaluation approach is the simplest to deploy and gives
    reasonable signal quality without any user interface changes.
    """
    
    def evaluate_chunk_relevance(self, query: str, chunk: str) -> float:
        """
        Score chunk relevance to query on 0-1 scale.
        This score is what gets stored as feedback.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Rate how relevant this document chunk is to the query. "
                    "Return only a number between 0.0 and 1.0.\n"
                    "0.0 = completely irrelevant\n"
                    "0.5 = somewhat relevant\n"
                    "1.0 = perfectly answers the query"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nChunk:\n{chunk[:1500]}\n\nRelevance score:"
            }
        ]
        score_text = self.llm.chat(messages).strip()
        try:
            return float(re.findall(r'\d+\.?\d*', score_text)[0])
        except (IndexError, ValueError):
            return 0.5  # fallback neutral score
```

---

## The Feedback Store

Feedback is persisted to disk (JSON) so it survives restarts and accumulates across sessions:

```python
class FeedbackStore:
    def __init__(self, feedback_file: str = "retrieval_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data: Dict[str, List[FeedbackRecord]] = {}
        self._load()
    
    def _load(self):
        """Load existing feedback from disk on startup."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
            print(f"Loaded {self._count()} feedback records")
    
    def add_feedback(self, record: FeedbackRecord):
        """Add a new feedback record for a query-chunk pair."""
        if record.query_hash not in self.feedback_data:
            self.feedback_data[record.query_hash] = []
        self.feedback_data[record.query_hash].append(asdict(record))
        self._save()
    
    def get_chunk_score(self, query_hash: str, chunk_id: int) -> Optional[float]:
        """
        Return the historical average relevance score for a chunk
        when retrieved for queries similar to the given query.
        
        Returns None if no historical data exists for this pair.
        """
        records = self.feedback_data.get(query_hash, [])
        chunk_records = [r for r in records if r["chunk_id"] == chunk_id]
        
        if not chunk_records:
            return None
        
        return sum(r["relevance_score"] for r in chunk_records) / len(chunk_records)
    
    def get_adjusted_k(self, query_hash: str) -> int:
        """
        Suggest how many chunks to retrieve based on historical performance.
        
        If past retrievals for similar queries consistently found that 1-2 chunks
        were highly relevant, reduce k to minimize noise.
        If past retrievals showed that many chunks were relevant (broad topics),
        increase k to capture full breadth.
        """
        records = self.feedback_data.get(query_hash, [])
        if len(records) < 5:  # not enough data
            return None
        
        avg_scores = [r["relevance_score"] for r in records]
        high_relevance_ratio = sum(1 for s in avg_scores if s > 0.7) / len(avg_scores)
        
        if high_relevance_ratio > 0.6:
            return 2  # many relevant chunks — narrow k
        elif high_relevance_ratio < 0.3:
            return 5  # few relevant chunks — broaden k
        return 3  # default
```

---

## Score Adjustment at Query Time

When a new query arrives, its hash is computed and the feedback store is checked for historical relevance scores for retrieved chunks:

```python
def query(self, question: str) -> Tuple[str, QueryResult]:
    """Query with feedback-adjusted retrieval."""
    
    query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    # Step 1: Standard FAISS retrieval
    question_embedding = self.embedder.embed_text(question)
    initial_results = self.vector_store.search(question_embedding, k=self.k * 2)
    
    # Step 2: Score adjustment using historical feedback
    scored_results = []
    for result in initial_results:
        chunk_id = result.document.metadata.get("chunk_index", 0)
        
        # Get adjusted score: combine cosine similarity + historical feedback
        faiss_score = result.score  # raw cosine similarity
        feedback_score = self.feedback_store.get_chunk_score(query_hash, chunk_id)
        
        if feedback_score is not None:
            # Blend: 60% FAISS + 40% historical feedback
            adjusted_score = 0.6 * faiss_score + 0.4 * feedback_score
            print(f"  Chunk {chunk_id}: FAISS={faiss_score:.3f}, Feedback={feedback_score:.3f}, Adjusted={adjusted_score:.3f}")
        else:
            # No historical data — use FAISS score only
            adjusted_score = faiss_score
        
        scored_results.append((result, adjusted_score))
    
    # Step 3: Re-rank by adjusted score, take top-k
    scored_results.sort(key=lambda x: x[1], reverse=True)
    final_results = [result for result, _ in scored_results[:self.k]]
    
    # Step 4: Generate answer
    contexts = [r.document.content for r in final_results]
    answer = self._generate_answer(question, contexts)
    
    # Step 5: Record feedback for future improvement
    self._record_feedback(question, query_hash, final_results)
    
    return answer, contexts
```

### The Scoring Weight (0.6 / 0.4)

The 60% FAISS / 40% feedback weighting is a deliberate balance:

- **Too much weight on feedback**: The system over-learns from past patterns and becomes inflexible. A chunk consistently helpful for "Q3 revenue" queries will be over-promoted for a different question that uses similar words.
- **Too much weight on FAISS**: Feedback has no effect — the system doesn't learn.
- **0.6 / 0.4 split**: FAISS remains the primary signal (semantic matching matters most) while feedback provides a meaningful adjustment for recognizably repeated query patterns.

---

## The Compounding Improvement Effect

Consider a knowledge base that's been live for 6 months, receiving 200 queries/day. By month 6:

| Query Type | Data Points | Feedback Confidence | Expected Improvement |
|-----------|-------------|-------------------|---------------------|
| New query (never seen) | 0 | None | Baseline FAISS quality |
| Recurring query (seen 50×) | 50 | High | 20-35% improvement in Precision@3 |
| Similar-pattern query (indirect match) | 10-20 | Moderate | 10-20% improvement |

The system becomes a **tacit expert** on its specific user base. It learns which chunks your specific users find useful for your specific query patterns — knowledge that no amount of general embedding model tuning would provide.

---

## Similarity-Based Feedback Transfer

A key design question: should historical feedback be applied *only* for queries with the exact same hash, or should it transfer to similar queries?

The simple implementation uses MD5 hashing — only exact same queries benefit from previous feedback. For real deployments, **semantic query matching** transfers feedback across similar queries:

```python
def get_similar_query_feedback(
    self, 
    query_embedding: List[float], 
    similarity_threshold: float = 0.85
) -> Dict[int, float]:
    """
    Find feedback records from similar past queries and aggregate their feedback.
    
    This allows: "What is Q3 profit?" to benefit from feedback recorded for
    "What was the Q3 revenue?" — semantically similar, different wording.
    """
    aggregated_feedback = {}
    
    for stored_query_hash, records in self.feedback_data.items():
        # Compare stored query embedding to current query
        stored_embedding = self.query_embedding_cache.get(stored_query_hash)
        if stored_embedding is None:
            continue
        
        similarity = cosine_similarity(query_embedding, stored_embedding)
        
        if similarity >= similarity_threshold:
            for record in records:
                chunk_id = record["chunk_id"]
                if chunk_id not in aggregated_feedback:
                    aggregated_feedback[chunk_id] = []
                # Weight by query similarity — closer queries have more influence
                aggregated_feedback[chunk_id].append(record["relevance_score"] * similarity)
    
    return {
        chunk_id: sum(scores) / len(scores)
        for chunk_id, scores in aggregated_feedback.items()
    }
```

---

## Cold Start and Warm State

| System State | Feedback Data | Retrieval Behavior |
|-------------|--------------|-------------------|
| Cold start (new deployment) | None | Pure FAISS quality — identical to standard RAG |
| Warm (100-500 records) | Sparse | Modest improvement on most common queries |
| Hot (1000+ records) | Dense | Significant improvement across frequent query patterns |

The system gracefully degrades — when no historical data exists, it falls through to standard FAISS retrieval with no overhead beyond the feedback-storage step.

---

## When to Use Feedback Loop RAG

Feedback Loop RAG provides the clearest return on investment in production systems with sustained, repeated usage — customer support platforms, internal knowledge hubs, and domain-specific Q&A tools that accumulate sessions over weeks and months. In these environments, the cold-start period transitions into a steadily improving warm state where frequently-asked questions get progressively more precise retrieval.

The technique is less appropriate for one-off or lightly-used deployments where query volumes never accumulate enough feedback to be statistically meaningful. If queries are highly diverse with little repetition, or if privacy constraints prevent storing query histories, the feedback store adds architectural complexity with diminishing returns. And in domains where the corpus changes very frequently, feedback scores can become stale before they accumulate enough weight to influence retrieval meaningfully.

---

## Summary

Feedback Loop RAG transforms a static retrieval system into an adaptive one. By recording per-chunk relevance scores for each query and blending this historical signal with FAISS similarity at query time, it progressively optimizes retrieval for the actual query distribution of its specific user base.

The improvement is invisible to users — there's no interface change — but measurable in retrieval metrics: chunks that have historically proven useful rank higher, and chunks that have been consistently irrelevant despite high cosine similarity are naturally down-weighted. Over weeks and months of use, this institutional memory makes the system substantially better than a freshly deployed RAG — without any retraining or re-indexing.
