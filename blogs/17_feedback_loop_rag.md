# Feedback Loop RAG: A System That Learns From Its Users

> **Technique:** Feedback Loop RAG  
> **Complexity:** Advanced  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

Every RAG system starts equal: no matter how well you tune chunking, embeddings, and retrieval, the system is static. Query 1,000 and query 1 are handled by the same mechanism. User signals — satisfaction ratings, follow-up questions, corrections — are discarded.

This is a lost opportunity. Your users know what's relevant. Their feedback, properly captured and applied, is the richest possible signal for improving retrieval quality over time.

**Feedback Loop RAG** builds a self-improving RAG system that learns from user ratings. After each interaction, users rate the response for relevance and quality (1-5 scale). This feedback is stored persistently and applied in two ways:

1. **Per-query score adjustment**: Past feedback on similar queries boosts or penalizes chunk relevance scores in real time
2. **Periodic index fine-tuning**: High-quality Q&A pairs are added to the vector store as new documents, enriching future retrievals

Over time, the system gets measurably better at serving its user population.

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│          FeedbackLoopRAG                     │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ FeedbackLoop    │  │   FeedbackStore  │  │
│  │ Retriever       │  │   (JSONL file)   │  │
│  │                 │←→│                  │  │
│  │ - FAISS search  │  │ - store()        │  │
│  │ - score adjust  │  │ - load_all()     │  │
│  │ - fine_tune     │  │ - load_high_q()  │  │
│  └─────────────────┘  └──────────────────┘  │
│           ↑                     ↑           │
│           │ retrieval           │ feedback   │
│  ┌─────────────────┐            │            │
│  │ RelevanceAdjuster│            │            │
│  │ (LLM scoring)  │            │            │
│  └─────────────────┘            │            │
│                                  │            │
│  User → query() → answer → submit_feedback() │
└─────────────────────────────────────────────┘
```

---

## The Two Feedback Mechanisms

### Mechanism 1: Per-Query Score Adjustment

Every new query checks past feedback to see if any previous interaction is relevant to the current one. If relevant feedback exists, chunk scores are adjusted proportionally:

```
adjusted_score = original_score × (avg_relevant_feedback_rating / neutral_score)
```

Where `neutral_score = 3.0` (the midpoint of the 1-5 rating scale).

- Feedback rating 5 (excellent) → multiplier 5/3 = 1.67 → **chunk score boosted**
- Feedback rating 3 (average) → multiplier 3/3 = 1.0 → **no change**
- Feedback rating 1 (poor) → multiplier 1/3 = 0.33 → **chunk score penalized**

```python
def adjust_score(self, query, results, feedback_data, neutral_score=3.0):
    for result in results:
        relevant_feedback = [
            fb for fb in feedback_data
            if self.is_feedback_relevant(query, result.document.content, fb)
        ]
        
        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            adjustment = avg_relevance / neutral_score
            result.score = result.score * adjustment
    
    # Re-sort by adjusted scores
    results.sort(key=lambda r: r.score, reverse=True)
    return results
```

### Mechanism 2: Index Fine-Tuning (Periodic)

High-quality Q&A pairs (rated ≥4 on both relevance and quality) are added to the knowledge base as new documents:

```python
def fine_tune_index(self) -> int:
    # Load only high-quality feedback
    good_feedback = self.feedback_store.load_high_quality(min_relevance=4, min_quality=4)
    
    # Format as Q&A documents
    additional_text = "\n\n".join(
        f"Question: {f['query']}\nAnswer: {f['response']}\n"
        for f in good_feedback
    )
    
    # Combine with original text and rebuild index
    combined = self._original_text + "\n\n" + additional_text
    self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
    num_chunks = self.index_document(combined)
    
    return num_chunks
```

This is powerful: questions users actually asked and answers that satisfied them become first-class content in the knowledge base. Future similar queries will retrieve the proven good answer directly, bypassing the need to synthesize from raw source text.

---

## The Relevance Adjuster: LLM-Powered Feedback Linking

The key challenge: how do you know if a past feedback entry is relevant to a current query-chunk pair? You use an LLM:

```python
class RelevanceAdjuster:
    def is_feedback_relevant(self, query, chunk_content, feedback) -> bool:
        messages = [
            {
                "role": "system",
                "content": (
                    "You determine if past feedback is relevant to a current query. "
                    "Return JSON with key 'relevant' set to true or false"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Current query: {query}\n"
                    f"Feedback's original query: {feedback['query']}\n"
                    f"Document Content: {chunk_content[:1000]}\n"
                    f"Feedback Response: {feedback['response']}\n\n"
                    "Is this feedback relevant to the current query and document?"
                )
            }
        ]
        result = self.llm.chat_json(messages)
        return result.get("relevant", False) is True
```

This LLM call is made for every (chunk, feedback) pair during retrieval. With 3 chunks and 10 feedback entries, that's 30 LLM calls. This is expensive for large feedback histories, but provides high-quality relevance linking that simple string matching cannot.

**Optimization opportunity**: For production systems, cache feedback-relevance decisions across queries. Two queries about "climate change causes" should produce the same feedback-chunk relevance judgments.

---

## The FeedbackStore

Feedback is persisted as JSON Lines (JSONL) — one JSON object per line:

```python
class FeedbackStore:
    def store(self, feedback: Dict) -> None:
        feedback["timestamp"] = datetime.now().isoformat()
        with open(self.file_path, "a", encoding="utf-8") as f:
            json.dump(feedback, f)
            f.write("\n")
    
    def load_all(self) -> List[Dict]:
        feedback_data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line))
        return feedback_data
    
    def load_high_quality(self, min_relevance=4, min_quality=4) -> List[Dict]:
        return [
            f for f in self.load_all()
            if f.get("relevance", 0) >= min_relevance
            and f.get("quality", 0) >= min_quality
        ]
```

JSONL format is ideal for append-only logging where new entries are added but existing ones are never modified. Reading is a simple line scan.

---

## The Usage Workflow

```python
# Initialize
rag = FeedbackLoopRAG(file_path="document.pdf", use_feedback=True)

# Query
answer, contexts = rag.query("What is the greenhouse effect?")
print(f"Answer: {answer}")

# User rates the response
rag.submit_feedback(
    relevance=4,          # 1-5: how relevant was the context?
    quality=5,            # 1-5: how good was the answer?
    comments="Clear and accurate explanation"
)

# ... more queries and feedback over time ...

# Periodic fine-tuning (daily/weekly, not per-query)
rag.fine_tune_index()

# View statistics
rag.show_feedback_stats()
```

---

## Feedback Statistics

```
Feedback statistics:
  Total entries:         47
  High-quality entries:  31  (for fine-tuning)
  AVG relevance:         4.2/5
  AVG quality:           4.0/5

  Recent Feedback:
  [2024-03-15] rel=5 qual=5  q="What are the main causes..."
  [2024-03-15] rel=3 qual=4  q="How does CO2 affect..."
  [2024-03-16] rel=4 qual=3  q="What are mitigation..."
```

Monitoring average scores tells you whether your RAG system is improving over time and which queries consistently receive low ratings (signaling retrieval gaps).

---

## The Compounding Improvement Effect

The feedback loop creates compounding quality improvements:

```
Week 1: Fresh system, no feedback
Week 2: 50 feedback entries → score adjustment begins → +5% quality
Week 4: 200 entries → rich adjustment signal → +12% quality
Week 8: 500 entries + first fine-tune → proven Q&A in index → +20% quality
Week 12: 1000 entries + regular fine-tunes → highly adapted system → +30%+ quality
```

This is dramatically different from a static RAG system that performs identically at week 12 as at week 1.

---

## When to Use Feedback Loop RAG

**Best for:**
- Internal knowledge bases with consistent user populations (same team using the same system)
- Customer support bots where query patterns repeat over time
- Research assistant systems where user expertise helps distinguish good answers
- Long-running deployments where system improvement over time justifies investment

**Skip when:**
- User population is anonymous and diverse (no consistent feedback signal)
- Responses are one-off and queries don't repeat
- Privacy constraints prevent storing user interaction data

---

## Summary

Feedback Loop RAG transforms a static retrieval system into a living, learning one. By capturing user ratings after each interaction and applying them — both as real-time score adjustments and as periodic index enrichments — the system becomes progressively better suited to its actual user population.

The core insight is deceptively simple: the users of your RAG system collectively know what constitutes a good answer. Capturing that knowledge and feeding it back into the retrieval and indexing processes turns user satisfaction signals into system intelligence.
