# Proposition Chunking: Atomic Facts as the Unit of Retrieval

> **Technique:** Proposition Chunking RAG  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

The quality of a RAG system is fundamentally bounded by the quality of its chunks. Standard chunking splits text by character count or sentence count — a mechanical process that treats all text as equally dense with information. But text isn't uniform. A single paragraph might contain one important fact buried in three sentences of contextual filler.

**Proposition Chunking** takes a radically different approach: it uses an LLM to decompose text into *atomic, self-contained propositions* — the smallest unit of meaning that stands on its own. Instead of retrieving paragraphs, you retrieve facts.

This technique was popularized by research into "dense x retrieval" and builds on the idea that retrieval precision improves dramatically when the indexed unit matches what users actually search for: individual claims, definitions, and relationships.

---

## The Core Idea: What Is a Proposition?

A proposition is a single, self-contained statement of fact. It must satisfy five criteria:

1. **Expresses a single fact**: One claim, one relationship.
2. **Self-contained**: Understandable without the surrounding text.
3. **Uses full names, not pronouns**: "Neil Armstrong", not "he".
4. **Includes qualifiers**: Dates, locations, conditions are preserved.
5. **Subject-predicate clarity**: Clear actor and action/attribute.

**Example transformation:**

Input text:
```
In 1969, Neil Armstrong became the first person to walk on the Moon 
during the Apollo 11 mission, which was part of NASA's broader effort 
to beat the Soviet Union in the Space Race.
```

Propositions:
```
• Neil Armstrong walked on the Moon in 1969.
• Neil Armstrong was the first person to walk on the Moon.
• Neil Armstrong walked on the Moon during the Apollo 11 mission.
• The Apollo 11 mission occurred in 1969.
• NASA ran the Apollo 11 mission.
• The Apollo program was part of NASA's effort to compete in the Space Race.
• The Space Race was a competition between the United States and the Soviet Union.
```

Each proposition is precise, independent, and directly searchable.

---

## How the Pipeline Works

```
Document
    ↓
Fixed-size chunks (for LLM processing manageability)
    ↓
[LLM] Generate propositions per chunk
    ↓
[LLM] Evaluate each proposition for quality (score 0.0–1.0)
    ↓
Filter: keep only high-quality propositions
    ↓
Embed each proposition → FAISS vector store
    ↓
[Query time] Retrieve by proposition similarity
    ↓
[LLM] Generate answer
```

The two-stage LLM process (generate then evaluate) ensures that low-quality, redundant, or trivially obvious propositions don't pollute the index.

---

## Code Deep Dive

### Proposition Generation

The generation prompt is carefully few-shot engineered:

```python
def generate_propositions(llm, chunk_text: str) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Please break down the following text into simple, self-contained propositions. "
                "Ensure that each proposition meets the following criteria:\n\n"
                "1. Express a Single Fact\n"
                "2. Be Understandable Without Context\n"
                "3. Use Full Names, Not Pronouns\n"
                "4. Include Relevant Dates/Qualifiers\n"
                "5. Contain One Subject-Predicate Relationship\n\n"
                'Respond with JSON: {"propositions": ["prop1", "prop2", ...]}'
            )
        },
        # Few-shot example included for output format alignment
        {"role": "user", "content": "In 1969, Neil Armstrong..."},
        {"role": "assistant", "content": '{"propositions": [...]}'},
        {"role": "user", "content": chunk_text}
    ]
    result = llm.chat_json(messages)
    return result.get("propositions", [])
```

The few-shot example is critical — it demonstrates exactly the JSON format expected, preventing parsing failures. JSON mode (`chat_json`) ensures structured output that can be reliably parsed.

### Quality Evaluation

Not all generated propositions are useful. The evaluator filters them:

```python
def evaluate_proposition(llm, proposition: str) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "Score this proposition on a scale of 0.0 to 1.0 based on:\n"
                "- Clarity: Is it clearly stated?\n"
                "- Factuality: Does it state a specific fact?\n"
                "- Usefulness: Would it be useful for answering questions?\n"
                'Respond with JSON: {"score": <float>}'
            )
        },
        {"role": "user", "content": f"Proposition: {proposition}"}
    ]
    result = llm.chat_json(messages)
    return result.get("score", 0.0)
```

Propositions scoring above a threshold (e.g., 0.6) are kept; the rest are discarded. This is the quality gate that keeps the index clean.

### Index-Time Processing

```python
def build_index(self, text: str):
    # Step 1: Initial chunking for LLM processing
    raw_chunks = chunk_text(text, chunk_size=2000, chunk_overlap=200)
    
    all_propositions = []
    for chunk in raw_chunks:
        # Step 2: Generate propositions
        props = generate_propositions(self.llm, chunk)
        
        # Step 3: Evaluate and filter
        for prop in props:
            score = evaluate_proposition(self.llm, prop)
            if score >= self.quality_threshold:
                all_propositions.append(prop)
    
    # Step 4: Embed and index
    documents = [Document(content=p) for p in all_propositions]
    documents = self.embedder.embed_documents(documents)
    self.vector_store.add_documents(documents)
```

---

## Why This Dramatically Improves Retrieval

### The Embedding Alignment Problem

When you embed a chunk like:

> "Climate change, driven primarily by the burning of fossil fuels since the Industrial Revolution, affects global temperature through the greenhouse effect, which traps heat in the atmosphere and disrupts weather patterns worldwide."

The embedding of this chunk is an *average* of many concepts: climate change, fossil fuels, Industrial Revolution, temperature, greenhouse effect, weather patterns. A query like "What causes the greenhouse effect?" competes against all of that noise.

When you embed the proposition:

> "The greenhouse effect traps heat in the atmosphere."

The embedding is *precisely* about that one concept. Query-to-proposition alignment is near perfect.

### Granularity Enables Precision

Each proposition has its own embedding vector, so each can be found independently. A query about the Apollo 11 crew will surface the "Neil Armstrong" proposition without competitors from the same paragraph about the Space Race.

---

## The Cost-Quality Trade-off

Proposition Chunking is expensive. For every chunk of text, you make at least two LLM calls (generate + evaluate) per chunk. For a 100-page document:

- ~300 initial chunks
- ~300 proposition generation calls
- ~3,000 evaluation calls (assuming ~10 propositions per chunk)

This is an **offline cost** — paid once at index time, not per query. Query-time latency is identical to Simple RAG. For applications where retrieval quality is paramount and the corpus is relatively stable, this trade-off is often very much worth it.

---

## Comparison: Fixed Chunking vs. Proposition Chunking

| Aspect | Fixed Chunking | Proposition Chunking |
|--------|---------------|----------------------|
| Indexed unit | Text window (1000 chars) | Atomic fact |
| Embedding precision | Medium | High |
| Retrieval recall | Good | Excellent |
| Index-time cost | Low | Very high (many LLM calls) |
| Query-time cost | Identical | Identical |
| Index size | Smaller | Larger (more granular items) |
| Handles dense paragraphs | Poorly | Excellently |

---

## When to Use Proposition Chunking

**Best for:**
- Knowledge bases where precision matters (medical, legal, technical docs)
- Q&A systems where users ask specific factual questions
- Applications where retrieval hallucination is unacceptable
- Corpora that are stable (infrequently updated)

**Not ideal for:**
- Frequently updated document sets (re-indexing is expensive)
- Documents with complex reasoning chains where propositions lose context
- Latency-critical applications where index-time is billed to users

---

## Summary

Proposition Chunking reframes the fundamental unit of RAG from a **text window** to a **fact**. By using an LLM to decompose documents into atomic, self-contained propositions and filtering them by quality, the vector index becomes a precision instrument — every entry is exactly the kind of information a user query seeks to retrieve.

The offline cost is real but bounded. The retrieval quality improvement, however, compounds at query time across every user interaction. For high-stakes RAG applications, this investment in index quality is often the highest-leverage optimization available.
