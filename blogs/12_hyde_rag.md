# HyDE: Retrieve With the Answer You Wish You Had

> **Technique:** Hypothetical Document Embeddings (HyDE)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

There's a subtle but persistent problem in dense retrieval: **query-document embedding asymmetry**. When a user types a question like "What causes the greenhouse effect?", its embedding represents the *question as a question* — syntactically and semantically. The documents in your corpus, however, contain *answers* — declarative statements like "Greenhouse gases trap infrared radiation emitted by the Earth's surface."

Even though the question and its answer are about the same topic, their embedding representations can differ enough that top-k retrieval ranks less-relevant answer documents higher merely because they happen to look more like questions.

**HyDE (Hypothetical Document Embeddings)** reframes the retrieval problem entirely: instead of searching for documents using the query, generate a *hypothetical answer document* and search using **that**. You retrieve documents that look like your hypothetical answer, which leads to finding actual answers — not just topically adjacent content.

---

## The Core Idea

Standard retrieval:
```
User Query → embed(query) → search → real documents
```

HyDE retrieval:
```
User Query → LLM → hypothetical answer → embed(hypothetical) → search → real documents
```

The embedding of the hypothetical answer lies in an area of vector space populated by answer-style content. Real documents in your corpus that contain actual answers cluster nearby. The cosine similarity between a well-crafted hypothetical answer and the real answer document is much higher than between the question and that same document.

### Visual Intuition

```
Embedding Space:

  [Questions]                     [Answers/Documents]
  "What causes X?"        ←——→   "X is caused by Y..."
  "How does Y work?"                                   
       ↑                                 ↑
     Query                         Hypothetical doc
     embedding                     embedding sits here!
       ↑                                 ↑
     Far from answers              Close to real answers
```

---

## How HyDE Works

### Step 1: Generate Hypothetical Document

```python
def generate_hypothetical_answer(self, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at generating detailed, in-depth documents "
                "that directly answer questions. Generate a document that would "
                "be found in a knowledge base as the perfect answer."
            )
        },
        {
            "role": "user",
            "content": (
                f"Given the question '{query}', generate a hypothetical document "
                f"that directly answers this question. The document should be "
                f"detailed and in-depth. The document size should be exactly "
                f"{self.chunk_size} characters."
            )
        }
    ]
    return self.llm.chat(messages=messages)
```

Note the instruction to match `chunk_size`. This is intentional — you want the hypothetical document to occupy the same region of embedding space as your indexed chunks (which are also `chunk_size` characters). A very short hypothetical or a very long one would have different embedding characteristics.

### Step 2: Embed the Hypothetical Document

```python
hypothetical_doc = self.generate_hypothetical_answer(query)
hypothetical_embedding = self.embedder.embed_text(hypothetical_doc)
```

The hypothetical document is embedded exactly as real chunks are — using the same embedding model and no special treatment.

### Step 3: Retrieve Using Hypothetical Embedding

```python
results = self.vector_store.search(hypothetical_embedding, k=self.k)
```

Instead of searching with `embed(query)`, we search with `embed(hypothetical_doc)`. The results are real documents from your corpus that are closest to the hypothetical answer.

---

## Code Deep Dive: The Full Retrieval Flow

```python
def retrieve(self, query: str, k: int = 3) -> Tuple[List[RetrievalResult], str]:
    if self.vector_store is None:
        raise ValueError("Vector store not initialized")
    
    # Generate hypothetical answer
    hypothetical_doc = self.generate_hypothetical_answer(query=query)
    
    # Embed hypothetical and search
    hypothetical_embedding = self.embedder.embed_text(hypothetical_doc)
    results = self.vector_store.search(hypothetical_embedding, k)
    
    return results, hypothetical_doc

def query(self, question: str) -> Tuple[str, List[str], str]:
    results, hypothetical_doc = self.retrieve(question, self.k)
    
    context = [r.document.content for r in results]
    context_text = "\n\n".join(context)
    
    # Generate final answer from real retrieved context
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant. Use the context below to answer the question.\n\n"
                f"Context:\n{context_text}"
            )
        },
        {"role": "user", "content": question}
    ]
    
    answer = self.llm.chat(messages=messages)
    return answer, context, hypothetical_doc
```

Note that the `hypothetical_doc` is returned alongside the answer. This is excellent for debugging and transparency — you can inspect what the model generated and understand why those real documents were retrieved.

---

## Why the Hypothetical Document Is (Intentionally) Imperfect

A common question: "The hypothetical document might be factually wrong. Doesn't that cause bad retrievals?"

This is a valid concern, but the answer reveals a key subtlety of HyDE: **factual accuracy is irrelevant to the effectiveness of embedding-based retrieval**. What matters is the *linguistic and stylistic pattern* of the hypothetical document.

If a user asks "What is the impact of deforestation on water cycles?" and the LLM generates a hypothetical answer that contains correct-sounding statements about evapotranspiration and rainfall patterns — even if some numbers are wrong — the embedding of that hypothetical will cluster near real documents about deforestation's hydrological effects. The vocabulary, topic distribution, and semantic content guide the embedding, not the factual correctness.

The real documents retrieved are used for final answer generation. The hypothetical is merely a retrieval key.

---

## HyDE vs. Standard Retrieval: When Each Shines

| Query Type | Standard Retrieval | HyDE |
|-----------|-------------------|------|
| Factual lookup: "When was X founded?" | Good | Marginal improvement |
| Complex explanations: "How does X affect Y?" | Moderate | **Significantly better** |
| Rare terminology present in docs | **Better** | Moderate |
| Open-ended analysis | Moderate | **Better** |
| Short factual queries | Equivalent | Slower (extra LLM call) |

HyDE shines brightest for complex, multi-faceted questions where the question itself is short and doesn't contain the vocabulary present in relevant answers.

---

## Trade-off: One Extra LLM Call Per Query

HyDE adds one LLM call per query (hypothetical generation) before retrieval. For `gpt-4o-mini` at typical usage:

- Hypothetical generation: ~300-500 tokens, ~$0.0001-0.0002 per query
- Added latency: ~500ms-1.5s

For applications where retrieval quality is paramount and slight latency increases are acceptable, this is an extremely favorable trade-off. For sub-100ms latency requirements, HyDE may be inappropriate without careful optimization (e.g., caching common query types' hypotheticals).

---

## HyDE vs. HyPE

These two techniques are commonly confused but serve different purposes:

| Aspect | HyDE | HyPE |
|--------|------|------|
| When it runs | At **query time** | At **index time** |
| What LLM generates | Hypothetical answer document | Hypothetical questions per chunk |
| Embedding used for search | Hypothetical answer embedding | Question embeddings (index) vs. query embedding (search) |
| Extra LLM calls per query | 1 | 0 (already paid at index time) |
| Extra LLM calls at index | 0 | N per chunk |

HyDE is a query-side technique; HyPE is an indexing-side technique. They can be combined.

---

## When to Use HyDE

**Best for:**
- Complex, open-ended questions requiring multi-sentence answers
- Domains where query language diverges from document language
- Academic, legal, or medical domains with formal document language and informal user queries
- Systems where retrieval quality improvements justify one extra LLM call

**Skip when:**
- Sub-200ms latency is critical
- Queries are naturally answer-like (already contain answer vocabulary)
- Documents and queries use consistent terminology

---

## Summary

HyDE elegantly reframes dense retrieval by asking: "What would the ideal answer look like, and what does it look like?" By generating a hypothetical answer and using its embedding as the retrieval key, you bridge the syntactic gap between questions and answers in embedding space.

The technique is particularly powerful for complex queries in domains with formal document language. It's one of the most creative insights in the RAG literature — using the LLM's generation capability not to answer the question directly, but to improve retrieval so that real documents can provide that answer reliably.
