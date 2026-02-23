# HyPE: The Indexing-Time Counterpart to HyDE

> **Technique:** Hypothetical Prompt Embeddings (HyPE)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`, `concurrent.futures`

---

## Introduction

HyDE solves the query-document embedding asymmetry at *query time* by generating hypothetical answer documents. **HyPE (Hypothetical Prompt Embeddings)** attacks the same fundamental problem from the opposite direction — at *index time* — by generating hypothetical *questions* for each document chunk.

Where HyDE says "make the query look like an answer," HyPE says "make the index look like questions." The indexed chunks are represented not just by their own embeddings, but by the embeddings of questions they could answer. At search time, a user's question directly matches the question-style vectors already in the index.

The result: **queries match against questions** — semantically near-identical objects — rather than against document text. This is the most natural alignment possible.

---

## The Core Idea

Standard indexing:
```
Chunk text → embed(chunk) → stored in FAISS
Query → embed(query) → search FAISS → chunk texts returned
```

HyPE indexing:
```
Chunk text → LLM → [Q1, Q2, Q3, Q4, Q5] (questions this chunk answers)
            ↓
Each Qi → embed(Qi) → stored in FAISS (pointing back to original chunk)
Query → embed(query) → search FAISS → matched question's parent chunk returned
```

The index stores question embeddings, not chunk embeddings. Real user queries are questions; the index contains questions; the match is direct and natural.

---

## The Vocabulary Alignment Advantage

Consider a chunk from a climate science paper:
> "Anthropogenic greenhouse gas emissions, primarily CO2 from fossil fuel combustion, have led to a 1.1°C increase in global average surface temperature since pre-industrial times."

Standard embedding: vector representing scientific assertions about emissions, temperature, CO2.

HyPE-generated questions:
- "How much has Earth's temperature increased since pre-industrial times?"
- "What is the main cause of global warming?"
- "What percentage of warming is due to CO2?"
- "How have fossil fuels affected global temperature?"
- "What are anthropogenic greenhouse gas emissions?"

A user asking "Why is Earth getting warmer?" may not match the chunk well in standard retrieval, but will strongly match one of these HyPE-generated questions — particularly "What is the main cause of global warming?" — because that question was generated *from the chunk that contains the answer*.

---

## Code Deep Dive

### Question Generation with Parallel Processing

```python
def embed_file(self):
    text = read_pdf(file_path=self.file_path)
    chunks = chunk_text(text, chunk_size=self.chunk_size, 
                        chunk_overlap=self.chunk_overlap)
    chunk_docs = []
    
    # Parallel question generation for efficiency
    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
        futures = [
            pool.submit(self.generate_hypothetical_prompt_embeddings, chunk)
            for chunk in chunks
        ]
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(chunks))):
            chunk_content, question_embeddings = future.result()
            
            # Each question embedding → Document pointing to original chunk
            for qe in question_embeddings:
                chunk_doc = Document(
                    content=chunk_content,    # ← ORIGINAL chunk, not the question
                    metadata={"source": self.file_path, "chunk_id": i},
                    embedding=qe              # ← QUESTION embedding for searching
                )
                chunk_docs.append(chunk_doc)
    
    self.vector_store.add_documents(chunk_docs)
```

`ThreadPoolExecutor` runs question generation in parallel. With `max_workers=10`, 200 chunks can be processed concurrently, reducing index build time from ~20 minutes to ~2 minutes. Each `gpu.submit()` call launches an independent LLM request.

### Question Generation

```python
def generate_hypothetical_prompt_embeddings(self, chunk_text_str: str) -> Tuple[str, List[List[float]]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You generate essential questions from text. "
                "Each question should be on one line, without numbering or prefixes."
            )
        },
        {
            "role": "user",
            "content": (
                "Analyze the input text and generate essential questions that, "
                "when answered, capture the main points of the text. "
                "Each question should be one line, without numbering or prefixes.\n\n"
                f"Text:\n{chunk_text_str}\n\nQuestions:"
            )
        }
    ]
    response = self.llm.chat(messages=messages)
    
    # Parse questions (one per line, filter short/empty)
    questions = [
        q.strip()
        for q in response.replace("\n\n", "\n").split("\n")
        if q.strip() and len(q.strip()) > 10
    ]
    
    # Embed all questions in one batch call
    question_embeddings = self.embedder.embed_texts(questions)
    
    return chunk_text_str, question_embeddings
```

The function returns both the original chunk text and all question embeddings. The question embeddings are used for FAISS; the chunk text is what gets returned to users. One chunk → N questions → N FAISS entries, all pointing to the same chunk text.

### Query-Time Deduplication

```python
def query(self, question: str) -> Tuple[str, List[str]]:
    query_embedding = self.embedder.embed_text(question)
    query_vec = np.array([query_embedding], dtype=np.float32)
    
    # Search wider than k (will deduplicate by chunk)
    search_k = min(self.vector_store.index.ntotal, self.k * 5)
    distances, indices = self.vector_store.index.search(query_vec, search_k)
    
    # Deduplicate: multiple questions may point to the same chunk
    seen_chunks = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(self.vector_store.documents):
            doc = self.vector_store.documents[idx]
            chunk_key = doc.content[:100]  # use first 100 chars as identity
            
            if chunk_key not in seen_chunks or dist < seen_chunks[chunk_key][0]:
                seen_chunks[chunk_key] = (dist, doc)
    
    # Sort by best match score, take top-k unique chunks
    sorted_results = sorted(seen_chunks.values(), key=lambda x: -x[0])[:self.k]
    context = [doc.content for _, (_, doc) in enumerate(sorted_results)]
```

**Why `k * 5` in search?** With 5 questions per chunk and `k=3` desired final chunks, you might need to look at 15 results to find 3 unique chunks. The over-search with deduplication guarantees you get at least `k` unique chunks in the final result.

---

## HyPE vs. HyDE: Side-by-Side

| Aspect | HyPE | HyDE |
|--------|------|------|
| **When** | Index time (offline) | Query time (online) |
| **LLM generates** | N questions per chunk | 1 hypothetical answer per query |
| **What's indexed** | Question embeddings | Original chunk embeddings |
| **Retrieval target** | Query matches questions | Hypothetical answer matches chunks |
| **Added query latency** | None (0 extra LLM calls) | 1 LLM call (~500ms) |
| **Index size increase** | N× (5-7× if 5-7 questions per chunk) | None |
| **Index rebuild cost** | High (N LLM calls per chunk) | None |
| **Best for** | High-query-volume systems | Low-latency-tolerant systems |

**Can you combine them?** Yes! Use HyPE-style question indexing AND HyDE-style hypothetical generation at query time. The hypothetical answer is matched against the question-indexed FAISS store. This is theoretically the most powerful combination but also the most expensive.

---

## Index Size Implications

If your corpus has 1,000 chunks and you generate 5 questions per chunk:

- Standard RAG: 1,000 FAISS vectors
- HyPE: 5,000 FAISS vectors (5× expansion)

FAISS handles millions of vectors efficiently, so this is not a performance bottleneck. However, it does increase memory usage (each vector = 6KB for 1536-dim):

- 1,000 chunks standard: ~6MB
- 5,000 chunks HyPE: ~30MB

For typical enterprise document corpora this is completely manageable.

---

## Choosing the Number of Questions

The `num_questions` implicitly set via the LLM response length affects the quality-cost trade-off:

| Questions per chunk | Quality | Index size | Index time |
|--------------------|---------|------------|------------|
| 2-3 | Good | 2-3× | Low |
| 4-7 | **Best** | 4-7× | Medium |
| 8+ | Diminishing returns | 8×+ | High |

5 questions per chunk is the standard recommendation: enough diversity to cover different query phrasings, not so many that questions become redundant.

---

## When to Use HyPE

**Best for:**
- High-volume production systems where per-query latency must be minimal
- Corpora that are stable and not updated frequently (index is built once)
- Domains with formal/technical language diverging from casual user queries
- Systems where you want to invest cost at index time rather than query time

**Skip when:**
- Document corpus changes frequently (requires expensive re-indexing)
- Index storage is severely constrained
- Documents already use question-answer format (FAQ pages, Stack Overflow)

---

## Summary

HyPE turns the vocabulary mismatch problem on its head: instead of making queries look like answers (HyDE), it makes the index look like questions. By pre-generating questions at index time and storing their embeddings alongside the original chunk text, HyPE creates a retrieval index that natively speaks the language of user queries.

The added index size and build cost are the price for zero added query latency — a trade-off that's ideal for production systems serving thousands of queries per day on stable document corpora. Think of HyPE as a high-quality pre-investment: pay once at index time, pay nothing extra for every query forever after.
