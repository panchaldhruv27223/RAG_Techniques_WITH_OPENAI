# Simple RAG: The Foundation of Retrieval-Augmented Generation

**Source:** `simple_rag/` · **Libraries:** `openai`, `faiss-cpu`, `PyMuPDF`

---

## Introduction

Every sophisticated system starts with a solid foundation. In the world of Retrieval-Augmented Generation (RAG), that foundation is **Simple RAG** — a clean, three-phase pipeline that connects a knowledge base to a language model, enabling it to answer questions grounded in real documents rather than relying purely on its training data.

Large Language Models are tremendous at reasoning, summarization, and synthesis. But they have two critical limitations that make them unsuitable as standalone tools for enterprise use cases:

1. **Knowledge cutoff**: LLMs are trained on corpora with a fixed end date. GPT-4's training data has a cutoff; anything after that simply doesn't exist in the model's world.
2. **Private data blindness**: No LLM has access to your company's internal reports, proprietary datasets, customer contracts, or confidential research. These documents live outside the training pipeline entirely.

RAG bridges this gap with a simple but profound idea: at question-answering time, *retrieve* the relevant information from your documents first, then *give* that information to the LLM as context before asking it to generate a response. The LLM, now reading from your private, up-to-date documents, can answer accurately.

---

## The Conceptual Model

Think of RAG as giving the LLM "open-book access" instead of forcing it to work from memory alone. 

In a closed-book exam, a student must recall everything from training. In an open-book exam, the student can look things up — but must still reason and synthesize. RAG turns every LLM query into an open-book exam: the model consults retrieved documents before writing its answer.

This has profound implications:
- **Factual accuracy improves** because the model reads the answer rather than recalling it
- **Hallucination decreases** because grounded responses stay tethered to real text
- **Private knowledge becomes accessible** without retraining or fine-tuning the model
- **Knowledge stays fresh** by updating the document store rather than retraining the model

---

## Architecture: The Three-Phase Pipeline

Simple RAG is organized into two modes separated by the first query:

### Phase 1: Indexing (Offline — runs once or at update time)

```
Raw Documents (PDF, CSV, TXT, etc.)
         │
         ▼
 ┌───────────────┐
 │ Text Extraction│   (PyMuPDF for PDF → plain text)
 └───────────────┘
         │
         ▼
 ┌────────────────────┐
 │ Text Chunking      │   Sliding window: chunk_size=1000 chars, overlap=200
 └────────────────────┘
         │
         ▼
 ┌─────────────────────────┐
 │ Embed each chunk        │   OpenAI text-embedding-3-small → 1536-dim vector
 └─────────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ FAISS vector store  │   L2-indexed flat store for nearest-neighbor search
 └─────────────────────┘
```

This phase can take seconds for small PDFs or minutes for very large corpora. It is designed to run **once** and then support unlimited queries without re-processing.

### Phase 2: Query Processing (Online — runs per user query)

```
User Question (natural language)
         │
         ▼
 Embed question → 1536-dim query vector
         │
         ▼
 FAISS similarity search → top-k most similar chunks
         │
         ▼
 Construct prompt:
   System: "Answer using this context: {chunks}"
   User:   "{question}"
         │
         ▼
 LLM (GPT-4o-mini) → generates grounded answer
         │
         ▼
 Answer returned to user
```

Query processing typically takes 500ms–2s total: ~50ms for embedding, ~5ms for FAISS search, ~500–2000ms for LLM generation.

---

## How FAISS Works Under the Hood

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional vector spaces. The standard configuration for Simple RAG uses a **flat L2 index** — which means every stored vector is compared to the query vector with no approximation.

**Cosine similarity vs. L2 distance:** The implementation uses cosine similarity (dot product of unit-normalized vectors), which measures the *angle* between vectors rather than their *distance*. Two vectors with identical semantic content will have cosine similarity → 1.0 regardless of their magnitude. L2 distance is also consistent with cosine similarity when vectors are L2-normalized (which is standard OpenAI embedding practice).

```python
# FAISS internal distance computation (simplified concept)
# For query vector q and stored vector d:
score = cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)
```

The FAISS flat index scans every stored vector on each search. This is O(n) per query but extremely vectorized using BLAS — it can search 1 million vectors in under 100ms on a CPU.

For very large corpora (>10M chunks), approximate search (IVF, HNSW) would be used to trade a small accuracy loss for dramatic speed improvements. For most enterprise use cases, the flat index is perfectly sufficient.

---

## Implementation Walkthrough

The full implementation is built on a handful of focused classes. Here's how each one works:

### OpenAIEmbedder

```python
class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()
        self.dimension = 1536  # fixed for text-embedding-3-small
    
    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        # Batch all documents in a single API call for efficiency
        texts = [d.content for d in documents]
        response = self.client.embeddings.create(
            model=self.model,
            input=texts  # OpenAI supports up to 2048 inputs per batch call
        )
        for doc, embedding_obj in zip(documents, response.data):
            doc.embedding = embedding_obj.embedding
        return documents
```

The `embed_documents` method batches all chunks into a single API call — crucially important for performance and cost when indexing large documents. A 100-page PDF producing 300 chunks would take 300 individual round-trips without batching, but only 1 with batching (or a small number if the batch exceeds the API limit).

### FAISSVectorStore

```python
class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Flat L2 index — exact search, no approximation
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Document] = []
    
    def add_documents(self, documents: List[Document]) -> None:
        embeddings = np.array(
            [d.embedding for d in documents],
            dtype=np.float32
        )
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: List[float], k: int) -> List[RetrievalResult]:
        query_vec = np.array([query_embedding], dtype=np.float32)
        # distances: L2 distances (lower = more similar)
        # indices: positions in self.documents
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                score = 1.0 / (1.0 + dist)  # convert L2 distance to similarity score
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=score
                ))
        return sorted(results, key=lambda r: r.score, reverse=True)
```

The score conversion `1.0 / (1.0 + dist)` transforms an L2 distance (0 = identical, ∞ = maximally different) into a bounded score (1.0 = identical, 0 = maximally different). This makes scores intuitive and comparable.

### Text Chunking: The Sliding Window

```python
def chunk_text(text: str, chunk_size: int = 1000, 
               chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # skip whitespace-only chunks
            chunks.append(chunk)
        
        # Advance by (chunk_size - chunk_overlap)
        # So next chunk starts 200 chars before the end of this one
        start += chunk_size - chunk_overlap
    
    return chunks
```

**Why overlap?** Without overlap, critical context that spans a chunk boundary is split. Consider this text at the boundary of chunk 1 and chunk 2:

```
... The main cause of this phenomenon is [boundary] carbon dioxide accumulation in the upper atmosphere ...
```

Without overlap, "this phenomenon" in chunk 1 has no referent (stored in one chunk), and "carbon dioxide" in chunk 2 has no question (stored in the next). With a 200-character overlap, both chunks contain the complete phrase around the boundary.

**A concrete example** — with `chunk_size=50, chunk_overlap=10` (shortened for clarity):

```
Text: "The quick brown fox jumped over the lazy dog near the river."

Chunk 1: "The quick brown fox jumped over the lazy"
Chunk 2: "y dog near the river."
             ↑
         10-char overlap from chunk 1
```

### Prompt Construction and Generation

```python
def query(self, question: str, k: int = 3) -> str:
    # Step 1: Embed the question
    query_embedding = self.embedder.embed_text(question)
    
    # Step 2: Retrieve top-k similar chunks
    results = self.vector_store.search(query_embedding, k=k)
    
    # Step 3: Build context string
    context_pieces = [r.document.content for r in results]
    context = "\n\n---\n\n".join(context_pieces)
    
    # Step 4: Construct the prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question "
                "using ONLY the information provided in the context below. "
                "If the context does not contain enough information to answer "
                "the question, say so clearly. Do not use knowledge outside the context.\n\n"
                f"Context:\n{context}"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    return self.llm.chat(messages)
```

The system prompt instructs the model to use **only** the provided context. This is the most important prompt design decision in a RAG system — without this constraint, the LLM will blend retrieved facts with training-data knowledge, making it impossible to know which parts of the answer are grounded.

The `temperature=0.0` setting on the LLM chat forces deterministic generation. For factual Q&A, you don't want creative variation — you want the same correct answer every time.

---

## CSV Support: RAG Over Tabular Data

Simple RAG also handles CSV files. Each row is converted to a structured string:

```python
def index_csv(self, file_path: str) -> int:
    df = pd.read_csv(file_path)
    
    # Convert each row to a key-value string
    rows_as_text = []
    for _, row in df.iterrows():
        # "Column1: Value1 | Column2: Value2 | Column3: Value3"
        row_text = " | ".join(
            f"{col}: {val}" for col, val in row.items()
            if pd.notna(val)  # skip empty cells
        )
        rows_as_text.append(row_text)
    
    # Concatenate and chunk (rows can be batched together)
    full_text = "\n".join(rows_as_text)
    return self.index_document(full_text)
```

This flattened representation — where every column becomes a labeled key-value pair — works well for semantic search. A query "orders from customer John in March" will semantically match rows containing "Customer: John | Month: March | ...".

**Important limitation**: This approach works for semantic lookup but poorly for analytical queries like "what is the total revenue by region?" Those require SQL-style aggregation, not semantic retrieval. For analytical CSV queries, consider text-to-SQL approaches instead.

---

## The Embedding Model: text-embedding-3-small

OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors and is the default for this implementation. Key properties:

| Property | Value |
|----------|-------|
| Dimensions | 1536 |
| Max input tokens | 8,192 |
| Context window | ~6,000 words |
| Cost | $0.02 per million tokens |
| Quality | Very good for semantic search |

A 1536-dimensional vector means each chunk is represented as a point in a 1536-dimensional space. Chunks about similar topics cluster near each other in this space. The astonishing power of modern embeddings is their ability to capture semantic relationships: "dog" and "canine" end up near each other; "climate change" and "global warming" cluster together even though they share no words.

---

## Performance Characteristics

### Index Time
For a typical 50-page PDF (~150,000 characters):
- Number of chunks: ~150 (at 1000 char/chunk, 200 overlap)
- Embedding API call: 1 batch call, ~2-3 seconds
- FAISS index build: instant (flat index)
- **Total index time**: ~3-5 seconds

For a 500-page document:
- Number of chunks: ~1,500
- Embedding time: ~20-30 seconds (multiple batches)
- **Total index time**: ~30-60 seconds

### Query Time
| Step | Typical Time |
|------|-------------|
| Embed question | 100-200ms |
| FAISS search (1,500 vectors) | <1ms |
| LLM generation (gpt-4o-mini) | 500-1500ms |
| **Total** | **~700ms-2s** |

---

## Key Configuration Parameters

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `chunk_size` | 1000 chars | More context per chunk, less precise retrieval | Fewer chars per chunk, more precise retrieval |
| `chunk_overlap` | 200 chars | Better boundary coverage, more storage | Risk of losing boundary context |
| `k` | 3 | More context, risk of dilution and cost | Less context, risk of missing relevant info |
| `temperature` | 0.0 | N/A (increase → more creative, less accurate) | Already minimum |

### Choosing k

`k=3` is a sensible default. Here's the practical guide:

- **k=1**: Use when the corpus is highly structured and one chunk will always contain the answer. Very fast, minimal noise.
- **k=3**: Standard for most Q&A applications. Covers 1 primary source + 2 supporting/alternative chunks.
- **k=5**: Use for complex, multi-faceted questions that may draw from multiple document sections.
- **k=10+**: Use cautiously. Beyond a point, adding more context confuses phrasing LLMs and blows up token costs.

---

## Failure Modes to Understand

### 1. Chunk Boundary Artifacts

When the answer straddles a chunk boundary, retrieval may return a chunk that *almost* contains the answer:

```
Perfect answer: "The treaty was signed on June 28, 1919"

Chunk retrieved: "negotiations culminated in an agreement, formalized"
Next chunk:      "on June 28, 1919, imposing heavy terms on..."
```

The chunk 200 characters before the date was retrieved because the query semantically matched "agreement/formalized", but the date itself is in the next chunk. Solutions: overlap, context enrichment window, or semantic chunking.

### 2. Top-k Averaging Problem

FAISS returns the k *most similar* chunks. But "most similar in embedding space" ≠ "most relevant to this specific question." A chunk about climate policy might score 0.82 cosine similarity to a climate mechanisms question purely because they share topic vocabulary — even though the chunk doesn't contain the answer.

Solutions: contextual compression, reranking.

### 3. Multi-hop Questions

"Compare the revenue performance in Q3 2023 to the strategic objectives stated in the 2022 annual report" requires retrieving from two separate sections of two separate documents. Simple RAG retrieves the k most similar chunks without any multi-hop reasoning. It may get one side of the comparison but miss the other.

Solutions: query decomposition, iterative retrieval, adaptive retrieval.

### 4. Long Documents Without Distinct Sections

For a 300-page textbook with dense, uniform writing, hundreds of chunks may be nearly equally relevant to most queries. FAISS returns the top-k but cannot distinguish truly relevant from semantically adjacent chunks.

Solutions: hierarchical indices, proposition chunking.

---

## When to Use Simple RAG

Simple RAG is the right starting point for almost any project — it's fast to build, easy to reason about, and surprisingly effective for well-structured documents with direct questions. If your corpus has clear, self-contained sections and your users ask specific questions that map to one or two document regions, Simple RAG may be all you ever need.

The signal that you need to go further is usually obvious in practice: users getting vague answers, chunks returning from the wrong sections, or queries that span multiple document parts coming back incomplete. Those are the failure modes that the subsequent techniques in this collection are designed to fix.

---

## Benchmarking: Simple RAG as Your Baseline

One of the most important functions of Simple RAG is serving as the **benchmark** against which every advanced technique is measured. Before adding complexity, always:

1. Implement Simple RAG
2. Create a representative evaluation set (20-50 question/answer pairs)
3. Measure Recall@k, Precision@k, and answer faithfulness
4. Apply an advanced technique
5. Measure again and compare

If a complex technique doesn't outperform Simple RAG on your specific dataset and query distribution, don't use it. Occam's razor applies strongly in production RAG systems: unnecessary complexity is a maintenance burden with no quality payoff.

---

## Comparison with Related Approaches

| Approach | Knowledge Source | Update Mechanism | Private Data | Hallucination Risk |
|----------|-----------------|-----------------|--------------|-------------------|
| Bare LLM | Training data | Retrain (expensive) | No | High |
| Fine-tuned LLM | Training + fine-tune data | Fine-tune (medium cost) | Yes, but baked in | Medium |
| **RAG** | **Live document store** | **Add/update documents** | **Yes, dynamic** | **Low** |
| Parametric only | Training data | None | No | High |

RAG's key advantage over fine-tuning is **dynamism**: the knowledge base can be updated, expanded, or corrected without touching the model.

---

## Summary

Simple RAG establishes the three-phase pattern — **index → retrieve → generate** — that all advanced RAG techniques build upon. It is simultaneously a useful tool in its own right and the conceptual foundation without which no advanced technique can be understood.

Understanding Simple RAG at a deep level means understanding:
- How embeddings represent semantic meaning numerically
- How FAISS finds nearest neighbors in high-dimensional space
- How chunking affects retrieval granularity
- How prompt design controls answer grounding
- How each parameter affects the cost/quality/latency triangle

Every advanced RAG technique you learn from this point is an incremental improvement to one or more of these fundamentals. Master this foundation, and every subsequent technique becomes intuitive rather than mysterious.
