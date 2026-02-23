# Context Enrichment Window: Giving Retrieved Chunks Their Neighbors

> **Technique:** Context Enrichment Window  
> **Complexity:** Beginner-Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

Imagine you're reading a 500-page textbook and you dog-ear a single page because it contains a relevant sentence. When you return to it, that one sentence needs the pages before and after it to make sense. RAG systems face the same problem: they retrieve precise chunks but often strip them of the surrounding narrative context needed to produce a coherent answer.

**Context Enrichment Window** (also called "Small-to-Big Retrieval" or "Sentence Window Retrieval") solves this elegantly: retrieve small, precise chunks for high accuracy, then expand them with their neighboring chunks before sending to the LLM. You get the best of both worlds — precise retrieval and rich context.

---

## The Two-Layer Architecture

This technique introduces a conceptual separation between the **retrieval unit** and the **synthesis unit**:

| Layer | Purpose | Size |
|-------|---------|------|
| Retrieval unit | What you search against | Small (precise) |
| Synthesis unit | What you send to the LLM | Large (context-rich) |

Standard RAG uses identical units for both. Context Enrichment Window decouples them.

### Visual Intuition

```
Document chunks: [C0] [C1] [C2] [C3] [C4] [C5] [C6]

Query hits C3.

Standard RAG sends: [C3]

Context Window (num_neighbors=1) sends: [C2] [C3] [C4]
Context Window (num_neighbors=2) sends: [C1] [C2] [C3] [C4] [C5]
```

The retrieved chunk `C3` is the "anchor". Its neighbors provide the surrounding context that makes `C3` interpretable.

---

## How the Pipeline Works

### Indexing

```
Document text
    ↓
Split into chunks (with overlap)
    ↓
Store ALL chunks in ChunkStore (simple list, indexed by position)
    ↓
Embed chunks → FAISS vector store
    (each embedding stores its chunk_index in metadata)
```

Both the `ChunkStore` (for positional access) and the `FAISSVectorStore` (for semantic search) are populated with the same chunks. The `chunk_index` in the FAISS metadata is the bridge between the two stores.

### Query Time

```
User query
    ↓
Embed query → vector
    ↓
FAISS search → top-k chunks (each has chunk_index in metadata)
    ↓
For each retrieved chunk_index i:
    → ChunkStore.get_window(center=i, num_neighbors=N)
    → Returns chunks [i-N, ..., i, ..., i+N]
    ↓
Deduplicate and merge windowed chunks
    ↓
LLM generates answer using expanded context
```

---

## Code Deep Dive

### The ChunkStore

The `ChunkStore` is the heart of this technique — a simple but purposeful data structure:

```python
class ChunkStore:
    def __init__(self):
        self._chunks: List[str] = []
        self._doc_id: str = ""

    def add_chunks(self, chunks: List[str], doc_id: str = "doc_0") -> None:
        self._chunks = chunks
        self._doc_id = doc_id

    def get(self, index: int) -> Optional[str]:
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None

    def get_window(self, center: int, num_neighbors: int) -> List[str]:
        start = max(0, center - num_neighbors)
        end = min(len(self._chunks), center + num_neighbors + 1)
        return [self._chunks[i] for i in range(start, end)]
```

`get_window` handles edge cases automatically — if `center=0` and `num_neighbors=2`, `start=max(0,-2)=0`, so you don't request non-existent chunks. The window is always bounded by document start and end.

### The ContextEnrichmentRetriever

```python
class ContextEnrichmentRetriever:
    def __init__(self, embedding_model, chunk_size=1000, 
                 chunk_overlap=200, num_neighbors=1, k=3):
        self.num_neighbors = num_neighbors
        self.k = k
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.chunk_store = ChunkStore()

    def index_document(self, text: str, doc_id: str = "doc_0") -> int:
        chunks = chunk_text(text, chunk_size=self.chunk_size,
                           chunk_overlap=self.chunk_overlap)
        
        # Store in ChunkStore for positional access
        self.chunk_store.add_chunks(chunks=chunks, doc_id=doc_id)
        
        # Build Documents with chunk_index metadata
        documents = [
            Document(
                content=chunk,
                metadata={"doc_id": doc_id, "chunk_index": i,
                         "total_chunks": len(chunks)}
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # Embed and store in FAISS
        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)
        return len(chunks)
```

**Critical design detail**: Every document stored in FAISS carries its `chunk_index` in metadata. This index is the key used to look up the window from `ChunkStore` at query time.

### Retrieval with Window Expansion

```python
def retrieve_with_context(self, query: str) -> List[str]:
    # Step 1: Retrieve top-k chunks from FAISS
    query_embedding = self.embedder.embed_text(query)
    results = self.vector_store.search(query_embedding, k=self.k)
    
    # Step 2: Expand each result to its window
    context_windows = []
    seen_indices = set()
    
    for result in results:
        center_idx = result.document.metadata["chunk_index"]
        window = self.chunk_store.get_window(center_idx, self.num_neighbors)
        
        # Deduplicate: windows from adjacent hits may overlap
        window_text = "\n".join(window)
        if window_text not in seen_indices:
            context_windows.append(window_text)
            seen_indices.add(window_text)
    
    return context_windows
```

---

## The Overlap Problem and Deduplication

When two retrieved chunks are adjacent (e.g., chunk 5 and chunk 6 are both top-k results), their windows will overlap significantly:

- Chunk 5 window (neighbors=1): chunks [4, **5**, 6]
- Chunk 6 window (neighbors=1): chunks [5, **6**, 7]

Without deduplication, chunk 5 and 6 appear twice. The implementation tracks seen window text to avoid redundant context, which keeps the LLM prompt clean.

---

## Tuning `num_neighbors`

The `num_neighbors` parameter controls the window width. Choosing it right depends on your chunk size and document structure:

| `num_neighbors` | Total chunks sent | Best for |
|----------------|------------------|----------|
| 0 | 1 (no window) | Identical to standard RAG |
| 1 | 3 | Well-structured docs, medium chunk size |
| 2 | 5 | Dense technical docs, small chunk size |
| 3+ | 7+ | Highly narrative content, tiny chunks |

**Practical guideline**: With chunk size 1,000 chars and `num_neighbors=1`, you're sending ~3,000 characters per retrieved hit. With `num_neighbors=2`, ~5,000. Most LLMs handle 10,000-30,000 characters of context comfortably, so `num_neighbors=1` or `2` is the sweet spot for typical configurations.

---

## When Context Windows Matter Most

### Narrative Continuity

In documents like research papers, books, or legal briefs, meaning flows across paragraphs. A retrieved chunk about "the study found X" is meaningless without the preceding chunk that explains the study design. Window expansion reconnects this broken narrative.

### Mid-Sentence Chunk Boundaries

With fixed-size chunking, chunks sometimes split in the middle of a thought:

- **Chunk 12 ends**: "...the primary mechanism responsible for this effect is"
- **Chunk 13 starts**: "the activation of the RAAS pathway which..."

If your query retrieves chunk 13, without its predecessor it contains a sentence fragment. Window expansion retrieves chunk 12 automatically.

### Multi-Step Reasoning

Some questions require assembling information from consecutive sections. "How does the introduction relate to the conclusion?" benefits from context around both the introduction and conclusion chunks.

---

## Comparison: Standard RAG vs. Context Enrichment Window

| Aspect | Standard RAG | Context Enrichment Window |
|--------|-------------|--------------------------|
| Retrieval unit | 1000-char chunk | 1000-char chunk (same precision) |
| LLM context | 1 chunk per hit | 3-5 chunks per hit |
| Boundary artifacts | Frequent | Rare |
| Narrative coherence | Poor | Good |
| Token cost | Lower | Higher (3-5x per hit) |
| Implementation complexity | Simple | Low-Medium |

---

## Summary

Context Enrichment Window is one of the highest ROI improvements you can make to a Simple RAG system. The core insight — that retrieval precision and synthesis context are different needs requiring different chunk sizes — is immediately intuitive once stated.

By maintaining a `ChunkStore` for positional lookups alongside the vector store for semantic search, this technique decouples the two concerns cleanly. You retrieve with precision and reason with context. The result is answers that actually make sense, even when the relevant information spans multiple paragraphs.

This technique is a sensible default for most production RAG systems, especially those dealing with long-form documents like reports, papers, or manuals.
