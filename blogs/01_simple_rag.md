# Simple RAG: The Foundation of Retrieval-Augmented Generation

> **Technique:** Simple RAG  
> **Complexity:** Beginner  
> **Key Libraries:** `openai`, `faiss-cpu`, `PyMuPDF`

---

## Introduction

Every sophisticated system starts with a solid foundation. In the world of Retrieval-Augmented Generation (RAG), that foundation is **Simple RAG** — a straightforward pipeline that connects a knowledge base to a language model, enabling it to answer questions grounded in real documents rather than relying purely on its training data.

If you're new to RAG, this is your starting point. If you're a seasoned practitioner, revisiting the basics regularly keeps your mental model sharp.

---

## The Core Problem RAG Solves

Large Language Models (LLMs) like GPT-4 are trained on enormous corpora of text, giving them broad world knowledge. But they have two critical limitations:

1. **Knowledge cutoff**: They don't know about events after their training date.
2. **Private data blindness**: They have no access to your proprietary documents, internal reports, or customer records.

RAG bridges this gap by giving the LLM a retrieval system — a way to look up relevant information at query time before generating a response. Think of it as giving the model open-book access instead of forcing it to work from memory.

---

## How Simple RAG Works

The pipeline has three distinct phases:

### Phase 1: Indexing (Offline)

Before any query is answered, the source documents must be processed and stored:

```
Raw Documents (PDF / CSV)
        ↓
   Text Extraction
        ↓
   Text Chunking  (fixed-size chunks with overlap)
        ↓
   Embed each chunk  →  Vector Store (FAISS)
```

**Chunking** breaks long documents into manageable pieces. A typical configuration uses a chunk size of 1,000 characters with a 200-character overlap. The overlap ensures that context spanning two adjacent chunks isn't lost.

**Embedding** converts each text chunk into a dense numerical vector using a model like `text-embedding-3-small`. These vectors capture the semantic meaning of the text, allowing similarity-based search later.

### Phase 2: Retrieval (Online, per query)

When a user asks a question:

```
User Query
    ↓
Embed query → query vector
    ↓
FAISS similarity search against all chunk vectors
    ↓
Top-k most semantically similar chunks returned
```

FAISS (Facebook AI Similarity Search) performs an extremely fast nearest-neighbor search in high-dimensional vector space. With even millions of chunks, retrieval takes milliseconds.

### Phase 3: Generation (Online, per query)

```
User Query + Retrieved Chunks (context)
    ↓
Prompt construction
    ↓
LLM (GPT-4o-mini) generates grounded answer
    ↓
Response returned to user
```

The LLM is instructed to answer using *only* the provided context. This is key: it grounds the response in real document text, dramatically reducing hallucination.

---

## Code Deep Dive

Here's how the pipeline is structured in the implementation:

```python
class SimpleRAG:
    def __init__(self, embedding_model="text-embedding-3-small",
                 chat_model="gpt-4o-mini", temperature=0.0):
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.llm = OpenAIChat(model_name=chat_model, temperature=temperature)
```

The system is initialized with three modular components:
- **`OpenAIEmbedder`**: Wraps OpenAI's embedding API
- **`FAISSVectorStore`**: Manages the vector index for fast similarity search
- **`OpenAIChat`**: Handles LLM generation

### Indexing a PDF

```python
def index_pdf(self, file_path, chunk_size=1000, chunk_overlap=200):
    text = read_pdf(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = [Document(content=c, metadata={"chunk_index": i})
                 for i, c in enumerate(chunks)]
    documents = self.embedder.embed_documents(documents)
    self.vector_store.add_documents(documents)
```

`chunk_text` uses a sliding-window approach: it moves through the text creating chunks of `chunk_size` characters, advancing by `chunk_size - chunk_overlap` each step. This is simple and reliable.

### Querying

```python
def query(self, question, k=3):
    query_embedding = self.embedder.embed_text(question)
    results = self.vector_store.search(query_embedding, k=k)
    context = "\n\n".join([r.document.content for r in results])
    
    messages = [
        {"role": "system", "content": f"Answer using this context:\n\n{context}"},
        {"role": "user", "content": question}
    ]
    return self.llm.chat(messages)
```

The query vector is compared against all stored chunk vectors. The top `k` (typically 3) most similar chunks are assembled into a context string and passed to the LLM.

---

## CSV Support

Simple RAG also handles structured CSV data. Each row is converted to a key-value string representation:

```
Column1: Value1 | Column2: Value2 | Column3: Value3
```

This flattened representation allows the same vector-search mechanism to work on tabular data, though more sophisticated approaches (like SQL generation) may be better for complex analytical queries.

---

## Key Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |
| `k` | 3 | Number of chunks to retrieve |
| `embedding_model` | `text-embedding-3-small` | Embedding model for semantic search |
| `chat_model` | `gpt-4o-mini` | LLM for answer generation |
| `temperature` | 0.0 | Generation determinism (0 = fully deterministic) |

---

## Strengths and Limitations

### Strengths

- **✅ Simple to implement and understand**: The pipeline has no complex dependencies.
- **✅ Works out of the box**: For most Q&A use cases on well-structured documents, Simple RAG delivers solid results.
- **✅ Low latency**: With FAISS, retrieval is extremely fast even at scale.
- **✅ Cost-efficient**: Only the retrieved chunks (not the entire document) are sent to the LLM.

### Limitations

- **❌ Fixed chunk boundaries**: Splitting by character count ignores sentence and paragraph structure. A chunk might end mid-sentence, cutting critical context.
- **❌ No query understanding**: The raw query is embedded as-is. Ambiguous or poorly phrased questions may retrieve irrelevant chunks.
- **❌ Context isolation**: Each chunk is retrieved independently. If the answer spans multiple conceptually related chunks that aren't near each other in the document, they may not all be retrieved.
- **❌ No relevance filtering**: All top-k results are included in the context regardless of their actual relevance to the query. Noisy context can mislead the LLM.

---

## When to Use Simple RAG

Simple RAG is the right choice when:

- You're prototyping or building a proof of concept
- Your documents are well-structured and self-contained in each section
- Query complexity is low (factual lookups, definitions, etc.)
- You need a quick, reliable baseline to benchmark other techniques against

---

## What Comes Next

The limitations of Simple RAG motivated the entire field of advanced RAG research. Every technique in this series addresses one or more of these gaps:

- **Proposition Chunking**: Breaks text into atomic facts instead of fixed windows
- **Semantic Chunking**: Uses embedding similarity to find natural topic boundaries
- **Context Enrichment Window**: Expands retrieved chunks with neighboring text
- **Reranking**: Adds a second-pass relevance filtering step
- **CRAG / Self-RAG**: Evaluates retrieval quality before answering

Understanding Simple RAG deeply is essential before exploring these advanced variants — knowing what you're improving upon makes each enhancement intuitive rather than arbitrary.

---

## Summary

Simple RAG is where every RAG journey begins. It establishes the three-phase pattern — **index → retrieve → generate** — that all advanced techniques build upon. Its elegance lies in its simplicity: a semantic search engine paired with a language model, each doing what it does best.

Master this foundation, and every advanced RAG technique becomes an incremental improvement rather than a black box.
