# Context Chunk Headers: Teaching Your Chunks Where They Come From

> **Technique:** Context Chunk Header  
> **Complexity:** Beginner-Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

One of the most underappreciated problems in RAG is **context amnesia**: a chunk doesn't know what document it came from, what topic it belongs to, or how it relates to the broader knowledge base. When a user asks "What does the Q3 report say about revenue growth?", the retrieved chunk might contain the answer — but if it reads "Revenue grew by 23% compared to the previous quarter", how does the LLM know which Q3 report produced this fact?

**Context Chunk Headers** solve this at the indexing stage by prepending each chunk with structured metadata — specifically, the document's title and a brief summary. This technique is deceptively simple but has outsized impact on retrieval quality and answer attribution.

Anthropic popularized a related technique called "contextual retrieval" in 2024, demonstrating that prepending chunk-specific context reduced retrieval failure rates by 49%.

---

## The Problem: Chunk Decontextualization

When you slice a 50-page document into 200 chunks of 1,000 characters each, each chunk loses its origin story. Consider this chunk:

> "The results showed significant improvement in the treated group, with a 34% reduction in symptoms compared to baseline. Adverse effects were minimal and resolved without intervention."

Completely useless without context. Which study? What condition? What treatment? The embedding of this chunk captures "medical results" but cannot distinguish it from thousands of similar chunks across your entire document collection.

Now consider the same chunk with a header:

> **Document: Stanford Trial of Drug X for Condition Y (2024)**  
> *This paper presents results of a randomized controlled trial testing Drug X against a placebo for Condition Y in 450 adult patients.*
>
> "The results showed significant improvement in the treated group, with a 34% reduction in symptoms compared to baseline. Adverse effects were minimal and resolved without intervention."

The embedding of the enriched chunk now captures the document context and dramatically improves retrieval precision when someone asks "What were the results of the Stanford Drug X trial?"

---

## How Context Chunk Header Works

### Indexing Pipeline

```
Document text
    ↓
[LLM] Generate document title
    ↓
[LLM] Generate 2-3 sentence document summary
    ↓
Chunk the document text
    ↓
Prepend "TITLE: {title}\nSUMMARY: {summary}\n\n" to each chunk
    ↓
Embed enriched chunks → FAISS vector store
```

The title and summary are generated **once per document** (not per chunk), making this very cost-efficient. If a document has 200 chunks, you pay for 2 LLM calls regardless.

### Query Pipeline

```
User query
    ↓
Embed query
    ↓
Search against enriched chunk embeddings
    ↓
Return top-k enriched chunks (with header context intact)
    ↓
LLM generates answer using context-rich chunks
```

---

## Code Deep Dive

### Generating the Document Title

```python
def get_document_title(self, document_text: str, guidance: str = "") -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "What is the title of the following document?\n\n"
                "Your response MUST be the title of the document, and nothing else. "
                "DO NOT respond with anything else.\n\n"
                f"{guidance}\n\n"
                f"DOCUMENT\n{document_text}"
            )
        }
    ]
    return self.llm.chat(messages)
```

The `guidance` parameter allows the user to provide hints — e.g., "This is a financial report from 2023". This is especially useful for documents without obvious titles embedded in their text (like scanned PDFs or exported reports).

### Generating the Document Summary

```python
def get_document_summary(self, document_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You generate concise document summaries for use as context in RAG systems."
        },
        {
            "role": "user",
            "content": (
                "Write a 2-3 sentence summary of this document. "
                "Focus on what the document is about and who it pertains to.\n\n"
                f"Document:\n{document_text}"
            )
        }
    ]
    return self.llm.chat(messages)
```

The system prompt is important here: it tells the model it's writing for a RAG system, not for human readers. This subtly changes the output — the model produces summaries optimized for retrieval rather than executive consumption.

### Building the Enriched Chunks

```python
def index_document(self, text: str, title_guidance: str = ""):
    # Generate document-level metadata (one-time cost)
    title = self.get_document_title(text[:3000], guidance=title_guidance)
    summary = self.get_document_summary(text[:3000])
    
    # Build header string
    header = f"Document Title: {title}\nDocument Summary: {summary}\n\n"
    
    # Chunk the original text
    chunks = chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    
    # Prepend header to every chunk
    enriched_chunks = [header + chunk for chunk in chunks]
    
    # Embed and store
    documents = [Document(content=ec, metadata={"title": title, "chunk_index": i})
                 for i, ec in enumerate(enriched_chunks)]
    documents = self.embedder.embed_documents(documents)
    self.vector_store.add_documents(documents)
```

Note the use of `text[:3000]` for title and summary generation. You don't need the entire document to understand what it's about — the first few thousand characters are almost always sufficient, and this keeps LLM costs predictable.

---

## Embedding Enrichment: What Changes in the Vector Space

When you prepend a header, you're changing the embedding of every chunk. The resulting vector captures both the chunk content and the document context simultaneously.

**Without header**, the chunk vector lies in a region of vector space defined by its local content.

**With header**, the chunk vector is a blend of:
- The document topic (from title/summary)
- The section content (from the chunk text)

This blended representation is more discriminative. A query like "What does the Apple annual report say about iPhone sales?" will now correctly rank chunks from the Apple report above chunks from Samsung or Google reports that discuss phone sales.

---

## Multi-Document Advantages

The header technique shines brightest in **multi-document corpora**. When your vector store contains:

- 20 annual reports
- 50 research papers
- 100 product manuals

...without headers, a query for "power consumption specifications" might retrieve the 3 most similar text chunks regardless of which product they describe. With headers, the query "power consumption specifications for the RTX 4090" will strongly prefer chunks from the RTX 4090 manual even if the chunk content is identical ("TDP: 450W") because the document title breaks the tie.

---

## Enriching vs. Restructuring

An important design note: the header is **prepended for embedding purposes**, but you have a choice about what to send to the LLM at query time:

1. **Send enriched chunks (with header)**: The LLM sees the document context, making attribution easier.
2. **Send raw chunks (without header)**: The LLM sees cleaner, more focused content.

The implementation prepends the header for both embedding and generation. This is the right default: attribution is valuable, and the header overhead is small compared to a 1,000-character chunk.

---

## When to Use Context Chunk Headers

**Best for:**
- Multi-document corpora where source attribution matters
- Documents without clear internal structure (exported PDFs, reports)
- Applications where users care about *which source* answered their question
- Corpora with many documents covering similar topics (risk of cross-document contamination)

**Less critical when:**
- Single-document RAG (there's only one source anyway)
- Documents with rich internal structure already (headers, footers, metadata)
- The corpus is very large and re-indexing cost must be minimized

---

## Comparison: With vs. Without Headers

| Scenario | Without Headers | With Headers |
|---------|----------------|--------------|
| Single document | Equivalent | Slightly better |
| Multi-document (different topics) | Good | Good |
| Multi-document (similar topics) | Poor | **Excellent** |
| Attribution in answer | Hard | Natural |
| Index cost | Low | Low (2 extra LLM calls per doc) |
| Chunk size impact | None | +100-200 chars per chunk |

---

## Summary

Context Chunk Headers are one of the most cost-effective improvements in the RAG toolkit. With just two LLM calls per document (title + summary), every chunk in your corpus gains a document identity — transforming dumb text windows into context-aware knowledge units.

The technique is particularly powerful in multi-document settings where without headers, the retrieval system is essentially guessing which source to trust. With headers, the document origin is embedded into the vector representation itself, giving the similarity search the signal it needs to make source-aware retrievals.

Start simple: add this technique to your existing RAG pipeline and watch cross-document retrieval precision improve immediately.
