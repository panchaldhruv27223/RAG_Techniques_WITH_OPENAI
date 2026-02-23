# Hierarchical Indices: Searching Big to Small for Precise Retrieval

> **Technique:** Hierarchical Indices  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

When searching a large document, two needs compete: finding the *right section* (broad scope) and finding the *specific detail* (narrow precision). Standard RAG with flat chunking struggles to satisfy both simultaneously — small chunks give precision but may miss section context; large chunks give context but dilute precision.

**Hierarchical Indices** solves this with a two-tier architecture: first search summaries to identify relevant sections, then search detailed chunks *within those sections* to find precise answers. It mirrors how a human expert searches: scan the table of contents to find the right chapter, then read within that chapter.

This technique is especially powerful for long documents (50+ pages) or large corpora with distinct topic sections.

---

## The Two-Tier Architecture

```
┌──────────────────────────────────────────┐
│           Tier 1: Summary Index          │
│  Section (page) summaries — broad scope  │
│  "Page 3: Discusses greenhouse effect    │
│   mechanisms and CO2 accumulation"       │
└────────────────┬─────────────────────────┘
                 │ Query → find relevant sections
                 ↓
┌──────────────────────────────────────────┐
│           Tier 2: Detail Index           │
│  Fine chunks filtered to matched pages  │
│  "...CO2 traps infrared radiation by..." │
└──────────────────────────────────────────┘
                 │
                 ↓
            Final Chunks → LLM Answer
```

**Tier 1** indexes page-level summaries. Each entry represents ~one page or major section.  
**Tier 2** indexes fine-grained chunks (e.g., 500 chars). Each chunk carries its page number in metadata.

At query time: Tier 1 identifies which pages are relevant → Tier 2 is searched only within those pages.

---

## Index Building

### Page-Level Summaries (Tier 1)

```python
def _build_summary_index(self, pages: List[str]) -> None:
    summary_docs = []
    for page_num, page_text in enumerate(pages):
        # Generate a concise summary for each page
        summary = self._generate_page_summary(page_text, page_num)
        summary_docs.append(
            Document(
                content=summary,
                metadata={"page_num": page_num, "type": "summary"}
            )
        )
    summary_docs = self.summary_embedder.embed_documents(summary_docs)
    self.summary_store.add_documents(summary_docs)
```

```python
def _generate_page_summary(self, page_text: str, page_num: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Write a concise 2-3 sentence summary that captures the main topics "
                "and key information on this page. Focus on what someone might search for."
            )
        },
        {
            "role": "user",
            "content": f"Page {page_num + 1}:\n\n{page_text}"
        }
    ]
    return self.llm.chat(messages)
```

The summary prompt explicitly asks for "what someone might search for" — optimizing the summary for retrieval use rather than human reading.

### Fine-Grained Chunks (Tier 2)

```python
def _build_detail_index(self, pages: List[str]) -> None:
    detail_docs = []
    for page_num, page_text in enumerate(pages):
        chunks = chunk_text(page_text, 
                           chunk_size=self.detail_chunk_size,  # e.g., 500 chars
                           chunk_overlap=self.detail_chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            detail_docs.append(
                Document(
                    content=chunk,
                    metadata={
                        "page_num": page_num,          # ← critical for filtering
                        "chunk_idx": chunk_idx,
                        "type": "detail"
                    }
                )
            )
    detail_docs = self.detail_embedder.embed_documents(detail_docs)
    self.detail_store.add_documents(detail_docs)
```

The `page_num` metadata is the bridge between the two tiers — it enables Tier 2 searches to be filtered to only the pages selected by Tier 1.

---

## Query-Time Two-Stage Retrieval

```python
def retrieve(self, query: str, top_pages: int = 3, top_chunks: int = 5) -> List[str]:
    # ── Tier 1: Find relevant pages ──
    query_embedding = self.summary_embedder.embed_text(query)
    summary_results = self.summary_store.search(query_embedding, k=top_pages)
    
    relevant_pages = set(
        r.document.metadata["page_num"] for r in summary_results
    )
    print(f"Relevant pages: {[p+1 for p in relevant_pages]}")
    
    # ── Tier 2: Search detail chunks within relevant pages only ──
    detail_results = self.detail_store.search(query_embedding, k=top_chunks * 5)
    
    # Filter to relevant pages
    filtered = [
        r for r in detail_results
        if r.document.metadata["page_num"] in relevant_pages
    ]
    
    # Take top-k after filtering
    filtered.sort(key=lambda r: r.score, reverse=True)
    top_results = filtered[:top_chunks]
    
    return [r.document.content for r in top_results]
```

**Critical implementation detail**: Tier 2 retrieves a large candidate set first (`k=top_chunks * 5`) then filters by page, rather than filtering first. This ensures enough candidates survive the filter to fill `top_chunks` slots even if some pages have fewer matching chunks.

---

## Why Page Filtering Matters

Without the hierarchical filter, a query like "What are the effects of CO2 on temperature?" in a climate document might retrieve chunks from:
- Page 3: CO2 mechanisms ← relevant
- Page 12: Economic costs of emissions ← adjacent topic
- Page 7: Methane and other gases ← related but not CO2/temperature
- Page 19: Carbon capture technologies ← different angle

With hierarchical filtering after Tier 1 identifies pages 3 and 4 as relevant:
- All 5 retrieved chunks come from pages 3 and 4
- No contamination from other sections
- Context coherence is dramatically improved

---

## Configuration Parameters

```python
class HierarchyIndicesRAG:
    def __init__(self,
                 file_path: str,
                 summary_chunk_size: int = 5000,    # ~1 page per summary entry
                 detail_chunk_size: int = 500,      # fine-grained chunks
                 detail_chunk_overlap: int = 50,
                 top_pages: int = 2,                # pages selected by Tier 1
                 top_chunks: int = 5,               # final chunks from Tier 2
                 summary_model: str = "gpt-4o-mini"):
```

**Key tuning decisions:**

| Parameter | Effect |
|-----------|--------|
| `summary_chunk_size` | Larger = fewer, broader summaries; good for structured docs |
| `detail_chunk_size` | Smaller = more precise retrieval; trade-off is more chunks |
| `top_pages` | More pages = more context but more noise in Tier 2 |
| `top_chunks` | Final context sent to LLM |

---

## Multi-Document Extension

Hierarchical indices extend naturally to multi-document corpora:

- **Tier 1**: Document summaries (one per document, not per page)
- **Tier 2**: Section or page chunks within matched documents

This architecture scales to thousands of documents. The first tier narrows from 1,000 documents to 5-10 relevant ones; the second tier then finds the precise passage within those documents.

---

## Comparison to Flat Indexing

| Aspect | Flat RAG | Hierarchical Indices |
|--------|---------|----------------------|
| Index complexity | Single FAISS store | Two FAISS stores + summaries |
| Index time | Low | Medium (LLM summary generation) |
| Query time | One search | Two sequential searches |
| Section coherence | Poor (chunks from any page) | Excellent (filtered by page) |
| Long document performance | Degrades | Maintains quality |
| Multi-document support | Native | Excellent with doc-level Tier 1 |

---

## When to Use Hierarchical Indices

**Best for:**
- Long documents (30+ pages) with distinct sections
- Multi-document corpora organized by topic or source
- Applications where section-level attribution matters
- Technical documentation with clear chapter structure (API docs, manuals, specifications)

**Less ideal when:**
- Documents are short and homogeneous
- Topics are uniformly distributed without section structure
- Build cost (LLM summary generation) is constrained

---

## Summary

Hierarchical Indices brings a top-down search strategy to RAG, mirroring how human experts navigate large documents: identify the right section, then find the precise detail. The two-tier architecture — page summaries for navigation, fine chunks for precision — delivers section-coherent retrieval that flat indexing fundamentally cannot match.

For long documents and multi-document corpora, the quality improvement justifies the added index complexity. The result is a retrieval system that finds not just semantically similar chunks, but the specifically relevant passage within the specifically relevant section.
