# Hierarchical Indices: Top-Down Navigation Through Long Documents

---

## Introduction

When you have a 500-page document — a legal code, a technical manual, a comprehensive research corpus — every page has some text, and many pages discuss the same broad topic. A query about "sentencing guidelines for white-collar crime" might have relevant content on 30 different pages. Standard RAG retrieves the top-k chunks by similarity — but which 3 chunks from those 30 pages are the most relevant? And can we even trust that FAISS's similarity search finds the right 30 from 3,000 total chunks?

The fundamental scaling problem: as corpus size grows, the "signal-to-noise ratio" in retrieval falls. With 3,000 chunks, the difference in cosine similarity between rank-1 and rank-50 may be only 0.05 — a practically meaningless distinction. Everything seems equally similar when the embedding space is crowded.

**Hierarchical Indices** solve this with a two-level navigation strategy modeled after how humans search through large documents. First, identify which *sections* of the document are relevant. Then, drill into those sections to find the specific *passages* that answer the question.

This is exactly how a lawyer uses a legal textbook: table of contents → chapter → section → specific clause. Hierarchical Indices encode this intuition into the RAG retrieval process.

---

## The Two-Tier Architecture

```
Level 1: Summary Index (coarse navigation)
┌──────────────────────────────────────────────────────────────┐
│  Summary-1: "Introduction to contract law principles..."     │  ← covers pages 1-15
│  Summary-2: "Breach of contract and remedies..."            │  ← covers pages 16-32
│  Summary-3: "Tort liability frameworks..."                   │  ← covers pages 33-58
│  Summary-4: "Intellectual property rights overview..."       │  ← covers pages 59-89
│  Summary-5: "Criminal law — elements and standards..."       │  ← covers pages 90-140
└──────────────────────────────────────────────────────────────┘
         ↓  Query: "what is the statute of limitations for IP infringement?"
         ↓  Level 1 search → Summary-4 (highest similarity)
         ↓
Level 2: Detail Index (precision retrieval)
┌──────────────────────────────────────────────────────────────┐
│  All detail chunks from pages 59-89 (Summary-4's scope)     │
│  ├── "Copyright protection lasts 70 years beyond author..."  │
│  ├── "Patent infringement actions must be filed within..."   │
│  ├── "Statute of limitations: IP infringement 3 years..."   │ ← answer is here
│  └── "Trade secret misappropriation claims..."               │
└──────────────────────────────────────────────────────────────┘
         ↓  Level 2 search → exact chunk with statute of limitations
```

By first identifying the IP section (Level 1), Level 2 search runs over only 30 chunks (pages 59-89) instead of the full 3,000. The signal-to-noise ratio is dramatically improved — all 30 chunks are about IP law, so the most relevant one stands out clearly.

---

## Building the Two-Level Index

### Page-Level Processing

The first step groups text into logical sections. The implementation uses page-level summaries as the natural grouping unit:

```python
def _split_into_pages(self, text: str, page_size: int = 2000) -> List[str]:
    """
    Split document text into page-sized segments.
    
    'page_size' is a character count approximation of one printed page.
    For a typical 12pt document: 2000 chars ≈ 1 printed page.
    """
    pages = []
    start = 0
    while start < len(text):
        end = start + page_size
        page_text = text[start:end]
        if page_text.strip():
            pages.append(page_text)
        start = end  # no overlap for pages — they're navigation units, not retrieval units
    return pages
```

### Summary Generation for Each Section

Each page/section gets an LLM-generated summary that captures its scope and key topics:

```python
def _generate_page_summary(self, page_text: str, page_num: int) -> str:
    """
    Generate a navigation-optimized summary for a document section.
    
    Unlike a human-facing summary that emphasizes conclusions or insights,
    this summary is designed to be maximally useful for routing queries
    to the right section. It emphasizes TOPICS and ENTITIES, not findings.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Create a concise summary of the following text section. "
                "The summary will be used to help route queries to the right "
                "section of a document. Focus on:\n"
                "1. The main topics covered\n"
                "2. Key entities (people, organizations, laws, products) mentioned\n"
                "3. The type of content (definitions, procedures, data, arguments)\n"
                "4. Important terms someone might search for\n\n"
                "Keep the summary to 2-4 sentences. Do not include page numbers."
            )
        },
        {
            "role": "user",
            "content": f"Summarize this document section:\n\n{page_text}"
        }
    ]
    summary = self.llm.chat(messages).strip()
    return summary
```

**Navigation-optimized vs. informational summaries:**

A standard executive summary might say: "This section argues that strict liability standards for product manufacturers provide better consumer protection outcomes than negligence-based standards."

A navigation-optimized summary says: "Covers product liability law, including strict liability standards, negligence-based standards, consumer protection, manufacturer responsibilities, and landmark cases including Greenman v. Yuba Power Products (1963)."

The second form is more useful for routing. A query about "Greenman v Yuba Power Products" would match the navigation summary but not the executive summary.

### Building Both FAISS Indices

```python
def _build_hierarchical_index(self, text: str):
    """
    Build two separate FAISS indices:
    - summary_index: one vector per document section (coarse navigation)
    - detail_index: one vector per chunk (precision retrieval)
    """
    pages = self._split_into_pages(text)
    
    summary_documents = []
    detail_documents = []
    
    for page_idx, page_text in enumerate(pages):
        # Level 1: Generate and index section summary
        summary = self._generate_page_summary(page_text, page_idx)
        
        summary_doc = Document(
            content=summary,
            metadata={
                "page_index": page_idx,
                "page_content": page_text,  # stored for chunk splitting
                "type": "summary"
            }
        )
        summary_documents.append(summary_doc)
        
        # Level 2: Split each page into fine chunks and index
        page_chunks = chunk_text(page_text, chunk_size=500, chunk_overlap=50)
        
        for chunk_idx, chunk_content in enumerate(page_chunks):
            detail_doc = Document(
                content=chunk_content,
                metadata={
                    "page_index": page_idx,
                    "chunk_index": chunk_idx,
                    "type": "detail"
                }
            )
            detail_documents.append(detail_doc)
    
    # Embed and index summaries (Level 1)
    summary_documents = self.embedder.embed_documents(summary_documents)
    self.summary_index.add_documents(summary_documents)
    
    # Embed and index detail chunks (Level 2)
    detail_documents = self.embedder.embed_documents(detail_documents)
    self.detail_index.add_documents(detail_documents)
    
    # Store cross-reference: page_index → which detail chunks belong to it
    self.page_to_chunks = {
        page_idx: [
            d for d in detail_documents
            if d.metadata["page_index"] == page_idx
        ]
        for page_idx in range(len(pages))
    }
    
    print(f"Summary index: {len(summary_documents)} section summaries")
    print(f"Detail index: {len(detail_documents)} detail chunks")
```

---

## The Two-Phase Query Process

```python
def _retrieve_hierarchical(
    self,
    query: str,
    top_sections: int = 3,    # Level 1: how many sections to explore
    chunks_per_section: int = 2  # Level 2: how many chunks from each section
) -> List[str]:
    
    query_embedding = self.embedder.embed_text(query)
    
    # ── Level 1: Section navigation ─────────────────────────────────────
    # Find the top-k most relevant sections by summary similarity
    summary_results = self.summary_index.search(query_embedding, k=top_sections)
    
    relevant_page_indices = [
        result.document.metadata["page_index"]
        for result in summary_results
    ]
    
    print(f"Level 1: Selected sections: {relevant_page_indices}")
    
    # ── Level 2: Precision retrieval within selected sections ─────────────
    # Collect all detail chunks from selected sections
    candidate_chunks = []
    for page_idx in relevant_page_indices:
        page_chunks = self.page_to_chunks.get(page_idx, [])
        candidate_chunks.extend(page_chunks)
    
    print(f"Level 2: Searching within {len(candidate_chunks)} candidate chunks")
    
    # Build temporary mini-index from candidate chunks
    mini_index = FAISSVectorStore(dimension=self.embedder.dimension)
    mini_index.add_documents(candidate_chunks)
    
    # Find best chunks within this filtered set
    final_k = top_sections * chunks_per_section
    detail_results = mini_index.search(query_embedding, k=final_k)
    
    return [r.document.content for r in detail_results]
```

**The mini-index trick**: Instead of filtering the full detail index by page — which requires reading metadata for every vector — a temporary mini FAISS index is built from only the candidate chunks. This is fast (building a 100-vector FAISS index is nearly instantaneous) and gives exact FAISS search semantics within the filtered set.

---

## How the Index Size Scales

For a 500-page document with `page_size=2000` chars and `chunk_size=500`:

| Index | Vectors | Memory |
|-------|---------|--------|
| Summary Index | 500 (one per page) | 3 MB |
| Detail Index | 2,000 (4 chunks/page avg) | 12 MB |
| **Total** | **2,500** | **15 MB** |

At query time, Level 1 narrows from 500 summaries to `top_sections=3` pages → 12 detail chunks. Level 2 searches only these 12. Compared to searching all 2,000 detail chunks directly, this is a 167× reduction in the Level 2 search space.

---

## The Information Gain of Two-Stage Navigation

### Why Level 1 Matters

Direct search on 2,000 detail chunks (500 chars each):
- Cosine similarity between rank-1 and rank-20 chunk: ~0.04 difference (crowded space)
- Many irrelevant-but-topically-adjacent chunks compete

Section-guided search on 12 chunks (from 3 relevant sections):
- All 12 chunks are from sections identified as relevant by the summary
- Cosine similarity between rank-1 and rank-5 within these 12: ~0.15 difference (cleaner space)
- The relevant chunk stands out clearly

### When Does Navigation Help Most?

| Corpus Size | Benefit |
|------------|---------|
| <50 pages | Marginal — direct search already precise |
| 50-200 pages | Noticeable improvement for domain-specific queries |
| 200-500 pages | **Significant** — where hierarchical wins clearly |
| 500+ pages | **Essential** — direct search degrades badly |

---

## LLM Call Count at Index Time

For a 500-page document:
- Summary generation: 500 LLM calls
- Each call: ~1,500 input tokens (page text) + ~100 output tokens (summary)

At `gpt-4o-mini` pricing (~$0.00015/1K input + $0.0006/1K output):
- Cost per call: ~$0.000285
- Total for 500 pages: ~**$0.14**

For a 1,000-page document: ~$0.28. Effectively free for the quality improvement delivered.

---

## Configuration Guide

```python
HierarchicalIndicesRAG(
    file_path="large_document.pdf",
    
    # Page size: characters per section in the summary index
    # Larger → fewer, coarser sections; Smaller → more, finer sections
    page_size=2000,  # ~1 printed page
    
    # Detail chunk size within each section
    chunk_size=500,   # ~4 detail chunks per page at default page_size
    chunk_overlap=50,
    
    # Query parameters
    top_sections=3,       # how many sections to explore in Level 1
    chunks_per_section=2, # how many chunks to select from each section
    
    # k = top_sections × chunks_per_section detail chunks reach the LLM
    # Default: 3 × 2 = 6 chunks
)
```

**Tuning `top_sections`:**
- `top_sections=1`: Precision-first. Explores only the most relevant section. Fast, but may miss multi-section answers.
- `top_sections=3`: Balanced (default). Covers the primary and two supporting sections.
- `top_sections=5`: Recall-first. Explores broadly. Better for multi-part questions, slower.

---

## Limitations and Mitigations

### Cross-Section Answers

Some queries span multiple sections that Level 1 might not detect together:
- Query: "How does contract breach affect IP licensing?" → requires Contract section + IP section

**Mitigation**: Increase `top_sections` to 5 or add a query decomposition step before Level 1 that identifies which aspects of the query might be in different sections.

### Section Boundary Issues

If a topic spans a page boundary, its summary may capture only half. The other half is in the summary for the adjacent page.

**Mitigation**: Add 20% overlap between adjacent pages at the page-splitting stage, or use semantic section detection (similar to semantic chunking) rather than fixed page sizes.

---

## When to Use Hierarchical Indices

Hierarchical Indices are most valuable for very long documents — textbooks, legal codes, technical manuals — anything with 200+ pages and clearly delineated sections. If your users regularly need attribution at the section level ("this came from Chapter 4"), or if retrieval from the wrong section is a recurring failure mode, this two-level architecture pays for itself quickly.

For shorter documents under 50 pages, or corpora where every section covers the same broad topic, the navigation layer adds complexity without a meaningful signal improvement — direct search over a smaller chunk set will be equally precise. Similarly, if index build latency is a hard constraint, note that section summarization requires one LLM call per page; for dense documents this can be substantial.

---

## Summary

Hierarchical Indices bring human intuition about document navigation into RAG retrieval. By building a two-level structure — coarse section summaries for navigation, fine detail chunks for retrieval — the system avoids searching a crowded, noisy embedding space of thousands of similar chunks.

Level 1 narrows the haystack. Level 2 finds the needle. Together, they combine the memory efficiency of dense retrieval with the precision of targeted search within a focused context — exactly what long-document RAG needs to scale without sacrificing answer quality.
