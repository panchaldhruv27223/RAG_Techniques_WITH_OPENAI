# Context Chunk Headers: Teaching Your Chunks Where They Come From

## Introduction

Every chunk in a standard RAG index suffers from a form of amnesia. It knows its own content — the 1,000 characters of text it contains — but has no memory of where it came from, what document it belongs to, what section it was in, or why it matters. When embedded, the resulting vector only captures the chunk's local content.

This creates a problem that becomes severe in multi-document corpora: two chunks from completely different documents may have nearly identical content and nearly identical embeddings if they discuss the same topic. FAISS has no way to prefer the chunk from the relevant document — it can only measure vector distance.

**Context Chunk Headers** fix this by giving every chunk a permanent identity. Before embedding, each chunk is prepended with a structured header containing the document's title and a brief summary. The resulting embedding vector captures both the local content *and* the document-level context, making every chunk's embedding unique and attributable.

Anthropic's "Contextual Retrieval" paper (2024) demonstrated that this concept — prepending context to chunks before embedding — reduces retrieval failure rates by approximately 49%. That single number justifies why this technique has become a near-universal best practice.

---

## The Root Problem: Chunk Decontextualization

When you slice a document into hundreds of chunks, each chunk loses its narrative thread. Consider these two chunks from completely different documents in a multi-document RAG system:

**Chunk from Document A** (Stanford oncology trial):
> "The treated group showed a statistically significant improvement, with a 34% reduction in tumor markers at 12 weeks. Side effects were mild and self-resolving, including nausea in 18% of patients and fatigue in 22%."

**Chunk from Document B** (Johnson & Johnson drug trial):
> "The treatment arm demonstrated meaningful clinical response, with 34% improvement in primary endpoints at week 12. Adverse events were predominantly mild: gastrointestinal effects in 20% of subjects, fatigue reported in 19%."

These two chunks are *nearly semantically identical*. Their embeddings will be extremely close in vector space. A query about either trial will retrieve both chunks indistinguishably — FAISS cannot tell them apart from the content alone.

**With headers:**

Chunk A becomes:
```
Document Title: Stanford Phase III Trial of Drug X for Pancreatic Cancer (2024)
Document Summary: This paper reports results of a 450-patient randomized controlled 
trial testing Drug X against placebo for Stage III pancreatic cancer over 12 weeks.

The treated group showed a statistically significant improvement, with a 34% reduction 
in tumor markers at 12 weeks...
```

Chunk B becomes:
```
Document Title: J&J Trial: Efficacy of Compound Y in Colorectal Adenocarcinoma (2023)
Document Summary: Phase II trial conducted across 8 centers evaluating Compound Y 
vs. standard of care in 312 colorectal cancer patients.

The treatment arm demonstrated meaningful clinical response, with 34% improvement 
in primary endpoints...
```

Now the embeddings are completely distinct. A query specifically about pancreatic cancer trials will strongly prefer Chunk A. A query about colorectal cancer will prefer Chunk B. The header has given each chunk a permanent, distinguishing identity.

---

## How Context Chunk Header Works

### The Indexing Pipeline

```
Document file (PDF, CSV, etc.)
        │
        ▼
 Extract full text
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │  [LLM] Generate document title      (1 call)     │
 │         ↓                                        │
 │  [LLM] Generate 2-3 sentence summary (1 call)    │
 │         ↓                                        │
 │  Build header string:                            │
 │  "Document Title: {title}\n                      │
 │   Document Summary: {summary}\n\n"               │
 └──────────────────────────────────────────────────┘
        │
        ▼
 Chunk the document text
    [chunk_1, chunk_2, ..., chunk_n]
        │
        ▼
 Prepend header to every chunk:
    [header + chunk_1, header + chunk_2, ..., header + chunk_n]
        │
        ▼
 Embed all enriched chunks → FAISS vector store
```

Two LLM calls per document. That's it. Regardless of whether the document has 10 chunks or 1,000 chunks, the cost is exactly 2 LLM calls. This makes Context Chunk Headers one of the highest-ROI techniques in this collection.

---

## Implementation Walkthrough

### Document Title Generation

```python
def get_document_title(
    self, 
    document_text: str, 
    guidance: str = ""
) -> str:
    """
    Extract or generate the document title.
    
    Uses the first ~3000 characters of the document, which almost always
    contains the title, abstract, or executive summary. This avoids
    processing the entire document just for title extraction.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "What is the title of the following document?\n\n"
                "Your response MUST be ONLY the title — no explanation, "
                "no prefix, no quotation marks. Just the title text itself.\n\n"
                f"Guidance (if any): {guidance}\n\n"
                f"DOCUMENT CONTENT:\n{document_text[:3000]}"
            )
        }
    ]
    title = self.llm.chat(messages).strip()
    return title
```

The `guidance` parameter is a powerful usability feature. Users can supply hints like:
- `"This is a Q4 2023 earnings call transcript from Apple"` → helps identify undated documents
- `"This is an internal HR policy document"` → helps when the document lacks a formal title
- `"This report was authored by Goldman Sachs in January 2024"` → fills in missing attribution

Without guidance, if the document starts with a table of contents or begins mid-paragraph (common in exported PDFs), the LLM might generate a vague title. Guidance prevents this.

**Why truncate to 3,000 characters?** For title extraction, the first 3,000 characters (roughly one page) are almost always sufficient. Processing less text means faster API response and lower cost. For a 500-page document, you avoid transmitting 297 unnecessary pages for title extraction.

### Document Summary Generation

```python
def get_document_summary(self, document_text: str) -> str:
    """
    Generate a 2-3 sentence summary optimized for retrieval use.
    
    Unlike a summary written for human reading, this is tailored
    to maximize the distinctiveness of the embeddings it produces —
    capturing what the document is about and who/what it pertains to.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You generate concise document summaries for use as context "
                "headers in a RAG knowledge retrieval system. Your summary "
                "will be prepended to every chunk from this document before "
                "embedding, so it must capture the document's topic, scope, "
                "and key entities in a way that makes each chunk findable by "
                "people searching for its content."
            )
        },
        {
            "role": "user",
            "content": (
                "Write a 2-3 sentence summary of this document. Include:\n"
                "- What the document is about (topic/purpose)\n"
                "- Who or what it primarily discusses (entities, people, organizations)\n"
                "- The document's scope or timeframe (if relevant)\n\n"
                f"Document (first portion):\n{document_text[:3000]}"
            )
        }
    ]
    return self.llm.chat(messages).strip()
```

**The "retrieval-optimized summary" distinction:** A human-facing executive summary might emphasize conclusions, recommendations, or narrative arc. A retrieval-optimized summary emphasizes entities, topics, and scope — exactly the dimensions that make embeddings distinctive.

For example, a human-facing summary might say:
> "This landmark study demonstrates that Drug X significantly reduces tumor burden and should be considered first-line therapy for eligible patients."

A retrieval-optimized summary says:
> "Phase III randomized controlled trial evaluating Drug X (pembrolizumab) for Stage III pancreatic cancer. Conducted by Stanford Medical Center in 2024 with 450 patients over 12 weeks. Primary endpoints: tumor marker reduction and overall survival."

The second version contains far more discriminating embedding signal: specific drug name, specific cancer type, specific institution, specific year, specific patient count, specific duration, specific endpoints. Every detail becomes a searchable dimension in the embedding space.

### Building the Enriched Chunks

```python
def index_document(
    self, 
    text: str, 
    title_guidance: str = ""
) -> int:
    """
    Full indexing pipeline with header enrichment.
    Returns number of chunks indexed.
    """
    # Step 1: Generate document-level metadata
    title = self.get_document_title(text, guidance=title_guidance)
    summary = self.get_document_summary(text)
    
    # Step 2: Build the header — prepended to every chunk
    header = (
        f"Document Title: {title}\n"
        f"Document Summary: {summary}\n\n"
    )
    
    # Step 3: Chunk the raw text
    raw_chunks = chunk_text(
        text, 
        chunk_size=self.chunk_size,      # e.g., 1000
        chunk_overlap=self.chunk_overlap  # e.g., 200
    )
    
    # Step 4: Enrich each chunk with the header
    enriched_chunks = [header + chunk for chunk in raw_chunks]
    
    # Step 5: Build Document objects with metadata for later filtering/attribution
    documents = [
        Document(
            content=enriched_chunk,
            metadata={
                "title": title,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "source_file": self.file_path
            }
        )
        for i, enriched_chunk in enumerate(enriched_chunks)
    ]
    
    # Step 6: Batch embed all enriched chunks
    documents = self.embedder.embed_documents(documents)
    
    # Step 7: Add to FAISS
    self.vector_store.add_documents(documents)
    
    return len(documents)
```

### What the Header Looks Like in Practice

For a climate science paper, the header might be:

```
Document Title: Global Warming Projections Under RCP 8.5 Scenario — IPCC AR6 Report (2023)
Document Summary: Sixth assessment report from the Intergovernmental Panel on Climate 
Change projecting global temperature rise scenarios under the high-emission RCP 8.5 
pathway through 2100. Analyzes impacts on sea levels, extreme weather frequency, and 
biodiversity loss across six continental regions.

[chunk text follows]
```

Every chunk in this document — all 500 of them — carries this header. When a user asks "What does the IPCC say about sea level rise by 2100?", the query vector aligns strongly with:
1. "IPCC AR6 Report" (from the header)
2. "sea levels" (from the header)
3. "sea level rise" content (from the chunk itself)

All three signals align simultaneously, producing near-perfect retrieval.

---

## The Vector Space Effect of Headers

### Without Headers

In embedding space, chunks about the same topic from different documents cluster together:

```
Vector Space:
  
  [climate sea level chunks from 12 different documents]
         ↑
  All very close together — indistinguishable by document origin
```

A query about sea level rise retrieves the 3 vectors closest to the query, regardless of which documents they came from. You might get 2 chunks from a general science article and 1 from the specific IPCC report you wanted.

### With Headers

Headers break these clusters apart by document:

```
Vector Space:
  
  [IPCC AR6 sea level chunks] ←── distinct region
  [NASA sea level monitor chunks] ←── distinct region  
  [journal article sea level chunks] ←── distinct region
```

Each document's chunks occupy a distinct sub-region of the space because every chunk carries the unique identity of its document in its embedding. Source-specific queries now retrieve source-specific chunks.

---

## Multi-Document Corpus: Where Headers Shine

### Example: Enterprise Knowledge Base

Imagine a RAG system containing:
- 50 annual reports from different companies (2019-2024)
- 100 product specification documents
- 200 regulatory filings

**Query**: "What were Apple's iPhone revenue figures in fiscal year 2023?"

**Without headers**: Any annual report discussing revenue or any product report mentioning iPhone might be retrieved. The query has no source-aware signal.

**With headers**: Every Apple 2023 annual report chunk contains `"Document Title: Apple Inc. Annual Report FY2023"` in its embedding. The query matches this specific header, routing directly to Apple's 2023 financials rather than any of the other 249 documents.

### Example: Medical Literature RAG

**Query**: "What were the efficacy results for pembrolizumab in the KEYNOTE-158 trial?"

**Without headers**: Every chunk discussing pembrolizumab efficacy from thousands of papers is in the same embedding cluster. FAISS can't distinguish KEYNOTE-158 from KEYNOTE-590 or KEYNOTE-811.

**With headers**: The KEYNOTE-158 paper chunks all contain `"Document Title: Pembrolizumab in KEYNOTE-158 Trial: Tumor Mutational Burden as a Predictive Biomarker"`. The trial name in the query matches the trial name in the header — precise retrieval.

---

## Handling Edge Cases

### Documents Without Clear Titles

Some documents — exported reports, scraped web pages, internal wikis — don't have obvious titles. Two strategies:

1. **Use the `guidance` parameter**: `index_document(text, title_guidance="This is a Q3 2023 sales report for the US region")`
2. **LLM inference from content**: The title-generation prompt will infer from the content, producing something like "US Region Q3 2023 Sales Performance Report" even without an explicit title field.

### Very Long Documents

For documents where even the first 3,000 characters don't contain useful metadata (e.g., raw data exports that start with rows of numbers), pass the actual title and summary directly:

```python
# Instead of using LLM generation
header = (
    f"Document Title: Q3 2023 US Sales Data Export\n"
    f"Document Summary: Raw sales transaction data for the United States region, "
    f"Q3 2023 (July-September). Contains order ID, customer, product, region, "
    f"and revenue columns for 45,000 transactions.\n\n"
)
```

### Frequently Updated Documents

If a document is updated daily (e.g., a live database export), re-indexing with header regeneration adds 2 LLM calls. This is minimal. However, if you're updating thousands of documents daily, batch LLM calls or cache titles and summaries separately.

---

## Token Impact Analysis

Adding a header increases each chunk's size by ~100-200 characters (roughly 25-50 tokens). For a document with 300 chunks:
- Additional tokens embedded: 300 × 50 = 15,000 tokens
- Additional embedding cost at $0.02/1M tokens: $0.0003 (negligible)
- Token increase in context sent to LLM per query (k=3 chunks): ~3 × 50 = 150 tokens
- Additional generation cost at $0.0006/1K tokens: $0.0001 per query (negligible)

The token overhead is essentially zero at any practical scale.

---

## Comparison: Standard vs. Header-Enriched Retrieval

| Scenario | Standard RAG | Context Chunk Header |
|---------|-------------|----------------------|
| Single document Q&A | Equivalent | Marginal improvement |
| 5-10 documents, different topics | Good | Slightly better |
| 50+ documents, same topic domain | **Poor** (cross-doc confusion) | **Excellent** (document-aware) |
| Named entity queries (specific report, specific study) | Unreliable | Highly reliable |
| Attribution ("what does Doc X say?") | No mechanism | Native support |
| Retrieval of specific named entities | Hit-or-miss | Consistent |
| LLM call overhead | 0 per document | 2 per document |
| Token overhead per chunk | 0 | ~50 tokens |

---

## When to Use Context Chunk Headers

Context Chunk Headers are a near-universal improvement for any multi-document system. If your corpus has five or more documents — especially ones that cover overlapping topics — this technique is effectively mandatory. The risk of cross-document confusion grows with corpus size, and headers are the most efficient corrective measure: just two LLM calls per document at index time, regardless of how many chunks that document produces.

The case is strongest when users ask source-specific questions ("what does the Q3 report say about margins?") or when source attribution matters for compliance or trust. In these scenarios, headers transform retrieval from "find the most similar text in the corpus" into "find the most relevant text *from the right document*."

For single-document pipelines or corpora where every document covers a completely different topic, the improvement is marginal — there's no cross-document contamination to prevent. But there's almost never a good reason to skip headers when you have meaningful document overlap. The token overhead is negligible (~50 tokens per chunk), and the quality uplift in multi-document settings is consistently positive.

---

## Summary

Context Chunk Headers are one of the simplest and most impactful improvements in the RAG toolkit. With just two LLM calls per document (title generation + summary generation), every chunk in your corpus receives a permanent document identity baked directly into its embedding vector.

In single-document RAG, this provides modest improvement. In multi-document RAG, it's transformative — the difference between "retrieve the most similar text in the entire corpus" and "retrieve the most relevant content *from the right document*." Source-aware retrieval becomes a native capability rather than an afterthought.

Add this technique whenever you have more than one document in your knowledge base. The return on that two-LLM-call investment is consistently positive.
