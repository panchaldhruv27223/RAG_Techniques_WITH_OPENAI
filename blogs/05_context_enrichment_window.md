# Context Enrichment Window: Precision Retrieval with Rich Context Delivery

## Introduction

Retrieval precision and context richness are fundamentally at odds in standard RAG. To maximize precision — ensuring retrieved chunks closely match the query — you want small chunks. Small chunks are specific enough that they match exactly the semantic meaning of the query. But small chunks lack the surrounding context needed for an LLM to write a coherent, fully-informed answer.

To maximize context richness, you want large chunks. Large chunks contain more surrounding information, giving the LLM a richer basis for generation. But large chunks are imprecise — their low-granularity embeddings represent an aggregate of many concepts rather than a specific point of meaning.

Standard RAG forces a compromise: pick a chunk size between "too small" and "too large." The compromise satisfies neither goal fully.

**Context Enrichment Window** resolves this tension by decoupling retrieval units from delivery units. The system maintains two representations of the same document simultaneously:

- **Small index** (child chunks): high-precision, fine-grained units for retrieval
- **Full text store** (original text): complete document for extracting rich context windows

When a child chunk is retrieved, the system looks up its position in the original text and returns a larger *window* of content centered on that position — providing the LLM with rich context while having benefited from precision retrieval.

Use small units to *find* information. Use large windows to *deliver* information. Retrieval and generation are now separately optimized.

---

## The Core Insight: Decoupling Two Separate Problems

The standard chunking approach conflates two distinct requirements:

| Requirement | Optimal unit | Reason |
|------------|-------------|--------|
| Retrieval precision | Small (~100-300 chars) | Small units have concentrated semantic signal; embeddings are specific |
| Generation context | Large (~1500-3000 chars) | LLMs need surrounding context to write coherent answers |

These requirements conflict. Context Enrichment Window satisfies both by serving different units at each step:

```
INDEXING TIME:
    Document → small child chunks (200 chars) → embedded → FAISS

QUERY TIME:
    User query → embed → FAISS retrieves child chunk C42
                              ↓
             C42 is chunk #42 in original document
                              ↓
             Return text from char 8200 to char 12200 (±2000 char window)
                              ↓
             LLM receives rich 4000-char window centered on the answer
```

The precision of a 200-character embedding. The richness of a 4000-character context. Both at once.

---

## How the Two-Layer Architecture Works

### Layer 1: The FAISS Child Index (for precision retrieval)

Small chunks — the "child" level — are indexed for retrieval. Their size is optimized for embedding quality, not for human readability:

```python
# Child chunks: small, precise, optimized for retrieval
child_chunks = chunk_text(
    text,
    chunk_size=200,     # much smaller than standard 1000
    chunk_overlap=0     # no overlap needed — windows handle boundary coverage
)
```

Each child chunk gets a positional identifier: `chunk_index=i`. This index is stored as metadata alongside the chunk's text in FAISS.

### Layer 2: The Original Text Store (for window extraction)

The full document text is kept verbatim. When a child chunk is retrieved at query time, the system uses the chunk's character position to extract a window around it:

```python
# Store full doc text keyed by doc_id
self.full_texts[doc_id] = original_text

# Retrieve window centered on a child chunk's position
def get_context_window(self, doc_id, chunk_idx, child_size=200, window_size=2000):
    full_text = self.full_texts[doc_id]
    
    # Calculate child chunk's center character position
    chunk_start = chunk_idx * child_size  # no overlap, so position is exact
    chunk_center = chunk_start + (child_size // 2)
    
    # Expand symmetrically by window_size on each side
    window_start = max(0, chunk_center - window_size)
    window_end = min(len(full_text), chunk_center + window_size)
    
    return full_text[window_start:window_end]
```

If child chunk #42 starts at character 8400 in the document, the window might span characters 6400 to 12400 — 6000 characters of rich continuous text, centered on the precise 200 characters where the answer was found.

---

## Complete Code Walkthrough

### The ContextEnrichmentWindowRAG Class

```python
class ContextEnrichmentWindowRAG:
    def __init__(
        self,
        file_path: str,
        child_chunk_size: int = 200,       # small child chunks for retrieval
        parent_window_size: int = 2000,    # ± characters to expand around child
        k: int = 3,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini"
    ):
        self.child_chunk_size = child_chunk_size
        self.parent_window_size = parent_window_size
        self.k = k
        
        # Two separate data structures
        self.vector_store = FAISSVectorStore(...)   # child chunks only
        self.full_texts = {}                         # full text by doc_id
        
        # Load and index the document
        text = read_pdf(file_path)
        self.doc_id = file_path
        self.full_texts[self.doc_id] = text
        self.index_document(text)
```

### Indexing: Building the Child Index

```python
def index_document(self, text: str):
    # Step 1: Create small, precise child chunks
    children = chunk_text(
        text, 
        chunk_size=self.child_chunk_size, 
        chunk_overlap=0  # Intentionally zero overlap
    )
    
    # Step 2: Build Document objects with positional metadata
    documents = []
    for i, chunk_content in enumerate(children):
        doc = Document(
            content=chunk_content,
            metadata={
                "chunk_index": i,           # position for window extraction
                "doc_id": self.doc_id,      # document identity
                "char_start": i * self.child_chunk_size  # character position
            }
        )
        documents.append(doc)
    
    # Step 3: Batch embed and add to FAISS
    documents = self.embedder.embed_documents(documents)
    self.vector_store.add_documents(documents)
    
    print(f"Indexed {len(documents)} child chunks")
    print(f"Full text stored: {len(text)} characters available for windowing")
```

**Why zero overlap for child chunks?** With window expansion, overlap is redundant. When window=2000 chars, child chunk #42 will be fully contained within the window returned for child chunk #41 (since 200 < 2000). Every piece of text will always be reachable through a neighboring chunk's window. Overlap would only increase index size and redundancy without improving coverage.

### Query Time: Retrieval + Window Expansion

```python
def query(self, question: str) -> Tuple[str, List[str]]:
    # Step 1: Embed question
    question_embedding = self.embedder.embed_text(question)
    
    # Step 2: Retrieve top-k small child chunks (precision retrieval)
    child_results = self.vector_store.search(question_embedding, k=self.k)
    
    # Step 3: Expand each retrieved child to its context window
    context_windows = []
    for result in child_results:
        chunk_idx = result.document.metadata.get("chunk_index", 0)
        doc_id = result.document.metadata.get("doc_id", self.doc_id)
        
        # Get the rich window centered on this child chunk
        window = self._get_context_window(doc_id, chunk_idx)
        context_windows.append(window)
    
    # Step 4: Deduplicate overlapping windows
    unique_windows = self._deduplicate_windows(context_windows)
    
    # Step 5: Generate answer from rich context
    answer = self._generate_answer(question, unique_windows)
    
    return answer, unique_windows

def _get_context_window(self, doc_id: str, chunk_idx: int) -> str:
    full_text = self.full_texts.get(doc_id, "")
    if not full_text:
        return ""
    
    # Precise character position of child chunk center
    char_position = chunk_idx * self.child_chunk_size + (self.child_chunk_size // 2)
    
    # Expand the window symmetrically
    start = max(0, char_position - self.parent_window_size)
    end = min(len(full_text), char_position + self.parent_window_size)
    
    return full_text[start:end]
```

### Window Deduplication

When multiple child chunks are retrieved from nearby positions (e.g., chunk #42 and chunk #43 are both retrieved), their windows will overlap substantially. Including both windows in the context would waste LLM tokens with duplicate content.

```python
def _deduplicate_windows(self, windows: List[str]) -> List[str]:
    """
    Remove windows that are largely subsets of other windows.
    Uses a simple prefix-matching approach.
    """
    if not windows:
        return windows
    
    unique = [windows[0]]
    for window in windows[1:]:
        # Check if this window substantially overlaps with any accepted window
        is_duplicate = False
        for accepted in unique:
            # If 80%+ of the new window is already contained in an accepted window
            overlap_chars = len(set(window[:200]) & set(accepted[:200]))  # simplified
            if window[:100] in accepted:  # prefix check
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(window)
    
    return unique
```

A more robust implementation would compute edit distance or use character-level hashing to identify substantial duplicates. The core principle: if window A and window B overlap by >70%, include only the longer one (which contains more unique content).

---

## Worked Example: The Boundary Value Problem

Consider a document about aviation regulation with this critical passage:

```
...Aviation authorities worldwide have adopted different standards for minimum 
spacing between aircraft during final approach. The critical threshold — the 
point below which collision risk increases exponentially — is determined by 
multiple factors including aircraft size, speed differential, atmospheric 
conditions, and runway configuration. The FAA specifies 3 nautical miles for 
light aircraft following heavy aircraft during IFR conditions, while extending 
this to 6 nautical miles in VMC when wake turbulence dissipation cannot be 
guaranteed by ground observation...
```

**Query**: "What is the minimum spacing for light aircraft following heavy aircraft?"

### Standard RAG Failure Mode

If this passage spans chunk #47 and chunk #48 (boundary at mid-sentence):

Chunk #47:
> "Aviation authorities worldwide have adopted different standards for minimum spacing between aircraft. The critical threshold — the point below which collision risk increases exponentially — is determined by multiple factors..."

Chunk #48:
> "...including aircraft size, speed differential, atmospheric conditions, and runway configuration. The FAA specifies 3 nautical miles for light aircraft following heavy aircraft during IFR conditions, while extending..."

FAISS retrieves chunk #47 (most semantically similar to "minimum spacing"). But chunk #47 doesn't contain the actual number! The answer is in chunk #48. The LLM generates a vague response: "The FAA specifies minimum spacing based on multiple factors including aircraft size..."

### Context Window Solution

Child chunk retrieved: #47 (200 chars, high similarity)  
Window expansion: ±2000 chars around chunk #47's center

The 4000-char window includes both chunk #47 AND chunk #48 AND surrounding context, containing the complete passage. The LLM reads "3 nautical miles for light aircraft following heavy aircraft during IFR conditions" and answers precisely.

---

## Window Size Optimization

Choosing the right `parent_window_size` is the main tuning challenge:

| `parent_window_size` | Total context returned | Best for |
|---------------------|----------------------|---------|
| 500 chars | ~1000 chars/chunk | Simple, self-contained facts |
| 1000 chars | ~2000 chars/chunk | Standard Q&A documents |
| 2000 chars | ~4000 chars/chunk | Complex technical documents with multi-part answers |
| 5000 chars | ~10000 chars/chunk | Very dense technical content with extensive cross-references |

**Cost impact:** Larger windows → more tokens in the LLM prompt → higher cost per query. A 2000-char window with k=3 results in ~12,000 chars ≈ 3,000 tokens of context. At `gpt-4o-mini` pricing, this is still very inexpensive (~$0.002/query).

**Quality impact:** Beyond a certain size, additional context begins to hurt rather than help. LLMs have a "lost in the middle" phenomenon: relevant information in the beginning and end of a long context is well-attended to, but information in the middle of a very long context may be missed. Window sizes above 5,000 chars (per chunk) should be used cautiously.

**Recommended approach**: Start at `window_size=1000` and evaluate retrieval quality. Increase to 2000 if multi-sentence answers are being truncated. Decrease to 500 if tokens cost is a concern and answers are adequate at lower window sizes.

---

## Memory Architecture

The two-layer architecture has a simple memory profile:

| Component | Memory usage (for 100-page PDF) |
|-----------|--------------------------------|
| FAISS index (child chunks) | ~1.5 MB (embedding vectors × dimension) |
| Full text store (raw text) | ~1.5 MB (pure text at ~15 chars/byte) |
| **Total** | **~3 MB** |

For large corpora (1000+ documents), the full text store grows proportionally to document count. This is a moderate memory overhead but remains manageable for server deployments.

---

## Comparison: Three Retrieval Strategies

| Strategy | Retrieval Unit | Context Delivered | Precision | Richness |
|---------|---------------|------------------|-----------|---------|
| Standard RAG (chunk_size=1000) | 1000 chars | 1000 chars | Medium | Medium |
| Small chunks (chunk_size=200) | 200 chars | 200 chars | **High** | **Low** |
| **Context Window** (child=200, window=2000) | 200 chars (find) | 4000 chars (deliver) | **High** | **High** |

Context Window achieves the best of both worlds. The precision graph looks like small-chunk RAG. The context richness looks like large-chunk RAG. Neither compromise has been made.

---

## When to Use Context Enrichment Window

This technique shines on documents where meaning unfolds across multiple sentences or paragraphs — technical manuals, research papers, legal contracts, medical literature — where a single sentence rarely stands fully on its own. If you're seeing retrieval precision that looks correct (the right sections are being found) but generated answers feel incomplete or truncated, context windowing is often the right fix. It directly addresses the gap between "found the right place" and "gave the LLM enough to answer well."

For documents where each chunk is already a discrete, self-contained unit — FAQ databases, product listings, structured reference tables — the enrichment step adds overhead without meaningful gain. Similarly, if your documents are short enough to fit entirely in a single LLM context window, you can skip the child-chunk architecture and just retrieve the full document. And if storing full text alongside the FAISS index is infeasible due to corpus size, the technique's positional expansion logic won't have source material to draw from.

---

## Summary

Context Enrichment Window elegantly resolves the retrieval-generation conflict at the heart of RAG design. By maintaining two representations — small child chunks for precision retrieval, and the full original text for window extraction — the technique delivers both the precision of fine-grained indexing and the richness of large-context delivery.

The implementation requires only modest additional storage (the full text alongside the FAISS index) and adds negligible query latency (window extraction is a text slice, not an LLM call). The result is a system where "retrieve the right place" and "give enough context to answer well" are no longer competing objectives.

For documents where answers span multiple paragraphs or where retrieval precision is high but answers feel incomplete, Context Enrichment Window is often the single most impactful structural change you can make to a RAG pipeline.
