# Relevant Segment Extraction: Context That Flows Like the Original Document

**Source:** `context_enrichment/relevant_segment_extraction_openai.py` · **Libraries:** `openai`, `faiss-cpu`, `numpy`

---

## Introduction

Standard top-k retrieval returns *isolated chunks* — fragments ripped from their context. Chunk 47 might be a perfect match for your query, but chunk 46 and 48, which together with 47 form a coherent argument, are not retrieved. The LLM answers from fragment 47 alone, missing the complete picture.

**Relevant Segment Extraction (RSE)** solves this by recognizing that relevant information in a document isn't randomly distributed — **it clusters**. When chunk 47 is highly relevant, there's a high probability that nearby chunks (46, 48, 49) are also relevant, because documents are written with local coherence. RSE exploits this clustering property to retrieve *contiguous segments* of text — multiple consecutive chunks that together form a logically complete unit — rather than isolated fragments.

The mathematical backbone is a **constrained maximum-sum-subarray** algorithm: find the contiguous sequence of chunks whose combined relevance score is maximized. This gives you not just the peak of relevance, but the full relevant region surrounding it.

---

## The Core Insight: Documents Have Spatial Coherence

Consider a 50-page technical report. A user asks "what were the methodology limitations?" Standard retrieval returns the 3 chunks with the highest embedding similarity to the query. RSE asks:

> Among all contiguous sections of this document, which section contains the most concentrated relevant information?

The difference is profound:
- **Standard retrieval**: 3 isolated sentences from pages 12, 28, and 41
- **RSE**: A complete 5-paragraph subsection from pages 27-29 labeled "Limitations" — logically coherent, contextually complete

---

## The Algorithm: Maximum-Sum-Subarray for Text

### Step 1: Zero-Overlap Chunking (Critical Design Requirement)

RSE requires chunks with **zero overlap**:

```python
# Standard chunking — overlap makes boundaries ambiguous
chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)

# RSE chunking — zero overlap for clean segment reconstruction
chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
```

With overlapping chunks, chunk 5 and chunk 6 share text. Concatenating them would duplicate content. Zero overlap ensures that concatenating chunks 5, 6, 7 exactly reconstructs the original text from that region.

### Step 2: Dual Storage

Chunks are stored in two structures simultaneously:

```python
@dataclass
class ChunkMeta:
    doc_id: str
    chunk_index: int    # position within document (0-based)
    text: str           # full text content
```

- **FAISS vector store**: for fast similarity search
- **Key-value store** (`KVStore`): for chunk lookup by `(doc_id, chunk_index)` — enables neighbor access

```python
class KVStore:
    """
    Key-value store for chunks indexed by (doc_id, chunk_index).
    Enables O(1) lookup of any chunk by its position — crucial for
    segment reconstruction after finding the optimal subarray bounds.
    """
    def __init__(self):
        self._data: Dict[Tuple[str, int], str] = {}
    
    def put(self, doc_id: str, chunk_index: int, text: str):
        self._data[(doc_id, chunk_index)] = text
    
    def get(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self._data.get((doc_id, chunk_index))
    
    def get_range(self, doc_id: str, start: int, end: int) -> List[str]:
        """Retrieve chunks [start, end] inclusive — used for segment reconstruction."""
        return [
            self._data[(doc_id, i)]
            for i in range(start, end + 1)
            if (doc_id, i) in self._data
        ]
```

### Step 3: Relevance Scoring

After FAISS retrieval, each candidate chunk is scored for relevance:

```python
def _score_chunks(
    self,
    query_embedding: List[float],
    candidate_results: List[RetrievalResult],
    threshold: float = 0.3
) -> Dict[Tuple[str, int], float]:
    """
    Compute relevance value for each candidate chunk.
    
    RSE chunk value = (absolute_similarity + rank_based_score) / 2 - threshold
    
    The threshold parameter controls what counts as "positive" relevance.
    Chunks below the threshold get negative values — penalizing them in
    the subarray sum. This ensures the algorithm doesn't extend segments
    into clearly irrelevant regions just to include more text.
    
    threshold=0.3:  Only chunks >30% similar are considered beneficial
                    to include in a segment. Below this, extension is
                    penalized — keeps segments tight and relevant.
    """
    n_candidates = len(candidate_results)
    chunk_values = {}
    
    for rank, result in enumerate(candidate_results):
        # Absolute similarity: how similar is this chunk to the query?
        absolute_score = cosine_similarity(
            query_embedding,
            result.document.embedding
        )
        
        # Rank-based score: top-ranked chunks get more weight
        # rank=0 → rank_score=1.0, rank=n-1 → rank_score→0
        rank_score = 1.0 - (rank / n_candidates)
        
        # Fuse: average of absolute + rank-based
        fused = (absolute_score + rank_score) / 2.0
        
        # Subtract threshold: makes "weakly relevant" chunks slightly negative
        value = fused - threshold
        
        doc_id = result.document.metadata["doc_id"]
        chunk_index = result.document.metadata["chunk_index"]
        chunk_values[(doc_id, chunk_index)] = value
    
    return chunk_values
```

### Step 4: Maximum-Sum-Subarray Segment Finding

This is the critical algorithmic step. Classic Kadane's algorithm finds the contiguous subarray with the maximum sum — but RSE applies it with document-awareness:

```python
def _find_best_segment(
    self,
    doc_id: str,
    chunk_values: Dict[Tuple[str, int], float],
    max_segment_length: int = 20
) -> Optional[Tuple[int, int, float]]:
    """
    Find the contiguous range of chunks [start, end] with maximum summed relevance.
    
    Standard Kadane's algorithm:
        current_sum = 0
        best_sum = 0, best_start = best_end = 0
        
        for each chunk i:
            current_sum += value[i]
            if current_sum > best_sum: update best
            if current_sum < 0: reset to 0 (start new subarray)
    
    RSE extension: max_segment_length caps the window to avoid returning
    entire documents for very broad queries. This keeps segments focused.
    
    Returns: (start_idx, end_idx, score) or None if no positive segment
    """
    # Get all chunk indices for this document, sorted
    doc_chunks = sorted([
        chunk_idx for (d, chunk_idx) in chunk_values 
        if d == doc_id
    ])
    
    if not doc_chunks:
        return None
    
    max_chunk_idx = max(doc_chunks)
    
    best_score = 0.0
    best_start = best_end = 0
    current_sum = 0.0
    current_start = 0
    
    for i in range(max_chunk_idx + 1):
        # Value is 0 for chunks not in candidates (not scored)
        val = chunk_values.get((doc_id, i), 0.0)
        
        if current_sum + val < 0:
            # Resetting: negative running sum means a new segment starts here
            current_sum = 0.0
            current_start = i + 1
        else:
            current_sum += val
            
            # Check segment length constraint
            segment_length = i - current_start + 1
            if segment_length <= max_segment_length and current_sum > best_score:
                best_score = current_sum
                best_start = current_start
                best_end = i
    
    if best_score <= 0:
        return None
    
    return (best_start, best_end, best_score)
```

**Kadane's algorithm explained for text retrieval:**

Imagine chunk values (relevance - threshold):

```
Position: 0     1     2     3     4     5     6     7     8
Values:   -0.1  0.4   0.5   0.3   -0.2  0.1   0.6   0.4   -0.3

Running sum:
  Start at 0: sum = 0 → chunk 0 = -0.1 → sum goes negative → RESET at pos 1
  Start at 1: sum = 0.4 → +0.5 → 0.9 → +0.3 → 1.2 → -0.2 → 1.0 → +0.1 → 1.1 → +0.6 → 1.7 → +0.4 → 2.1 → BEST
  Next: sum + (-0.3) = 1.8 < 2.1 — try continuing
  → Best segment: [1, 7], score=2.1

Reconstructed segment: chunks 1, 2, 3, 4, 5, 6, 7 concatenated
= a coherent 7-chunk passage containing the most concentrated relevant content
```

### Step 5: Segment Reconstruction

Once the optimal `[start, end]` range is found:

```python
def _reconstruct_segment(
    self,
    doc_id: str,
    start: int,
    end: int
) -> str:
    """
    Retrieve chunks [start, end] from the KV store and concatenate.
    
    Because we used zero-overlap chunking, this concatenation exactly
    reconstructs the original document text for this range — no
    duplicates, no gaps.
    """
    chunk_texts = self.kv_store.get_range(doc_id, start, end)
    return "\n\n".join(chunk_texts)
```

---

## RSE vs. Standard RAG: A Worked Example

**Document**: 50-page pharmaceutical report. Query: "What are the adverse effects of the drug in elderly patients?"

**FAISS scores** (top 5 by cosine similarity):
- Chunk 23 (0.89): "In patients over 65, the incidence of adverse events was 34%..."
- Chunk 31 (0.84): "Adverse effects were more pronounced in the geriatric subgroup..."
- Chunk 24 (0.81): "The most common adverse effects in elderly patients were..."
- Chunk 11 (0.72): "Patient demographics included 42% aged over 65..."
- Chunk 25 (0.68): "These effects were dose-dependent and reversible..."

**Standard RAG returns**: Chunks 23, 31, 24 — isolated, jumping from page 12 to page 16 to page 12 again. The LLM gets three fragments with no narrative flow.

**RSE analysis**:
- Chunks 23, 24, 25 are consecutive (pages 12-13) with high relevance → strong subarray starting here
- Chunk 31 is isolated (page 16) with a gap of low relevance chunks 26-30

RSE finds the maximum-sum subarray = [23, 25] (score = 2.38).

**RSE returns**: One coherent segment = chunks 23+24+25 concatenated — a complete, continuous discussion of elderly patient adverse effects, preserving the document's narrative structure.

---

## Configuration Tuning

```python
RSERetrievalRAG(
    file_path="report.pdf",
    
    # Zero overlap is mandatory for RSE
    chunk_size=1000,
    chunk_overlap=0,       # MUST be 0 — otherwise segment concatenation duplicates text
    
    # How many candidate chunks to retrieve before segment finding
    k=10,                  # More candidates = wider search for optimal segments
    
    # If max similarity < this, chunk has negative value → segment won't extend
    relevance_threshold=0.3,   # Lower = more permissive segments (longer)
                               # Higher = tighter, higher-precision segments (shorter)
    
    # Maximum chunks a single segment can span
    max_segment_length=15,     # 15 × 1000 chars ≈ 15,000 char context window cap
)
```

**`relevance_threshold` guide:**

| Value | Effect | Use when |
|-------|--------|----------|
| 0.2 | Long segments, permissive extension | Dense, academic text with slow topic transitions |
| 0.3 | Balanced (default) | General enterprise documents |
| 0.4 | Short, high-precision segments | Heterogeneous documents (FAQ, mixed-topic reports) |
| 0.5 | Very tight — only high-relevance chunks | Precision-critical applications |

---

## When to Use RSE

RSE is most valuable for long, structured documents — research papers, financial reports, legal briefs — where the answer to a complex question spans multiple consecutive paragraphs rather than a single isolated sentence. If users frequently need multi-part answers that need context to make sense ("what were the methodology limitations and how did they affect the results?"), RSE is the right tool. It's less worth the overhead for short corpora, FAQ-style documents where each chunk is already self-contained, or situations where real-time latency is critical.

---

## Summary

RSE is a retrieval paradigm shift: instead of "find the best k chunks," it asks "find the best contiguous region." By scoring chunk relevance, applying Kadane's maximum-sum-subarray algorithm to find optimal segment bounds, and reconstructing segments from a zero-overlap chunk store, RSE delivers context that reads like a coherent extracted passage — not a collage of fragments.

The result is that the LLM receives narratively complete context, with each piece connected to the next by the document's own structure. This is especially powerful for long, complex documents where the answer to a query spans a complete subsection — which isolated top-k retrieval can never capture in full.
