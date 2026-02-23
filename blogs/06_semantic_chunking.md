# Semantic Chunking: Let the Content Define Its Own Boundaries

> **Technique:** Semantic Chunking  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`, `numpy`

---

## Introduction

Every RAG practitioner has wrestled with chunking. Too large, and retrieval becomes imprecise — you retrieve entire sections when you only needed one paragraph. Too small, and you fragment coherent ideas across dozens of tiny pieces. Fixed character limits ignore the most fundamental property of text: **meaning doesn't distribute uniformly**.

A 1,000-character chunk might span three unrelated topics — or it might cut a single complex argument in half. Neither outcome is good for retrieval.

**Semantic Chunking** abandons the idea of predetermined chunk sizes entirely. Instead, it analyzes the semantic similarity between consecutive sentences to detect where one topic ends and another begins. Chunk boundaries are placed at these natural topic transitions, producing chunks that are semantically cohesive by construction.

---

## The Core Insight: Sentences About the Same Topic Are Similar

If you embed consecutive sentences in a document and compute their pairwise cosine similarities, you'll observe a pattern:

- **High similarity** (0.8–1.0): Adjacent sentences discussing the same topic
- **Low similarity** (0.3–0.6): Sentences at topic boundaries

This similarity signal reveals the document's natural structure. By identifying "breakpoints" — positions where similarity drops sharply — we can slice the document along its semantic seams rather than at arbitrary character positions.

---

## How Semantic Chunking Works

### Step-by-Step Pipeline

```
Document text
    ↓
Split into sentences (using regex)
    ↓
Embed all sentences (OpenAI text-embedding-3-small)
    ↓
Compute cosine similarity between consecutive sentences
    similarities = [sim(s0,s1), sim(s1,s2), ..., sim(s_{n-1},s_n)]
    ↓
Detect breakpoints using threshold method
    ↓
Slice sentences at breakpoints → raw semantic chunks
    ↓
Merge small chunks (below min_chunk_size)
    ↓
Split oversized chunks (above max_chunk_size)
    ↓
Embed final chunks → FAISS vector store
```

### Three Breakpoint Detection Methods

The implementation supports three statistical methods for identifying breakpoints:

```python
class SemanticChunker:
    def __init__(self, embedder, 
                 method="percentile",  # or "standard_deviation", "interquartile"
                 threshold=90.0,       # percentile value
                 min_chunk_size=1000,
                 max_chunk_size=20000):
```

#### 1. Percentile Method (Default)
Place breakpoints where similarity drops below the `threshold`-th percentile of all similarities. With `threshold=90`, a breakpoint is detected wherever similarity is in the bottom 10% — marking the 10% most dramatic topic shifts.

```python
def _find_breakpoints_percentile(self, similarities):
    # Low similarity = topic change = breakpoint
    percentile_value = np.percentile(similarities, 100 - self.threshold)
    return [i for i, s in enumerate(similarities) if s < percentile_value]
```

**Intuition**: "A breakpoint is where similarity is unusually low relative to the rest of the document."

#### 2. Standard Deviation Method
Place breakpoints where similarity falls more than `threshold` standard deviations below the mean.

```python
def _find_breakpoints_std(self, similarities):
    mean = np.mean(similarities)
    std = np.std(similarities)
    cutoff = mean - (self.threshold * std)
    return [i for i, s in enumerate(similarities) if s < cutoff]
```

**Best for**: Documents with broadly distributed similarity scores where absolute thresholds may be unreliable.

#### 3. Interquartile Method
Uses the IQR (Q1 - 1.5*IQR) as the lower fence — any similarity below this fence is a breakpoint.

```python
def _find_breakpoints_iqr(self, similarities):
    q1, q3 = np.percentile(similarities, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    return [i for i, s in enumerate(similarities) if s < lower_fence]
```

**Best for**: Documents with outlier-heavy similarity distributions; IQR is more robust to extremes.

---

## Code Deep Dive

### Sentence Splitting

The sentence splitter uses a carefully crafted regex to detect sentence boundaries:

```python
def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", ' ', text).strip()
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+',
        text
    )
    cleaned = [s.strip() for s in sentences if s.strip() and len(s) > 5]
    return cleaned
```

This regex catches two common sentence boundary patterns:
- Sentence-ending punctuation followed by whitespace and a capital letter
- Sentence-ending punctuation followed by a newline

The `len(s) > 5` filter eliminates abbreviations ("Dr.", "et al.") and spurious fragments.

### The Full Chunking Logic

```python
def chunk(self, text: str) -> List[str]:
    sentences = split_into_sentences(text)
    embeddings = self.embedder.embed_texts(sentences)
    similarities = self._compute_similarity(embeddings)
    breakpoints = self._find_breakpoints(similarities)
    
    # Slice into raw chunks at breakpoints
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk_sentences = sentences[start:bp+1]
        chunks.append(" ".join(chunk_sentences))
        start = bp + 1
    if start < len(sentences):
        chunks.append(" ".join(sentences[start:]))
    
    # Merge small chunks
    merged = []
    buffer = ""
    for chunk in chunks:
        if buffer:
            chunk = buffer + " " + chunk
            buffer = ""
        if len(chunk) < self.min_chunk_size:
            buffer = chunk  # accumulate until big enough
        else:
            merged.append(chunk)
    if buffer:
        if merged:
            merged[-1] += " " + buffer  # attach remainder to last chunk
        else:
            merged.append(buffer)
    
    # Split oversized chunks
    final_chunks = []
    for c in merged:
        if len(c) > self.max_chunk_size:
            sub = chunk_text(c, chunk_size=self.max_chunk_size, chunk_overlap=0)
            final_chunks.extend(sub)
        else:
            final_chunks.append(c)
    
    return final_chunks
```

The two-pass merge-and-split step ensures chunks stay within `[min_chunk_size, max_chunk_size]` bounds. A 50-sentence semantic unit about one highly technical procedure might exceed `max_chunk_size` — it gets split. A 3-sentence transition paragraph that mentions a topic briefly stays below `min_chunk_size` — it gets merged with its neighbor.

---

## Similarity Computation

```python
def _compute_similarity(self, embeddings: List[List[float]]) -> List[float]:
    similarities = []
    for i in range(len(embeddings) - 1):
        vec_a = np.array(embeddings[i])
        vec_b = np.array(embeddings[i+1])
        # Cosine similarity
        sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        similarities.append(float(sim))
    return similarities
```

With `n` sentences, this produces `n-1` similarity scores — one score per consecutive pair. The computational cost is `O(n)` in distance computations.

---

## The Embedding Cost Consideration

Semantic chunking requires embedding **every sentence** before any retrieval happens. For a 10,000-word document with ~700 sentences, this is 700 embedding API calls (or one batched call). This is an **index-time cost only** — once computed, queries run identically to standard RAG.

Modern embedding APIs allow batch embedding, bringing this cost to a few API calls regardless of document length. The OpenAI `text-embedding-3-small` model processes 8,192 tokens per request, and most sentences are 10-30 tokens, so batching is extremely efficient.

---

## The Max Context Length Problem

One critical issue: if individual sentences are very long (e.g., technical specs, tables extracted from PDFs), they may exceed the embedding model's token limit (typically 8,192 tokens).

The implementation guards against this by capping text at the embedding model's limit before embedding. This was the source of a `BadRequestError` in earlier usage and is now handled explicitly.

---

## Semantic Chunking vs. Fixed Chunking: A Visual

Consider a document about Climate Change with three sections:
1. Causes of climate change (paragraphs 1-4)
2. Effects on biodiversity (paragraphs 5-8)
3. Policy responses (paragraphs 9-13)

**Fixed chunking (1000 chars)** might produce:
- Chunk A: End of causes + start of biodiversity effects
- Chunk B: Middle of biodiversity section
- Chunk C: End of biodiversity + start of policy

**Semantic chunking** produces:
- Chunk 1: All of "Causes" (paragraphs 1-4)
- Chunk 2: All of "Effects on biodiversity" (5-8)
- Chunk 3: All of "Policy responses" (9-13)

Queries about policy will retrieve Chunk 3 and *only* Chunk 3 — no contamination from biodiversity content that happened to be near a fixed-size boundary.

---

## Tuning Guide

| Parameter | Description | Recommended Starting Value |
|-----------|-------------|---------------------------|
| `method` | Breakpoint detection method | `"percentile"` |
| `threshold` | Percentile cutoff (higher = fewer chunks) | `90.0` |
| `min_chunk_size` | Minimum chars per chunk | `1000` |
| `max_chunk_size` | Maximum chars per chunk | `20000` |

**Increasing `threshold`** (e.g., 95) creates fewer, larger chunks — better for documents with smooth topic transitions. **Decreasing `threshold`** (e.g., 80) creates more, smaller chunks — better for documents with rapid topic switching.

---

## When to Use Semantic Chunking

**Best for:**
- Long-form documents with distinct sections (books, reports, papers)
- Documents where topic clarity in each chunk is critical
- Corpora with variable information density across sections
- Applications where chunk quality directly impacts answer quality

**Less ideal when:**
- Documents are short (semantic analysis doesn't add value)
- Documents are highly fragmented with no coherent sections
- Index time is extremely constrained (embedding every sentence takes time)

---

## Summary

Semantic Chunking represents a fundamental paradigm shift: instead of imposing a structure on text, let the text reveal its own structure. By detecting topic transitions through embedding similarity, chunks become semantically coherent units rather than arbitrary text windows.

The result is a retrieval index where each chunk represents a complete idea — and queries find precisely the idea they need, without the noise of adjacent unrelated content. Combined with thoughtful min/max size bounds, semantic chunking produces index quality that fixed-size chunking simply cannot match for complex, structured documents.
