# Contextual Compression: Extract Only What's Relevant

> **Technique:** Contextual Compression  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

Imagine retrieving a 1,200-character chunk to answer "What is the boiling point of water at sea level?" — but the chunk covers three paragraphs of thermodynamic principles, of which only one sentence contains the actual answer ("Water boils at 100°C / 212°F at standard atmospheric pressure").

You're sending 1,200 characters to the LLM when 60 characters would suffice.

**Contextual Compression** solves this precise problem. After retrieving relevant chunks, a secondary LLM pass extracts *only the query-relevant portions* from each chunk, discarding surrounding filler. The LLM then generates its answer from concentrated, noise-free context rather than diluted paragraphs.

This technique was introduced by LangChain and has since become a standard optimization in production RAG systems.

---

## Why Context Noise Hurts RAG Quality

LLMs are impressively capable, but they can still be misled by irrelevant context. When the answer is buried in a chunk alongside unrelated content, two failure modes emerge:

1. **Attention dilution**: The model's attention distributes across all tokens. Irrelevant sentences compete with the answer for attention weight, sometimes causing the model to overlook or misweight the key information.

2. **Token budget waste**: LLM context windows have limits. Sending bloated chunks means you can include fewer retrieved chunks total. Compression extends how much *useful* information fits in a single context window.

3. **Hallucination risk**: Tangentially related content in a chunk can trigger the model to weave it into the answer incorrectly, mixing accurate and fabricated information.

---

## How Contextual Compression Works

### Pipeline

```
User Query
    ↓
Standard vector retrieval → top-k chunks
    ↓
For each chunk:
    [LLM Compressor] "Extract only the parts relevant to the query"
    ↓
    Compressed chunk (or empty string if nothing relevant)
    ↓
Filter: discard empty (irrelevant) chunks
    ↓
[Answer LLM] generates response from compressed context
```

The compressor is a secondary LLM call — a small, fast model like `gpt-4o-mini` at temperature 0.0. It operates in extraction mode, not generation mode: it preserves original wording rather than paraphrasing.

### Two Possible Outcomes Per Chunk

1. **Relevant content found**: Returns the extracted relevant sentences/phrases
2. **No relevant content**: Returns `"NO_RELEVANT_CONTENT"` — the chunk is discarded entirely

This makes contextual compression also a **relevance filter**: chunks that happened to be top-k by vector similarity but don't actually address the query get eliminated before reaching the answer LLM.

---

## Code Deep Dive

### The Compressor

```python
class ContextualCompressor:
    def __init__(self, model_name="gpt-4o-mini", 
                 temperature=0.0, max_tokens=5000):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def compress(self, chunk_text: str, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert text extractor. Given a user query and a text chunk, "
                    "extract ONLY the parts of the text that are directly relevant to answering "
                    "the query. Preserve the original wording — do not paraphrase or summarize. "
                    "If no part of the text is relevant to the query, respond with exactly: "
                    "NO_RELEVANT_CONTENT"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Text chunk:\n{chunk_text}\n\n"
                    "Extract only the relevant parts:"
                ),
            },
        ]
        result = self.llm.chat(messages)
        if "NO_RELEVANT_CONTENT" in result.strip():
            return ""
        return result.strip()
```

**Key design choices:**
- `temperature=0.0`: Deterministic extraction. We don't want creative variations.
- `"Preserve the original wording"`: This is extraction, not summarization. The LLM should be a scalpel, not a summarizer.
- `NO_RELEVANT_CONTENT` signal: A defined sentinel value that's easy to check programmatically.
- `max_tokens=5000`: Generous upper bound — some chunks may have multiple relevant sections.

### Batch Compression with Filtering

```python
def compress_documents(self, chunks: List[str], query: str) -> List[str]:
    compressed = []
    for chunk in chunks:
        result = self.compress(chunk, query)
        if result:  # non-empty = relevant content was found
            compressed.append(result)
    return compressed
```

Chunks returning empty strings are silently dropped. The answer LLM receives only the compressed, query-relevant excerpts.

### The Full RAG Pipeline

```python
class ContextualCompressionRAG:
    def query(self, question: str) -> Tuple[str, List[str]]:
        # Step 1: Retrieve broadly
        results = self.retriever.retrieve(question, k=self.k)
        raw_chunks = [r.document.content for r in results]
        
        # Step 2: Compress each chunk
        compressed_chunks = self.compressor.compress_documents(raw_chunks, question)
        
        # Step 3: Fallback if all chunks were irrelevant
        if not compressed_chunks:
            return "I could not find relevant information.", []
        
        # Step 4: Generate from compressed context
        answer = self.llm.chat_with_context(question, compressed_chunks)
        return answer, compressed_chunks
```

---

## The Extraction vs. Summarization Distinction

It's worth emphasizing a critical point in the compressor's instructions: **"Preserve the original wording — do not paraphrase or summarize."**

This is intentional and important. If the LLM summarizes instead of extracts, it introduces a transformation step that:
- Can introduce inaccuracies
- May omit important nuances
- Makes the source text unverifiable

By instructing extraction of original text, contextual compression is lossless with respect to accuracy — it only removes, never rewrites.

---

## Performance Analysis

### Token Savings

Consider a retrieval of k=5 chunks, each 1,000 characters (~250 tokens):

| Scenario | Tokens sent to answer LLM |
|----------|--------------------------|
| Standard RAG | 5 × 250 = 1,250 tokens |
| With compression (50% relevant) | 5 × 125 = 625 tokens |
| With compression + filtering (2 irrelevant) | 3 × 125 = 375 tokens |

At scale, this can reduce LLM costs by 50-70% for answer generation.

### Latency Trade-off

Compression adds one LLM call per retrieved chunk (k calls total) before the final answer generation call. This increases latency by roughly:
- k=3: ~3 additional API calls
- k=5: ~5 additional API calls

For applications where response quality outweighs response time, this trade-off is favorable. For real-time applications, consider caching compressed chunks or using a faster/cheaper model for compression.

---

## Comparison With Other Techniques

| Technique | What it does | When to prefer |
|---------|-------------|---------------|
| **Standard RAG** | Retrieves full chunks | Baseline; fast |
| **Contextual Compression** | Extracts query-relevant excerpts | When chunks are dense with mixed topics |
| **Reranking** | Reorders chunks by relevance | When ordering matters; doesn't reduce chunk size |
| **CRAG** | Evaluates retrieval quality holistically | When local retrieval may be inadequate entirely |

Contextual Compression and Reranking are complementary — you can apply both: rerank first to get the most relevant chunks, then compress each to extract only the relevant portions.

---

## When to Use Contextual Compression

**Best for:**
- Chunks containing multiple topics or dense paragraphs
- Applications where LLM token cost is significant
- Corpora where a single chunk may address multiple possible questions
- Systems where answer precision is more important than latency

**Skip when:**
- Chunks are already very focused (e.g., proposition chunks or semantic chunks)
- Latency is paramount and cannot afford extra LLM calls
- The corpus is so large that pre-compression at index time isn't feasible

---

## Summary

Contextual Compression is a powerful post-retrieval refinement that transforms retrieved chunks from "approximately relevant paragraphs" into "precisely relevant excerpts." By using an LLM as a precision extractor, it:

1. **Reduces noise** in the answer LLM's context
2. **Filters irrelevant chunks** that made it through vector search
3. **Saves tokens** on answer generation
4. **Improves answer focus** by presenting concentrated signal

It's a natural complement to any retrieval strategy and particularly shines when your chunks are broad and topic-diverse. Think of it as giving your answer LLM a pre-digested, query-specific context rather than a pile of raw material to sift through.
