# Contextual Compression: Delivering Only What's Relevant

## Introduction

Standard RAG delivers entire chunks to the LLM. A chunk might be 1000 characters, and your user's query might be answered perfectly by 50 characters buried inside it. The other 950 characters are noise — they add token cost, dilute the signal, and in long-context scenarios can actually cause the LLM to lose focus on the core answer.

**Contextual Compression** adds a focused extraction step between retrieval and generation. After retrieving chunks in the usual way, it passes each chunk through an LLM "compressor" that extracts only the text directly relevant to the user's query. If a 1000-character chunk contains the answer in 80 characters, only those 80 characters make it to the final LLM generation prompt.

The term comes from LangChain's Contextual Compression Retriever, introduced in 2023 as a way to reduce context noise and token waste in RAG pipelines.

---

## The Signal-to-Noise Problem in RAG Context

Consider a query: "What is the melting point of tungsten?"

**Retrieved chunk (1000 chars)**:
> "Tungsten is a chemical element with the symbol W and atomic number 74. It is a hard, rare metal under standard conditions when uncombined, and has the highest melting point of all known elements. Pure tungsten is a steel-gray to tin-white metal. Naturally occurring tungsten has five stable isotopes with atomic masses varying from 180 to 184. Tungsten's melting point is **3,422°C (6,192°F)**, making it indispensable in applications requiring extreme heat resistance. Its boiling point is 5,555°C. Tungsten is commonly used in light bulb filaments, X-ray targets, and as a steel additive for hardness in high-speed tool steels. The element was discovered in 1783 by the de Elhujar brothers in Spain."

The answer is "3,422°C (6,192°F)". It's 18 characters in a 1000-character chunk. The LLM receives all 1000 characters but could have answered correctly from 18.

**After compression**:
> "Tungsten's melting point is 3,422°C (6,192°F), the highest of all known elements."

The LLM receives 84 characters. The answer is identical. The noise is eliminated.

---

## The Two-Stage Architecture

```
User Query
     │
     ▼
[Stage 1: Standard Vector Retrieval]
     │  FAISS similarity search → top-k chunks
     │  Returns: [Chunk_1, Chunk_2, Chunk_3]
     ▼
[Stage 2: Contextual Compression]
     │  For each chunk:
     │    LLM: "Extract only the text relevant to the query from this chunk"
     │    LLM: "If the chunk contains nothing relevant, return empty"
     │
     │  Chunk_1 → "Tungsten's melting point is 3,422°C..."  (relevant)
     │  Chunk_2 → ""  (nothing relevant — filtered out)
     │  Chunk_3 → "3,422°C is higher than any other element..."  (relevant)
     ▼
[Stage 3: Generation]
     │  LLM receives only compressed, relevant excerpts
     ▼
Answer
```

Two benefits flow from compression:
1. **Token reduction**: Compressed context is typically 10-30% of original chunk size
2. **Relevance filtering**: Chunks that contain nothing relevant are eliminated entirely, avoiding LLM distraction from off-topic retrieved content

---

## Implementation Walkthrough

### The Compressor

```python
class ContextualCompressor:
    """
    Compresses retrieved chunks by extracting only query-relevant content.
    Uses an LLM to perform extraction — more accurate than keyword matching
    or simple substring search.
    """
    
    def compress(
        self, 
        query: str, 
        context: str,
        max_length: int = 500  # character limit on compressed output
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a contextual compression system. Your task is to extract "
                    "the most relevant information from the given context that directly "
                    "answers or supports the given query.\n\n"
                    "Rules:\n"
                    "1. Extract ONLY the relevant portions — do not paraphrase or add information\n"
                    "2. Maintain the original wording where possible — quote directly\n"
                    "3. If the context contains NO relevant information, return exactly: "
                    "   'No relevant information found'\n"
                    "4. Be concise — return the minimum text needed to answer the query\n"
                    f"5. Maximum length: {max_length} characters"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Context to compress:\n{context}\n\n"
                    "Extract only the relevant information:"
                )
            }
        ]
        
        compressed = self.llm.chat(messages).strip()
        return compressed
    
    def is_relevant(self, compressed: str) -> bool:
        """
        Check if the compressed text actually contains useful information.
        Filters chunks where the compressor found nothing relevant.
        """
        irrelevant_phrases = [
            "no relevant information",
            "no information found", 
            "not found in the context",
            "the context does not contain",
            "nothing relevant"
        ]
        compressed_lower = compressed.lower().strip()
        return not any(phrase in compressed_lower for phrase in irrelevant_phrases)
```

**Why use an LLM for extraction instead of keyword matching?**

Keyword matching ("does the chunk contain the word 'melting point'?") fails for paraphrases:
- "highest known thermal tolerance" = melting point concept but shares zero keywords
- "the temperature at which it liquefies" = melting point concept but shares zero keywords

LLM extraction understands semantic equivalence. It correctly identifies "highest known thermal tolerance" as relevant to "melting point" even without shared vocabulary.

### Full RAG Pipeline with Compression

```python
def query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
    # Over-retrieve: fetch more than needed
    # Compression will reduce k=5 to only the actually-relevant subset
    raw_results = self.vector_store.search(question_embedding, k=k)
    
    relevant_contexts = []
    compression_stats = []
    
    for result in raw_results:
        original_chunk = result.document.content
        original_length = len(original_chunk)
        
        # Compress this chunk against the query
        compressed = self.compressor.compress(
            query=question,
            context=original_chunk,
            max_length=500
        )
        
        if self.compressor.is_relevant(compressed):
            relevant_contexts.append(compressed)
            compression_ratio = len(compressed) / original_length
            compression_stats.append({
                "original_length": original_length,
                "compressed_length": len(compressed),
                "ratio": f"{compression_ratio:.1%}"
            })
    
    # Generate from compressed context only
    context_text = "\n\n---\n\n".join(relevant_contexts)
    answer = self._generate_answer(question, context_text)
    
    return answer, relevant_contexts
```

**Over-retrieval strategy**: Retrieve k=5 or k=7 initially. Compression will filter out irrelevant chunks, often leaving k=2 or k=3 high-quality excerpts. This means the final LLM context is simultaneously *smaller* and *more relevant* than a standard k=3 retrieval.

---

## The Compression Ratio

Compression ratios vary significantly by document type and query specificity:

| Scenario | Typical compression ratio |
|---------|--------------------------|
| Highly specific factual query | 3–10% (a number in a paragraph) |
| Directed technical query | 15–30% (a few sentences from a section) |
| Broad thematic query | 40–70% (most of a chunk is relevant) |
| Off-topic chunk (filtered out) | 0% (eliminated entirely) |

For a typical factual Q&A system: k=5 initial retrieval, average 20% compression ratio, 2 chunks filtered → the LLM receives ~60% of the context it would in standard RAG, with much less noise.

### Token Cost Analysis

| Stage | Tokens (standard k=3) | Tokens (compression k=5, avg 20% keep) |
|-------|----------------------|----------------------------------------|
| Retrieval context in final prompt | 3,000 | 600 (20% × 3,000 from 3 relevant chunks) |
| Compression LLM calls | 0 | 5 × 1,200 avg input = 6,000 |
| **Net context reduction** | — | **80% less context noise** |

**The trade-off**: Compression costs 5 extra LLM calls but delivers 80% reduction in context noise. For expensive LLM generation (GPT-4 etc.), the compression calls in `gpt-4o-mini` are dramatically cheaper than the context tokens they save in the generator.

---

## Practical Example: Multi-Fact Chunk Filtering

**Query**: "Who invented the transistor and in what year?"

**Chunk 1** (retrieved, cosine sim = 0.82):
> "The transistor was invented in 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs. The invention was one of the most significant technological achievements of the 20th century. Bardeen, Brattain, and Shockley received the Nobel Prize in Physics in 1956 for their invention. The transistor replaced the vacuum tube and enabled the miniaturization of electronic circuits, eventually leading to the microprocessor revolution of the 1970s and 1980s."

**After compression for "who invented it and when"**:
> "The transistor was invented in 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs."

76 characters extracted from 572. A 13% retention ratio — exactly the information needed, nothing else.

**Chunk 2** (retrieved, cosine sim = 0.78):
> "Semiconductor devices rely on the properties of materials that have conductivity between conductors and insulators. Silicon is the most widely used semiconductor material. Its band gap of 1.12 eV at room temperature makes it ideal for use in transistors and integrated circuits operating at standard temperatures."

**After compression**: "No relevant information found" → filtered out.

Despite cosine similarity 0.78 suggesting relevance (it's about transistors), the chunk contains nothing about *invention* or *year*. Compression correctly eliminates it. Standard RAG would have included it as context, potentially confusing the LLM with semiconductor physics.

---

## Error Handling and Edge Cases

### Chunk Shorter Than Compressed Output

Occasionally the compressor returns text longer than the original chunk (hallucination). Guard against this:

```python
def compress(self, query: str, context: str, max_length: int = 500) -> str:
    compressed = self.llm.chat(messages).strip()
    
    # Safety: if compressed is longer than original, return original
    if len(compressed) > len(context):
        return context  # Compressor hallucinated — safe fallback
    
    return compressed
```

### API Rate Limits During Parallel Compression

For k=5 with sequential compression calls, the pipeline runs 5 LLM calls. For high-throughput production systems, these can be parallelized:

```python
from concurrent.futures import ThreadPoolExecutor

def compress_all_parallel(self, query, chunks):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(self.compressor.compress, query, chunk)
            for chunk in chunks
        ]
        return [f.result() for f in futures]
```

Parallel compression reduces total compression latency from `5 × t_per_call` to `max(t_per_call) ≈ 1 × t_per_call`.

---

## Configuration Guide

```python
ContextualCompressionRAG(
    file_path="document.pdf",
    
    # Initial retrieval — over-fetch since some chunks will be filtered
    k=5,
    
    # Max characters the compressor returns per chunk
    max_compressed_length=500,  # ~125 tokens — enough for 2-3 sentences
    
    # Models — compressor should be fast and cheap (gpt-4o-mini)
    # Generator can be more powerful (gpt-4o) since it receives cleaner input
    compression_model="gpt-4o-mini",  # fast, cheap
    generation_model="gpt-4o-mini",   # or gpt-4o for highest quality
)
```

**Key design choice: different models for compression vs. generation**

Compression is a mechanical extraction task — simple, short, well-defined. `gpt-4o-mini` is ideal.  
Generation requires synthesis and nuanced language. If your use case demands high-quality prose, use `gpt-4o` for generation while keeping `gpt-4o-mini` for compression.

---

## When to Use Contextual Compression

Contextual Compression is most valuable when your retrieved chunks are large and information-dense but only a fraction of each chunk is relevant to any given query. This is common in general-purpose document corpora: a 1000-character chunk about a chemical element might contain historical, physical, chemical, and industrial information, but the user only asked about one property. Compression surgically extracts the answer and discards the rest, reducing token cost and eliminating the LLM distraction that comes from irrelevant context.

It's also well-suited for mixed-topic documents where retrieved chunks frequently contain tangential material. If users report that answers feel "confused" or reference details they didn't ask about, that's often a signal that context noise is the culprit — and compression is the targeted fix.

The technique adds real latency: compressing five candidates requires five extra LLM calls, adding several seconds per query in sequential mode. If sub-second responses are required, or if your chunks are already small and targeted (proposition-level), the compression overhead won't be worth it. Broad, thematic queries — where most of each chunk is relevant — also see minimal compression benefit and pay full compression cost.

---

## Summary

Contextual Compression transforms retrieval from "give the LLM a collection of chunks" to "give the LLM exactly the information it needs." By using an LLM to extract query-relevant text from each retrieved chunk — and to filter out chunks that contain nothing relevant — the technique delivers dramatically cleaner, more focused context to the generation step.

The result: lower token costs, less LLM confusion from irrelevant context, and answers that are more directly grounded in the precise facts that answer the question. For RAG systems where retrieval is good but generation quality feels imprecise, contextual compression is one of the most targeted improvements available.
