# Proposition Chunking: Atomic Facts as the Unit of Retrieval

**Source:** `context_enrichment/proposition_chunking_rag.py` · **Libraries:** `openai`, `faiss-cpu`

---

## Introduction

The quality ceiling of any RAG system is set by the quality of its indexed units. You can have the best embedding model, the fastest vector store, and the most powerful LLM — but if the units being indexed are poorly formed, retrieval will be imprecise.

Standard chunking is mechanical: it splits text at fixed character intervals, ignoring whether the resulting pieces are semantically coherent or independently useful. A 1,000-character chunk might contain one brilliant insight surrounded by 800 characters of contextual scaffolding that dilutes its retrievability.

**Proposition Chunking** rethinks the fundamental unit entirely. Instead of indexing text windows, it uses an LLM to decompose every chunk of text into *atomic propositions* — the smallest, most independently useful statements of fact. Then it indexes each proposition individually.

The result is a knowledge base made of clean, precise, self-contained facts — exactly the kind of units that user queries try to find.

---

## What Is a Proposition?

In logic and linguistics, a proposition is a statement that has a definite truth value — it can be true or false. In the context of RAG, we define a proposition more practically as **a single, self-contained, verifiable statement of fact**.

A good proposition satisfies five criteria:

### Criterion 1: Expresses a Single Fact
Bad: "Climate change affects biodiversity, sea levels, and weather patterns."  
Good:
- "Climate change affects biodiversity."
- "Climate change causes sea levels to rise."
- "Climate change alters weather patterns."

Each good proposition is one claim. You never need to retrieve all three together just to answer a question about sea levels.

### Criterion 2: Self-Contained (No Pronoun Dependencies)
Bad: "It was discovered in 1928 by Fleming."  
Good: "Penicillin was discovered in 1928 by Alexander Fleming."

A proposition must be understandable by someone who has read nothing else. "It" is meaningless without context; "Penicillin" is not.

### Criterion 3: Uses Full Names, Not Pronouns
Bad: "He won the Nobel Prize in 1921."  
Good: "Albert Einstein won the Nobel Prize in Physics in 1921."

This is a stricter version of criterion 2. Propositions must resolve all coreferences to their canonical form: full names, complete dates, specific entities.

### Criterion 4: Includes Relevant Qualifiers
Bad: "CO₂ levels rose."  
Good: "Atmospheric CO₂ levels rose from 280 ppm to 420 ppm between 1800 and 2024."

Qualifiers make propositions more precise and more distinctly searchable. Two queries — "CO₂ before industrialization" and "current CO₂ levels" — both find this proposition because it contains both pieces of information.

### Criterion 5: Clear Subject-Predicate Relationship
Bad: "Regarding Einstein and the photoelectric effect, Nobel 1921."  
Good: "Albert Einstein won the 1921 Nobel Prize in Physics for explaining the photoelectric effect."

Every proposition should have a clear subject (actor/entity) and predicate (action/property/relationship).

---

## Worked Example: From Paragraph to Propositions

**Input paragraph:**

> "The Apollo 11 mission, launched on July 16, 1969, successfully landed the first humans on the Moon on July 20, 1969. Mission commander Neil Armstrong and lunar module pilot Buzz Aldrin spent about two hours on the lunar surface, while Michael Collins orbited Moon in the command module. Armstrong's famous words, 'That's one small step for man, one giant leap for mankind,' were broadcast live to millions of viewers on Earth."

**Generated Propositions:**
1. Apollo 11 launched on July 16, 1969.
2. Apollo 11 was a NASA Moon landing mission.
3. Apollo 11 successfully landed the first humans on the Moon.
4. The first humans landed on the Moon on July 20, 1969.
5. Neil Armstrong was the mission commander of Apollo 11.
6. Buzz Aldrin was the lunar module pilot of Apollo 11.
7. Michael Collins was the command module pilot of Apollo 11.
8. Neil Armstrong and Buzz Aldrin spent approximately two hours on the lunar surface.
9. Michael Collins orbited the Moon in the command module during Apollo 11.
10. Neil Armstrong said "That's one small step for man, one giant leap for mankind" upon landing on the Moon.
11. Armstrong's Moon landing statement was broadcast live to viewers on Earth.

Now consider what these propositions enable:
- Query: "Who orbited the Moon during Apollo 11?" → Proposition 9
- Query: "How long did the astronauts stay on the Moon?" → Proposition 8
- Query: "What did Armstrong say on the Moon?" → Proposition 10

Each query retrieves *exactly one* precise proposition rather than the entire paragraph with surrounding noise.

---

## How the Pipeline Works

### Phase 1: Initial Chunking (for LLM manageability)

Proposition extraction requires an LLM to process the text. LLMs have context limits, and processing a 300-page document in one shot isn't feasible. The first step creates intermediate "processing chunks" — typically larger than final retrieval chunks — solely for the LLM to work with.

```
Document text (long)
    ↓
chunk_text(chunk_size=2000, chunk_overlap=200)
    ↓
Processing chunks: [C0, C1, C2, ..., Cn]   ← LLM processes each
```

These are not the indexed chunks — they're temporary units for extraction purposes only.

### Phase 2: Proposition Generation

For each processing chunk, the LLM generates a list of atomic propositions:

```python
def generate_propositions(llm, chunk_text: str) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Break down the following text into simple, "
                "self-contained propositions. Each proposition must:\n\n"
                "1. Express exactly ONE fact\n"
                "2. Be fully understandable without surrounding context\n"
                "3. Use full names — no pronouns (he/she/it/they)\n"
                "4. Include relevant dates, quantities, and qualifiers\n"
                "5. Have a clear subject-predicate structure\n\n"
                "Return JSON: {\"propositions\": [\"prop1\", \"prop2\", ...]}"
            )
        },
        {
            # Few-shot example: demonstrates expected format and quality
            "role": "user",
            "content": "In 1955, Rosa Parks refused to give up her seat on..."
        },
        {
            "role": "assistant",
            "content": "{\"propositions\": [\"Rosa Parks refused to give up her bus seat in 1955.\", ...]}"
        },
        {
            "role": "user",
            "content": chunk_text
        }
    ]
    result = llm.chat_json(messages)
    return result.get("propositions", [])
```

**Why the few-shot example matters:** Without a demonstration, LLMs sometimes return entire sentences, compound sentences, or reformulations rather than truly atomic propositions. The few-shot example anchors the model's behavior to the exact format and granularity required.

**Why JSON mode:** Using `chat_json` (which sets `response_format={"type": "json_object"}`) ensures perfectly parseable output 100% of the time. Without JSON mode, even well-prompted models occasionally produce prose explanations that wrap the list, corrupting parsing.

### Phase 3: Quality Evaluation

Not every generated proposition is valuable. The evaluator filters them:

```python
def evaluate_proposition(llm, proposition: str) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "Score this proposition on a scale of 0.0 to 1.0. Evaluate:\n\n"
                "- Clarity (0-0.33): Is the proposition clearly and unambiguously stated?\n"
                "- Factuality (0-0.33): Does it state a specific, verifiable fact?\n"
                "- Independence (0-0.34): Can it be understood without any prior context?\n\n"
                "Sum the three sub-scores for a total between 0.0 and 1.0.\n"
                "Return JSON: {\"score\": <float>, \"reason\": \"<brief explanation>\"}"
            )
        },
        {
            "role": "user",
            "content": f"Proposition: {proposition}"
        }
    ]
    result = llm.chat_json(messages)
    return float(result.get("score", 0.0))
```

**Propositions that fail evaluation** (score < 0.6):
- "The situation improved significantly." ← No subject, vague, not verifiable
- "This has been discussed extensively in academic literature." ← No subject, no specific fact
- "It." ← Fragment from parsing error
- "Related to the earlier point about X..." ← Not independent

**Propositions that pass** (score ≥ 0.6):
- "The unemployment rate in the US fell to 3.4% in January 2023." ← Specific, verifiable, independent
- "Python 3.11 was released in October 2022 and included significant performance improvements." ← Two related facts about the same event, both specific

### Phase 4: Embedding and Indexing

```python
def build_index(self, text: str):
    raw_chunks = chunk_text(text, chunk_size=2000, chunk_overlap=200)
    
    all_propositions = []
    all_metadata = []
    
    for chunk_idx, chunk in enumerate(raw_chunks):
        # Generate propositions from this chunk
        raw_propositions = generate_propositions(self.llm, chunk)
        
        for prop in raw_propositions:
            # Quality gate
            score = evaluate_proposition(self.llm, prop)
            if score >= self.quality_threshold:  # default: 0.6
                all_propositions.append(prop)
                all_metadata.append({
                    "source_chunk": chunk_idx,
                    "quality_score": score
                })
    
    # Embed all accepted propositions
    prop_documents = [
        Document(content=p, metadata=m)
        for p, m in zip(all_propositions, all_metadata)
    ]
    prop_documents = self.embedder.embed_documents(prop_documents)
    self.vector_store.add_documents(prop_documents)
    
    print(f"Indexed {len(all_propositions)} propositions from {len(raw_chunks)} chunks")
```

---

## The Embedding Alignment Problem: Why Propositions Win

To understand why proposition chunking dramatically improves retrieval, you need to understand the **embedding alignment problem** in standard RAG.

### Standard Chunk Embedding

Consider a 1,000-character chunk about climate science:

> "Global warming, caused primarily by increasing concentrations of CO₂ and other greenhouse gases in the atmosphere, has led to rising global temperatures. This warming affects weather patterns, causing more frequent extreme weather events. Ocean temperatures have also risen, leading to coral bleaching..."

When you embed this chunk, the resulting vector is an *aggregate representation* of all these topics: CO₂, greenhouse gases, global temperatures, weather patterns, extreme weather, ocean temperatures, coral bleaching. The embedding is a blended cocktail of meanings.

Query: "What causes coral bleaching?"  
The query vector is specifically about "coral bleaching causes."  
The chunk vector is about everything. Its distance from the query vector is mediated by the entire paragraph, not just the coral bleaching sentence.

### Proposition Embedding

Instead of the full paragraph, you've indexed these individual propositions:
- "Rising CO₂ concentrations are the primary cause of global warming."
- "Global warming leads to rising global temperatures."
- "Global warming causes more frequent extreme weather events."
- "Rising ocean temperatures cause coral bleaching."

Query: "What causes coral bleaching?"  
Now the query is compared to a proposition *specifically about coral bleaching causes*. The cosine similarity is near 1.0. The match is essentially perfect.

This is the mathematical reason proposition chunking works: by reducing each indexed unit from an *aggregate* of many concepts to a *single* concept, you make embeddings more discriminative and queries more precise.

---

## The Cost-Quality Trade-off: A Quantitative Analysis

Proposition chunking is expensive. Let's be precise about the costs:

For a 100-page PDF (~300,000 characters):
- Processing chunks: 300,000 / 2,000 ≈ 150 chunks
- Proposition generation calls: 150 LLM calls
- Average propositions per chunk: ~8
- Total propositions: ~1,200
- Evaluation calls: 1,200 LLM calls
- **Total offline LLM calls: 1,350**

Compared to Simple RAG with the same document:
- Embedding call: 1 batch call
- **Total offline LLM calls: 0**

At `gpt-4o-mini` pricing (~$0.00015/1K input tokens, ~$0.0006/1K output tokens), 1,350 calls with ~200 input tokens each ≈ **~$0.04 total**. For most enterprise use cases, this is negligible.

The cost is **paid once** at index time. Every query thereafter runs at standard RAG speed with zero additional overhead.

### When is the cost justified?

| Corpus Stability | Query Volume | Verdict |
|-----------------|-------------|---------|
| Stable (rarely changes) | High (many queries/day) | **Strongly justified** |
| Stable | Low | Likely justified |
| Frequently updated | High | Consider caching unchanged propositions |
| Frequently updated | Low | Marginal — evaluate whether quality gain is needed |

---

## Retrieval Quality Comparison

**Test query**: "Who was the first person to walk on the Moon, and when did it happen?"

**Standard RAG** result (returns 1 chunk):
> "The Apollo 11 mission, launched on July 16, 1969, successfully landed the first humans on the Moon on July 20, 1969. Mission commander Neil Armstrong and lunar module pilot Buzz Aldrin spent about two hours on the lunar surface..."

The LLM receives 250 characters before getting to the relevant facts, plus additional facts about Aldrin and Collins that aren't needed.

**Proposition RAG** result (returns 3 propositions):
1. "Neil Armstrong was the first human to walk on the Moon."
2. "The first humans landed on the Moon on July 20, 1969."
3. "Neil Armstrong was the mission commander of Apollo 11."

The LLM receives exactly the information it needs — clean, precise, no noise. The answer practically writes itself from these propositions.

---

## Advanced Configuration

```python
PropositionChunkingRAG(
    file_path="document.pdf",
    
    # Processing chunk size (for LLM input)
    processing_chunk_size=2000,
    processing_chunk_overlap=200,
    
    # Quality threshold for proposition acceptance
    quality_threshold=0.6,      # 0.0-1.0; higher = stricter filtering
    
    # How many propositions to retrieve per query
    k=5,                        # propositions are small, so k can be higher
    
    # Models
    embedding_model="text-embedding-3-small",
    chat_model="gpt-4o-mini",
)
```

**Tuning `quality_threshold`:**
- `0.4`: Lenient — more propositions indexed, including some vague ones
- `0.6`: Standard — good balance of quantity and quality  
- `0.8`: Strict — only crystal-clear, specific propositions; fewer indexed items

**Tuning `k` for propositions**: Because propositions are small (typically 1-2 sentences), you can afford a higher k than standard RAG. `k=5` for propositions provides roughly the same context volume as `k=3` for 1000-character chunks, but with better precision since each of the 5 items is specifically relevant.

---

## Integration with Other Techniques

Proposition chunking composites naturally with other techniques:

- **Proposition Chunking + Context Chunk Headers**: Prepend the document title/summary to each proposition before embedding. Now each proposition carries document context AND is semantically precise.

- **Proposition Chunking + Reranking**: First retrieve the top-15 propositions by vector similarity, then rerank them with a cross-encoder to select the top-5. This catches cases where proposition embedding alignment is good but not perfect.

- **Proposition Chunking + Contextual Compression**: Since propositions are already atomic, compression is less needed — but it can still remove propositions that are retrieved by theme but don't contain the specific fact needed.

---

## When to Use Proposition Chunking

Proposition chunking gives the best returns on knowledge bases where every retrieved fact matters — medical literature, legal documents, financial reports. It shines when users ask specific, factual questions and when the corpus is stable enough that the offline indexing cost isn't a recurring pain. If your documents change multiple times a day or if your questions are inherently broad and analytical ("summarize the market landscape"), the overhead isn't justified. But for high-precision Q&A over a stable corpus, it often delivers the largest single retrieval quality improvement you can make.

---

## Summary

Proposition Chunking is a fundamental rethinking of what belongs in a RAG index. By replacing arbitrarily-sized text windows with precisely-scoped, independently-meaningful atomic facts, the index becomes a precision instrument rather than a pile of text shards.

The offline cost — multiple LLM calls per document — is real but bounded and amortized across all queries. The retrieval quality improvement is permanent and compounds with every interaction: users always receive precisely the facts they asked for, not paragraphs they must sift through.

For high-stakes knowledge bases where answer precision is paramount, proposition chunking often delivers the largest retrieval quality improvement of any single technique in this collection.
