# HyPE: Pre-Computing the Questions Your Documents Answer

---

## Introduction

HyDE (Hypothetical Document Embeddings) bridges the query-document embedding gap from the query side — by expanding the query into something more document-like at query time.

**HyPE (Hypothetical Prompt Embeddings)** bridges the same gap from the document side — by expanding each document chunk into questions at *index time*.

The insight: if you know what questions each chunk of text answers, you can store those question embeddings in the index alongside (or instead of) the chunk embeddings. When a user poses a question at query time, embedding-to-embedding similarity between the user's question and stored question embeddings will be dramatically higher than embedding-to-embedding similarity between the user's question and a chunk of declarative text.

HyPE is the index-time complement to HyDE. Together, they attack the query-document mismatch from both ends.

---

## Why Questions Embed Better Against Questions

### The Linguistic Symmetry Principle

Consider a scientific fact and the question it answers:

**Fact** (document form): "Water reaches its maximum density at 4°C (39.2°F), which is why ice floats on water and aquatic environments remain liquid at depth during winter freezing."

**Question** (query form): "Why does ice float on water?"

In embedding space, the *question* "Why does ice float?" embeds much closer to a user's query "Why does ice float?" than the *fact* does — because both are in question form, both are short, and both use the same interrogative structure and vocabulary.

The embedding of "Water reaches its maximum density at 4°C" places the fact in document-statement embedding space. The embedding of "Why does ice float?" places the question in query-question embedding space. A user asking "Why does ice float?" is in the same space as the stored question — high similarity — but further from the raw fact statement.

### The Matching Gain

Empirically, on retrieval benchmarks (NQ, TriviaQA), pre-generated question embeddings increase Recall@5 by 15-25% compared to embedding the source document text directly — for query distributions that are question-form (which covers most user-facing RAG applications).

---

## Index-Time: Generating Hypothetical Prompts

The core difference from standard RAG is entirely at indexing:

```python
def index_document(self, text: str) -> int:
    # Step 1: Standard chunking
    chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
    total_indexed = 0
    
    for chunk_idx, chunk_content in enumerate(chunks):
        # Step 2: Generate questions this chunk answers
        questions = self._generate_hypothetical_prompts(chunk_content)
        
        # Step 3: Index the QUESTIONS (not the chunk text)
        for question in questions:
            question_doc = Document(
                content=question,          # QUESTION is embedded
                metadata={
                    "chunk_index": chunk_idx,
                    "source_content": chunk_content,  # original chunk stored here
                    "type": "hype_question"
                }
            )
            question_docs.append(question_doc)
        
        # Step 4: Also index the chunk itself (for direct content matching)
        chunk_doc = Document(
            content=chunk_content,
            metadata={"chunk_index": chunk_idx, "type": "source_chunk"}
        )
        chunk_docs.append(chunk_doc)
    
    # Embed and index both questions and source chunks
    all_docs = chunk_docs + question_docs
    all_docs = self.embedder.embed_documents(all_docs)
    self.vector_store.add_documents(all_docs)
    
    return len(all_docs)
```

### The Question Generation Prompt

```python
def _generate_hypothetical_prompts(
    self,
    chunk_text: str,
    num_questions: int = 5
) -> List[str]:
    """
    Generate questions a user might ask whose answer is this chunk.
    
    Unlike Document Augmentation (which also generates questions),
    HyPE's question generation is optimized specifically to match
    the natural query phrasings of real users — not just semantically
    accurate paraphrases.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Generate realistic questions that users would naturally ask "
                "when searching for the information contained in the given text. "
                "Focus on:\n"
                "1. Natural user language (not formal academic phrasing)\n"
                "2. Diverse question types: 'what', 'how', 'why', 'when', 'who'\n"
                "3. Both specific (fact-seeking) and conceptual questions\n"
                "4. Questions at different expertise levels (beginner to expert)\n"
                "5. Common phrasings someone would type into a search bar\n\n"
                "Return JSON: {\"questions\": [\"q1\", \"q2\", ...]}"
            )
        },
        {
            "role": "user",
            "content": (
                f"For this text, generate {num_questions} questions a user "
                f"might ask whose answer is found here:\n\n{chunk_text}"
            )
        }
    ]
    result = self.llm.chat_json(messages)
    return result.get("questions", [])
```

#### HyPE vs. Document Augmentation Question Generation

Both techniques generate questions. The difference is subtle but important:

| Aspect | Document Augmentation | HyPE |
|--------|----------------------|------|
| Goal | Match diverse query formulations | Match natural search-bar queries |
| Style | Diverse paraphrases | Natural user language |
| Coverage | Wide semantic coverage | User intent alignment |
| Query-time use | Still has source chunks in index | Primarily uses question embeddings |

In practice, HyPE's question generation emphasizes match to how users *actually type queries*, while Document Augmentation emphasizes semantic coverage breadth. The implementation is similar; the prompt engineering differs.

---

## Query Time: Question-to-Question Matching

At query time, HyPE is identical to standard RAG — the difference was entirely at indexing:

```python
def query(self, question: str) -> Tuple[str, List[str]]:
    # Embed the user's question
    question_embedding = self.embedder.embed_text(question)
    
    # FAISS searches across ALL embeddings (both questions and source chunks)
    # Questions stored in the index embed much closer to the query
    results = self.vector_store.search(question_embedding, k=self.k * 3)
    
    # Deduplicate: multiple question embeddings point to the same source chunk
    seen_chunks = set()
    final_contexts = []
    
    for result in results:
        metadata = result.document.metadata
        
        # Retrieve the SOURCE CONTENT (the original chunk), not the question
        if metadata.get("type") == "hype_question":
            content = metadata.get("source_content", "")
            chunk_id = metadata.get("chunk_index")
        else:
            content = result.document.content
            chunk_id = metadata.get("chunk_index")
        
        if chunk_id not in seen_chunks and content:
            seen_chunks.add(chunk_id)
            final_contexts.append(content)
            
            if len(final_contexts) >= self.k:
                break
    
    answer = self._generate_answer(question, final_contexts)
    return answer, final_contexts
```

**The matching chain**:
1. User query → embedded as question vector Q
2. Stored hypothetical question → embedded as question vector Q'
3. Q and Q' are both questions — highly similar in embedding space
4. Q' has `source_content` metadata → original document chunk
5. LLM receives original chunk (not the question) to generate the answer

---

## Index Size and Query Time Analysis

For a document with N chunks and M questions per chunk:

| Component | Vectors in FAISS |
|-----------|-----------------|
| Source chunks | N |
| HyPE questions | N × M |
| **Total** | **N × (M + 1)** |

For M=5: index is 6× larger than standard RAG. FAISS flat search over 6N vectors is still fast (sub-millisecond for N < 100K).

**Query-time overhead**: Zero additional LLM calls. The extra vectors are in FAISS, which is fast. Query time is essentially identical to standard RAG.

This is the key contrast with HyDE:
- HyDE adds 1 LLM call at *query time* (generating hypothetical)
- HyPE adds N LLM calls at *index time* (generating questions), zero at query time

For high-query-volume systems, HyPE's front-loaded cost is much better for latency. For rarely-queried corpora, HyDE's no-index-cost approach is more efficient.

---

## Worked Example

**Chunk text**: "The Maillard reaction occurs when amino acids and reducing sugars react at temperatures above 140°C, producing hundreds of different flavor compounds and the characteristic brown color in cooked foods. The reaction rate depends on pH (faster in alkaline environments), water activity (aw < 0.6 increases rate), and specific amino acid types present."

**Generated HyPE questions**:
1. "Why does meat turn brown when you cook it?"
2. "What chemical reaction causes food to brown?"
3. "How does the Maillard reaction work?"
4. "What temperature does the Maillard reaction happen?"
5. "Why does baking soda make cookies brown faster?"

**User query at query time**: "How do you get better browning when cooking?"

Direct chunk embedding vs. query: moderate similarity (complex technical text vs. simple question)  
"Why does meat turn brown when you cook it?" vs. query: high similarity (both are browning questions)

The question bridges the linguistic form gap. The Maillard chunk is retrieved at rank 1 through its question embedding, then the original technical chunk text is delivered to the LLM.

---

## HyPE + HyDE: The "Meet in the Middle" Stack

The maximum retrieval benefit comes from combining both:

```
Standard RAG:
    User query → embed(query) ←→ embed(chunk)
    [query-form vs. document-form — ASYMMETRIC]

HyDE:
    User query → generate_hypothetical → embed(hypothetical) ←→ embed(chunk)
    [document-form vs. document-form — SYMMETRIC]

HyPE:
    User query → embed(query) ←→ embed(stored_question)
    [question-form vs. question-form — SYMMETRIC]

HyDE + HyPE:
    User query → generate_hypothetical → embed(hypothetical) ←→ embed(stored_question)
    [both sides bridged — MAXIMUM ALIGNMENT]
```

The combined pipeline:
1. HyPE populates the index with question embeddings at index time
2. HyDE generates a hypothetical document at query time
3. The hypothetical's embedding (document-form) matches stored questions better than the raw query

This "meet in the middle" strategy maximizes the chance that the relevant chunk is found, regardless of vocabulary gaps or form-factor asymmetry.

---

## Configuration Recommendations

```python
HyPERAG(
    file_path="document.pdf",
    
    # Questions per chunk — primary tuning parameter
    num_questions_per_chunk=5,  # default; increase for vocabulary-heavy domains
    
    # Standard chunking
    chunk_size=1000,
    chunk_overlap=200,
    
    # Retrieval — over-fetch to account for deduplication of questions→chunks
    k=3,  # final unique chunks delivered to LLM
    
    embedding_model="text-embedding-3-small",
    chat_model="gpt-4o-mini"
)
```

**Per-domain `num_questions_per_chunk` recommendations:**

| Domain | Recommended |
|--------|-------------|
| General enterprise Q&A | 5 (default) |
| Medical / clinical | 7 (diverse user expertise levels) |
| Legal documents | 6 (multiple query phrasings for clauses) |
| Technical manuals | 5 |
| FAQ / customer support | 3 (already question-optimized content) |

---

## When to Use HyPE

HyPE adds the most value in systems where indexing time is under your control and query latency is a hard constraint. Because all the question-generation work happens at index time, query time is identical to standard RAG — no extra LLM calls, no additional latency. For long-lived knowledge bases where the same corpus is queried hundreds or thousands of times, the per-chunk indexing cost amortizes quickly against the retrieval quality gains.

The technique is most effective when your user population reliably poses question-form queries — which describes most enterprise search, internal chatbots, and customer support applications. For corpora with high vocabulary mismatch between source text and user language, increasing `num_questions_per_chunk` sharpens the coverage significantly.

If the corpus changes frequently, regenerating HyPE questions on every update can become expensive — each updated chunk requires new LLM calls. In those cases, HyDE offers a cheaper alternative since its cost is incurred at query time rather than index time. Similarly, if index size must be kept minimal, HyPE's M+1 vector expansion per chunk may be a constraint. And if your users submit commands, code snippets, or other non-question-form queries, question-embedding matching provides little advantage over direct chunk embeddings.

---

## Summary

HyPE inverts the standard RAG indexing assumption: instead of indexing what documents *say*, index what questions they *answer*. By pre-generating question embeddings for each chunk at index time, HyPE ensures that question-form queries find their way to relevant chunks through question-to-question similarity — a much more direct and reliable path than question-to-document-text matching.

The result is a retrieval system that "speaks the same language" as its users — matching natural query expressions to the natural questions that source content answers. Combined with HyDE at query time, it forms a complete, dual-sided bridge across the query-document linguistic gap that limits conventional RAG.
