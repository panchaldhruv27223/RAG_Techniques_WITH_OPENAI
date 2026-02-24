# Document Augmentation: Indexing the Questions Your Chunks Answer

## Introduction

At the heart of RAG lies an asymmetry problem: documents are written as *answers*, but users query as *questions*. A research paper's abstract says "We investigate the effects of X on Y in the context of Z." A user asks "Does X cause changes in Y?" The semantic content is similar, but the linguistic form is radically different.

Embedding models are trained to place semantically similar text close together in vector space. They're generally good at this — "president of the United States" and "American head of government" will cluster together. But they're less reliable when the form of the text diverges: a statement and the question it answers don't always embed similarly, even when they're directly semantically related.

This form-based embedding asymmetry is the root cause of a common RAG failure: the system fails to retrieve clearly relevant chunks when the query phrasing doesn't match the chunk phrasing — even when the chunk *would* answer the query perfectly.

**Document Augmentation** bridges this gap by using an LLM at index time to generate a set of questions that each chunk answers. Both the chunk and its question embeddings are added to the index. At query time, the user's question is compared not just against chunk embeddings, but against question embeddings — matching question to question, form to form.

---

## The Embedding Asymmetry Problem: A Concrete Analysis

### The Linguistic Gap

Consider this medical document chunk:

> "Metformin reduces hepatic glucose production by activating AMP-activated protein kinase (AMPK), which subsequently inhibits gluconeogenesis. Clinical studies demonstrate HbA1c reduction of approximately 1.5% with first-line metformin therapy in type 2 diabetes patients."

**User query**: "How does metformin work for type 2 diabetes?"

The chunk embeds as: [biochemical mechanism, AMPK pathway, hepatic glucose, HbA1c, clinical study]  
The query embeds as: [mechanism of action, drug, diabetes treatment, metformin]

These embeddings are semantically related but not perfectly aligned. The chunk contains mechanism-of-action information but uses domain-specific technical vocabulary (AMPK, gluconeogenesis, HbA1c) that the query doesn't use at all. Cosine similarity might be 0.68 — retrievable, but not at the top.

**Pre-generated questions** stored for this chunk:
- "How does metformin lower blood sugar in type 2 diabetes?"
- "What is the mechanism of action of metformin?"
- "How effective is metformin as a first-line diabetes treatment?"
- "What does metformin do to glucose production in the liver?"

User query: "How does metformin work for type 2 diabetes?"  
Question embedding: "How does metformin lower blood sugar in type 2 diabetes?"

These two embed as nearly identical — both are how-does-drug-work questions about metformin and diabetes. Cosine similarity → 0.94. The relevant chunk is now retrieved at rank 1 instead of potentially rank 3 or 4.

---

## How Document Augmentation Works

### Index-Time Pipeline

```
Document chunks
     │
     ├── chunk_0: "Metformin reduces hepatic glucose..."
     ├── chunk_1: "Insulin resistance occurs when..."
     └── chunk_2: "HbA1c is a measure of..."
     │
     ▼
[For each chunk, LLM generates N questions the chunk could answer]
     │
     ├── chunk_0 → ["How does metformin work?", "What is AMPK?", ...]
     ├── chunk_1 → ["What causes insulin resistance?", "How does insulin resistance develop?", ...]
     └── chunk_2 → ["What is HbA1c?", "How do doctors measure diabetes control?", ...]
     │
     ▼
[Embed each question; link it to its source chunk via metadata]
     │
     ▼
FAISS index contains:
  ├── Original chunk embeddings (content-to-content matching)
  └── Question embeddings (question-to-question matching)
          Each with metadata: {"source_chunk_id": i, "type": "question"}
```

When a query arrives, it's compared against *all* embeddings — both chunk and question embeddings. If a question embedding matches the query more closely than any chunk embedding, that question's source chunk is returned.

### Query-Time Pipeline

```
User query: "How does metformin work?"
     │
     ▼
Embed query → query_vector
     │
     ▼
FAISS search across ALL embeddings (chunks + questions)
     │
     ├── question "How does metformin lower blood sugar?" → sim 0.94 → source: chunk_0
     ├── chunk "Metformin reduces hepatic glucose..." → sim 0.72 → chunk_0
     ├── question "What is metformin's mechanism?" → sim 0.88 → source: chunk_0
     └── chunk "Type 2 diabetes treatments..." → sim 0.65 → chunk_7
     │
     ▼
Deduplicate: multiple hits pointing to chunk_0 → return chunk_0 once
Return: [chunk_0, chunk_7]
     │
     ▼
LLM generates answer from chunk content (not from the questions)
```

---

## Implementation Walkthrough

### Question Generation

```python
def generate_questions(self, chunk_text: str, num_questions: int = 5) -> List[str]:
    """
    Generate diverse questions that this chunk could answer.
    
    Diversity is critical — if all questions are too similar, they cover
    the same query angles. Diverse questions extend the "retrieval surface area"
    of the chunk across more query types.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at generating diverse, realistic questions that "
                "a user might ask whose answer is contained in a given text. "
                "Generate questions that:\n"
                "1. Are genuinely answered by the text\n"
                "2. Use different phrasings and vocabulary\n"
                "3. Cover different aspects of the text\n"
                "4. Range from specific to general\n"
                "5. Match the natural language a user would actually type\n\n"
                "Do NOT generate questions about topics only tangentially mentioned.\n"
                f"Return JSON: {{\"questions\": [\"q1\", \"q2\", ..., \"q{num_questions}\"]}}"
            )
        },
        {
            "role": "user",
            "content": f"Generate {num_questions} questions for this text:\n\n{chunk_text}"
        }
    ]
    result = self.llm.chat_json(messages)
    return result.get("questions", [])
```

**Why diversity matters**: If you generate 5 questions and 4 of them are near-synonyms:
- "What does metformin do?" 
- "How does metformin work?"
- "What is metformin's action?"
- "What is the mechanism of metformin?"
- "How does metformin affect diabetes?"

The 4 near-synonyms add minimal coverage — they cluster in the same embedding region. Better to generate:
- "What is the biochemical pathway activated by metformin?" (specific, technical)
- "How does metformin lower blood sugar?" (practical, patient-facing)
- "Is metformin effective as a first-line diabetes medication?" (clinical question)
- "What enzyme does metformin activate?" (knowledge-test style)
- "Why is metformin preferred for type 2 diabetes?" (reasoning style)

These 5 questions cover 5 different positions in embedding space, each acting as a "landing zone" that captures a different class of user queries.

### Building the Augmented Index

```python
def index_document(self, text: str) -> int:
    # Step 1: Create standard chunks
    chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
    total_indexed = 0
    
    for chunk_idx, chunk_content in enumerate(chunks):
        # Step 2: Store the chunk itself
        chunk_doc = Document(
            content=chunk_content,
            metadata={
                "type": "chunk",
                "chunk_id": chunk_idx,
                "source_chunk_id": chunk_idx  # points to itself
            }
        )
        
        # Step 3: Generate questions for this chunk
        questions = self.generate_questions(chunk_content, num_questions=5)
        
        # Step 4: Create a Document for each question
        question_docs = []
        for q in questions:
            if q.strip():  # skip empty/malformed
                question_docs.append(Document(
                    content=q,                     # the QUESTION text is embedded
                    metadata={
                        "type": "question",
                        "chunk_id": chunk_idx,
                        "source_chunk_id": chunk_idx,  # links back to parent chunk
                        "source_content": chunk_content  # full chunk for retrieval
                    }
                ))
        
        # Step 5: Embed chunk + questions together (one batch call)
        all_docs = [chunk_doc] + question_docs
        all_docs = self.embedder.embed_documents(all_docs)
        self.vector_store.add_documents(all_docs)
        
        total_indexed += len(all_docs)
    
    print(f"Indexed {len(chunks)} chunks + {total_indexed - len(chunks)} questions")
    return total_indexed
```

**Crucial detail**: The question Document's `content` field is the **question text** — not the chunk text. This is what gets embedded. But the metadata stores the full `source_content` (the chunk), which is what gets returned to the LLM at query time.

When a question's embedding matches the query, we retrieve the *chunk* (via `source_content`), not the question. The LLM never sees the questions — it only sees the source passages.

### Query and Deduplication

```python
def query(self, question: str) -> Tuple[str, List[str]]:
    query_embedding = self.embedder.embed_text(question)
    
    # Retrieve top results — some may be question embeddings, some chunk embeddings
    results = self.vector_store.search(query_embedding, k=self.k * 3)  # over-fetch
    
    # Deduplicate: multiple question embeddings may point to the same source chunk
    seen_chunk_ids = set()
    unique_contexts = []
    
    for result in results:
        chunk_id = result.document.metadata.get("source_chunk_id")
        
        # Get the actual content to return: either direct chunk or question's source
        if result.document.metadata.get("type") == "question":
            content = result.document.metadata.get("source_content", "")
        else:
            content = result.document.content
        
        # Only include each chunk once, even if matched by multiple question embeddings
        if chunk_id not in seen_chunk_ids and content:
            seen_chunk_ids.add(chunk_id)
            unique_contexts.append(content)
            
            if len(unique_contexts) >= self.k:  # stop at k unique chunks
                break
    
    answer = self._generate_answer(question, unique_contexts)
    return answer, unique_contexts
```

Deduplication is critical. Without it, a single chunk might appear 5 times (once for its own embedding + 4 times for its question embeddings), flooding the context with duplicates.

---

## The Cost Equation

### Index Time
For a document with N chunks, each with M questions generated:

| Operation | Calls |
|-----------|-------|
| Question generation | N LLM calls |
| Embedding (chunks + questions) | ceil((N × M + N) / 2048) embedding API calls |
| **Total for 100-chunk document, M=5** | 100 LLM calls |

At `gpt-4o-mini`: ~$0.02 for 100 calls (each ~200 input tokens, ~100 output tokens).

### Index Size
The FAISS index grows by M× (where M is questions per chunk):
- Standard RAG: 100 chunks → 100 vectors
- Augmented RAG: 100 chunks → 100 chunks + 500 questions = 600 vectors

For M=5, the index is 6× larger. For typical document sizes this remains well within memory budgets (600 × 1536-dim vectors × 4 bytes = ~3.7 MB).

### Query Time
No additional overhead. Query time is identical to standard RAG — FAISS search now searches over 600 vectors instead of 100, but FAISS flat search over 600 vectors is still sub-millisecond.

---

## Performance Analysis: When Does Augmentation Help?

### Vocabulary Mismatch (Largest Benefit)

| Query Style | Chunk Style | Standard RAG | Augmented RAG |
|------------|-------------|-------------|--------------|
| Lay language ("how does X work?") | Technical jargon | Often fails | Strong |
| Technical query | Technical document | Good | Similar |
| Question form | Statement form | Mixed | Strong |
| Different domain vocabulary | Specialized terms | Poor | Good |

### Typical Recall@3 Improvement on Medical/Legal Corpora

Academic benchmarks show document augmentation improving Recall@3 by 10-25% on vocabulary-divergent queries. For general-purpose corpora where query and document vocabulary overlaps naturally, improvement is smaller (5-10%).

---

## Practical Configuration Guide

```python
DocumentAugmentationRAG(
    file_path="document.pdf",
    
    # How many questions per chunk
    # More questions = wider retrieval coverage, higher index cost
    num_questions_per_chunk=5,  # typical sweet spot
    
    # Standard chunking parameters
    chunk_size=1000,
    chunk_overlap=200,
    
    # Over-fetch factor to account for deduplication
    # With k=3 and 6 candidates, deduplication usually leaves 3 unique chunks
    k=3,
    
    embedding_model="text-embedding-3-small",
    chat_model="gpt-4o-mini"
)
```

**Tuning `num_questions_per_chunk`:**

| Questions/chunk | Retrieval coverage | Index overhead | Recommended for |
|----------------|-------------------|----------------|-----------------|
| 2 | Narrow | Minimal | Simple, homogeneous queries |
| 5 | Good | Moderate | General use (default) |
| 8 | Wide | Significant | High-vocabulary-mismatch corpora |
| 10+ | Very wide | High | Specialized technical domains |

---

## Integration with Other Techniques

- **Augmentation + Context Headers**: Generate questions after adding the header, so questions embed with document context included: "According to the IPCC AR6 Report, how does CO₂ affect global temperatures?" This creates hyper-specific question embeddings.

- **Augmentation + Proposition Chunking**: Generate questions for each proposition. Since propositions are atomic facts, questions about them are highly specific — excellent for detailed Q&A.

- **Augmentation + Reranking**: Retrieve an over-large set (using both chunk and question embeddings), then rerank all candidates by a cross-encoder. Best of question-match retrieval + cross-encoder precision.

---

## When to Use Document Augmentation

Document Augmentation is most effective when there's a meaningful vocabulary gap between how your documents are written and how users ask questions. This is endemic to specialized domains: a medical knowledge base uses clinical terminology ("hepatic gluconeogenesis," "AMPK activation") while patients and generalist users ask in lay language ("how does metformin work?"). Pre-generating questions that bridge this linguistic gap gives the embedding index a "translation layer" that catches queries the document vocabulary alone would miss.

Multi-language or cross-terminology scenarios benefit for the same reason — even within a single language, different communities use different words for the same concept. For knowledge bases that serve users across expertise levels, question augmentation ensures that both a specialist and a novice asking about the same underlying fact have a good chance of retrieving the right content.

The main constraints are index-time cost (one LLM call per chunk) and index size growth (M× larger for M questions per chunk). This makes it less suitable for corpora that update frequently, since re-generating questions on every update compounds the cost. If your documents and user queries already share vocabulary naturally, the improvement will be modest and may not justify the overhead.

---

## Summary

Document Augmentation bridges the linguistic gap between how information is *stored* (statements, reports, descriptions) and how it is *sought* (questions, queries, conversational prompts). By pre-generating diverse questions for each chunk at index time and indexing both the questions and the chunks, the system creates multiple "retrieval hooks" per piece of content — each hook catching a different formulation of the same underlying information need.

The technique shines brightest when the vocabulary of users and the vocabulary of source documents diverge — the most common scenario in enterprise, medical, legal, and research RAG systems. The index-time investment (one LLM call per chunk) pays dividends on every query for the lifetime of the knowledge base.
