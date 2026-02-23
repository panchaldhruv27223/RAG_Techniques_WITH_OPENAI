# Document Augmentation via Question Generation: Closing the Query-Document Gap

> **Technique:** Document Augmentation (Question-Driven Indexing)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

There's a fundamental asymmetry in standard RAG: documents are written as statements of fact, but users query with questions. A clinical trial paper says "The intervention reduced mortality by 23%"; a user asks "Does Drug X reduce death rates?" The semantic distance between these two phrasings — even though they're about the same thing — can cause retrieval to miss the match entirely.

**Document Augmentation via Question Generation** bridges this gap by enriching the index at build time. For every text chunk, an LLM generates a set of questions that the chunk could answer. These questions — along with the original text — are embedded and indexed. At query time, user questions are matched against both document text and pre-generated questions, dramatically improving retrieval recall.

It's a form of **query-document alignment** performed proactively at index time rather than reactively at query time.

---

## The Vocabulary Mismatch Problem

This technique directly addresses what information retrieval researchers call the *vocabulary mismatch problem*: the same concept expressed differently by authors versus users. Some examples:

| Document says | User might ask |
|---------------|---------------|
| "CO2 emissions have risen 2.4ppm annually" | "How much is carbon dioxide increasing?" |
| "The patient exhibited bradycardia" | "Did the patient have a slow heart rate?" |
| "Revenue CAGR of 18% over five years" | "How fast was the company growing?" |

In standard dense retrieval, the embedding of "CO2 emissions have risen 2.4ppm annually" is statistically closer to other emission statistics than to "How much is carbon dioxide increasing?" — even though one perfectly answers the other.

By pre-generating the question "How much has CO2 increased annually?" and indexing it alongside the chunk, we create a direct semantic bridge between document content and user query style.

---

## How the Pipeline Works

### Index Time (Offline)

```
Document text
    ↓
Split into chunks
    ↓
For each chunk:
    [LLM] Generate N hypothetical questions this chunk answers
    ↓
    Embed chunk text → vector(s)
    Embed each question → vector(s)
    ↓
    Store ALL vectors linked to this chunk in FAISS
    ↓
[Query time] search against all vectors (text + questions)
→ the matched vector's metadata points back to the original chunk
```

The key insight: multiple vectors (original text + N questions) all point to the **same chunk content**. Whichever vector matches the user's query, you retrieve the same reliable original text to send to the LLM.

### Query Time (Online)

```
User query
    ↓
Embed query → query vector
    ↓
FAISS search against ALL indexed vectors (text + questions)
    ↓
Matched vectors → retrieve their associated chunk content
    ↓
Deduplicate chunks (multiple question-vectors may point to same chunk)
    ↓
LLM generates answer from original chunk text
```

Deduplication is critical: if the query matches 3 question-vectors that all point to the same chunk, that chunk should appear only once in the context.

---

## Code Deep Dive

### Question Generation

```python
def generate_questions_for_chunk(self, chunk_text: str) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a question generator for a RAG knowledge base. "
                "Given text, generate diverse questions that this text answers. "
                "Questions should vary in phrasing and cover different aspects. "
                "Return JSON: {\"questions\": [\"q1\", \"q2\", ...]}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Generate {self.num_questions} questions that this text chunk answers:\n\n"
                f"{chunk_text}\n\n"
                "Focus on the key facts and ensure questions are standalone."
            )
        }
    ]
    result = self.llm.chat_json(messages)
    return result.get("questions", [])
```

The prompt asks for `num_questions` diverse questions. Diversity is important: three questions asking essentially the same thing in the same phrasing are wasteful. Good questions cover:
- Direct factual queries ("What was the growth rate?")
- Comparative queries ("How does X compare to Y?")
- Causal queries ("Why did X happen?")
- Definitional queries ("What is X?")

### Building the Augmented Index

```python
def index_with_augmentation(self, text: str):
    chunks = chunk_text(text, chunk_size=self.chunk_size, 
                        chunk_overlap=self.chunk_overlap)
    
    for i, chunk in enumerate(chunks):
        # Embed the original chunk text
        chunk_embedding = self.embedder.embed_text(chunk)
        chunk_doc = Document(
            content=chunk,
            metadata={"chunk_index": i, "type": "original"},
            embedding=chunk_embedding
        )
        self.vector_store.add_documents([chunk_doc])
        
        # Generate and embed questions
        questions = self.generate_questions_for_chunk(chunk)
        for j, question in enumerate(questions):
            q_embedding = self.embedder.embed_text(question)
            q_doc = Document(
                content=chunk,          # ← Return original chunk, not question
                metadata={"chunk_index": i, "type": "question", 
                         "question": question},
                embedding=q_embedding   # ← But search on question embedding
            )
            self.vector_store.add_documents([q_doc])
```

**The critical detail**: question documents store `content=chunk` (the original text), not `content=question`. This means retrieval always returns meaningful document text — never a question. The question's embedding is used for *matching*, but the chunk's text is returned for *answering*.

### Query-Time Deduplication

```python
def retrieve(self, query: str, k: int = 3) -> List[str]:
    query_embedding = self.embedder.embed_text(query)
    
    # Search more results since we'll deduplicate
    raw_results = self.vector_store.search(query_embedding, k=k*3)
    
    seen_chunks = set()
    unique_chunks = []
    
    for result in raw_results:
        chunk_idx = result.document.metadata["chunk_index"]
        if chunk_idx not in seen_chunks:
            seen_chunks.add(chunk_idx)
            unique_chunks.append(result.document.content)
        
        if len(unique_chunks) >= k:
            break
    
    return unique_chunks
```

We search for `k*3` results to ensure that after deduplication we still have at least `k` unique chunks. Multiple question matches for the same chunk don't count as separate results.

---

## Quantifying the Index Expansion

If you have 100 chunks and generate 5 questions per chunk, your FAISS index grows from 100 to 600 vectors (100 original + 500 questions). This is a 6x index expansion in exchange for dramatically improved recall.

This trade-off is generally favorable because:
- FAISS handles millions of vectors with sub-millisecond search
- Index storage cost is modest (each vector is ~6KB for 1536 dimensions)
- The retrieval coverage improvement is worth the storage overhead

---

## Relationship to HyPE (Hypothetical Prompt Embeddings)

Document Augmentation is closely related to **HyPE** (which we'll cover in a dedicated blog). The key distinction:

| Aspect | Document Augmentation | HyPE |
|--------|----------------------|------|
| What gets embedded | Original chunk + generated questions | Generated questions only |
| How queries are matched | Against original text OR questions | Against questions only |
| Chunk returned on match | Original chunk text | Original chunk text |
| Conceptual emphasis | Dual indexing | Full question-mode indexing |

Document Augmentation maintains the original text embedding alongside questions. HyPE goes further, making questions the *sole* retrieval target.

---

## Practical Configuration

```python
rag = DocumentAugmentedRAG(
    file_path="document.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    num_questions=5,    # questions per chunk
    k=3                 # chunks to retrieve per query
)
```

**Choosing `num_questions`:**
- Too few (1-2): Minimal coverage improvement over standard RAG
- Optimal (3-7): Good diversity without excessive LLM cost
- Too many (10+): Diminishing returns; questions become repetitive

**Choosing model for generation**: A cheaper model (`gpt-4o-mini`) is appropriate for question generation since this is a structured task with clear constraints, not a nuanced reasoning problem.

---

## When to Use Document Augmentation

**Best for:**
- Corpora where user queries are written very differently from document content (jargon-laden docs, academic papers, legal text)
- Multi-turn QA systems where question phrasing varies significantly
- Knowledge bases that receive diverse query types from diverse user populations
- Domains with rich synonym and paraphrase variation

**Less critical when:**
- Documents already contain question-like structures (FAQ pages, Q&A forums)
- The user population is technical and queries closely mirror document terminology
- Index rebuild frequency is high (question generation adds indexing cost)

---

## Summary

Document Augmentation via Question Generation tackles the vocabulary mismatch problem where it's most impactful: at index time. By asking "what questions does this chunk answer?" and indexing those questions alongside the original text, the retrieval system gains the ability to match query-style phrasing against document-style facts.

The result is a RAG system that meets users where they are, regardless of whether they phrase their question the same way the document was written. This is especially valuable in enterprise settings where documents are authored by domain experts and queries come from a much broader audience.
