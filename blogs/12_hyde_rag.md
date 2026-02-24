# HyDE: Searching With the Answer You Wish You Had

---

## Introduction

Every embedding model has a representational asymmetry between queries and documents. Documents are typically long, declarative, information-dense. Queries are short, interrogative, information-sparse. A query like "what causes inflation?" is a fundamentally different type of text from the encyclopedic text that answers it — and embedding models, trained on corpora of similar-form text pairs, can struggle to bridge this gap.

**HyDE (Hypothetical Document Embeddings)**, introduced by Gao et al. (2022) at Carnegie Mellon University, solves this asymmetry with a simple but powerful idea:

> Instead of embedding the query and searching for similar documents, generate a *hypothetical answer* to the query and embed *that*. Search document space with a document-type embedding, not a query-type embedding.

The intuition: if you're trying to find documents that answer "what causes inflation?", the embedding of "inflation is caused by excessive money supply growth, demand shocks, and supply-side constraints, as analyzed by Milton Friedman..." is much closer to real documents in embedding space than the embedding of the question itself.

You're not searching with the question. You're searching with the form of thing you're looking for.

---

## The Embedding Asymmetry, Concretely

### Query vs. Document Embedding Space

Modern embedding models are trained to place semantically similar texts near each other. But "semantic similarity" has a subtle form-factor dependency:

```
Query embedding:     [short, interrogative, topic-sparse]
                          ↓ maps to ↓
                     Region of embedding space for QUESTIONS
                     
Document embedding:  [long, declarative, information-dense]
                          ↓ maps to ↓
                     Region of embedding space for TEXT PASSAGES
```

These two regions overlap, but they're not identical. A question and its direct answer do not always embed as nearest neighbors, even when the answer directly addresses the question — because their linguistic forms are different.

### Measuring the Gap

On standard benchmarks (BEIR, MS-MARCO), direct question→document retrieval achieves Recall@10 of ~0.65-0.75 with `text-embedding-3-small`. HyDE-augmented retrieval (hypothetical-answer→document) achieves ~0.78-0.85 on the same benchmarks — a 10-15% improvement in recall by using a document-form query.

---

## The HyDE Pipeline

```
User Query: "What are the economic causes of hyperinflation?"
         │
         ▼
[Step 1: Generate Hypothetical Answer]
    
    LLM prompt: "Write a short passage that answers: 
                 'What are the economic causes of hyperinflation?'
                 Write it as if from an economics textbook."
    
    LLM output: "Hyperinflation typically results from a combination of 
                 excessive money supply expansion by central banks, loss 
                 of public confidence in currency, supply chain disruptions, 
                 and external debt obligations in foreign currency.
                 Historical examples include Weimar Germany (1921-1923) 
                 where monthly inflation peaked at 29,525%..."
         │
         ▼
[Step 2: Embed the Hypothetical Answer (not the original query)]
    
    embed("Hyperinflation typically results from...") → H ∈ ℝ¹⁵³⁶
         │
         ▼
[Step 3: FAISS Search Using Hypothetical Embedding]
    
    FAISS finds actual documents nearest to H
    
    Result: chunks from economics textbooks discussing hyperinflation causes
         │
         ▼
[Step 4: Generate Final Answer from Retrieved Real Documents]
    
    LLM uses retrieved real content (not the hypothetical) to answer
```

The hypothetical answer serves *only* as a retrieval probe. It is never shown to the user. Only the real retrieved documents are used for generation.

---

## Implementation Walkthrough

### Generating the Hypothetical Document

```python
def generate_hypothetical_document(self, query: str) -> str:
    """
    Generate a plausible document passage that would answer the query.
    
    Key prompt design decisions:
    1. "Passage from a document" — encourages document-form text, not answer-form
    2. "Approximately 100-150 words" — long enough to have document-like embedding
       but short enough to be factually plausible even for obscure queries
    3. "Factual and informative" — discourages hedged, conversational responses
    
    Note: factual accuracy of the hypothetical doesn't matter for retrieval.
    We care about the FORM and TOPICAL COVERAGE of the hypothetical, not its truth.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at generating hypothetical document passages "
                "for information retrieval purposes. When given a query, generate "
                "a plausible, factual-sounding passage (approximately 100-150 words) "
                "that would be found in a document answering that query.\n\n"
                "The passage should:\n"
                "- Be written as if extracted from an informative document or textbook\n"
                "- Use domain-appropriate vocabulary and sentence structure\n"
                "- Cover the key aspects of the query topic\n"
                "- Sound authoritative and informative\n\n"
                "IMPORTANT: The passage is for retrieval purposes only. "
                "Its accuracy is secondary to its form and topic coverage."
            )
        },
        {
            "role": "user",
            "content": f"Generate a hypothetical document passage that would answer: {query}"
        }
    ]
    
    hypothetical = self.llm.chat(messages).strip()
    
    print(f"Generated hypothetical ({len(hypothetical)} chars):")
    print(f"  '{hypothetical[:100]}...'")
    
    return hypothetical
```

**Why 100-150 words?** 

An embedding of a single sentence ("Hyperinflation is caused by money supply growth.") has limited topical coverage — it's in embedding space as a short, specific claim. An embedding of 150 words, touching multiple related concepts (causes, mechanisms, historical examples) covers more of the relevant embedding space neighborhood. The richer the hypothetical, the closer its embedding lands to the cluster of real documents on this topic.

**Does factual accuracy matter?**

No — and this is counterintuitive but important. The hypothetical is never shown to the user. If the hypothetical gets a historical fact wrong ("Weimar inflation peaked at 10,000%" when it was 29,525%), the retrieved documents will correct this because we search for real documents that match the *topical form* of the hypothetical, then generate the answer from those real documents.

The hypothetical's job is to act as a navigation probe in embedding space — not to provide information.

### The Full Query Process

```python
def query(self, question: str) -> Tuple[str, List[str]]:
    # Step 1: Generate hypothetical document (costs 1 LLM call)
    hypothetical_doc = self.generate_hypothetical_document(question)
    
    # Step 2: Embed the hypothetical (not the question!)
    hypothetical_embedding = self.embedder.embed_text(hypothetical_doc)
    
    # Step 3: FAISS search with the hypothetical embedding
    results = self.vector_store.search(hypothetical_embedding, k=self.k)
    context_chunks = [r.document.content for r in results]
    
    # Step 4: Generate answer from REAL retrieved documents
    # The hypothetical is discarded here — never passed to the answering LLM
    answer = self._generate_answer(question, context_chunks)
    
    return answer, context_chunks
```

### The Optional Dual-Embedding Strategy

A more robust variant combines query embedding + hypothetical embedding:

```python
def query_dual(self, question: str, alpha: float = 0.5) -> Tuple[str, List[str]]:
    """
    Combine direct query embedding with hypothetical document embedding.
    
    alpha=0.0 → pure query embedding (no HyDE benefit)
    alpha=0.5 → equal mix (balanced)
    alpha=1.0 → pure hypothetical embedding (maximum HyDE)
    """
    # Standard query embedding
    query_embedding = np.array(self.embedder.embed_text(question))
    
    # Hypothetical document embedding
    hypothetical = self.generate_hypothetical_document(question)
    hyp_embedding = np.array(self.embedder.embed_text(hypothetical))
    
    # Linear interpolation in embedding space
    combined_embedding = (1 - alpha) * query_embedding + alpha * hyp_embedding
    
    # Normalize the combined embedding (required for cosine similarity)
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    
    results = self.vector_store.search(combined_embedding.tolist(), k=self.k)
    context_chunks = [r.document.content for r in results]
    
    answer = self._generate_answer(question, context_chunks)
    return answer, context_chunks
```

The linear interpolation `(1-α) × query + α × hypothetical` creates a point in embedding space that blends both representations. At α=0.5, the combined embedding is equidistant between "query form" and "document form" — capturing the best of both.

---

## When HyDE Is Most Beneficial

### High-Benefit Scenarios

**Short, ambiguous queries:**
Query: "transformer architecture" → ambiguous (electrical? ML model? audio?)  
Hypothetical: "The Transformer architecture, introduced by Vaswani et al. (2017) in 'Attention is All You Need', replaced recurrent networks..." → disambiguates to ML context, embedding lands in ML documents cluster

**Cross-vocabulary queries:**
User query: "how do cars float on water"  
Document vocabulary: "vehicle buoyancy depends on density-to-volume ratio"  
HyDE bridges: hypothetical uses "buoyancy" and "density" → matches document vocabulary

**Simple factual questions with complex source text:**
Query: "Napoleon's height" (short, simple)  
Documents: verbose historical passages  
Hypothetical: "Historical accounts of Napoleon Bonaparte (1769-1821) generally indicate a height of approximately 5 feet 7 inches (170 cm) in modern measurements..."  
The hypothetical's length and vocabulary better matches document text

### Lower-Benefit Scenarios

- **Terminology-matched queries**: If query and document use identical vocabulary, the query embedding already lands near the right documents
- **Very long queries**: Already document-like in form — adding a hypothetical adds noise
- **Highly technical/specialized domains**: LLM may not generate plausible hypotheticals for obscure domains (astrophysics, niche legal areas), and an inaccurate hypothetical embedding misdirects search

---

## Cost Analysis

| Operation | LLM Calls | Tokens | Relative cost |
|-----------|-----------|--------|--------------|
| Standard RAG | 1 (generation) | ~2,000 | Baseline |
| HyDE RAG | 2 (generation + hypothetical) | ~2,500 | 1.25× |

HyDE adds exactly one extra LLM call at query time (generating the hypothetical). The hypothetical is short (~150 words / ~200 tokens output) so the marginal cost is small.

---

## HyDE vs. Related Techniques

| Technique | When query runs | What is embedded for retrieval |
|-----------|----------------|-------------------------------|
| Standard RAG | Directly | Query |
| HyDE | Generates hypothetical first | Hypothetical (document-form) |
| HyPE | Embeds questions at index time | Pre-generated questions in index |
| Document Augmentation | Generates questions at index time | Questions in index |

HyDE and HyPE are complementary:
- HyDE adds a query-expansion step at query time (bridges query→document gap from the query side)
- HyPE bridges the same gap from the document side (by putting question-form embeddings in the index)

Combining HyDE + HyPE creates a "meet in the middle" effect: the index contains question embeddings, and the query probe is itself question-form (hypothetical). Similarity between them is maximized.

---

## When to Use HyDE

HyDE earns its extra LLM call most clearly when queries are short and ambiguous — where the raw query embedding lands in a sparse region of embedding space that doesn't reliably find the right documents. Cross-vocabulary scenarios, where users phrase things colloquially and documents use technical language, are another natural fit. Research and knowledge discovery applications — Q&A over academic papers, technical reports — consistently benefit because the gap between question form and document form is large.

If latency is a hard constraint, HyDE requires one additional generation call that adds roughly 0.5–1s depending on model speed. The technique also assumes the LLM can generate a plausible hypothetical — for highly specialized or obscure domains where the model has thin training coverage, the hypothetical may be inaccurate enough to misdirect retrieval rather than improve it. Queries that are already long and document-like in form gain little from an additional hypothetical, which can introduce noise rather than signal.

---

## Summary

HyDE is a beautifully simple idea: the best way to find a document is to describe what you're looking for in the same form as the documents you're searching. By generating a hypothetical answer at query time and using its embedding as the retrieval probe, HyDE bridges the form-factor asymmetry between queries and documents that limits all embedding-based retrieval.

The implementation is minimal — one extra LLM call at query time, then standard FAISS retrieval with the hypothetical embedding. The benefit is consistent and measurable: better recall across diverse query types, with especially pronounced gains for short, ambiguous, or cross-vocabulary queries. For systems where retrieval quality matters and one extra LLM call is acceptable, HyDE is one of the most bang-for-buck query enhancements available.
