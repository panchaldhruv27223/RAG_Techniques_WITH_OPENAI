# CRAG: Corrective RAG with Intelligent Fallback

---

## Introduction

Standard RAG has a fundamental trust problem: it trusts that what it retrieved is relevant. If your vector database returns chunks that happen to score high cosine similarity but don't actually address the query — due to embedding noise, topical adjacency, or corpus coverage gaps — the LLM will receive bad context and produce a bad answer. No alarm is raised. No correction is made. The system confidently answers from wrong evidence.

**Corrective RAG (CRAG)**, introduced by Shi et al. (2024), addresses this by adding a self-evaluation step between retrieval and generation. Before passing retrieved chunks to the LLM for answer generation, a "corrector" evaluates each chunk's relevance and routes the system through one of three decision paths:

1. **CORRECT**: Retrieved content is highly relevant → proceed directly to generation
2. **INCORRECT**: Retrieved content is irrelevant → discard it, search the web, generate from web results
3. **AMBIGUOUS**: Partial relevance → refine the retrieved content AND supplement with web search

This corrective loop transforms RAG from a single-pass pipeline into a quality-aware system that can detect and recover from retrieval failures — the most common cause of RAG hallucinations.

---

## The Three-Action Decision Framework

```
User Query
     │
     ▼
[Standard Vector Retrieval]
     │
     ▼
[Relevance Evaluation — per retrieved chunk]
     │
     ├─── Score > 0.7 (HIGH) ───► CORRECT
     │                                │
     │                                ▼
     │                        Use retrieved docs directly
     │                        → Generate answer
     │
     ├─── Score 0.3-0.7 (MEDIUM) ─► AMBIGUOUS
     │                                │
     │                                ▼
     │                        Strip irrelevant sentences
     │                        + Web search for supplemental info
     │                        → Generate from refined + web docs
     │
     └─── Score < 0.3 (LOW) ───► INCORRECT
                                      │
                                      ▼
                                Discard retrieved docs entirely
                                Rewrite query for web search
                                → Generate from web docs only
```

The evaluation score is a per-chunk relevance assessment made by the LLM, not the embedding similarity. Embedding similarity is a retrieval signal; CRAG adds a second, independent evaluation using the LLM's semantic understanding.

---

## The Relevance Evaluator

```python
class RelevanceEvaluator:
    """
    Evaluates how relevant a retrieved document chunk is to a given query.
    Uses the LLM as an 'intelligent judge' — can understand nuance, negation,
    paraphrase, and domain relevance that cosine similarity cannot.
    """
    
    # Thresholds for the three action categories
    CORRECT_THRESHOLD = 0.7    # High: directly useful
    INCORRECT_THRESHOLD = 0.3  # Low: discard and use web
    # AMBIGUOUS: 0.3 < score ≤ 0.7 (between thresholds)
    
    def evaluate(self, query: str, document: str) -> RelevanceResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance evaluation expert. Given a query and a "
                    "document, assess how relevant the document is to answering "
                    "the query on a scale of 0.0 to 1.0.\n\n"
                    "Scoring criteria:\n"
                    "  0.9-1.0: Document directly answers the query completely\n"
                    "  0.7-0.9: Document substantially addresses the query\n"
                    "  0.5-0.7: Document is partially relevant, addresses some aspects\n"
                    "  0.3-0.5: Document is tangentially related but doesn't address the query well\n"
                    "  0.1-0.3: Document is on a related topic but not relevant to the query\n"
                    "  0.0-0.1: Document is completely irrelevant\n\n"
                    'Return JSON: {"score": <float>, "reasoning": "<brief explanation>"}'
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument:\n{document[:2000]}"
            }
        ]
        
        result = self.llm.chat_json(messages)
        score = float(result.get("score", 0.5))
        reasoning = result.get("reasoning", "")
        
        # Classify action based on thresholds
        if score >= self.CORRECT_THRESHOLD:
            action = "CORRECT"
        elif score >= self.INCORRECT_THRESHOLD:
            action = "AMBIGUOUS"
        else:
            action = "INCORRECT"
        
        return RelevanceResult(score=score, action=action, reasoning=reasoning)
```

**Why use an LLM evaluator instead of cosine similarity?**

Consider this pair:
- Query: "What is the GDP of France?"
- Retrieved chunk: "France has a rich cultural heritage, including world-famous cuisine, art, and architecture. Paris is the capital city and houses the Louvre museum..."

Cosine similarity might be 0.71 (both are about France — semantically related). An LLM evaluator correctly scores this 0.1: "Document discusses French culture but does not mention GDP or economic statistics."

Without CRAG, the LLM would receive this irrelevant chunk and potentially hallucinate a GDP figure. With CRAG, it's flagged as INCORRECT and the system falls back to web search.

---

## The Web Search Fallback

When retrieval is deemed INCORRECT or AMBIGUOUS, CRAG uses web search to cover the gap:

```python
class WebSearcher:
    def search(self, query: str, max_results: int = 3) -> List[str]:
        """
        Search the web for information not available in the vector store.
        Uses DuckDuckGo (no API key required) as the default provider.
        """
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                # Combine title + body for richer context
                snippet = f"Source: {result.get('title', 'Unknown')}\n{result.get('body', '')}"
                results.append(snippet)
        
        return results if results else ["No web results found"]
    
    def rewrite_query_for_web(self, original_query: str) -> str:
        """
        Optimize the query for web search engines.
        Web search prefers keyword-heavy, specific queries.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the query for a web search engine. "
                    "Make it more specific, keyword-focused, and remove "
                    "conversational elements. Keep it under 10 words if possible."
                )
            },
            {
                "role": "user",
                "content": f"Original: {original_query}\n\nWeb-optimized version:"
            }
        ]
        return self.llm.chat(messages).strip()
```

**Why rewrite the query for web search?**

Natural language queries work well for embedding retrieval but poorly for web search engines. "What is the biological mechanism by which aspirin reduces fever?" → web-optimized: "aspirin antipyretic mechanism COX inhibition fever". The reformulation matches how web documents are indexed — keyword-dense, not conversational.

---

## The Complete CRAG Pipeline

```python
def query(self, question: str) -> Tuple[str, CRAGResult]:
    """Execute the corrective RAG pipeline."""
    
    # Step 1: Initial vector retrieval
    results = self.vector_store.search(query_embedding, k=self.k)
    retrieval_action = "CORRECT"  # optimistic default
    all_context = []
    
    # Step 2: Evaluate each retrieved chunk
    per_chunk_actions = []
    for result in results:
        chunk_text = result.document.content
        evaluation = self.relevance_evaluator.evaluate(question, chunk_text)
        
        per_chunk_actions.append({
            "chunk": chunk_text[:100],
            "action": evaluation.action,
            "score": evaluation.score,
            "reasoning": evaluation.reasoning
        })
        
        # The overall action is determined by the WORST chunk action
        # If any chunk is INCORRECT, we must fall back for that chunk
        if evaluation.action == "AMBIGUOUS" and retrieval_action == "CORRECT":
            retrieval_action = "AMBIGUOUS"
        elif evaluation.action == "INCORRECT":
            retrieval_action = "INCORRECT"
    
    print(f"Overall retrieval action: {retrieval_action}")
    
    # Step 3: Route based on evaluation
    if retrieval_action == "CORRECT":
        # All chunks are highly relevant — use them directly
        all_context = [r.document.content for r in results]
    
    elif retrieval_action == "AMBIGUOUS":
        # Some chunks partially relevant — refine them + add web search
        for r in results:
            chunk_text = r.document.content
            # Distill: keep only sentences relevant to the query
            refined = self._refine_chunk(question, chunk_text)
            if refined:
                all_context.append(refined)
        
        # Supplement with web search
        web_query = self.web_searcher.rewrite_query_for_web(question)
        web_results = self.web_searcher.search(web_query)
        all_context.extend(web_results)
    
    elif retrieval_action == "INCORRECT":
        # All chunks irrelevant — discard completely, use web only
        web_query = self.web_searcher.rewrite_query_for_web(question)
        web_results = self.web_searcher.search(web_query)
        all_context = web_results
    
    # Step 4: Generate answer from selected context
    answer = self._generate_answer(question, all_context)
    
    return answer, CRAGResult(
        action=retrieval_action,
        per_chunk_evaluations=per_chunk_actions,
        web_searched=(retrieval_action != "CORRECT")
    )
```

---

## Chunk Refinement (AMBIGUOUS Path)

When an action is AMBIGUOUS, the relevant sentences are extracted:

```python
def _refine_chunk(self, query: str, chunk_text: str) -> str:
    """
    For AMBIGUOUS chunks: extract only the sentences relevant to the query.
    Discards off-topic sentences while preserving useful content.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Extract and return only the sentences from the context that are "
                "relevant to answering the query. If no sentences are relevant, "
                "return an empty string. Do not add, summarize, or modify content."
            )
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nContext:\n{chunk_text}\n\nExtract relevant sentences:"
        }
    ]
    refined = self.llm.chat(messages).strip()
    return refined if refined and "no sentences" not in refined.lower() else ""
```

---

## LLM Call Budget

Understanding the cost of CRAG is essential for deployment decisions:

| Action Path | LLM Calls |
|------------|-----------|
| CORRECT (k=3 chunks, no web) | k evaluations + 1 generation = 4 |
| AMBIGUOUS (k=3 chunks, web) | k evaluations + k refinements + 1 generation = 7 |
| INCORRECT (k=3, web only) | k evaluations + 1 web rewrite + 1 generation = 5 |

**Baseline standard RAG**: 1 LLM call (generation only)  
**CRAG overhead**: 3-6 additional LLM calls  
**CRAG latency**: 5-15 additional seconds (depending on path)

For applications where accuracy is paramount (medical, legal, research) this overhead is entirely justified. For high-throughput, latency-sensitive consumer apps, standard RAG or cheaper evaluation methods should be used.

---

## When CRAG Prevents Hallucination

### Scenario: Knowledge Cutoff Gap

User query: "What is OpenAI's latest released model as of 2025?"

**Vector store** (indexed in 2024): Returns chunks about GPT-4o and o1 models. These are the latest models *in the vector store* — but not as of 2025.

**Without CRAG**: LLM receives outdated context, asserts "the latest model is GPT-4o" — incorrect answer.

**With CRAG**: 
- Evaluator scores retrieval: "Document discusses models from 2024, but the query asks for 2025 — potentially outdated" → AMBIGUOUS
- Web search executes → retrieves current 2025 model information
- LLM correctly answers from web content

### Scenario: Corpus Coverage Gap

User query: "How did Hurricane Milton affect Tampa Bay in 2024?"

**Vector store**: No documents about Hurricane Milton (not in the corpus).

**Without CRAG**: FAISS returns the most similar vectors (other hurricane-related content). LLM may confuse details from other storms.

**With CRAG**:
- Evaluator scores: "Document discusses hurricane preparedness generally, no mention of Milton or Tampa Bay in 2024" → INCORRECT
- Web search retrieves Hurricane Milton coverage → accurate answer

---

## Configuring Action Thresholds

```python
# Conservative (higher bar for CORRECT, more web fallback)
evaluator = RelevanceEvaluator(
    correct_threshold=0.8,   # Must be very relevant to avoid web
    incorrect_threshold=0.4  # Tolerates more ambiguity before fallback
)

# Aggressive (lower bar for CORRECT, less web fallback)
evaluator = RelevanceEvaluator(
    correct_threshold=0.6,   # Proceed with moderate relevance
    incorrect_threshold=0.2  # Only fully irrelevant triggers web
)
```

**When to be conservative**: Knowledge-critical applications (medical Q&A, financial research) — prefer web fallback over risk of wrong context.

**When to be aggressive**: Latency-sensitive applications, stable corpora, high cost of web calls — prefer fewer web searches when retrieval is "good enough."

---

## When to Use CRAG

CRAG is the right choice when your vector store cannot be trusted to cover all the queries users will ask. If the corpus has a knowledge cutoff, is narrowly scoped while users ask broad questions, or changes slowly while user needs evolve faster, CRAG's web fallback provides a meaningful safety net against confident-but-wrong answers.

For high-stakes applications — medical Q&A, financial research, legal lookups — the cost of a wrong answer far outweighs the overhead of additional LLM evaluation calls and web search latency. CRAG's three-action decision framework makes that tradeoff explicit and configurable.

Where CRAG is inappropriate: air-gapped deployments with no web access, systems where every answer must come from the existing corpus for compliance or policy reasons, and latency-sensitive applications where the AMBIGUOUS or INCORRECT paths (each adding several LLM calls and web round trips) would breach SLAs.

---

## Summary

CRAG transforms RAG from a pipeline that blindly trusts retrieval into one that critically evaluates it. The three-action decision framework — CORRECT, AMBIGUOUS, INCORRECT — routes each query through the minimum amount of additional work needed to ensure high-quality context for generation.

The key insight is that embedding similarity is a good *first filter*, but an insufficient *quality guarantee*. By adding an LLM-based relevance evaluator between retrieval and generation, CRAG catches the cases that embedding similarity misses and corrects them — either by refining partial context or by falling back to web search for completely missed queries.

For applications where the cost of a wrong answer exceeds the cost of additional LLM calls, CRAG is a principled, measurable improvement in reliability.
