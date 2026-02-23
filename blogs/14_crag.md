# CRAG: Corrective RAG — When Retrieved Documents Aren't Good Enough

> **Technique:** Corrective RAG (CRAG)  
> **Complexity:** Advanced  
> **Key Libraries:** `openai`, `faiss-cpu`, `duckduckgo-search`

---

## Introduction

Standard RAG has a silent assumption baked in: *the retrieved documents are relevant*. It retrieves the top-k chunks and feeds them to the LLM, trusting that vector similarity translates to answer quality. But what if the query asks about something your document corpus doesn't cover? What if the user asks about a recent event that post-dates your knowledge base?

In these cases, standard RAG doesn't fail loudly — it fails quietly, generating confident-sounding answers from irrelevant context. This is arguably the worst failure mode in production RAG systems.

**Corrective RAG (CRAG)** introduces an explicit quality check on retrieved documents and *actively corrects* when local context is insufficient by falling back to web search. It transforms retrieval from a passive step into an informed decision.

---

## The Three-Action Decision Framework

CRAG evaluates retrieved documents using an LLM-based relevance scorer and commits to one of three actions based on the highest relevance score among retrieved documents:

```
Retrieved documents: [Doc1, Doc2, Doc3]
                        ↓
          [LLM scores each: 0.0 – 1.0]
                        ↓
          max_score = max(scores)
                        ↓
        ┌───────────────────────────────────┐
        │ max_score > 0.7                   │
        │ → CORRECT: Use best local doc     │
        ├───────────────────────────────────┤
        │ max_score < 0.3                   │
        │ → INCORRECT: Discard, use web     │
        ├───────────────────────────────────┤
        │ 0.3 ≤ max_score ≤ 0.7           │
        │ → AMBIGUOUS: Combine both         │
        └───────────────────────────────────┘
```

This is the key innovation: retrieval isn't just a passive lookup — it's an evaluated decision with three distinct outcomes.

---

## Code Deep Dive

### Relevance Evaluation

```python
def _evaluate_relevance(self, query: str, document: str) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a relevance evaluator. "
                "Score how relevant the document is to answering the query. "
                'Respond with JSON: {"relevance_score": <float between 0.0 and 1.0>}'
            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nDocument: {document}",
        },
    ]
    result = self.llm.chat_json(messages)
    score = result.get("relevance_score", 0.5)
    return max(0.0, min(1.0, float(score)))
```

A score of 0.0 means "completely irrelevant"; 1.0 means "perfectly answers the query." The LLM evaluates each document independently and returns a structured JSON score.

### The Full CRAG Pipeline

```python
def crag(self, query: str) -> CRAGResponse:
    # Step 1: Retrieve
    results = self._retrieve(query)
    docs = [(r.document.content, r.score) for r in results]
    
    # Step 2: Evaluate relevance
    relevance_scores = []
    for content, sim_score in docs:
        rel_score = self._evaluate_relevance(query, content)
        relevance_scores.append(rel_score)
    
    max_score = max(relevance_scores)
    best_doc = docs[relevance_scores.index(max_score)][0]
    
    # Step 3: Decide action
    if max_score > self.high_threshold:       # 0.7
        action = "correct"
        final_knowledge = best_doc
        sources = [("Retrieved document", "")]
        
    elif max_score < self.low_threshold:      # 0.3
        action = "incorrect"
        final_knowledge, sources = self._web_search(query)
        
    else:                                      # ambiguous
        action = "ambiguous"
        refined_local = self._refine_knowledge(best_doc)
        web_knowledge, web_sources = self._web_search(query)
        final_knowledge = f"Local Knowledge:\n{refined_local}\n\nWeb Knowledge:\n{web_knowledge}"
        sources = [("Retrieved document", "")] + web_sources
    
    # Step 4: Generate response
    answer = self._generate_response(query, final_knowledge, sources)
    
    return CRAGResponse(answer=answer, action=action, sources=sources,
                       relevance_scores=relevance_scores, max_score=max_score,
                       context_used=[final_knowledge])
```

### Web Search Fallback

When action is INCORRECT or AMBIGUOUS, CRAG performs a web search to supplement or replace local knowledge:

```python
def _web_search(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
    # Rewrite query for better web results
    rewritten = self._rewrite_query_for_web(query)
    
    # Search DuckDuckGo (no API key required)
    web_results = web_search_duckduckgo(rewritten, max_results=3)
    
    sources = [(r["title"], r["link"]) for r in web_results]
    raw_text = "\n\n".join(
        f"Source: {r['title']}\n{r['snippet']}" for r in web_results
    )
    
    # Refine: extract key points from web snippets
    refined = self._refine_knowledge(raw_text)
    return refined, sources
```

#### Query Rewriting for Web

```python
def _rewrite_query_for_web(self, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer. "
                "Rewrite the query to get better web search results. "
                'Respond with JSON: {"rewritten_query": "<optimized query>"}'
            ),
        },
        {"role": "user", "content": f"Original query: {query}"},
    ]
    result = self.llm.chat_json(messages)
    return result.get("rewritten_query", query)
```

User queries are often conversational ("what does the report say about X?"). Web search engines work better with keyword-style queries. The rewriter transforms the former into the latter.

### Knowledge Refinement

Raw web results contain noise — navigation text, ads, irrelevant snippets. CRAG filters them:

```python
def _refine_knowledge(self, raw_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract the key information from the following text as concise bullet points. "
                "Focus on facts that would help answer a question. "
                'Respond with JSON: {"key_points": ["point1", "point2", ...]}'
            ),
        },
        {"role": "user", "content": raw_text},
    ]
    result = self.llm.chat_json(messages)
    points = result.get("key_points", [])
    return "\n".join(f"• {p}" for p in points)
```

---

## The AMBIGUOUS Case: Best of Both Worlds

The most interesting action is AMBIGUOUS. Here's what happens:

1. Local document is *partially* relevant (score 0.3–0.7)
2. CRAG refines the local document — extracting its most relevant points
3. CRAG also performs a web search
4. Both refined local knowledge AND web knowledge are combined
5. The LLM generates an answer with access to both sources

This is the most powerful outcome: using domain-specific local knowledge (from your carefully curated corpus) augmented with fresh web information. Neither source alone would have been optimal.

---

## The CRAGResponse Dataclass

```python
@dataclass
class CRAGResponse:
    answer: str
    action: str           # "correct" | "incorrect" | "ambiguous"
    sources: List[Tuple[str, str]]
    relevance_scores: List[float]
    max_score: float
    context_used: List[str]
```

Full metadata is returned for monitoring, logging, and debugging. Production systems can track action distributions to measure how often local retrieval is adequate vs. requiring web fallback.

---

## Cost Analysis

CRAG adds LLM API calls compared to standard RAG:

| Step | API calls |
|------|-----------|
| Relevance evaluation | k calls (1 per retrieved doc) |
| Query rewriting (if web) | 1 call |
| Knowledge refinement (if web) | 1 call |
| Answer generation | 1 call |
| **Total (CORRECT action)** | k + 1 calls |
| **Total (INCORRECT action)** | k + 3 calls |
| **Total (AMBIGUOUS action)** | k + 4+ calls |

With k=3 retrieved docs, CORRECT action costs 4 LLM calls vs. 1 for standard RAG. This is the price of validated, reliable retrieval.

---

## When to Use CRAG

**Best for:**
- Queries that may reach beyond your corpus (current events, niche topics)
- High-stakes applications where "I don't know" is better than a confident wrong answer
- Production systems serving diverse user queries over constrained knowledge bases
- Medical, legal, or financial domains where retrieval failures have real consequences

**Skip when:**
- Corpus is comprehensive for the query domain and web fallback isn't allowed
- Cost per query is tightly constrained (CRAG is significantly more expensive)
- Low latency is critical (evaluation + web search adds several seconds)

---

## Summary

CRAG transforms retrieval from a blind lookup into an intelligent, evaluated decision. By scoring retrieved documents and choosing between local context, web search, or both — based on explicit relevance thresholds — CRAG dramatically reduces the risk of confidently answering from irrelevant context.

The three-action framework (CORRECT / INCORRECT / AMBIGUOUS) is elegantly practical: it handles the full spectrum from "our corpus is perfect for this query" to "we have no relevant information locally" to "we have partial information worth combining with external data." Production RAG systems operating over bounded knowledge bases should seriously consider CRAG as a reliability layer.
