# Self-RAG: A System That Questions Its Own Answers

> **Technique:** Self-Reflective RAG (Self-RAG)  
> **Complexity:** Advanced  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

Most RAG systems are optimistic: retrieve some documents, generate an answer, return it. They trust their retrieval and trust their generation. But what if retrieval is unnecessary for a particular query? What if the retrieved documents don't actually support the generated answer? What if one context produced a better response than another?

**Self-RAG** incorporates structured self-reflection at every step of the pipeline. Inspired by the 2023 paper "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," this technique adds evaluation checkpoints throughout the RAG process — turning a single-pass pipeline into a multi-stage critique loop.

The result is a system that knows when it doesn't know, knows when it retrieved correctly, and knows which of its generated answers is the best one.

---

## The 7-Step Self-RAG Pipeline

```
User Query
    ↓
[Step 1] Retrieval Decision → "Do I need to retrieve for this query?"
    ↓ (if yes)
[Step 2] Document Retrieval → Fetch top-k from vector store
    ↓
[Step 3] Relevance Evaluation → "Is each document relevant or irrelevant?"
    → Filter to relevant documents only
    ↓
[Step 4] Response Generation → Generate one response per relevant context
    ↓
[Step 5] Support Assessment → "Is this response grounded in the context?"
    → Categorize: fully_supported / partially_supported / no_support
    ↓
[Step 6] Utility Evaluation → "How useful is this response?" (score 1-5)
    ↓
[Step 7] Response Selection → Pick best by support level + utility score
```

Each step uses an LLM with structured JSON output for reliable parsing.

---

## Code Deep Dive

### Step 1: Retrieval Decision

```python
def _decide_retrieval(self, query: str) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval decision system. "
                "Determine if the query requires retrieving external documents to answer accurately. "
                'Respond with JSON: {"needs_retrieval": true} or {"needs_retrieval": false}'
            ),
        },
        {"role": "user", "content": f"Query: {query}"},
    ]
    result = self.llm.chat_json(messages)
    return result.get("needs_retrieval", True)
```

For queries like "What is 25 × 4?" or "What year were you trained?", the system correctly answers without retrieval. This avoids unnecessary API calls and latency for simple queries.

### Step 3: Relevance Evaluation

```python
def _evaluate_relevance(self, query: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a relevance evaluator. "
                "Determine if the given context is relevant to answering the query. "
                'Respond with JSON: {"relevance": "relevant"} or {"relevance": "irrelevant"}'
            ),
        },
        {"role": "user", "content": f"Query: {query}\n\nContext: {context}"},
    ]
    result = self.llm.chat_json(messages)
    return result.get("relevance", "irrelevant").strip().lower()
```

Unlike CRAG's 0.0-1.0 float scoring, Self-RAG uses binary relevant/irrelevant classification per document. All irrelevant documents are filtered out before generation — the LLM never sees irrelevant context at all.

### Step 5: Support Assessment

```python
def _assess_support(self, response: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a support assessment system. "
                "Determine how well the response is supported by the given context. "
                'Respond with JSON: {"support": "fully_supported"} or '
                '{"support": "partially_supported"} or {"support": "no_support"}'
            ),
        },
        {"role": "user", "content": f"Context: {context}\n\nResponse: {response}"},
    ]
    result = self.llm.chat_json(messages)
    return result.get("support", "no_support").strip().lower()
```

This step checks *grounding* — whether the response is actually supported by the context provided. A response can sound plausible while mixing LLM knowledge with retrieved context. "fully_supported" means everything in the response can be traced to the context; "no_support" means the answer is effectively hallucinated.

### Step 6: Utility Evaluation

```python
def _evaluate_utility(self, query: str, response: str) -> int:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a utility evaluator. "
                "Rate how useful the response is for answering the query. "
                'Respond with JSON: {"utility": <score>} where score is 1-5. '
                "1=useless, 2=poor, 3=acceptable, 4=good, 5=excellent"
            ),
        },
        {"role": "user", "content": f"Query: {query}\n\nResponse: {response}"},
    ]
    result = self.llm.chat_json(messages)
    score = result.get("utility", 3)
    return max(1, min(5, int(score)))
```

This is where the system gives itself a grade. A response can be fully grounded in context but still score utility=2 if the context was only tangentially related to the query. This catches technically-accurate-but-not-helpful responses.

### Step 7: Best Response Selection

```python
@staticmethod
def _select_best(candidates: List[Dict]) -> Dict:
    support_priority = {
        "fully_supported": 3,
        "partially_supported": 2,
        "no_support": 1,
    }
    return max(
        candidates,
        key=lambda c: (support_priority.get(c["support"], 0), c["utility"]),
    )
```

The selection is a compound key sort: primary sort by support level (fully > partially > no), secondary sort by utility score (5 > 1). This ensures grounding always trumps raw usefulness — we prefer a well-supported response of utility 4 over an unsupported response of utility 5.

---

## Generation Per Context

The self_rag pipeline generates **one response per relevant context** — not one response from all contexts combined:

```python
candidates = []
for i, context in enumerate(relevant_contexts):
    response = self._generate_response(query, context)
    support = self._assess_support(response, context)
    utility = self._evaluate_utility(query, response)
    
    candidates.append({
        "response": response,
        "support": support,
        "utility": utility,
        "context": context,
    })

best = self._select_best(candidates)
```

This is key to Self-RAG's design philosophy: by generating from individual contexts rather than concatenating everything, you can evaluate how well each response is grounded in its specific source. A response generated from Context A that's only partially supported is distinguishable from a response generated from Context B that's fully supported. The selection picks the best single context-response pair.

---

## The SelfRAGResponse Dataclass

```python
@dataclass
class SelfRAGResponse:
    answer: str
    support: str           # "fully_supported" | "partially_supported" | "no_support"
    utility: int           # 1-5
    context_used: List[str]
    retrieval_needed: bool
    total_docs_retrieved: int
    relevant_docs_count: int
    all_candidates: List[Dict]  # full evaluation data for each generated response
```

Full transparency: the caller can inspect not just the final answer but every evaluation score for every candidate. This is invaluable for debugging and system monitoring.

---

## LLM Call Count

Self-RAG is expensive. For a query with k=3 retrieved docs (all relevant):

| Step | LLM calls |
|------|-----------|
| Retrieval decision | 1 |
| Relevance evaluation | k = 3 |
| Response generation | k = 3 |
| Support assessment | k = 3 |
| Utility evaluation | k = 3 |
| **Total** | **1 + 4k = 13 calls** |

Compare to standard RAG: 1 call. Self-RAG costs 13× as many API calls for k=3. This is the price of thorough self-reflection. For applications where answer quality is paramount and cost is secondary, this is very much worth it.

---

## Self-RAG vs. CRAG

Both systems add evaluation on top of standard RAG, but they have distinct philosophies:

| Aspect | Self-RAG | CRAG |
|--------|----------|------|
| Focus | Evaluating generated *responses* | Evaluating retrieved *documents* |
| Web fallback | No | Yes |
| Per-doc response generation | Yes (one per doc) | No |
| Self-critique mechanism | Support assessment + utility scoring | Relevance thresholds only |
| Output selection | Multi-candidate ranking | Best single action |
| LLM call overhead | 4k + 1 per query | k + 1-4 per query |

Self-RAG and CRAG are complementary: CRAG ensures you're working with good raw material (relevant docs); Self-RAG ensures you're generating well-supported, useful responses from whatever context you have.

---

## When to Use Self-RAG

**Best for:**
- High-stakes domains where hallucinations are unacceptable (medicine, law, compliance)
- Applications where answer grounding must be explicitly verified and audited
- Systems requiring detailed provenance — "why did you say that?"
- Offline evaluation frameworks benchmarking answer quality

**Skip when:**
- Real-time applications requiring sub-second responses
- Cost per query is tightly constrained
- Simple Q&A over a well-curated, query-appropriate corpus

---

## Summary

Self-RAG is the most introspective technique in the RAG landscape. By evaluating whether retrieval is needed, whether retrieved documents are relevant, whether generated responses are supported, and how useful those responses are — all before returning a single answer — Self-RAG transforms the LLM from a confident generator into a careful, self-aware reasoner.

The computational cost is real and significant. But for applications where an incorrect confident answer is worse than a slow careful one, Self-RAG provides a level of output quality assurance that no other single technique matches.
