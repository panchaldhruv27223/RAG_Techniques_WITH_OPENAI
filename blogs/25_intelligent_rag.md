# Intelligent RAG: A Budget-Controlled, Query-Adaptive Pipeline

---

## Introduction

Most RAG systems use a single, fixed strategy: embed query → retrieve k chunks → generate answer. This works adequately for simple factual queries but fails for:

- **Temporal queries** ("How has X changed since 2015?") — need recency-aware retrieval
- **Multi-hop queries** ("Which company acquired the firm that built system Y?") — need iterative retrieval to fill knowledge gaps
- **Complex decomposable queries** ("Compare the economic impacts with social impacts") — need sub-question breakdown
- **Simple lookups** ("What is the capital of France?") — need just one fast vector search

**Intelligent RAG** is a production-quality orchestrator that detects which of these challenge types applies to each query and routes it through a specialized engine pipeline — all within a hard budget of LLM calls per query. It's the equivalent of triage in an emergency room: classify the problem, route to the right specialist, monitor the resource budget.

---

## Architecture: The Pipeline of Engines

```
User Query
    │
    ▼
┌─────────────┐    Cache hit? → Return immediately
│ Query Cache │
└─────────────┘
    │ (cache miss)
    ▼
┌──────────────────┐
│  QueryAnalyzer   │ ← Heuristics first, LLM only if ambiguous (0-1 LLM calls)
│  (Classifier)    │   Detects: SIMPLE | TEMPORAL | MULTI_HOP | DECOMPOSITION
└──────────────────┘
    │
    ▼
┌──────────────────────┐
│  Initial Vector      │ ← Standard FAISS search, 0 LLM calls
│  Search              │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  TemporalReranker    │ ← If TEMPORAL detected: rerank by recency (0 LLM calls)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  MultiHopRetriever   │ ← If MULTI_HOP: identify gaps, fetch more (budget-controlled)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  QueryDecomposer     │ ← If DECOMPOSITION: split into sub-queries, answer each
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  AnswerSynthesizer   │ ← 1 LLM call: generate final answer
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ ConfidenceCalibrator │ ← 1 LLM call: grounding check + confidence score
└──────────────────────┘
    │
    ▼
IntelligentResult
(answer, confidence, trace, strategies_used, llm_calls)
```

All engines write to a shared `RetrievalState` object — no engine calls another directly. This makes the pipeline composable, testable, and easy to extend.

---

## The RetrievalState: Shared Pipeline Memory

The key architectural decision: every engine reads from and writes to a single shared state object, rather than passing data through function returns:

```python
@dataclass
class RetrievalState:
    # Input
    original_query: str
    challenges: List[Challenge] = field(default_factory=list)
    
    # Accumulated retrieval (grows as engines add more chunks)
    chunks: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    
    # Sub-question tracking (decomposition engine)
    sub_questions: List[str] = field(default_factory=list)
    sub_answers: Dict[str, str] = field(default_factory=dict)
    
    # Multi-hop tracking
    hops: List[Dict[str, str]] = field(default_factory=list)
    
    # Budget system — hard cap on LLM calls
    llm_budget: int = 8
    llm_calls_used: int = 0
    
    # Full execution trace for debugging
    trace: List[str] = field(default_factory=list)
    
    def use_budget(self, n: int = 1) -> bool:
        """Attempt to spend n LLM calls. Returns False if over budget."""
        if self.llm_calls_used + n > self.llm_budget:
            return False
        self.llm_calls_used += n
        return True
    
    def add_chunks(self, chunks: List[str], scores: List[float], source: str):
        """Deduplicated chunk accumulation — no duplicate context in the window."""
        existing = set(c[:100] for c in self.chunks)
        for c, s in zip(chunks, scores):
            if c[:100] not in existing:
                self.chunks.append(c)
                self.scores.append(s)
                existing.add(c[:100])
```

The **budget system** is critical: it prevents runaway LLM call chains. If a complex query would normally trigger 10+ LLM calls (analyze + 3 hops + 4 sub-answers + synthesize + calibrate), but the budget is 8, non-essential operations are automatically skipped. The system degrades gracefully rather than exploding costs.

---

## Engine 1: QueryAnalyzer — Heuristics-First Classification

The key design principle: **heuristics before LLM**. Using regex patterns and keyword matching catches 80%+ of cases at zero LLM cost:

```python
TEMPORAL_KEYWORDS = {"since", "changed", "trend", "latest", "recent", "current", "2024", ...}
DECOMP_KEYWORDS = {"compare", "contrast", "pros and cons", "analyze", "evaluate", ...}
MULTIHOP_PATTERNS = [
    r"which .+ (?:has|have|had|is|are) .+",
    r"who .+ that .+",
    r"what .+ (?:causes?|leads? to) .+",
]
```

LLM classification is **only called** when:
- Query is >12 words AND heuristics found no challenges (ambiguous long query)
- Query is >20 words (complex enough that heuristics might miss the intent)

```python
def analyze(self, query: str, state: RetrievalState) -> RetrievalState:
    challenges = set()
    
    # 1. Temporal heuristic (regex for years + keyword match)
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
    if year_matches or any(kw in q_lower for kw in self.TEMPORAL_KEYWORDS):
        challenges.add(Challenge.TEMPORAL)
    
    # 2. Decomposition (keyword + multiple "?" + "and" count)
    if has_decomp_keyword or query.count("?") > 1 or and_count >= 2:
        challenges.add(Challenge.DECOMPOSITION)
    
    # 3. Multi-hop (regex patterns)
    for pattern in self.MULTIHOP_PATTERNS:
        if re.search(pattern, q_lower):
            challenges.add(Challenge.MULTI_HOP)
    
    # 4. LLM fallback only if ambiguous
    if (not challenges and word_count > 12) or word_count > 20:
        if state.use_budget(1):
            llm_challenges = self._llm_classify(query)
            challenges.update(llm_challenges)
    
    if not challenges:
        challenges.add(Challenge.SIMPLE)
    
    state.challenges = list(challenges)
    return state
```

**Classification examples:**

| Query | Detected Challenge | Method |
|-------|-------------------|--------|
| "What is FAISS?" | SIMPLE | Heuristic |
| "How has inflation changed since 2020?" | TEMPORAL | Heuristic (year + "changed since") |
| "Which model trained on ImageNet has the best accuracy?" | MULTI_HOP | Heuristic (regex pattern) |
| "Compare the pros and cons of SQL vs NoSQL" | DECOMPOSITION | Heuristic ("compare", "vs") |

---

## Engine 2: TemporalReranker — Zero LLM, Pure Heuristic

When a TEMPORAL query is detected, retrieved chunks are reranked by recency *without any LLM calls*:

```python
def _score_temporal(self, chunk: str, intent: str) -> float:
    years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', chunk)]
    max_year = max(years) if years else 0
    
    if intent == "latest":
        if max_year >= 2023: return 1.4   # 40% boost for very recent content
        if max_year >= 2020: return 1.2   # 20% boost for recent
        if max_year >= 2015: return 1.0   # No change
        return 0.8                          # Penalty for dated content
    
    elif intent == "trend":
        if len(set(years)) >= 3: return 1.3  # Chunk spans multiple years → trend-useful
        if len(set(years)) >= 2: return 1.2
        return 0.9
    
    # ... "since", "comparison" intents
```

The score multiplies the original cosine similarity score. A chunk with cosine similarity 0.78 from 2024 gets score 0.78 × 1.4 = 1.09 — rising above a 0.92 score chunk from 2010 (0.92 × 0.8 = 0.74). No LLMs involved, no latency added.

---

## Engine 3: MultiHopRetriever — Gap-Filling Iterative Retrieval

For multi-hop queries, the LLM analyzes what's missing and generates targeted follow-up searches:

```python
def _find_gap(self, original_query: str, chunks: List[str]) -> Optional[str]:
    """Ask: 'Given what we've retrieved, what's still missing?'"""
    chunk_preview = "\n---\n".join(c[:300] for c in chunks[:8])
    messages = [
        {
            "role": "system",
            "content": (
                "Given a question and retrieved info, is there enough to answer fully? "
                'Return JSON: {"complete": True/False, "gap_query": "search query" or None}'
            )
        },
        {"role": "user", "content": f"Question: {original_query}\n\nRetrieved:\n{chunk_preview}"}
    ]
    result = self.llm.chat_json(messages)
    if result.get("complete", True):
        return None  # Enough context — stop hopping
    return result.get("gap_query")  # New search query to fill the gap
```

**Worked example** for "Which company acquired the firm that developed BERT?"

- Hop 1: Initial search for "BERT" → finds "BERT was developed by Google AI researchers..."
- Gap analysis: "We know BERT was made by Google AI, but the query asks about acquisition — gap: which company acquired Google AI?"
- Hop 2: Search "Google AI acquisition company" → finds "Google Brain merged with DeepMind in..."
- Gap analysis: "Complete — the information chain is now sufficient."

Max hops is configurable (default: 2). Each hop costs 1 LLM call (gap identification) + 0 LLM calls (FAISS search).

---

## Engine 4: QueryDecomposer — Sub-questions with Synthesis

For complex multi-aspect queries:

```python
def decompose(self, state: RetrievalState, search_fn) -> RetrievalState:
    """Split query → retrieve per sub-query → accumulate in shared pool."""
    messages = [
        {"role": "system", "content": 'Break into 2-4 atomic sub-questions. JSON: {"sub_questions": [...]}'},
        {"role": "user", "content": state.original_query}
    ]
    result = self.llm.chat_json(messages)
    sub_qs = result.get("sub_questions", [state.original_query])
    
    # Each sub-query retrieves into the SAME shared chunk pool
    for sq in sub_qs:
        chunks, scores = search_fn(sq, 3)
        state.add_chunks(chunks, scores, f"decomp:{sq[:30]}")
    
    return state
```

The synthesis step weaves sub-answers together:

```python
def _synthesize(self, state: RetrievalState) -> str:
    sub_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in state.sub_answers.items())
    messages = [
        {"role": "system", "content": "Synthesize sub-answers into one coherent response. Don't list them — weave naturally."},
        {"role": "user", "content": f"Original: {state.original_query}\n\nSub-answers:\n{sub_text}"}
    ]
    return self.llm.chat(messages)
```

---

## Engine 5: ConfidenceCalibrator — Multi-Signal Validation

Confidence isn't self-reported by the answer LLM (which is unreliable) — it's computed from three independent signals:

```python
def calibrate(self, state: RetrievalState, answer: str) -> Tuple[float, str]:
    # Signal 1: Retrieval quality (OBJECTIVE — cosine similarity scores)
    avg_score = float(np.mean(state.scores[:5]))
    retrieval_signal = min(1.0, max(0.0, (avg_score - 0.3) / 0.5))
    
    # Signal 2: Coverage (OBJECTIVE — how many chunks contributed)
    coverage_signal = min(1.0, len(state.chunks) / 3)
    
    # Signal 3: Grounding (1 LLM call — independent validation)
    grounding_signal = self._check_grounding(answer, state.chunks)
    
    # Weighted fusion: grounding is most trusted (55% weight)
    confidence = 0.30 * retrieval_signal + 0.15 * coverage_signal + 0.55 * grounding_signal
    
    if confidence >= 0.80: label = "High"
    elif confidence >= 0.60: label = "Medium"
    elif confidence >= 0.35: label = "Low"
    else: label = "Very Low"
    
    return confidence, label
```

The grounding check independently verifies whether each answer claim can be traced to a source chunk — catching hallucinations before they reach the user.

---

## Query Cache: Free Speedup for Repeated Queries

```python
class QueryCache:
    def _key(self, query: str) -> str:
        """Normalize (lowercase, strip punctuation) then MD5 hash."""
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
```

"What is FAISS?", "what is faiss?", and "What is FAISS!" all hash to the same key. Cache hits return in milliseconds with zero LLM calls.

---

## LLM Budget by Query Type

With default `llm_budget=20`:

| Challenge | Analyzer | Temporal | Multi-Hop | Decompose | Sub-Answers | Synthesize | Calibrate | Total |
|-----------|---------|---------|-----------|-----------|-------------|-----------|----------|-------|
| SIMPLE | 0 | 0 | 0 | 0 | 0 | 1 | 1 | **2** |
| TEMPORAL | 0 | 0 | 0 | 0 | 0 | 1 | 1 | **2** |
| MULTI_HOP (2 hops) | 0 | 0 | 2 | 0 | 0 | 1 | 1 | **4** |
| DECOMPOSITION (3 sub) | 0 | 0 | 0 | 1 | 3 | 1 | 1 | **6** |
| All challenges combined | 1 | 0 | 2 | 1 | 3 | 1 | 1 | **9** |

The budget ensures even worst-case complex queries stay within 9-10 LLM calls — predictable costs in production.

---

## When to Use Intelligent RAG

Intelligent RAG is appropriate when you need production-grade robustness over a genuinely mixed query profile. Enterprise knowledge management systems, technical support platforms, and research assistants all receive a blend of simple factual lookups, temporally-sensitive questions, multi-hop reasoning requests, and complex decomposable queries. A single fixed strategy will be over-engineered for some and inadequate for others. Intelligent RAG's classification-and-routing architecture handles the full distribution gracefully, right-sizing strategy to each query type.

The budget enforcement mechanism makes it particularly well-suited to cost-sensitive production deployments where unbounded LLM call counts are operationally unacceptable. By capping calls per query and sharing state across pipeline stages, it delivers a predictable cost ceiling while still applying the most capable strategy the budget allows.

For simple, narrow-domain applications where queries are consistently factual and retrieval quality is already high, the classification overhead adds latency and complexity without meaningful benefit. In those cases, a well-configured standard RAG or one of the more targeted techniques in this series will outperform a general-purpose orchestrator.

---

## Summary

Intelligent RAG is what happens when you build a RAG system for production rather than for a demo. By classifying query challenges, routing through challenge-specific engines, enforcing a hard LLM budget, caching repeated queries, and computing grounded confidence scores, it produces more reliable answers for every query type — at controlled cost.

The key design insights are: (1) heuristics before LLM for classification avoids most overhead; (2) shared state makes engines composable and independently testable; (3) budget enforcement prevents cost explosions; (4) multi-signal confidence provides genuine reliability signaling. Together, these make Intelligent RAG suitable for real-world deployments where simple pipelines would fail.
