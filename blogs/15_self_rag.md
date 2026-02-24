# Self-RAG: Teaching the Model to Know What It Doesn't Know

---

## Introduction

Standard RAG always retrieves context — whether the query needs it or not. Ask "what is 2+2?" and the system still does an embedding lookup, retrieves chunks, and builds a context window before generating "4." This is wasteful but harmless. More damaging: standard RAG always *trusts* whatever it retrieved and always includes the entirety of retrieved context in the answer — even if some of it is irrelevant or contradicted by better sources.

**Self-RAG**, introduced by Asai et al. (2023) at the University of Washington, trains LLMs to make explicit retrieval and quality decisions as part of the generation process. It extends the RAG loop with a five-phase self-assessment pipeline:

1. **Retrieve?** — Does this query even need retrieval?
2. **Retrieve** — Get candidate chunks (only if step 1 says yes)
3. **ISREL** — Is each retrieved chunk *relevant* to the query?
4. **ISSUP** — Does the generated answer *follow* from the retrieved context?
5. **ISUSE** — Is the answer *useful* and complete?

Each decision is a dedicated LLM call, making Self-RAG the most thorough — and most LLM-intensive — quality control approach in the RAG family.

---

## The Five-Phase Pipeline

### Phase 1: Should We Retrieve?

Not all queries benefit from retrieval. Self-RAG first decides whether to retrieve:

```python
def _should_retrieve(self, query: str) -> bool:
    """
    Decide if retrieval is necessary for this query.
    
    Simple factual arithmetic, definitional questions, or queries the LLM
    clearly knows from training need no retrieval.
    
    Multi-hop, specific, recent, or domain-specific queries benefit from retrieval.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Determine if this query requires retrieval from a knowledge base "
                "to answer accurately and completely.\n\n"
                "Retrieval is NEEDED when:\n"
                "- The answer requires specific facts, statistics, or data\n"
                "- The answer involves specialized domain knowledge\n"
                "- The answer may have changed over time (current events)\n"
                "- The query asks about specific entities, places, or events\n\n"
                "Retrieval is NOT NEEDED when:\n"
                "- The answer is basic arithmetic or logic\n"
                "- The answer is a commonly known general fact\n"
                "- The query is a creative writing or opinion request\n\n"
                'Return JSON: {"needs_retrieval": true/false, "reasoning": "..."}'
            )
        },
        {"role": "user", "content": f"Query: {query}"}
    ]
    result = self.llm.chat_json(messages)
    needs_retrieval = result.get("needs_retrieval", True)  # default: retrieve
    
    print(f"Retrieve? {needs_retrieval} -- {result.get('reasoning', '')[:80]}")
    return needs_retrieval
```

**Examples:**
- "What is 15% of 240?" → `needs_retrieval: false` (arithmetic)
- "What is the molecular weight of aspirin?" → `needs_retrieval: true` (specific fact)
- "Write a poem about autumn." → `needs_retrieval: false` (creative, no facts needed)
- "What were the key findings of the IPCC AR6 report?" → `needs_retrieval: true` (specific document)

### Phase 2: Retrieval (conditional)

Standard FAISS-based retrieval, executed only if Phase 1 returns `true`.

### Phase 3: ISREL — Is Each Chunk Relevant?

```python
def _evaluate_relevance(self, query: str, chunk: str) -> RelevanceScore:
    """
    ISREL: Evaluate whether the retrieved chunk is relevant to the query.
    
    This is similar to CRAG's evaluator but is framed as a binary+confidence
    assessment rather than a 0-1 score. The binary framing is intentional —
    in Self-RAG, relevance is a filter, not a weight.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Evaluate if the provided document is relevant to the query. "
                "A relevant document contains information that would help answer the query.\n\n"
                '{"is_relevant": true/false, "confidence": 0.0-1.0, "reasoning": "..."}'
            )
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nDocument:\n{chunk[:1500]}"
        }
    ]
    result = self.llm.chat_json(messages)
    return RelevanceScore(
        is_relevant=result.get("is_relevant", False),
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", "")
    )
```

Chunks that fail the ISREL check are discarded. Only relevant chunks proceed to answer generation.

### Phase 4: ISSUP — Does the Answer Follow From Context?

After generating a candidate answer from relevant chunks, Self-RAG checks whether the answer is actually *supported* by the retrieved context:

```python
def _evaluate_support(self, query: str, context: str, answer: str) -> SupportScore:
    """
    ISSUP: Does the generated answer follow from the provided context?
    
    This detects hallucination: statements in the answer that are not
    derivable from the retrieved context — even if they sound plausible.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Evaluate if the generated answer is fully supported by the provided context.\n\n"
                "Support levels:\n"
                "  FULLY_SUPPORTED: Every claim in the answer is directly stated in context\n"
                "  PARTIALLY_SUPPORTED: Some claims are in context, some are not\n"
                "  NOT_SUPPORTED: Claims in the answer are not in the context (hallucination risk)\n\n"
                '{"support_level": "FULLY_SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED", '
                '"unsupported_claims": ["..."], "reasoning": "..."}'
            )
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Context:\n{context[:2000]}\n\n"
                f"Generated Answer:\n{answer}\n\n"
                "Evaluate whether the answer is supported by the context:"
            )
        }
    ]
    result = self.llm.chat_json(messages)
    return SupportScore(
        support_level=result.get("support_level", "NOT_SUPPORTED"),
        unsupported_claims=result.get("unsupported_claims", []),
        reasoning=result.get("reasoning", "")
    )
```

**What ISSUP catches that ISREL misses:**

ISREL asks "is this chunk relevant?" (document-level check)
ISSUP asks "did the generation stay within the retrieved context?" (generation-level check)

A chunk can be highly relevant but the LLM might extrapolate beyond it. For example:
- Chunk: "Aspirin inhibits COX-1 and COX-2 enzymes at doses of 300-600mg."
- Generated: "Aspirin inhibits COX enzymes. At high doses (>1000mg), it can cause liver damage."

The chunk is relevant. But "liver damage" is not in the chunk — ISSUP flags this as NOT_SUPPORTED.

### Phase 5: ISUSE — Is the Answer Useful?

```python
def _evaluate_usefulness(self, query: str, answer: str) -> UsefulnessScore:
    """
    ISUSE: Is the final answer actually useful and complete?
    
    Even a fully supported answer might be incomplete, vague, or evasive.
    ISUSE is a final quality gate before returning to the user.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Rate the usefulness of this answer on a scale of 1-5:\n"
                "  5: Complete, direct, well-explained answer\n"
                "  4: Good answer with minor gaps\n"
                "  3: Partial answer, addresses the question roughly\n"
                "  2: Vague or incomplete, doesn't fully address the query\n"
                "  1: Doesn't answer the question at all\n\n"
                '{"usefulness_score": 1-5, "reasoning": "...", "is_useful": true/false}'
            )
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nAnswer: {answer}"
        }
    ]
    result = self.llm.chat_json(messages)
    score = int(result.get("usefulness_score", 3))
    return UsefulnessScore(
        score=score,
        is_useful=(score >= 3),
        reasoning=result.get("reasoning", "")
    )
```

---

## The Complete Self-RAG Decision Tree

```python
def query(self, question: str) -> SelfRAGResult:
    
    # Phase 1: Retrieval decision
    if not self._should_retrieve(question):
        # No retrieval — generate directly from LLM knowledge
        answer = self._generate_without_context(question)
        return SelfRAGResult(
            answer=answer,
            used_retrieval=False,
            phases_executed=["RETRIEVE_DECISION: NO"]
        )
    
    # Phase 2: Retrieve
    candidates = self.vector_store.search(query_emb, k=self.k)
    
    # Phase 3: ISREL — filter irrelevant chunks
    relevant_chunks = []
    for chunk in candidates:
        rel = self._evaluate_relevance(question, chunk.content)
        if rel.is_relevant:
            relevant_chunks.append(chunk.content)
    
    if not relevant_chunks:
        # All retrieved chunks were irrelevant — generate with uncertainty signal
        answer = self._generate_with_uncertainty(question)
        return SelfRAGResult(answer=answer, used_retrieval=True, phases=["ISREL: NONE RELEVANT"])
    
    # Generate candidate answer from relevant chunks
    context = "\n\n".join(relevant_chunks)
    candidate_answer = self._generate_with_context(question, context)
    
    # Phase 4: ISSUP — check if answer is grounded in context
    support = self._evaluate_support(question, context, candidate_answer)
    
    if support.support_level == "NOT_SUPPORTED":
        # Answer contains hallucinated content — regenerate with stricter grounding instructions
        candidate_answer = self._regenerate_grounded(question, context)
    
    # Phase 5: ISUSE — final quality check
    usefulness = self._evaluate_usefulness(question, candidate_answer)
    
    final_answer = candidate_answer
    if not usefulness.is_useful and relevant_chunks:
        # Attempt one more generation with different prompting
        final_answer = self._generate_expanded(question, context)
    
    return SelfRAGResult(
        answer=final_answer,
        used_retrieval=True,
        phases_executed=["RETRIEVE_DECISION: YES", "ISREL", "ISSUP", "ISUSE"],
        support_level=support.support_level,
        usefulness_score=usefulness.score
    )
```

---

## LLM Call Budget: Self-RAG vs. Standard RAG

For a query requiring retrieval with k=3 candidates:

| Phase | LLM calls |
|-------|-----------|
| Should retrieve? | 1 |
| ISREL × k | 3 |
| Generation | 1 |
| ISSUP | 1 |
| ISUSE | 1 |
| **Total (happy path)** | **7** |
| **If ISSUP fails → regenerate** | +1 = 8 |
| **If ISUSE fails → regenerate** | +1 = 9 |

**Standard RAG**: 1 call  
**Self-RAG**: 7-9 calls

Self-RAG is 7-9× more expensive per query. This is justified only for:
- Applications where errors are costly (medical decisions, legal research, financial analysis)
- Use cases where answer quality audit is a strict requirement
- Corpora where hallucination risk is known to be high

---

## Self-RAG vs. CRAG: Choosing the Right Corrective Strategy

| Dimension | Self-RAG | CRAG |
|-----------|---------|------|
| Retrieval decision | Yes (explicit "should retrieve?" phase) | No (always retrieves) |
| Relevance check | Per-chunk (ISREL) | Per-chunk (evaluator score) |
| Generation grounding | Yes (ISSUP) | No |
| Usefulness check | Yes (ISUSE) | No |
| Web fallback | No (by default) | Yes (on INCORRECT/AMBIGUOUS) |
| LLM calls per query | 7-9 | 4-7 |
| Best for | Self-contained corpora needing generation quality control | Corpora with coverage gaps needing web fallback |

**Use Self-RAG when**: Your corpus covers the domain well, but you need strong guarantees about grounding and you can't afford web access.

**Use CRAG when**: Your corpus may have knowledge gaps, and web search is available to fill them.

**Use both**: Enterprise knowledge bases with web access for fallback + generation quality control.

---

## When to Use Self-RAG

Self-RAG is the appropriate choice when correctness failures carry real-world consequences. In medical decision support, legal research, and financial analysis, the cost of a hallucinated answer or an unsupported claim can be significant — and the 7-9 LLM calls per query is a straightforward tradeoff against that risk. Systems where retrieval and generation decisions must be auditable also benefit from Self-RAG's five logged phases, which create a clear decision trail for each response.

For general-purpose applications where queries are simple or hallucination risk is inherently low, the computational overhead is difficult to justify. Standard RAG produces one LLM call; Self-RAG produces at minimum seven. At scale, that difference is substantial both in cost and latency. The technique is best reserved for the narrowly defined, high-accountability use cases where each answer is genuinely consequential.

---

## Summary

Self-RAG is the most thorough quality-control framework in the RAG family. By making explicit, logged decisions at each phase — retrieval necessity, chunk relevance, generation grounding, and answer usefulness — it transforms RAG from a pipeline that *outputs* answers into one that *evaluates* them at every step.

The tradeoff is substantial: 7-9 LLM calls per query vs. 1 for standard RAG. For applications where accuracy is the primary metric and cost is secondary, Self-RAG provides the strongest available guarantee of quality — both in what was retrieved and in how faithfully the generation used it.
