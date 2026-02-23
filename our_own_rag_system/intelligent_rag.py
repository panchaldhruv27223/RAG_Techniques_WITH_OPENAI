import os
import sys
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

import numpy as np

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    chunk_text,
)


class Challenge(str, Enum):
    SIMPLE = "simple"
    TEMPORAL = "temporal"
    MULTI_HOP = "multi_hop"
    DECOMPOSITION = "decomposition"


@dataclass
class RetrievalState:
    """
    Shared state object passed through the pipeline.

    Every engine reads from and writes to this state.
    This is the composability fix — engines don't call each other,
    they all operate on the same state sequentially.
    """
    # Input
    original_query: str
    challenges: List[Challenge] = field(default_factory=list)

    # Accumulated retrieval
    chunks: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)

    # Sub-question tracking (decomposition)
    sub_questions: List[str] = field(default_factory=list)
    sub_answers: Dict[str, str] = field(default_factory=dict)

    # Multi-hop tracking
    hops: List[Dict[str, str]] = field(default_factory=list)

    # Temporal tracking
    temporal_intent: Optional[str] = None
    temporal_conflict: Optional[str] = None

    # Budget tracking
    llm_budget: int = 8
    llm_calls_used: int = 0

    # Execution trace
    trace: List[str] = field(default_factory=list)

    def use_budget(self, n: int = 1) -> bool:
        """Attempt to spend n LLM calls. Returns False if over budget."""
        if self.llm_calls_used + n > self.llm_budget:
            return False
        self.llm_calls_used += n
        return True

    @property
    def budget_remaining(self) -> int:
        return max(0, self.llm_budget - self.llm_calls_used)

    def add_chunks(self, chunks: List[str], scores: List[float], source: str):
        """Add chunks with deduplication."""
        existing = set(c[:100] for c in self.chunks)
        for c, s in zip(chunks, scores):
            key = c[:100]
            if key not in existing:
                self.chunks.append(c)
                self.scores.append(s)
                self.chunk_sources.append(source)
                existing.add(key)



@dataclass
class IntelligentResult:
    """Final result with full transparency."""
    answer: str
    confidence: float
    confidence_label: str
    strategies_used: List[str]
    trace: List[str]
    chunks: List[str]
    chunks_used: int
    llm_calls: int
    sub_answers: Optional[Dict[str, str]] = None
    temporal_info: Optional[str] = None
    hops: Optional[List[Dict]] = None



class QueryAnalyzer:
    """
    Hybrid analyzer: heuristics first, LLM only when ambiguous.
    """

    TEMPORAL_KEYWORDS = {
        "since", "changed", "trend", "latest", "recent", "recently",
        "current", "now", "today", "this year", "last year", "over time",
        "historically", "evolution", "before", "after", "post-pandemic",
        "decade", "century", "updated", "new", "newest", "oldest",
    }
    TEMPORAL_INTENTS = {
        "latest": {"latest", "newest", "current", "most recent", "now", "today"},
        "trend": {"trend", "over time", "evolution", "historically", "changed", "changing"},
        "since": {"since", "after", "post", "from"},
        "comparison": {"before", "compared to", "vs", "versus", "difference between"},
    }
    DECOMP_KEYWORDS = {
        "compare", "contrast", "differences", "similarities",
        "pros and cons", "advantages and disadvantages",
        "analyze", "evaluate", "assess",
    }
    MULTIHOP_PATTERNS = [
        r"which .+ (?:has|have|had|is|are|was|were) .+",
        r"who .+ that .+",
        r"what .+ (?:causes?|leads? to|results? in) .+",
    ]

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = OpenAIChat(
            model_name=model_name, temperature=0.0, max_tokens=5000,
        )

    def analyze(self, query: str, state: RetrievalState) -> RetrievalState:
        """Analyze query using heuristics first, LLM only if ambiguous."""
        q_lower = query.lower()
        challenges = set()
        temporal_intent = None
        needs_llm = False

        # --- HEURISTIC: Temporal ---
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        has_temporal_keyword = any(kw in q_lower for kw in self.TEMPORAL_KEYWORDS)
        if year_matches or has_temporal_keyword:
            challenges.add(Challenge.TEMPORAL)
            for intent, signals in self.TEMPORAL_INTENTS.items():
                if any(s in q_lower for s in signals):
                    temporal_intent = intent
                    break
            if not temporal_intent and year_matches:
                temporal_intent = "since"

        # --- HEURISTIC: Decomposition ---
        has_decomp_keyword = any(kw in q_lower for kw in self.DECOMP_KEYWORDS)
        has_multiple_questions = query.count("?") > 1
        and_count = len(re.findall(r'\band\b', q_lower))
        has_listing = and_count >= 2 or ("," in query and "and" in q_lower)
        if has_decomp_keyword or has_multiple_questions or has_listing:
            challenges.add(Challenge.DECOMPOSITION)

        # --- HEURISTIC: Multi-hop ---
        for pattern in self.MULTIHOP_PATTERNS:
            if re.search(pattern, q_lower):
                challenges.add(Challenge.MULTI_HOP)
                break

        # --- LLM fallback for ambiguous cases ---
        word_count = len(query.split())
        if (not challenges and word_count > 12) or (word_count > 20):
            needs_llm = True

        if needs_llm and state.use_budget(1):
            state.trace.append("Heuristics inconclusive → LLM classification")
            llm_challenges = self._llm_classify(query)
            challenges.update(llm_challenges)

        if not challenges:
            challenges.add(Challenge.SIMPLE)

        state.challenges = list(challenges)
        state.temporal_intent = temporal_intent
        state.trace.append(
            f"Analysis: {[c.value for c in state.challenges]}"
            f"{f' (temporal: {temporal_intent})' if temporal_intent else ''}"
            f" [{'heuristic' if not needs_llm else 'heuristic+LLM'}]"
        )
        return state

    def _llm_classify(self, query: str) -> set:
        """LLM classification — only called when heuristics are ambiguous."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify this query's challenges. Return JSON:\n"
                    '{"challenges": ["simple"|"temporal"|"multi_hop"|"decomposition"]}\n'
                    "challenges is an List, can have multiple values."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            result = self.llm.chat_json(messages)
            raw = result.get("challenges", ["simple"])
            return {Challenge(c) for c in raw if c in [e.value for e in Challenge]}
        except Exception:
            return set()


# ENGINE 1: TEMPORAL RERANKER
class TemporalReranker:
    """
    Reranks chunks based on temporal relevance.
    """

    RECENCY_KEYWORDS = {"recent", "latest", "new", "current", "updated", "2024", "2025"}

    def process(self, state: RetrievalState) -> RetrievalState:
        """Rerank chunks by temporal relevance. Zero LLM calls."""
        if Challenge.TEMPORAL not in state.challenges:
            return state
        if not state.chunks:
            return state

        intent = state.temporal_intent or "latest"
        scored_pairs = []

        for i, (chunk, score) in enumerate(zip(state.chunks, state.scores)):
            temporal_boost = self._score_temporal(chunk, intent)
            scored_pairs.append((chunk, score * temporal_boost, state.chunk_sources[i] if i < len(state.chunk_sources) else "initial"))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        state.chunks = [c for c, _, _ in scored_pairs]
        state.scores = [s for _, s, _ in scored_pairs]
        state.chunk_sources = [src for _, _, src in scored_pairs]

        state.trace.append(f"Temporal rerank: intent={intent}, {len(state.chunks)} chunks (0 LLM calls)")
        return state

    def _score_temporal(self, chunk: str, intent: str) -> float:
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', chunk)]
        chunk_lower = chunk.lower()
        has_dates = len(years) > 0
        max_year = max(years) if years else 0

        if intent == "latest":
            if max_year >= 2023: return 1.4
            if max_year >= 2020: return 1.2
            if max_year >= 2015: return 1.0
            if any(kw in chunk_lower for kw in self.RECENCY_KEYWORDS): return 1.1
            if has_dates: return 0.8
            return 0.9
        elif intent == "trend":
            if len(set(years)) >= 3: return 1.3
            if len(set(years)) >= 2: return 1.2
            if has_dates: return 1.1
            return 0.9
        elif intent == "since":
            if max_year >= 2015: return 1.3
            if has_dates: return 1.0
            return 0.8
        elif intent == "comparison":
            if len(set(years)) >= 2: return 1.3
            if has_dates: return 1.1
            return 0.9
        return 1.0


# ENGINE 2: MULTI-HOP RETRIEVER
class MultiHopRetriever:
    """
    Iterative retrieval that identifies and fills knowledge gaps.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_hops: int = 2):
        self.llm = OpenAIChat(
            model_name=model_name, temperature=0.0, max_tokens=5000,
        )
        self.max_hops = max_hops

    def process(self, state: RetrievalState, search_fn) -> RetrievalState:
        """Execute multi-hop retrieval. Adds chunks to shared state."""
        if Challenge.MULTI_HOP not in state.challenges:
            return state

        for hop_num in range(self.max_hops):
            if not state.use_budget(1):
                state.trace.append(f"Multi-hop stopped: budget exhausted after {hop_num} hops")
                break

            # Gap analysis on RAW chunks (not summaries)
            gap_query = self._find_gap(state.original_query, state.chunks)
            if gap_query is None:
                state.trace.append(f"Multi-hop complete: no gaps found (checked {hop_num+1} times)")
                break

            state.trace.append(f"Hop {hop_num+1}: gap='{gap_query[:60]}...'")
            new_chunks, new_scores = search_fn(gap_query, 3)
            state.add_chunks(new_chunks, new_scores, f"hop_{hop_num+1}")
            state.hops.append({"query": gap_query, "chunks_added": len(new_chunks)})

        return state

    def _find_gap(self, original_query: str, chunks: List[str]) -> Optional[str]:
        """Identify what's still missing. Operates on RAW chunks."""
        chunk_preview = "\n---\n".join(c[:300] for c in chunks[:8])
        messages = [
            {
                "role": "system",
                "content": (
                    "Given a question and retrieved info, is there enough to answer fully? "
                    'Return JSON: {"complete": True/False, "gap_query": "search query" or None}'
                ),
            },
            {
                "role": "user",
                "content": f"Question: {original_query}\n\nRetrieved:\n{chunk_preview}",
            },
        ]
        try:
            result = self.llm.chat_json(messages)
            if result.get("complete", True):
                return None
            return result.get("gap_query")
        except Exception:
            return None


# ENGINE 3: QUERY DECOMPOSER
class QueryDecomposer:
    """
    Splits complex queries into atomic sub-questions.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = OpenAIChat(
            model_name=model_name, temperature=0.0, max_tokens=5000,
        )

    def decompose(self, state: RetrievalState, search_fn) -> RetrievalState:
        """Decompose query and retrieve for sub-questions into shared pool."""
        if Challenge.DECOMPOSITION not in state.challenges:
            return state
        if not state.use_budget(1):
            state.trace.append("Decomposition skipped: budget exhausted")
            return state

        messages = [
            {
                "role": "system",
                "content": 'Break this into 2-4 atomic sub-questions. Return JSON: {"sub_questions": ["q1", "q2"]}',
            },
            {"role": "user", "content": state.original_query},
        ]
        try:
            result = self.llm.chat_json(messages)
            sub_qs = result.get("sub_questions", [state.original_query])
        except Exception:
            sub_qs = [state.original_query]

        state.sub_questions = sub_qs
        state.trace.append(f"Decomposed into {len(sub_qs)} sub-questions: {sub_qs}")

        # Retrieve per sub-question into SHARED pool
        for sq in sub_qs:
            chunks, scores = search_fn(sq, 3)
            state.add_chunks(chunks, scores, f"decomp:{sq[:30]}")

        return state

    def answer_sub_questions(self, state: RetrievalState) -> RetrievalState:
        """Answer each sub-Q from shared chunk pool. Budget-controlled."""
        if not state.sub_questions:
            return state

        context_text = "\n---\n".join(state.chunks[:10])

        for sq in state.sub_questions:
            if not state.use_budget(1):
                state.trace.append(f"Sub-answer skipped (budget): {sq[:40]}...")
                break
            messages = [
                {"role": "system", "content": "Answer concisely from context. Say if unsure."},
                {"role": "user", "content": f"Question: {sq}\n\nContext:\n{context_text}"},
            ]
            try:
                state.sub_answers[sq] = self.llm.chat(messages)
            except Exception:
                state.sub_answers[sq] = "Could not answer."

        state.trace.append(f"Answered {len(state.sub_answers)}/{len(state.sub_questions)} sub-questions")
        return state



# ENGINE 4: CONFIDENCE CALIBRATOR (Independent Validation)
class ConfidenceCalibrator:
    """
    Multi-signal confidence with INDEPENDENT grounding validation.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = OpenAIChat(
            model_name=model_name, temperature=0.0, max_tokens=5000,
        )

    def calibrate(self, state: RetrievalState, answer: str) -> Tuple[float, str]:
        # Signal 1: Retrieval quality (OBJECTIVE)
        if state.scores:
            avg_score = float(np.mean(state.scores[:min(5, len(state.scores))]))
            retrieval_signal = min(1.0, max(0.0, (avg_score - 0.3) / 0.5))
        else:
            retrieval_signal = 0.0

        # Signal 2: Coverage (OBJECTIVE)
        coverage_signal = min(1.0, len(state.chunks) / 3)

        # Signal 3: Grounding check (1 LLM call, INDEPENDENT)
        grounding_signal = 0.5
        if state.use_budget(1):
            grounding_signal = self._check_grounding(answer, state.chunks)

        confidence = 0.30 * retrieval_signal + 0.15 * coverage_signal + 0.55 * grounding_signal
        confidence = max(0.05, min(0.99, confidence))

        if confidence >= 0.80: label = "High"
        elif confidence >= 0.60: label = "Medium"
        elif confidence >= 0.35: label = "Low"
        else: label = "Very Low"

        state.trace.append(
            f"Confidence: {confidence:.0%} ({label}) "
            f"[retrieval={retrieval_signal:.2f}, coverage={coverage_signal:.2f}, "
            f"grounding={grounding_signal:.2f}]"
        )
        return confidence, label

    def _check_grounding(self, answer: str, chunks: List[str]) -> float:
        """Check if claims are grounded in sources. Catches hallucination."""
        source_preview = "\n---\n".join(c[:400] for c in chunks[:6])
        messages = [
            {
                "role": "system",
                "content": (
                    "Fact-check: can EACH claim in the answer be traced to the sources? "
                    'Return JSON: {"grounded_ratio": 0.0-1.0, "unsupported_claims": [...]}'
                ),
            },
            {
                "role": "user",
                "content": f"Answer:\n{answer}\n\nSources:\n{source_preview}",
            },
        ]
        try:
            result = self.llm.chat_json(messages)
            return float(result.get("grounded_ratio", 0.5))
        except Exception:
            return 0.5


# ANSWER SYNTHESIZER
class AnswerSynthesizer:
    """Generates final answer. Handles both direct and decomposed cases."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = OpenAIChat(
            model_name=model_name, temperature=temperature, max_tokens=1500,
        )

    def generate(self, state: RetrievalState) -> str:
        if not state.use_budget(1):
            return f"Based on available information: {' '.join(state.chunks[:2])[:500]}"

        if state.sub_answers:
            return self._synthesize(state)
        return self._direct(state)

    def _direct(self, state: RetrievalState) -> str:
        context_parts = []
        if state.temporal_conflict:
            context_parts.append(f"[Note: {state.temporal_conflict}]")
        context_parts.extend(state.chunks[:min(3, len(state.chunks))])
        context = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer from context. Be concise and accurate. "
                    "Acknowledge uncertainty when context is insufficient."
                ),
            },
            {"role": "user", "content": f"Question: {state.original_query}\n\nContext:\n{context}"},
        ]
        return self.llm.chat(messages)

    def _synthesize(self, state: RetrievalState) -> str:
        sub_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in state.sub_answers.items())
        messages = [
            {
                "role": "system",
                "content": "Synthesize sub-answers into one coherent response. Don't list them — weave naturally.",
            },
            {
                "role": "user",
                "content": f"Original: {state.original_query}\n\nSub-answers:\n{sub_text}",
            },
        ]
        return self.llm.chat(messages)




# QUERY CACHE

class QueryCache:
    """In-memory cache. Prevents recomputation of identical queries."""

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, IntelligentResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query: str) -> str:
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[IntelligentResult]:
        result = self._cache.get(self._key(query))
        if result: self.hits += 1
        else: self.misses += 1
        return result

    def put(self, query: str, result: IntelligentResult) -> None:
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[self._key(query)] = result




# INTELLIGENT RAG — THE UNIFIED PIPELINE
class IntelligentRAG:
    """
    Production-quality query-adaptive RAG system.

    Pipeline (each stage transforms shared RetrievalState):

        State → Analyzer → Initial Search → Temporal Rerank
              → Multi-Hop → Decomposer → Sub-Answers
              → Answer Synthesis → Confidence Calibration
              → IntelligentResult

    Each engine checks state.challenges to decide if it should run.
    Budget system caps total LLM calls per query.

    Usage:
        rag = IntelligentRAG(file_path="report.pdf")
        result = rag.query("How has climate policy changed since 2015?")
        print(result.answer)
        print(f"Confidence: {result.confidence:.0%}")
        rag.show_trace(result)
        rag.compare("What is the greenhouse effect?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 3,
        llm_budget: int = 20,
        max_hops: int = 4,
        embedding_model: str = "text-embedding-3-small",
        engine_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        cache_size: int = 1000,
    ):
        self.k = k
        self.llm_budget = llm_budget

        # Retrieval infrastructure
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

        # Pipeline engines
        self.analyzer = QueryAnalyzer(model_name=engine_model)
        self.temporal = TemporalReranker()
        self.multi_hop = MultiHopRetriever(model_name=engine_model, max_hops=max_hops)
        self.decomposer = QueryDecomposer(model_name=engine_model)
        self.synthesizer = AnswerSynthesizer(model_name=chat_model, temperature=temperature)
        self.confidence = ConfidenceCalibrator(model_name=engine_model)
        self.cache = QueryCache(max_size=cache_size)

        # Simple RAG chat for comparison
        self.simple_chat = OpenAIChat(model_name=chat_model, temperature=temperature)

        # Index
        self._index(file_path, chunk_size, chunk_overlap)

    def _index(self, file_path: str, chunk_size: int, chunk_overlap: int) -> None:
        if file_path.endswith(".pdf"):
            text = read_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        documents = [
            Document(content=c, metadata={"chunk_index": i, "source": file_path})
            for i, c in enumerate(chunks)
        ]
        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)
        print(f"[IntelligentRAG] Indexed '{os.path.basename(file_path)}' → {len(documents)} chunks")
        print(f"[IntelligentRAG] Budget: {self.llm_budget} LLM calls/query")

    def _search(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """Base vector search → (chunk_texts, scores)."""
        if k is None:
            k = self.k
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=k)
        return [r.document.content for r in results], [r.score for r in results]

    def query(self, question: str) -> IntelligentResult:
        """
        Main query method. Full pipeline execution.

        Pipeline order:
            1. Check cache
            2. Create state with budget
            3. Analyzer (heuristic-first)
            4. Initial vector search
            5. Temporal rerank (if temporal, 0 LLM calls)
            6. Multi-hop (if needed, budget-controlled)
            7. Decomposition + sub-answers (if needed, budget-controlled)
            8. Answer synthesis (1 LLM call)
            9. Confidence calibration (1 LLM call)
            10. Cache result

        Args:
            question: User's question.

        Returns:
            IntelligentResult with answer, confidence, trace.
        """
        # Step 1: Cache check
        cached = self.cache.get(question)
        if cached:
            print(f"  [Cache] Hit! Returning cached result.")
            return cached.answer, cached.chunks

        # Step 2: Initialize state
        state = RetrievalState(
            original_query=question,
            llm_budget=self.llm_budget,
        )
        print(f"\n  [Intelligent] Query: '{question[:70]}...'")
        print(f"  [Intelligent] Budget: {self.llm_budget} LLM calls")

        # Step 3: Analyze query (heuristic-first, uses 0-1 LLM calls)
        state = self.analyzer.analyze(question, state)
        print(f"  [Intelligent] Challenges: {[c.value for c in state.challenges]}")

        # Step 4: Initial vector search (0 LLM calls)
        initial_chunks, initial_scores = self._search(question)
        state.add_chunks(initial_chunks, initial_scores, "initial")
        state.trace.append(f"Initial search: {len(initial_chunks)} chunks")

        # Step 5: Temporal rerank (0 LLM calls)
        state = self.temporal.process(state)

        # Step 6: Multi-hop (budget-controlled)
        state = self.multi_hop.process(state, self._search)

        # Step 7: Decomposition + sub-answers (budget-controlled)
        state = self.decomposer.decompose(state, self._search)
        state = self.decomposer.answer_sub_questions(state)

        # Step 8: Generate answer (1 LLM call)
        print(f"  [Intelligent] Generating answer (budget remaining: {state.budget_remaining})...")
        answer = self.synthesizer.generate(state)
        state.trace.append(f"Answer generated ({len(answer)} chars)")

        # Step 9: Confidence calibration (1 LLM call)
        conf, conf_label = self.confidence.calibrate(state, answer)

        # Step 10: Build result
        result = IntelligentResult(
            answer=answer,
            confidence=conf,
            confidence_label=conf_label,
            strategies_used=[c.value for c in state.challenges],
            trace=state.trace,
            chunks= state.chunks,
            chunks_used=len(state.chunks),
            llm_calls=state.llm_calls_used,
            sub_answers=state.sub_answers if state.sub_answers else None,
            temporal_info=state.temporal_conflict,
            hops=state.hops if state.hops else None,
        )

        # Cache it
        self.cache.put(question, result)

        print(f"  [Intelligent] Done: {conf:.0%} confidence, {state.llm_calls_used} LLM calls")

        return result.answer, result.chunks


if __name__ == "__main__":
    pdf_file_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"

    rag = IntelligentRAG(
        file_path=pdf_file_path,
        chunk_size=1000,
        chunk_overlap=200,
        k=5,
        llm_budget=10,
        max_hops=2,
    )

    
    user_question = input("Ask a question: ").lower().strip()
    answer, context = rag.query(user_question)

    print("\n" + "="*60)
    print("Answer: \n", answer)
    print("Context: \n", context)
    print("="*60)


