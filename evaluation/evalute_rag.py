"""
RAG Evaluator
"""

import os
import json
import time
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


## Data classes
@dataclass
class RAGTestCase:
    """
    A single test case for RAG evaluation.

    Args:
        input_user_query:              The user query.
        actual_output:      The generated answer from your RAG system.
        expected_output:    Ground truth answer (needed for correctness/completeness).
        retrieval_context:  Retrieved documents/chunks (needed for faithfulness/relevancy).
    """
    input_user_query: str
    actual_output: str
    expected_output: str = ""
    retrieval_context: List[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    metric: str
    score: Optional[float]
    raw_score: Optional[Any] = None
    reason: str = ""
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCaseResult:
    """Results for one test case across all metrics."""
    test_case: RAGTestCase
    metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def passed_all(self) -> bool:
        return all(
            m.passed for m in self.metrics.values()
            if m.passed is not None
        )

    def summary_line(self) -> str:
        parts = []
        for name, m in self.metrics.items():
            icon = "✓" if m.passed else ("✗" if m.passed is not None else "—")
            score = f"{m.score:.2f}" if m.score is not None else "N/A"
            parts.append(f"{icon} {name}={score}")
        return f"[{' | '.join(parts)}]  Q: {self.test_case.input_user_query[:60]}"


@dataclass
class EvaluationReport:
    """Aggregated evaluation results across all test cases."""
    results: List[TestCaseResult]
    metrics_used: List[str]
    model: str
    elapsed_seconds: float = 0.0

    @property
    def averages(self) -> Dict[str, Optional[float]]:
        """Average score per metric (excluding None scores)."""
        avgs = {}
        for metric_name in self.metrics_used:
            scores = [
                r.metrics[metric_name].score
                for r in self.results
                if metric_name in r.metrics and r.metrics[metric_name].score is not None
            ]
            avgs[metric_name] = sum(scores) / len(scores) if scores else None
        return avgs

    @property
    def pass_rates(self) -> Dict[str, Optional[float]]:
        """Pass rate per metric."""
        rates = {}
        for metric_name in self.metrics_used:
            verdicts = [
                r.metrics[metric_name].passed
                for r in self.results
                if metric_name in r.metrics and r.metrics[metric_name].passed is not None
            ]
            rates[metric_name] = sum(verdicts) / len(verdicts) if verdicts else None
        return rates

    def summary(self) -> str:
        """Pretty-printed summary."""
        lines = [
            f"\n{'='*70}",
            f"  RAG EVALUATION REPORT",
            f"  Model: {self.model} | Cases: {len(self.results)} | Time: {self.elapsed_seconds:.1f}s",
            f"{'='*70}",
        ]

        # Per test case
        for i, r in enumerate(self.results):
            lines.append(f"  {i+1}. {r.summary_line()}")

        # Averages
        lines.append(f"\n{'─'*70}")
        lines.append("  AVERAGES")
        lines.append(f"{'─'*70}")
        for name, avg in self.averages.items():
            rate = self.pass_rates.get(name)
            avg_str = f"{avg:.3f}" if avg is not None else "N/A"
            rate_str = f"{rate*100:.0f}%" if rate is not None else "N/A"
            lines.append(f"  {name:20s}  avg={avg_str}  pass_rate={rate_str}")

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dict for JSON serialization."""
        return {
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "num_cases": len(self.results),
            "averages": self.averages,
            "pass_rates": self.pass_rates,
            "results": [
                {
                    "input_user_query": r.test_case.input_user_query,
                    "actual_output": r.test_case.actual_output,
                    "metrics": {
                        name: {
                            "score": m.score,
                            "raw_score": m.raw_score,
                            "passed": m.passed,
                            "reason": m.reason,
                        }
                        for name, m in r.metrics.items()
                    }
                }
                for r in self.results
            ]
        }


# Evaluation Prompts


CORRECTNESS_PROMPT = """You are an evaluation judge. Determine whether the actual output is \
factually correct based on the expected output.

Score from 0.0 to 1.0:
- 1.0: Completely correct, conveys the same facts as expected output
- 0.7-0.9: Mostly correct, minor differences that don't change meaning
- 0.4-0.6: Partially correct, some facts match but others missing or wrong
- 0.1-0.3: Mostly incorrect, only tangentially related
- 0.0: Completely incorrect or irrelevant

Respond with JSON: {"score": float, "reason": "brief explanation"}"""

FAITHFULNESS_PROMPT = """You are an evaluation judge assessing faithfulness of a RAG answer.

Faithfulness = every factual claim in the answer must be supported by the provided references.

Steps:
1. Extract ALL factual claims from the actual output
2. For each claim, determine if it is supported by the references
3. Score = (number of supported claims) / (total claims)

If the answer contains no factual claims, score is 1.0.
If a claim is partially supported, count it as 0.5.

Respond with JSON:
{
    "claims": [
        {"claim": "...", "supported": true/false, "reference_used": "which ref or none"}
    ],
    "supported_count": int,
    "total_count": int,
    "score": float,
    "reason": "brief explanation"
}"""

RELEVANCY_PROMPT = """You are an evaluation judge assessing answer relevancy.

Evaluate whether the answer accurately and directly responds to the user's question.
- Truthfulness does NOT impact relevancy — only whether the content addresses the question
- Missing information does NOT reduce relevancy — only evaluate what IS in the answer
- Irrelevant/off-topic content DOES reduce relevancy

Rating scale (1-5):
5 - Excellent: all information precisely answers the question
4 - Good: relevant info with minor tangential content
3 - Average: mix of relevant and superfluous content
2 - Low: mostly off-topic with some related elements
1 - Very low: does not address the question at all

Respond with JSON: {"score_1_to_5": int, "reason": "brief explanation"}"""

COMPLETENESS_PROMPT = """You are an evaluation judge assessing completeness.

Completeness = does the answer contain all relevant information from the references \
that pertains to the question?

Steps:
1. Identify what information in the references is relevant to the question
2. Check how much of that relevant info appears in the actual output
3. Compare against the expected output to see what's missing

Rating scale (1-5):
5 - Very complete: all relevant info from references is present
4 - Mostly complete: covers majority of key points, minor gaps
3 - Partial: addresses some relevant aspects, notable gaps
2 - Minimal: covers only a small portion of relevant info
1 - Incomplete: none of the relevant information is covered

Respond with JSON:
{
    "relevant_info_in_refs": "what info from refs is relevant to the question",
    "info_present_in_answer": "what relevant info IS in the answer",
    "info_missing_from_answer": "what relevant info is MISSING from the answer",
    "score_1_to_5": int,
    "reason": "brief explanation"
}"""

USEFULNESS_PROMPT = """You are an evaluation judge.

The answer indicates that no document precisely answers the question, but it may still \
provide supplementary related information.

Evaluate whether the supplementary information is useful and related to the question.

Score:
1 - The supplementary info is related and adds value
0 - The supplementary info is completely off-topic
null - No supplementary info provided (just says "no document found")

Respond with JSON:
{
    "has_supplementary_info": bool,
    "score": 0 or 1 or null,
    "reason": "brief explanation"
}"""

CONTEXT_PRECISION_PROMPT = """You are an evaluation judge assessing retrieval quality.

Given a question and the retrieved context chunks, evaluate how precise the retrieval is:
- What fraction of retrieved chunks are actually relevant to answering the question?

For each chunk, determine if it contains information useful for answering the question.
Score = (relevant chunks) / (total chunks)

Respond with JSON:
{
    "chunk_assessments": [
        {"chunk_index": int, "relevant": true/false, "reason": "brief"}
    ],
    "relevant_count": int,
    "total_count": int,
    "score": float,
    "reason": "overall assessment"
}"""

HALLUCINATION_PROMPT = """You are an evaluation judge checking for hallucinations.

A hallucination is any factual claim in the answer that:
1. Is NOT supported by the provided references, AND
2. Cannot be trivially inferred from the references

Steps:
1. Extract all factual claims from the answer
2. For each claim, check if it's in the references or trivially inferable
3. Flag any unsupported claims as hallucinations

Score = 1.0 - (hallucinated claims / total claims)
A score of 1.0 means NO hallucinations.

Respond with JSON:
{
    "claims": [
        {"claim": "...", "hallucinated": true/false, "explanation": "..."}
    ],
    "hallucination_count": int,
    "total_claims": int,
    "score": float,
    "reason": "brief explanation"
}"""


# Core Evaluator

class RAGEvaluator:
    """
    RAG evaluation engine.
    """


    # All available metrics
    AVAILABLE_METRICS = [
        "correctness", "faithfulness", "relevancy", "completeness", "usefulness", "context_precision", "hallucination"
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model:        OpenAI model for evaluation (gpt-4o recommended).
            temperature:  0.0 for deterministic evaluation.
            max_tokens:   Max response tokens for evaluation calls.
            api_key:      OpenAI API key (defaults to OPENAI_API_KEY env var).
            thresholds:   Custom pass/fail thresholds per metric (0.0-1.0).
                          Defaults: correctness=0.5, faithfulness=0.7, relevancy=0.75,
                                    completeness=0.75, hallucination=0.8, context_precision=0.5
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.thresholds = {
            "correctness": 0.80,
            "faithfulness": 0.85,
            "relevancy": 0.75,
            "completeness": 0.75,
            "usefulness": 0.50,
            "context_precision": 0.60,
            "hallucination": 0.90,
        }
        if thresholds:
            self.thresholds.update(thresholds)


    def _llm_json(self, system_prompt: str, user_content: str) -> Dict:
        """Make an LLM call expecting JSON response."""
        messages = [
            {"role": "system", "content": system_prompt + "\nRespond with valid JSON only. No markdown."},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _format_context(self, contexts: List[str]) -> str:
        """Format retrieval context for prompts."""
        return "\n\n".join(
            f"Reference {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        )

    # ── Individual Metrics ──

    def correctness(self, tc: RAGTestCase) -> MetricResult:
        """Is the actual output factually correct vs expected output?"""
        result = self._llm_json(
            CORRECTNESS_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"Expected Output: {tc.expected_output}\n\n"
            f"Actual Output: {tc.actual_output}"
        )
        score = float(result.get("score", 0.0))
        return MetricResult(
            metric="correctness",
            score=score,
            raw_score=score,
            reason=result.get("reason", ""),
            passed=score >= self.thresholds["correctness"],
            details=result,
        )

    def faithfulness(self, tc: RAGTestCase) -> MetricResult:
        """Is the answer supported by the retrieval context? (No hallucinations)"""
        result = self._llm_json(
            FAITHFULNESS_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"References:\n{self._format_context(tc.retrieval_context)}\n\n"
            f"Actual Output: {tc.actual_output}"
        )
        score = float(result.get("score", 0.0))
        return MetricResult(
            metric="faithfulness",
            score=score,
            raw_score=score,
            reason=result.get("reason", ""),
            passed=score >= self.thresholds["faithfulness"],
            details=result,
        )

    def relevancy(self, tc: RAGTestCase) -> MetricResult:
        """Is the answer relevant to the question?"""
        result = self._llm_json(
            RELEVANCY_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"References:\n{self._format_context(tc.retrieval_context)}\n\n"
            f"Answer: {tc.actual_output}"
        )
        raw = int(result.get("score_1_to_5", 1))
        normalized = (raw - 1) / 4  # 1-5 → 0.0-1.0
        return MetricResult(
            metric="relevancy",
            score=normalized,
            raw_score=raw,
            reason=f"[{raw}/5] {result.get('reason', '')}",
            passed=normalized >= self.thresholds["relevancy"],
            details=result,
        )

    def completeness(self, tc: RAGTestCase) -> MetricResult:
        """Does the answer cover all relevant info from context?"""
        result = self._llm_json(
            COMPLETENESS_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"References:\n{self._format_context(tc.retrieval_context)}\n\n"
            f"Expected Output: {tc.expected_output}\n\n"
            f"Actual Output: {tc.actual_output}"
        )
        raw = int(result.get("score_1_to_5", 1))
        normalized = (raw - 1) / 4
        return MetricResult(
            metric="completeness",
            score=normalized,
            raw_score=raw,
            reason=f"[{raw}/5] {result.get('reason', '')}",
            passed=normalized >= self.thresholds["completeness"],
            details=result,
        )

    def usefulness(self, tc: RAGTestCase) -> MetricResult:
        """When system says 'no document found', is supplementary info useful?"""
        no_doc_phrases = ["no document", "no relevant", "cannot find", "unable to find", "don't have"]
        is_no_doc = any(p in tc.actual_output.lower() for p in no_doc_phrases)

        if not is_no_doc:
            return MetricResult(
                metric="usefulness",
                score=None,
                raw_score=None,
                reason="N/A — answer does not indicate no document found",
                passed=None,
            )

        result = self._llm_json(
            USEFULNESS_PROMPT,
            f"Question: {tc.input_user_query}\n\nAnswer: {tc.actual_output}"
        )
        raw = result.get("score")
        score = float(raw) if raw is not None else None
        return MetricResult(
            metric="usefulness",
            score=score,
            raw_score=raw,
            reason=result.get("reason", ""),
            passed=bool(raw) if raw is not None else None,
            details=result,
        )

    def context_precision(self, tc: RAGTestCase) -> MetricResult:
        """What fraction of retrieved chunks are relevant to the question?"""
        if not tc.retrieval_context:
            return MetricResult(
                metric="context_precision",
                score=None,
                reason="No retrieval context provided",
                passed=None,
            )

        result = self._llm_json(
            CONTEXT_PRECISION_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"Retrieved Chunks:\n{self._format_context(tc.retrieval_context)}"
        )
        score = float(result.get("score", 0.0))
        return MetricResult(
            metric="context_precision",
            score=score,
            raw_score=score,
            reason=result.get("reason", ""),
            passed=score >= self.thresholds["context_precision"],
            details=result,
        )

    def hallucination(self, tc: RAGTestCase) -> MetricResult:
        """Check for hallucinated claims not in the context."""
        result = self._llm_json(
            HALLUCINATION_PROMPT,
            f"Question: {tc.input_user_query}\n\n"
            f"References:\n{self._format_context(tc.retrieval_context)}\n\n"
            f"Answer: {tc.actual_output}"
        )
        score = float(result.get("score", 0.0))
        return MetricResult(
            metric="hallucination",
            score=score,
            raw_score=result.get("hallucination_count", 0),
            reason=result.get("reason", ""),
            passed=score >= self.thresholds["hallucination"],
            details=result,
        )



    # ── Evaluation Entry Points ──

    def _resolve_metrics(self, metrics: Optional[List[str]] = None) -> List[str]:
        """Resolve metric names, expanding 'all'."""
        if metrics is None or metrics == ["all"] or "all" in metrics:
            return list(self.AVAILABLE_METRICS)
        for m in metrics:
            if m not in self.AVAILABLE_METRICS:
                raise ValueError(
                    f"Unknown metric '{m}'. Available: {self.AVAILABLE_METRICS}"
                )
        return metrics

    def _get_metric_fn(self, name: str) -> Callable:
        """Get metric function by name."""
        return getattr(self, name)

    def evaluate_single(
        self,
        test_case: RAGTestCase,
        metrics: Optional[List[str]] = None,
    ) -> TestCaseResult:
        """
        Evaluate a single test case against specified metrics.

        Args:
            test_case: The RAG test case to evaluate.
            metrics:   List of metric names, or ["all"] for all metrics.

        Returns:
            TestCaseResult with all metric results.
        """
        metric_names = self._resolve_metrics(metrics)
        tc_result = TestCaseResult(test_case=test_case)

        for name in metric_names:
            fn = self._get_metric_fn(name)
            tc_result.metrics[name] = fn(test_case)

        return tc_result

    def evaluate_batch(
        self,
        test_cases: List[RAGTestCase],
        metrics: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> EvaluationReport:
        """
        Evaluate multiple test cases and generate an aggregate report.

        Args:
            test_cases: List of RAG test cases.
            metrics:    List of metric names, or ["all"].
            verbose:    Print progress and per-case results.

        Returns:
            EvaluationReport with per-case results and averages.
        """
        metric_names = self._resolve_metrics(metrics)
        start = time.time()
        results = []

        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"\n[{i+1}/{len(test_cases)}] Evaluating: {tc.input_user_query[:60]}...")

            tc_result = self.evaluate_single(tc, metric_names)
            results.append(tc_result)

            if verbose:
                print(f"  {tc_result.summary_line()}")

        elapsed = time.time() - start

        report = EvaluationReport(
            results=results,
            metrics_used=metric_names,
            model=self.model,
            elapsed_seconds=elapsed,
        )

        if verbose:
            print(report.summary())

        return report



# Convenience Functions

def create_test_cases(
    questions: List[str],
    generated_answers: List[str],
    ground_truths: Optional[List[str]] = None,
    retrieved_contexts: Optional[List[List[str]]] = None,
) -> List[RAGTestCase]:
    """
    Create test cases from parallel lists.

    Args:
        questions:          List of queries.
        generated_answers:  List of RAG-generated answers.
        ground_truths:      List of expected answers (optional).
        retrieved_contexts: List of context lists (optional).

    Returns:
        List of RAGTestCase objects.
    """
    n = len(questions)
    ground_truths = ground_truths or [""] * n
    retrieved_contexts = retrieved_contexts or [[]] * n

    return [
        RAGTestCase(
            input_user_query=q,
            actual_output=a,
            expected_output=gt,
            retrieval_context=ctx,
        )
        for q, a, gt, ctx in zip(questions, generated_answers, ground_truths, retrieved_contexts)
    ]



def quick_evaluate(
    question: str,
    answer: str,
    context: List[str],
    expected: str = "",
    metrics: Optional[List[str]] = None,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    One-liner evaluation for quick testing.

    Args:
        question: The user query.
        answer:   The RAG-generated answer.
        context:  Retrieved context chunks.
        expected: Ground truth answer.
        metrics:  Which metrics to run.
        model:    Evaluation model.

    Returns:
        Dict of {metric_name: {"score": float, "passed": bool, "reason": str}}

    Example:
        result = quick_evaluate(
            question="What is X?",
            answer="X is a thing.",
            context=["X is defined as a thing in the field of Y."],
            expected="X is a thing in Y.",
        )
    """
    evaluator = RAGEvaluator(model=model)
    tc = RAGTestCase(
        input_user_query=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=context,
    )
    result = evaluator.evaluate_single(tc, metrics=metrics)
    return {
        name: {"score": m.score, "passed": m.passed, "reason": m.reason}
        for name, m in result.metrics.items()
    }


if __name__ == "__main__":
    print("RAG Evaluator — Demo\n")

    evaluator = RAGEvaluator(model="gpt-4o")

    # demo_cases = [
    #     RAGTestCase(
    #         input_user_query="What is the capital of Spain?",
    #         actual_output="Madrid.",
    #         expected_output="Madrid is the capital of Spain.",
    #         retrieval_context=["Madrid is the capital of Spain, located in the center of the Iberian Peninsula."],
    #     ),
    #     RAGTestCase(
    #         input_user_query="Where is the Eiffel Tower located?",
    #         actual_output="The Eiffel Tower is located at Rue Rabelais in Paris.",
    #         expected_output="The Eiffel Tower is on the Champ de Mars in Paris.",
    #         retrieval_context=[
    #             "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    #             "Gustave Eiffel died in his apartment at Rue Rabelais in Paris.",
    #         ],
    #     ),
    #     RAGTestCase(
    #         input_user_query="What are the effects of climate change on agriculture?",
    #         actual_output=(
    #             "No document seems to precisely answer your question. "
    #             "However, climate change is known to increase extreme weather events."
    #         ),
    #         expected_output="",
    #         retrieval_context=["Climate change leads to more frequent extreme weather events including droughts and floods."],
    #     ),
    # ]

    # report = evaluator.evaluate_batch(
    #     demo_cases,
    #     metrics=["correctness", "faithfulness", "relevancy", "hallucination", "completeness", "context_precision"],
    #     verbose=True,
    # )

    # # Export
    # print("\nJSON export:")
    # print(json.dumps(report.to_dict(), indent=2, default=str))
