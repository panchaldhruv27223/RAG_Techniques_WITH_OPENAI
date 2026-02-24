# Reliable RAG: Building Systems That Know What They Don't Know

---

## Introduction

LLMs hallucinate. This is not a bug to be fixed but a fundamental characteristic of probabilistic token prediction: when the model lacks strong grounding evidence, it generates plausible-but-wrong text with the same fluent confidence as correct text. In RAG systems, hallucination takes a specific form — the model generates claims that go *beyond* or *contradict* the retrieved context.

**Reliable RAG** addresses this directly by adding a post-generation verification stage. After generating an answer, it:

1. **Extracts individual claims** from the answer (each discrete factual assertion)
2. **Verifies each claim** against the retrieved context
3. **Labels claims** as SUPPORTED, UNSUPPORTED, or UNCERTAIN
4. **Computes a reliability score** from the claim verification results
5. **Makes a reliability decision**: Output the answer with a confidence tag, modify it to remove unsupported claims, or decline to answer with caveats

The output is not just an answer — it's an audited answer with a documented confidence level, enabling users and systems to make appropriate trust decisions.

---

## Why LLM-Generated Answers Can't Be Trusted Unconditionally

### The Extrapolation Problem

RAG retrieves relevant context and asks the LLM to answer from it. But LLMs don't strictly confine themselves to the retrieved text. Consider:

**Retrieved context**: "Aspirin reduces fever by inhibiting COX enzymes. Studies show it is effective for mild-to-moderate pain management."

**LLM-generated answer**: "Aspirin reduces fever by inhibiting COX enzymes and is effective for mild-to-moderate pain. The recommended adult dose is 325-650mg every 4-6 hours, and it should not be given to children under 12 due to Reye's syndrome risk."

The bolded text is *not* in the retrieved context. The LLM has drawn on prior knowledge — which may be correct, but isn't grounded in the retrieved evidence. For high-stakes applications (medical, legal, financial), this unverified extrapolation is unacceptable.

### The Confidence Illusion

LLMs generate text with equal fluency regardless of how confident they are. "The capital of France is Paris" and "The GDP of Zambia in 1973 was $X billion" are expressed in identically confident syntax. Users have no way to distinguish grounded claims from confabulated ones — unless the system explicitly detects and labels them.

---

## The Claim Extraction Pipeline

### Step 1: Decompose the Answer into Atomic Claims

```python
def extract_claims(self, answer: str) -> List[Claim]:
    """
    Decompose the answer into individual, atomic, verifiable claims.
    
    An 'atomic claim' is the smallest unit of verifiable assertion:
    - Not a question
    - Not an opinion (unless the opinion is attributed)
    - Not a conjunction of two independent claims
    
    Examples of atomic claims:
    ✓ "Aspirin inhibits COX-1 and COX-2 enzymes"
    ✓ "The boiling point of water is 100°C at sea level"
    ✗ "Aspirin reduces pain and fever and has anti-inflammatory properties"  (too many claims combined)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Extract all factual claims from the given text as a list of "
                "simple, atomic statements. Each claim should:\n"
                "1. Contain exactly ONE verifiable fact\n"
                "2. Be a complete, self-contained statement\n"
                "3. Use specific language (no vague terms like 'some' or 'many')\n"
                "4. Not overlap significantly with other claims\n\n"
                "Do not include opinions, definitions, or methodological descriptions.\n\n"
                'Return JSON: {"claims": ["claim 1", "claim 2", ...]}'
            )
        },
        {
            "role": "user",
            "content": f"Extract all factual claims from:\n\n{answer}"
        }
    ]
    result = self.llm.chat_json(messages)
    claim_texts = result.get("claims", [])
    
    return [
        Claim(text=claim_text, status=ClaimStatus.UNVERIFIED)
        for claim_text in claim_texts
        if claim_text.strip()
    ]
```

### Step 2: Verify Each Claim Against the Context

```python
def verify_claim(self, claim: str, context: str) -> VerificationResult:
    """
    Check whether a specific claim is supported by the retrieved context.
    
    Verification labels:
    - SUPPORTED: Claim is explicitly stated or directly inferrable from context
    - UNSUPPORTED: Claim makes an assertion not found in context (potential hallucination)
    - UNCERTAIN: Context is ambiguous about this claim, or partially addresses it
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Verify if the given claim is supported by the provided context.\n\n"
                "SUPPORTED: The claim is directly stated or clearly follows from the context.\n"
                "UNSUPPORTED: The claim makes an assertion not in the context "
                "(hallucination risk — the model may be using prior knowledge).\n"
                "UNCERTAIN: The context partially addresses the claim but is ambiguous.\n\n"
                'Return JSON: {"status": "SUPPORTED|UNSUPPORTED|UNCERTAIN", '
                '"evidence": "relevant quote from context or empty string", '
                '"reasoning": "brief explanation"}'
            )
        },
        {
            "role": "user",
            "content": (
                f"Claim to verify: {claim}\n\n"
                f"Context:\n{context[:3000]}\n\n"
                "Is this claim supported by the context?"
            )
        }
    ]
    result = self.llm.chat_json(messages)
    
    return VerificationResult(
        status=result.get("status", "UNCERTAIN"),
        evidence=result.get("evidence", ""),
        reasoning=result.get("reasoning", "")
    )
```

### Step 3: Compute Reliability Score

```python
def compute_reliability_score(self, claims: List[VerifiedClaim]) -> ReliabilityReport:
    """
    Aggregate claim verification results into a overall reliability score.
    
    Formula:
    reliability = (supported + 0.5 × uncertain) / total_claims
    
    Why 0.5 for uncertain? These claims might be correct (partial support)
    but shouldn't count fully toward the reliability score.
    """
    if not claims:
        return ReliabilityReport(score=1.0, level="HIGH", claims=claims)
    
    supported = sum(1 for c in claims if c.status == "SUPPORTED")
    uncertain = sum(1 for c in claims if c.status == "UNCERTAIN")
    unsupported = sum(1 for c in claims if c.status == "UNSUPPORTED")
    total = len(claims)
    
    reliability_score = (supported + 0.5 * uncertain) / total
    
    # Convert to reliability level
    if reliability_score >= 0.85:
        level = "HIGH"
        recommendation = "Answer is well-grounded in retrieved context"
    elif reliability_score >= 0.65:
        level = "MEDIUM"
        recommendation = "Answer is mostly grounded; review uncertain claims"
    else:
        level = "LOW"
        recommendation = "Answer contains unsupported claims; do not use without verification"
    
    print(f"📊 Reliability: {reliability_score:.2f} ({level})")
    print(f"   Supported: {supported} | Uncertain: {uncertain} | Unsupported: {unsupported}")
    
    return ReliabilityReport(
        score=reliability_score,
        level=level,
        recommendation=recommendation,
        supported_count=supported,
        uncertain_count=uncertain,
        unsupported_count=unsupported,
        total_claims=total,
        claims=claims
    )
```

---

## The Reliability Decision Engine

Based on the reliability score and level, the system makes one of three decisions:

```python
def make_reliability_decision(
    self,
    original_answer: str,
    reliability_report: ReliabilityReport,
    query: str,
    context: str
) -> FinalOutput:
    
    if reliability_report.level == "HIGH":
        # All (or nearly all) claims are grounded — output with confidence marker
        return FinalOutput(
            answer=original_answer,
            confidence="HIGH",
            caveat=None,
            modified=False
        )
    
    elif reliability_report.level == "MEDIUM":
        # Partially grounded — remove unsupported claims and add a disclaimer
        cleaned_answer = self._remove_unsupported_claims(
            original_answer,
            reliability_report.claims
        )
        return FinalOutput(
            answer=cleaned_answer,
            confidence="MEDIUM",
            caveat=(
                "Note: Some claims in the original answer could not be verified "
                "against the retrieved context and have been removed. "
                "Please verify with primary sources before relying on this answer."
            ),
            modified=True
        )
    
    else:  # LOW
        # Too many unsupported claims — decline to answer fully, offer what's verified
        verified_facts = [
            c.text for c in reliability_report.claims 
            if c.status == "SUPPORTED"
        ]
        if verified_facts:
            safe_answer = (
                "Based on the retrieved context, the following can be confirmed:\n"
                + "\n".join(f"• {fact}" for fact in verified_facts)
                + "\n\nAdditional information could not be verified from the available context."
            )
        else:
            safe_answer = (
                "The retrieved context does not contain sufficient information to "
                "reliably answer this question. Please consult authoritative sources."
            )
        
        return FinalOutput(
            answer=safe_answer,
            confidence="LOW",
            caveat="Answer limited to context-verified claims.",
            modified=True
        )
```

---

## Worked Example: Claim-Level Verification in Action

**Query**: "What is the mechanism and dosing of metformin for type 2 diabetes?"

**Retrieved context** (abbreviated): "Metformin reduces hepatic glucose production by activating AMPK. In clinical trials, first-line metformin therapy reduces HbA1c by approximately 1.5%. It is generally well-tolerated with the most common side effects being GI upset."

**Generated answer**: "Metformin works by activating AMPK to reduce hepatic glucose output. It typically reduces HbA1c by 1.5%. The standard starting dose is 500mg twice daily, titrating to 2000mg/day maximum. It should be avoided in patients with eGFR < 30."

**Claim extraction**:
1. "Metformin works by activating AMPK to reduce hepatic glucose output"
2. "It typically reduces HbA1c by 1.5%"
3. "The standard starting dose is 500mg twice daily"
4. "Maximum dose is 2000mg/day"
5. "Should be avoided in patients with eGFR < 30"

**Verification results**:

| Claim | Status | Evidence |
|-------|--------|---------|
| 1 | SUPPORTED | "reduces hepatic glucose production by activating AMPK" |
| 2 | SUPPORTED | "reduces HbA1c by approximately 1.5%" |
| 3 | UNSUPPORTED | Dosing not in retrieved context |
| 4 | UNSUPPORTED | Maximum dose not in retrieved context |
| 5 | UNSUPPORTED | eGFR threshold not in retrieved context |

**Reliability score**: (2 supported + 0.5 × 0 uncertain) / 5 = 0.40 → **LOW**

**Decision**: The dosing claims (3, 4, 5) are plausible (and may even be correct) but they're extrapolated from the LLM's training data, not from the retrieved context. For a medical application, this distinction is critical.

**Final output**: The system returns only the two verified claims (mechanism + HbA1c) with a disclaimer that dosing should be verified from authoritative clinical sources.

---

## LLM Call Budget

For k=3 retrieved chunks and N extracted claims (typically 3-8 claims per answer):

| Phase | Calls |
|-------|-------|
| Standard retrieval | 0 |
| Generation | 1 |
| Claim extraction | 1 |
| Claim verification (N claims) | N |
| Total | 2 + N ≈ 6-10 |

For N=5 claims: 7 total LLM calls. Comparable to Self-RAG in cost but different focus:
- Self-RAG prevents hallucination at generation time (proactive)
- Reliable RAG detects it after generation and allows controlled degradation (reactive)

---

## Production Integration Patterns

### Pattern 1: Silent Confidence Scoring
Return the answer + a reliability score to the UI, which displays a confidence indicator:
```json
{"answer": "...", "reliability": 0.91, "level": "HIGH"}
```

### Pattern 2: User-Facing Caveats
Append reliability-dependent disclaimers to answers. "HIGH" answers get none. "MEDIUM" get soft disclaimers. "LOW" get explicit warnings.

### Pattern 3: Automatic Escalation
When reliability falls below a threshold, automatically route to a human expert or premium LLM for verification:
```python
if reliability.score < 0.65:
    return escalate_to_expert(query, answer, reliability)
```

### Pattern 4: Audit Logging
Log all claim-level verifications for compliance and quality monitoring:
```python
audit_log.record(
    query=query,
    claims=reliability.claims,
    unsupported=[c for c in reliability.claims if c.status == "UNSUPPORTED"]
)
```

---

## When to Use Reliable RAG

Reliable RAG is appropriate when the downstream cost of a hallucinated answer is high enough to justify its computational overhead. Medical question answering, legal research, financial advisory tools, and compliance systems all share the property that a confident wrong answer is meaningfully worse than an explicit low-confidence response. For these cases, the claim extraction and verification pipeline is a straightforward tradeoff against the risk.

For general-purpose information lookup where hallucination rates are inherently low and users can tolerate occasional imprecision, the overhead is harder to justify. High-traffic applications also need to account for the cost multiplication — at 7+ extra LLM calls per query, the economics require careful modeling. Where query volume is high and the domain is lower-stakes, a single-pass generation with standard retrieval will often suffice.

---

## Summary

Reliable RAG acknowledges a fundamental truth about LLMs: they don't know what they don't know. By decomposing generated answers into atomic claims and verifying each against the retrieved context, it transforms opaque LLM outputs into auditable, confidence-labeled responses.

The reliability score is not just a monitoring metric — it's an operational signal. LOW reliability triggers answer modification or decline. MEDIUM triggers caveat injection. HIGH grants full confidence. This three-tier decision framework allows the system to be maximally useful when ground truth is available and appropriately cautious when it isn't.

For any RAG application where incorrect answers are costly, Reliable RAG changes the system's stance from "generate and hope" to "generate and verify" — a fundamental improvement in trustworthiness.
