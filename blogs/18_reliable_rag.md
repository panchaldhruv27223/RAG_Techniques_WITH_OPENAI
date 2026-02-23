# Reliable RAG: Catching Hallucinations Before They Reach the User

> **Technique:** Reliable RAG (Hallucination Detection)  
> **Complexity:** Intermediate  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

The central promise of RAG is grounding: by providing the LLM with retrieved context, we expect answers to be rooted in real documents rather than fabricated from training data. But RAG doesn't *guarantee* grounding. LLMs can still hallucinate details, blend retrieved facts with invented ones, or produce technically accurate-sounding statements that aren't supported by the provided context.

**Reliable RAG** adds an explicit hallucination detection layer that validates generated answers against retrieved context *before returning them to the user*. If the answer can't be verified against the source material, the system flags it — or in strict mode, refuses to return it.

This technique is about building trust: not just generating answers, but proving those answers are supported.

---

## The Core Problem: RAG Doesn't Eliminate Hallucination

Consider this failure case:

**Context retrieved**: "The Treaty of Versailles was signed on June 28, 1919."

**LLM answer**: "The Treaty of Versailles, signed on June 28, 1919, which ended World War I and imposed heavy reparations of $442 billion on Germany, was ratified by the League of Nations in 1920."

Two problems:
1. The $442 billion figure is not in the context (and is inaccurate)
2. The ratification detail is not in the context

RAG retrieved a good chunk. The LLM then added plausible-sounding additional "facts" from its training knowledge, creating a response that mixes verified and unverified information — and the user has no way to tell the difference.

Reliable RAG catches this.

---

## How Reliable RAG Works

### Pipeline

```
User Query
    ↓
Standard vector retrieval → top-k chunks
    ↓
LLM generates answer (standard generation)
    ↓
[Hallucination Detector]
    ↓ Breaks answer into individual claims
    ↓ Checks each claim against retrieved context
    ↓ Scores: verified / partially_verified / unverified / irrelevant
    ↓
Compute hallucination_rate = unverified_claims / total_claims
    ↓
If rate > threshold → respond = "Cannot verify this answer"
Else → return answer + transparency report
```

### Claim Extraction

```python
def _extract_claims(self, answer: str) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Break down the answer into individual factual claims. "
                "Each claim should be a single verifiable statement. "
                'Return JSON: {"claims": ["claim1", "claim2", ...]}'
            )
        },
        {"role": "user", "content": f"Answer:\n{answer}"}
    ]
    result = self.llm.chat_json(messages)
    return result.get("claims", [])
```

This step decomposes the answer into atomic, individually verifiable statements. "The Treaty of Versailles, signed in 1919, imposed reparations on Germany" becomes three claims:
1. "The Treaty of Versailles was signed in 1919"
2. "The Treaty of Versailles imposed reparations"
3. "The reparations were imposed on Germany"

### Claim Verification

```python
def _verify_claim(self, claim: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Verify if the claim is supported by the context. "
                'Return JSON: {"status": "verified"} or {"status": "partially_verified"} '
                'or {"status": "unverified"} or {"status": "irrelevant"}\n\n'
                "verified: directly supported by context text\n"
                "partially_verified: partially supported, some details not confirmed\n"
                "unverified: contradicts or not found in context\n"
                "irrelevant: claim is about general knowledge, not specific to the question"
            )
        },
        {
            "role": "user",
            "content": f"Claim: {claim}\n\nContext: {context}"
        }
    ]
    result = self.llm.chat_json(messages)
    return result.get("status", "unverified")
```

Four verification states capture the full spectrum between "definitely in the context" and "definitely not in the context":

| Status | Meaning | Action |
|--------|---------|--------|
| `verified` | Directly supported by context text | ✅ Safe |
| `partially_verified` | Partially supported, some details unclear | ⚠️ Flag |
| `unverified` | Not found in or contradicts context | ❌ Hallucination |
| `irrelevant` | General background knowledge | ℹ️ Neutral |

### Hallucination Rate Calculation

```python
status_counts = {
    "verified": 0, "partially_verified": 0, 
    "unverified": 0, "irrelevant": 0
}
for claim in claims:
    status = self._verify_claim(claim, context_text)
    claim_verifications.append({"claim": claim, "status": status})
    status_counts[status] += 1

factual_claims = len(claims) - status_counts["irrelevant"]
if factual_claims > 0:
    hallucination_rate = status_counts["unverified"] / factual_claims
else:
    hallucination_rate = 0.0
```

`irrelevant` claims (general background knowledge) are excluded from the denominator — it's unfair to penalize the LLM for correctly stating that "gravity is the force that attracts objects" when the context doesn't explicitly contain that fact.

### The Reliability Decision

```python
is_reliable = hallucination_rate <= self.hallucination_threshold  # default: 0.2

if is_reliable:
    final_answer = answer
else:
    final_answer = (
        "I cannot provide a verified answer to this question. "
        "The retrieved information may be insufficient or the question "
        "may require knowledge not present in the documents."
    )
```

A `hallucination_threshold=0.2` means: if more than 20% of factual claims are unverifiable against retrieved context, refuse to return the answer. Adjust this based on your application's risk tolerance:

- **High-risk domain (medical, legal)**: threshold = 0.05 (5% max unverified)
- **Enterprise knowledge base**: threshold = 0.20 (20%)
- **General Q&A**: threshold = 0.30 (30%)

---

## The Reliability Report

Reliable RAG returns a rich structured response:

```python
@dataclass
class ReliabilityReport:
    answer: str
    is_reliable: bool
    hallucination_rate: float   # 0.0 = no hallucinations, 1.0 = fully hallucinated
    claim_verifications: List[Dict[str, str]]  # [{claim, status}, ...]
    status_counts: Dict[str, int]
    context_used: List[str]
    final_answer: str           # may be refusal if unreliable
```

Every claim is individually tagged with its verification status. This enables:

1. **User transparency**: "Here's the answer, and here are the claims we could verify."
2. **Developer debugging**: "This claim about X is always unverified — maybe we need more source documents?"
3. **Audit trails**: Full verification records for compliance or review processes.

---

## Configuration

```python
class ReliableRAG:
    def __init__(self,
                 file_path: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 k: int = 3,
                 hallucination_threshold: float = 0.2,  # max fraction unverified
                 embedding_model: str = "text-embedding-3-small",
                 chat_model: str = "gpt-4o-mini"):
```

---

## LLM Call Overhead

For a query producing an answer with N claims:

| Standard RAG | Reliable RAG |
|-------------|-------------|
| 1 generation call | 1 generation + 1 extraction + N verification calls |

With N=5 claims and k=3, Reliable RAG runs 7 LLM calls vs. 1. Each verification call is short (a claim + context, scored as one of four options), so they're fast and inexpensive.

---

## Beyond Hallucination: A Forcing Function for Better Source Documents

Reliable RAG has a productive side effect: it highlights gaps in your knowledge base. If certain query types consistently produce high hallucination rates, it means your corpus doesn't contain sufficient information for those queries. This is *actionable intelligence* — add more source documents covering those topics and watch the hallucination rate drop.

---

## When to Use Reliable RAG

**Best for:**
- Medical, legal, financial, or compliance applications where false statements have real consequences
- Enterprise systems where the LLM may confabulate policy details
- Public-facing systems where reputational risk from hallucinations is high
- Audit-required environments requiring traceable answer verification

**Skip when:**
- Your corpus is broad and queries are general knowledge (most claims will be "irrelevant")
- Latency is critical and 7× LLM overhead is unacceptable
- Cost per query is tightly constrained

---

## Summary

Reliable RAG treats answer generation not as the final step but as a draft requiring verification. By decomposing answers into claims, checking each claim against retrieved context, and computing a hallucination rate, it makes the RAG system's groundedness explicit and measurable.

For high-stakes applications, this transforms RAG from "trustworthy by assumption" to "trustworthy by proof." Users don't just receive answers — they receive verified answers, with a clear indication when verification fails.

In a world where LLM hallucination is still an unsolved problem, building an explicit verification layer like Reliable RAG is the responsible choice for production deployments where accuracy matters.
