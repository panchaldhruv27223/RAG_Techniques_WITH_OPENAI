# Query Transformations: Fixing the Query Before It Hits the Index

---

## Introduction

Most RAG systems treat the user's query as sacred — whatever the user typed is embedded as-is and used to search the vector store. This works well when user queries are precise, well-formed, and vocabulary-matched to the indexed documents. It fails badly when queries are vague, colloquial, too narrow, or too complex.

**Query Transformations** are a pre-retrieval intervention: use the LLM to rewrite, generalize, or decompose the user's query *before* embedding it and searching the vector store. The transformation produces a better retrieval query — one that more closely matches the vocabulary, specificity, and structure of the indexed content.

Three complementary strategies:

1. **Rewrite**: Make the query more specific and document-aligned
2. **Step-Back**: Generalize the query to retrieve broader background context
3. **Decompose**: Split a complex multi-part query into focused sub-queries

---

## Why Query Transformation Works

### The Vocabulary Mismatch Problem

A user asks: "why is my code slow?"

The indexed document contains: "Computational complexity analysis and performance optimization strategies for algorithmic bottleneck identification."

In embedding space, "why is my code slow" and "computational complexity analysis" are not as close as they should be — different vocabulary registers for the same concept.

A rewritten query: "methods for identifying and resolving software performance bottlenecks and computational efficiency issues" — lands much closer in embedding space to the technical documentation vocabulary.

### The Specificity-Recall Tradeoff

Some queries are too specific for retrieval:
- "What is the exact market cap of NVIDIA as of Q3 2024?" — Very specific. Only documents with that exact figure will match well.
- Rewritten: "NVIDIA financial performance market capitalization 2024" — matches broader relevance criteria.

Some queries are too vague:
- "Tell me about machine learning" — Matches everything tangentially.
- Decomposed into: ["What is supervised learning?", "What is unsupervised learning?", "What are the main machine learning algorithms?"] — Each sub-query retrieves targeted context.

---

## Strategy 1: Query Rewrite

Produces a single, improved query replacing the original:

```python
def rewrite_query(llm: OpenAIChat, original_query: str) -> str:
    """
    Reformulate the query to be more specific, document-aligned, and
    likely to retrieve relevant information from a knowledge base.
    
    Key transformations the LLM typically applies:
    - Replaces colloquial terms with technical vocabulary
    - Adds specificity where the original is vague
    - Removes ambiguous pronouns and references
    - Expands abbreviations and acronyms
    - Clarifies implied intent
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with reformulating user queries "
                "to improve retrieval in a RAG system. Given the original query, "
                "rewrite it to be more specific, detailed, and likely to retrieve "
                "relevant information."
            )
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}\n\nRewritten query:"
        }
    ]
    return llm.chat(messages)
```

### Rewrite Examples

| Original | Rewritten |
|----------|-----------|
| "why is inflation so high?" | "economic causes and contributing factors to elevated inflation rates and consumer price increases" |
| "how do I fix the bug?" | "software debugging techniques and common error resolution strategies for code defects" |
| "what did they say about climate?" | "climate change assessment findings and key statements from the document" |
| "explain AI" | "overview of artificial intelligence concepts, machine learning methodologies, and neural network applications" |

**Best for**: Vague, colloquial, or ambiguous queries where the user's vocabulary doesn't match the document vocabulary.

---

## Strategy 2: Step-Back Query

Generalize the specific query into a broader base question to retrieve foundational context:

```python
def generate_step_back_query(llm: OpenAIChat, original_query: str) -> str:
    """
    Generate a "step-back" query — a more general question whose answer
    provides the background context needed to properly answer the original.
    
    Step-Back RAG (from Google DeepMind paper, 2023):
    "Prompting to Learn from Abstracted Principles"
    
    Insight: Answering a specific question is easier when you first retrieve
    the relevant general principles. The step-back query retrieves those principles.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with generating broader, more general "
                "queries to improve context retrieval in a RAG system. Given the original "
                "query, generate a step back query that is more general and can help "
                "retrieve relevant background information."
            )
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}\n\nStep-Back query:"
        }
    ]
    return llm.chat(messages)
```

### Step-Back Examples

| Original (specific) | Step-Back (general) |
|--------------------|--------------------|
| "What is the effective dose of metformin for HbA1c reduction in elderly diabetics?" | "How does metformin work to treat Type 2 diabetes?" |
| "What was Apple's revenue impact from iPhone sales in Q4 2023?" | "How does Apple's revenue depend on iPhone product cycles?" |
| "How do I resolve a deadlock in PostgreSQL transaction isolation?" | "What are database transaction isolation levels and concurrent access patterns?" |

**Why step-back helps**: The original query needs specific context, but the ideal retrieval chunks are in background/explainer sections. The step-back query retrieves the conceptual foundation that makes the specific answer interpretable.

**Two-phase step-back RAG** (most powerful variant):

```python
def query_with_step_back(self, question: str) -> str:
    """Retrieve both background (step-back) and specific (direct) context."""
    
    # Phase 1: Step-back → retrieve background concepts
    step_back = generate_step_back_query(self.llm, question)
    background_context = self.rag_retriever.retrieve_context(step_back)
    
    # Phase 2: Direct query → retrieve specific facts
    specific_context = self.rag_retriever.retrieve_context(question)
    
    # Combine: background provides framing, specific provides the answer
    all_context = background_context + specific_context
    return self.llm.chat_with_context(question, all_context)
```

---

## Strategy 3: Query Decomposition

Break a complex, multi-aspect query into focused sub-queries and aggregate results:

```python
def decompose_query(llm: OpenAIChat, original_query: str) -> List[str]:
    """
    Decompose a complex query into 2-4 simpler sub-queries.
    
    Each sub-query retrieves targeted context for one aspect of the complex question.
    The sub-query answers are then synthesized into a comprehensive response.
    
    This is particularly powerful for:
    - Multi-hop queries (A causes B, which causes C?)
    - Comparison queries (compare X and Y on dimensions 1, 2, 3)
    - Multi-aspect analyses (economic, social, and environmental impacts of X)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with breaking down complex queries into "
                "simpler sub-queries for a RAG system. Given the original query, decompose "
                "it into 2-4 simpler sub-queries that, when answered together, would provide "
                "a comprehensive response to the original query.\n\n"
                'Respond with JSON: {"sub_queries": ["query1", "query2", ...]}'
            )
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}"
        }
    ]
    result = llm.chat_json(messages)
    return result.get("sub_queries", [])
```

### Decomposition Examples

| Complex Query | Decomposed Sub-Queries |
|--------------|----------------------|
| "What are the economic, social, and environmental impacts of climate change?" | 1. "What are the economic costs of climate change on GDP and industry?" 2. "How does climate change affect human health and displacement?" 3. "What are the environmental and ecological impacts of climate change?" |
| "Compare SQL and NoSQL databases for use in large-scale web applications" | 1. "What are the strengths of SQL databases for web applications?" 2. "What are the strengths of NoSQL databases for web applications?" 3. "How do SQL and NoSQL databases compare in terms of scalability?" |

### The Decomposition Pipeline

```python
def query(self, question: str, method: str = "decompose") -> Tuple[str, List]:
    if method == "decompose":
        sub_queries = decompose_query(self.llm, question)
        
        results = []
        all_contexts = []
        
        for sub_query in sub_queries:
            # Each sub-query retrieves focused context
            context = self.rag_retriever.retrieve_context(sub_query)
            sub_answer = self.llm.chat_with_context(sub_query, context)
            results.append({"query": sub_query, "answer": sub_answer})
            all_contexts.append(context)
        
        # Final synthesis: combine sub-answers into one comprehensive response
        sub_answer_text = "\n\n".join([
            f"Query: {r['query']}\nAnswer: {r['answer']}" 
            for r in results
        ])
        final_answer = self.llm.chat_with_context(question, sub_answer_text)
        
        return final_answer, all_contexts
```

---

## Choosing a Strategy

| Signal | Best Strategy |
|--------|--------------|
| User query uses casual language; documents use technical terms | **Rewrite** |
| Query is too specific; answer requires background understanding | **Step-Back** |
| Query has multiple aspects or requires comparison | **Decompose** |
| Query fails to retrieve relevant results with standard RAG | **Rewrite → retry** |
| Query about a complex causal chain or multi-hop relationship | **Decompose** |
| Query about a specific event but needs broader context | **Step-Back** |

**Auto-routing** (production pattern):

```python
def auto_transform(self, query: str) -> Tuple[str, str]:
    """LLM-decides which transformation to apply."""
    routing_messages = [
        {
            "role": "system",
            "content": (
                "Given the query, choose the best transformation strategy:\n"
                "- REWRITE: query is vague, colloquial, or vocabulary-mismatched\n"
                "- STEP_BACK: query is very specific, context/background would help\n"
                "- DECOMPOSE: query has multiple distinct aspects\n"
                "- NONE: query is already precise and well-formed\n\n"
                'Return JSON: {"strategy": "REWRITE|STEP_BACK|DECOMPOSE|NONE", "reasoning": "..."}'
            )
        },
        {"role": "user", "content": f"Query: {query}"}
    ]
    result = self.llm.chat_json(routing_messages)
    return result.get("strategy", "NONE"), result.get("reasoning", "")
```

---

## Cost Analysis

| Strategy | LLM Calls | Retrieval Calls | Best For |
|----------|-----------|-----------------|---------|
| None (standard RAG) | 1 | 1 | Precise queries |
| Rewrite | 2 | 1 | Vague/colloquial queries |
| Step-Back | 2 | 2 | Context-needing queries |
| Decompose (3 sub) | 4 | 3 | Complex multi-aspect queries |
| Step-Back + Rewrite | 3 | 2 | Context + vocab mismatch |

Decomposition is the most expensive (up to 4+ LLM calls) but provides the most comprehensive retrieval for complex queries where different aspects are in different document regions.

---

## When to Use Query Transformations

Query Transformations earn their cost most clearly in systems with a diverse, non-expert user base who tend to phrase queries casually or colloquially, and in enterprise knowledge bases where user vocabulary differs meaningfully from document vocabulary. Research assistants are another strong fit — queries often span multiple concepts or require background context that the user hasn't phrased explicitly.

The technique is less beneficial when users are domain experts who already phrase queries with precise technical vocabulary, when latency budget is tight (each transformation adds roughly 0.5–1 second of LLM inference), or when the query distribution is narrow and predictable. In those cases, optimizing the index for the dominant query pattern will yield better returns than adding a pre-retrieval transformation layer.

---

## Summary

Query Transformations acknowledge that the bottleneck in many RAG failures isn't the retrieval algorithm — it's the query itself. By applying Rewrite, Step-Back, or Decompose transformations before search, the system bridges the gap between how users ask questions and how documents are written.

Each strategy targets a different query failure mode: Rewrite fixes vocabulary mismatch, Step-Back adds missing context, and Decompose handles multi-aspect complexity. Used together with an auto-routing classifier, they form a comprehensive pre-retrieval quality layer that improves retrieval for nearly all non-trivial queries.
