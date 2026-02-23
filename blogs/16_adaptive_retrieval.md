# Adaptive Retrieval: One RAG System, Four Retrieval Strategies

> **Technique:** Adaptive Retrieval RAG  
> **Complexity:** Advanced  
> **Key Libraries:** `openai`, `faiss-cpu`

---

## Introduction

A factual question and an analytical question are fundamentally different retrieval tasks. "When was the WHO founded?" needs pinpoint precision — find the one chunk containing that date. "How does climate change affect ecosystems?" needs diversity — find chunks covering terrestrial ecosystems, marine ecosystems, species extinction, food chains, and multiple time scales.

Standard RAG applies the same retrieval strategy regardless of query type. That's like using the same tool for every job. **Adaptive Retrieval RAG** classifies each incoming query into one of four types and routes it to a specialized retrieval strategy purpose-built for that query's nature.

The result is a system that's simultaneously precise when precision is needed and comprehensive when breadth is required — all from the same index.

---

## The Four Query Types and Their Strategies

### 1. FACTUAL — "What is the greenhouse effect?"
*Needs*: Precision. Find the exact chunk with the answer.

**Strategy:**
1. LLM rewrites the query for better retrieval precision
2. Vector search retrieves 2k candidates (over-fetch)
3. LLM scores each candidate 1-10 for relevance
4. Return top-k by score

### 2. ANALYTICAL — "How does climate change affect marine and terrestrial ecosystems?"
*Needs*: Diversity. Different sub-aspects of the question need different source chunks.

**Strategy:**
1. LLM generates k sub-questions covering different facets
2. Vector search retrieves docs for each sub-question
3. LLM selects the most diverse+relevant subset across all sub-queries

### 3. OPINION — "What are different views on nuclear energy?"
*Needs*: Balance. Multiple perspectives, not just the most popular one.

**Strategy:**
1. LLM identifies k distinct viewpoints on the topic
2. Vector search retrieves docs for each viewpoint
3. LLM selects the most representative diverse set of perspectives

### 4. CONTEXTUAL — "How should I prepare for climate change?"
*Needs*: Personalization. The answer depends on the user's specific situation.

**Strategy:**
1. LLM reformulates the query incorporating user context (`user_context` parameter)
2. Vector search retrieves 2k candidates
3. LLM ranks considering both relevance AND user context

---

## Code Deep Dive

### Query Classification

```python
class QueryClassifier:
    def classify(self, query: str) -> QueryType:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the query into exactly one category. Return JSON with "
                    "key 'category' set to one of: Factual, Analytical, Opinion, Contextual.\n"
                    "- Factual: seeks specific, verifiable information\n"
                    "- Analytical: requires comprehensive analysis or explanation\n"
                    "- Opinion: about subjective matters or seeking diverse viewpoints\n"
                    "- Contextual: depends on user-specific context or situation"
                ),
            },
            {"role": "user", "content": f"Classify: {query}"},
        ]
        result = self.llm.chat_json(messages, schema={
            "type": "object",
            "properties": {"category": {
                "type": "string",
                "enum": ["Factual", "Analytical", "Opinion", "Contextual"]
            }},
            "required": ["category"]
        })
        return QueryType(result.get("category", "Factual"))
```

JSON schema validation is used to constrain the output to valid categories, preventing parsing failures.

### Factual Strategy (Precision-Focused)

```python
class FactualStrategy(BaseStrategy):
    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        # Step 1: Enhance query for precision
        messages = [
            {"role": "system", "content": "Rewrite this factual query for better information retrieval. Return only the enhanced query."},
            {"role": "user", "content": query},
        ]
        enhanced = self.llm.chat(messages)
        
        # Step 2: Over-fetch candidates
        results = self._search(enhanced, k=k * 2)
        
        # Step 3: LLM-rank by relevance
        scored = []
        for r in results:
            rank_msgs = [
                {"role": "system", "content": "Rate document relevance to the query on 1-10. Return JSON with key 'score'."},
                {"role": "user", "content": f"Query: {enhanced}\nDocument: {r.document.content[:1000]}"},
            ]
            score = float(self.llm.chat_json(rank_msgs).get("score", 5))
            scored.append((r.document.content, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:k]]
```

### Analytical Strategy (Diversity-Focused)

```python
class AnalyticalStrategy(BaseStrategy):
    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        # Step 1: Generate sub-questions covering different aspects
        messages = [
            {"role": "system", "content": f"Generate {k} sub-questions covering different aspects. Return JSON with key 'sub_queries'."},
            {"role": "user", "content": query},
        ]
        sub_queries = self.llm.chat_json(messages).get("sub_queries", [query])
        
        # Step 2: Retrieve for each sub-query
        all_docs = []
        for sq in sub_queries:
            results = self._search(sq, k=2)
            all_docs.extend(results)
        
        # Step 3: Select most diverse+relevant subset
        docs_text = "\n".join(f"{i}: {r.document.content[:80]}..." 
                              for i, r in enumerate(all_docs))
        select_msgs = [
            {"role": "system", "content": f"Select the {k} most diverse and relevant documents. Return JSON with key 'indices'."},
            {"role": "user", "content": f"Query: {query}\nDocuments:\n{docs_text}"},
        ]
        indices = self.llm.chat_json(select_msgs).get("indices", list(range(min(k, len(all_docs)))))
        
        return [all_docs[i].document.content for i in indices[:k]]
```

The "diversity" selection is key: the LLM is explicitly asked to pick documents that cover *different aspects*, not just the top-k most similar ones. This ensures the analytical answer has breadth.

### Contextual Strategy (Personalization-Focused)

```python
class ContextualStrategy(BaseStrategy):
    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        user_context = kwargs.get("user_context", "No specific context provided")
        
        # Step 1: Reformulate query with user context
        messages = [
            {"role": "system", "content": "Reformulate this query to best address the user's needs given their context."},
            {"role": "user", "content": f"Context: {user_context}\nQuery: {query}"},
        ]
        contextualized = self.llm.chat_json(messages).get("reformulated_query", query)
        
        # Step 2: Retrieve broadly
        results = self._search(contextualized, k=k * 2)
        
        # Step 3: Rank considering user context
        scored = []
        for r in results:
            rank_msgs = [
                {"role": "system", "content": "Rate relevance considering both query and user context (1-10). Return JSON with key 'score'."},
                {"role": "user", "content": f"Query: {contextualized}\nContext: {user_context}\nDocument: {r.document.content[:1000]}"},
            ]
            score = float(self.llm.chat_json(rank_msgs).get("score", 5))
            scored.append((r.document.content, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:k]]
```

The user context is threaded through both query reformulation and candidate ranking. "How should I prepare for climate change?" answered for a coastal farmer is very different from the same query answered for an urban office worker.

### The Adaptive Router

```python
class AdaptiveRetriever:
    def retrieve(self, query: str, user_context: Optional[str] = None) -> Tuple[List[str], QueryType]:
        # Classify
        query_type = self.classifier.classify(query)
        
        # Route to appropriate strategy
        strategy = self._strategies.get(query_type, self._strategies[QueryType.FACTUAL])
        contexts = strategy.retrieve(query, k=self.k, user_context=user_context)
        
        return contexts, query_type
```

Routing is a simple dictionary lookup after classification. The `QueryType` enum keys directly map to strategy instances.

---

## Strategy Performance by Query Type

| Query Type | Standard RAG | Adaptive RAG |
|-----------|-------------|--------------|
| Factual (precision needed) | Good (if chunk boundaries align) | **Excellent** (LLM-ranked) |
| Analytical (breadth needed) | Poor (top-k are often similar) | **Excellent** (sub-query diversity) |
| Opinion (balance needed) | Very poor (popularity bias) | **Excellent** (viewpoint targeting) |
| Contextual (personalization) | Poor (ignores user context) | **Excellent** (context-aware ranking) |

---

## LLM Call Overhead Per Strategy

| Strategy | Extra LLM calls (vs standard) |
|----------|-------------------------------|
| Factual | 1 (query enhance) + 2k (scoring) |
| Analytical | 1 (sub-queries) + 1 (selection) |
| Opinion | 1 (viewpoints) + 1 (selection) |
| Contextual | 1 (reformulate) + 2k (ranking) |

Factual and Contextual are heavier (per-candidate scoring). Analytical and Opinion are lighter (one selection call over all candidates).

---

## Usage

```python
rag = AdaptiveRetrievalRAG(file_path="document.pdf", k=3)

# Adaptive retrieval — strategy is chosen automatically
answer, contexts = rag.query("What is the greenhouse effect?")

# Explicitly pass user context for contextual queries
answer, contexts = rag.query(
    "How should I prepare for climate change?",
    user_context="I live in a coastal city and work in agriculture"
)
```

---

## When to Use Adaptive Retrieval

**Best for:**
- Systems serving diverse user populations with varied query patterns
- Knowledge bases where users ask a mix of factual, research, and opinion questions
- Applications with rich user profiles that can supply contextual metadata

**Skip when:**
- Query types are homogeneous (all factual or all analytical)
- The extra LLM calls per query are cost-prohibitive
- Simplicity is valued and standard RAG performs adequately for your use case

---

## Summary

Adaptive Retrieval is the "specialist team" model for RAG: rather than one generalist retriever handling every query, route each query to the specialist best suited for it. The four specialized strategies — precision LLM-ranking for factual, diversity-seeking sub-queries for analytical, viewpoint targeting for opinion, and context-aware reformulation for contextual — each excel where standard retrieval falls short.

For systems that genuinely serve heterogeneous user queries, adaptive retrieval is one of the most powerful architectural patterns available — using LLM intelligence not just for generation but for the retrieval process itself.
