"""
Adaptive Retrieval RAG

A RAG system that classifies each query into one of four types and applies
a different retrieval strategy for each. Different questions need different
approaches — a factual lookup is nothing like an opinion-seeking question.

Four query types and their strategies:

    FACTUAL — "What is the greenhouse effect?"
        1. LLM enhances query for precision
        2. Retrieve kX2 candidates
        3. LLM scores each candidate's relevance (1-10)
        4. Return top-k by score

    ANALYTICAL — "How does climate change affect ecosystems?"
        1. LLM generates sub-queries covering different aspects
        2. Retrieve docs for each sub-query
        3. LLM selects most diverse+relevant subset
        4. Return diverse set

    OPINION — "What are different views on nuclear energy?"
        1. LLM identifies distinct viewpoints on the topic
        2. Retrieve docs for each viewpoint
        3. LLM selects most representative diverse opinions
        4. Return balanced set

    CONTEXTUAL — "How should I prepare for climate change?"
        1. LLM incorporates user context into the query
        2. Retrieve with contextualized query
        3. LLM ranks considering both relevance and user context
        4. Return context-aware results

Why adaptive matters:
    Standard RAG uses the same retrieval for every query type.
    But "What year was X founded?" needs precision (factual),
    while "Analyze the causes of Y" needs breadth (analytical).
    One-size-fits-all leaves performance on the table.

Usage:
    from adaptive_retrieval_rag import AdaptiveRetrievalRAG

    rag = AdaptiveRetrievalRAG(file_path="document.pdf")
    answer, strategy = rag.query("What is the greenhouse effect?")
"""


import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Literal
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    chunk_text,
)


# Query Types
class QueryType(str, Enum):
    FACTUAL = "Factual"
    ANALYTICAL = "Analytical"
    OPINION = "Opinion"
    CONTEXTUAL = "Contextual"

# Query Classifier
class QueryClassifier:
    """
    Classifies queries into one of four types using LLM.

    Uses JSON mode for reliable structured output.

    Args:
        model_name:   OpenAI model for classification.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=0.0,
            max_tokens=50,
        )

    def classify(self, query: str) -> QueryType:
        """
        Classify a query into Factual, Analytical, Opinion, or Contextual.

        Args:
            query:  User's question.

        Returns:
            QueryType enum value.
        """
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

        try:
            result = self.llm.chat_json(messages, schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Factual", "Analytical", "Opinion", "Contextual"],
                    }
                },
                "required": ["category"]
            })

            category = result.get("category", "Factual")
            return QueryType(category)
        
        except (ValueError, KeyError):
            return QueryType.FACTUAL 


# Base Retrieval Strategy
class BaseStrategy:
    """
    Base class for all retrieval strategies.


    Provides shared vector store and LLM access.
    Each subclass overrides retrieve() with its specific logic.
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        vector_store: FAISSVectorStore,
        llm: OpenAIChat,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def _search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """Basic vector search helper."""
        query_emb = self.embedder.embed_text(query)
        return self.vector_store.search(query_emb, k=k)

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        """Override in subclasses."""
        results = self._search(query, k)
        return [r.document.content for r in results]



# Factual Strategy
class FactualStrategy(BaseStrategy):
    """
    For factual queries: enhance query → retrieve kX2 → LLM-rank by relevance.

    Steps:
        1. LLM rewrites query for better precision
        2. Retrieve kX2 candidates (over-fetch)
        3. LLM scores each on 1-10 relevance scale
        4. Return top-k by score

    """

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        print(f"  [Factual] Enhancing query...")

        messages = [
            {"role": "system", "content": "Rewrite this factual query for better information retrieval. Return only the enhanced query."},
            {"role": "user", "content": query},
        ]
        enhanced = self.llm.chat(messages)
        print(f"  [Factual] Enhanced: {enhanced[:100]}...")

        results = self._search(enhanced, k=k * 2)

        print(f"  [Factual] Ranking {len(results)} candidates...")
        scored = []
        for r in results:
            rank_msgs = [
                {"role": "system", "content": "Rate document relevance to the query on 1-10. Return JSON with key 'score'."},
                {"role": "user", "content": f"Query: {enhanced}\nDocument: {r.document.content[:1000]}\nRelevance score:"},
            ]
            try:
                score = float(self.llm.chat_json(rank_msgs, schema={"type": "object", "properties": {"score": {"type": "number", "min": 1, "max": 10}}, "required": ["score"]}).get("score", 5))
            except (ValueError, TypeError):
                score = 5.0
            scored.append((r.document.content, score))  

        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:k]]


# Analytical Strategy
class AnalyticalStrategy(BaseStrategy):
    """
    For analytical queries: generate sub-queries → retrieve for each → select diverse set.

    Steps:
        1. LLM generates k sub-questions covering different aspects
        2. Retrieve docs for each sub-query
        3. LLM selects most diverse+relevant subset
    """

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        print(f"  [Analytical] Generating sub-queries...")

        messages = [
            {"role": "system", "content": f"Generate {k} sub-questions that cover different aspects of the query. Return JSON with key 'sub_queries' as a list."},
            {"role": "user", "content": query},
        ]
        try:
            sub_queries = self.llm.chat_json(messages, schema={"type": "object", "properties": {"sub_queries": {"type": "list", "items": {"type": "string"}}}, "required": ["sub_queries"]}).get("sub_queries", [query])
        except Exception:
            sub_queries = [query]

        print(f"  [Analytical] Sub-queries: {sub_queries}")

        all_docs = []
        for sq in sub_queries:
            results = self._search(sq, k=2)
            all_docs.extend(results)

        if not all_docs:
            return []

        docs_text = "\n".join(
            f"{i}: {r.document.content[:80]}..."
            for i, r in enumerate(all_docs)
        )
        select_msgs = [
            {"role": "system", "content": f"Select the {k} most diverse and relevant documents. Return JSON with key 'indices' as a list of integers. indices should be between 0 and {len(all_docs) - 1}, both are inclusive."},
            {"role": "user", "content": f"Query: {query}\nDocuments:\n{docs_text}"},
        ]
        try:
            indices = self.llm.chat_json(select_msgs, schema={"type": "object", "properties": {"indices": {"type": "list", "items": {"type": "integer"}}}, "required": ["indices"]}).get("indices", list(range(min(k, len(all_docs)))))
        except Exception:
            indices = list(range(min(k, len(all_docs))))

        print(f"  [Analytical] Selected {len(indices)} diverse documents")
        indices = indices[:min(k, len(indices))]
        return [
            all_docs[i].document.content
            for i in indices
        ]



# Opinion Strategy
class OpinionStrategy(BaseStrategy):
    """
    For opinion queries: identify viewpoints → retrieve per viewpoint → select diverse opinions.

    Steps:
        1. LLM identifies distinct viewpoints on the topic
        2. Retrieve docs for each viewpoint
        3. LLM selects most representative diverse set

    """

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        print(f"  [Opinion] Identifying viewpoints...")

        messages = [
            {"role": "system", "content": f"Identify {k} distinct viewpoints or perspectives on this topic. Return JSON with key 'viewpoints' as a list of short descriptions."},
            {"role": "user", "content": query},
        ]
        try:
            viewpoints = self.llm.chat_json(messages, schema={"type": "object", "properties": {"viewpoints": {"type": "list", "items": {"type": "string"}}}, "required": ["viewpoints"]}).get("viewpoints", [query])
        except Exception:
            viewpoints = [query]

        print(f"  [Opinion] Viewpoints: {viewpoints}")

        all_docs = []
        for vp in viewpoints:
            results = self._search(f"{query} {vp}", k=2)
            all_docs.extend(results)

        if not all_docs:
            return []

        docs_text = "\n".join(
            f"{i}: {r.document.content[:100]}..."
            for i, r in enumerate(all_docs)
        )
        select_msgs = [
            {"role": "system", "content": f"Select {k} documents representing the most diverse viewpoints. Return JSON with key 'indices' as a list of integers."},
            {"role": "user", "content": f"Query: {query}\nDocuments:\n{docs_text}"},
        ]
        try:
            indices = self.llm.chat_json(select_msgs, schema={"type": "object", "properties": {"indices": {"type": "list", "items": {"type": "integer"}}}, "required": ["indices"]}).get("indices", list(range(min(k, len(all_docs)))))
        except Exception:
            indices = list(range(min(k, len(all_docs))))

        print(f"  [Opinion] Selected {len(indices)} diverse opinions")
        indices = indices[:min(k, len(indices))]
        return [
            all_docs[i].document.content
            for i in indices
        ]




# Contextual Strategy

class ContextualStrategy(BaseStrategy):
    """
    For contextual queries: incorporate user context → retrieve → context-aware ranking.

    Steps:
        1. LLM reformulates query with user context
        2. Retrieve kX2 candidates
        3. LLM ranks considering both relevance and user context

    """

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[str]:
        user_context = kwargs.get("user_context", "No specific context provided")
        print(f"  [Contextual] Incorporating user context...")

        messages = [
            {"role": "system", "content": "Reformulate this query to best address the user's needs given their context. Return only the reformulated query. if no context is provided, return the original query."},
            {"role": "user", "content": f"Context: {user_context}\nQuery: {query}"},
        ]
        try:
            contextualized = self.llm.chat_json(messages, schema={"type": "object", "properties": {"reformulated_query": {"type": "string"}}, "required": ["reformulated_query"]}).get("reformulated_query", query)
        except Exception:
            contextualized = query

        print(f"  [Contextual] Reformulated: {contextualized[:100]}...")

        results = self._search(contextualized, k=k * 2)

        print(f"  [Contextual] Ranking {len(results)} candidates with context...")
        scored = []
        for r in results:
            rank_msgs = [
                {"role": "system", "content": "Rate document relevance considering both the query and user context (1-10). Return JSON with key 'score'."},
                {"role": "user", "content": f"Query: {contextualized}\nContext: {user_context}\nDocument: {r.document.content[:1000]}"},
            ]
            try:
                score = float(self.llm.chat_json(rank_msgs).get("score", 5))
            except (ValueError, TypeError):
                score = 5.0
            scored.append((r.document.content, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:k]]



# Adaptive Retriever - Routes to the right strategy
class AdaptiveRetriever:
    """
    Routes queries to the appropriate retrieval strategy based on classification.

    Args:
        embedding_model:  OpenAI embedding model.
        strategy_model:   OpenAI model used by strategies (enhancement, ranking, etc).
        classifier_model: OpenAI model for query classification.
        chunk_size:       Characters per chunk.
        chunk_overlap:    Overlap between chunks.
        k:                Default number of results.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        strategy_model: str = "gpt-4o-mini",
        classifier_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.llm = OpenAIChat(model_name=strategy_model, temperature=0.0, max_tokens=1000)
        self.classifier = QueryClassifier(model_name=classifier_model)

        self._strategies: Dict[QueryType, BaseStrategy] = {}

    def _init_strategies(self):
        """Initialize strategies with shared components."""
        shared = (self.embedder, self.vector_store, self.llm)
        self._strategies = {
            QueryType.FACTUAL: FactualStrategy(*shared),
            QueryType.ANALYTICAL: AnalyticalStrategy(*shared),
            QueryType.OPINION: OpinionStrategy(*shared),
            QueryType.CONTEXTUAL: ContextualStrategy(*shared),
        }

    def index_text(self, text: str, doc_id: str = "doc_0") -> int:
        """Chunk and index text."""
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        documents = [
            Document(content=c, metadata={"doc_id": doc_id, "chunk_index": i})
            for i, c in enumerate(chunks)
        ]
        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)
        self._init_strategies()
        return len(chunks)

    def index_pdf(self, file_path: str) -> int:
        text = read_pdf(file_path)
        return self.index_text(text, os.path.basename(file_path))

    def retrieve(self, query: str, user_context: Optional[str] = None) -> Tuple[List[str], QueryType]:
        """
        Classify query and route to appropriate strategy.

        Args:
            query:         User's question.
            user_context:  Optional user-specific context (for contextual queries).

        Returns:
            Tuple of (context_texts, query_type).
        """
        query_type = self.classifier.classify(query)
        print(f"  [Adaptive] Query classified as: {query_type.value}")

        strategy = self._strategies.get(query_type, self._strategies[QueryType.FACTUAL])
        contexts = strategy.retrieve(query, k=self.k, user_context=user_context)

        return contexts, query_type



# Complete Adaptive Retrieval RAG Pipeline

class AdaptiveRetrievalRAG:
    """
    Complete adaptive RAG pipeline that classifies queries and applies
    type-specific retrieval strategies.

    Usage:
        rag = AdaptiveRetrievalRAG(file_path="report.pdf")
        answer, query_type = rag.query("What is the greenhouse effect?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 3,
        embedding_model: str = "text-embedding-3-small",
        strategy_model: str = "gpt-4o-mini",
        classifier_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.file_path = file_path

        # Initialize adaptive retriever
        self.retriever = AdaptiveRetriever(
            embedding_model=embedding_model,
            strategy_model=strategy_model,
            classifier_model=classifier_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )

        # Initialize answer LLM
        self.chat = OpenAIChat(model_name=chat_model, temperature=temperature)

        # Index
        if file_path.endswith(".pdf"):
            num = self.retriever.index_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                num = self.retriever.index_text(f.read())

        print(f"[Adaptive] Indexed '{os.path.basename(file_path)}' → {num} chunks (k={k})")

    def query(
        self,
        question: str,
        user_context: Optional[str] = None,
        return_context: bool = True,
    ) -> Tuple[str, str]:
        """
        Query with adaptive strategy selection.

        Args:
            question:      User's question.
            user_context:  Optional user context (for contextual queries).
            return_context: Whether to return contexts.

        Returns:
            Tuple of (answer_string, query_type_string).
        """
        contexts, query_type = self.retriever.retrieve(question, user_context)

        if not contexts:
            return "No relevant information found.", query_type.value

        print(f"Found query type: {query_type.value}")
        answer = self.chat.chat_with_context(question, contexts)
        return answer, contexts

    def show_strategy(self, question: str, user_context: Optional[str] = None) -> None:
        """
        Debug helper: show classification and strategy details.

        Args:
            question:      Search query.
            user_context:  Optional user context.
        """
        print(f"\nQuery: {question}")
        print("=" * 70)

        contexts, query_type = self.retriever.retrieve(question, user_context)

        emoji_map = {
            QueryType.FACTUAL: "📌",
            QueryType.ANALYTICAL: "🔬",
            QueryType.OPINION: "💬",
            QueryType.CONTEXTUAL: "🎯",
        }
        emoji = emoji_map.get(query_type, "❓")

        print(f"\n  {emoji} Strategy: {query_type.value}")
        print(f"  Retrieved {len(contexts)} chunks:")
        print("-" * 50)
        for i, ctx in enumerate(contexts):
            preview = ctx[:200].replace('\n', ' ')
            print(f"  {i+1}. {preview}...")

        print("\n" + "=" * 70)


# Main
if __name__ == "__main__":

    pdf_path = r"data\Understanding_Climate_Change.pdf"

    rag = AdaptiveRetrievalRAG(
        file_path=pdf_path,
        chunk_size=1000,
        k=3,
    )

    # Test all four query types
    print("\n--- FACTUAL ---")
    ans, contexts = rag.query("What is the distance between the Earth and the Sun?")
    print(f"Answer: {ans}\n")
    print("Contexts:")
    for i, ctx in enumerate(contexts):
        print(f"  {i+1}. {ctx}")

    print("\n--- ANALYTICAL ---")
    ans, contexts = rag.query("How does climate change affect both marine and terrestrial ecosystems?")
    print(f"Answer: {ans}\n")
    print("Contexts:")
    for i, ctx in enumerate(contexts):
        print(f"  {i+1}. {ctx}")


    print("\n--- OPINION ---")
    ans, contexts = rag.query("What are different perspectives on nuclear energy as a climate solution?")
    print(f"Answer: {ans}\n")
    print("Contexts:")
    for i, ctx in enumerate(contexts):
        print(f"  {i+1}. {ctx}")


    print("\n--- CONTEXTUAL ---")
    ans, contexts = rag.query(
        "How should I prepare for climate change impacts?",
        user_context="I live in a coastal city and work in agriculture",
    )
    print(f"Answer: {ans}\n")
    print("Contexts:")
    for i, ctx in enumerate(contexts):
        print(f"  {i+1}. {ctx}")

    # Interactive loop
    print("\n[Adaptive RAG] Ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() == "exit":
            break

        answer, contexts = rag.query(question)
        print(f"\nAnswer: {answer}\n")
        print("Contexts:")
        for i, ctx in enumerate(contexts):
            print(f"  {i+1}. {ctx}")
