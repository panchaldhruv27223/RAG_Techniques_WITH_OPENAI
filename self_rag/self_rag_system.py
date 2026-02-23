"""

Self-RAG: Self-Reflective Retrieval-Augmented Generation

Self-RAG dynamically decides whether to retrieve, evaluates retrieved documents
for relevance, generates responses, then self-critiques them for support and
utility — selecting the best response through a multi-step evaluation pipeline.

Pipeline:
    1. Retrieval Decision  → Should we even retrieve? (yes/no)
    2. Document Retrieval   → Fetch top-k from vector store
    3. Relevance Evaluation → Filter irrelevant docs (relevant/irrelevant)
    4. Response Generation   → Generate response per relevant context
    5. Support Assessment    → Is response grounded in context? (fully/partially/no)
    6. Utility Evaluation    → How useful is the response? (1-5)
    7. Response Selection    → Pick best by support + utility score

Uses:
    - OpenAIChat.chat_json() for structured LLM outputs
    - RAGRetriever for vector search
    - FAISSVectorStore for embeddings

"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from helper_function_openai import (
    RAGRetriever,
    OpenAIChat,
    Document,
    RetrievalResult,
)


@dataclass
class SelfRAGResponse:
    """Complete Self-RAG response with evaluation metadata."""
    answer: str
    support: str
    utility: int
    context_used: List[str]
    retrieval_needed: bool
    total_docs_retrieved: int
    relevant_docs_count: int
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)



class SelfRAGOpenAI:
    """
    Self-Reflective RAG using pure OpenAI SDK.
    
    The system uses structured JSON outputs (chat_json) for each evaluation
    step, replacing LangChain's with_structured_output pattern.
    
    Args:
        file_path:      Path to PDF or CSV to index
        chunk_size:     Characters per chunk
        chunk_overlap:  Overlap between chunks
        top_k:          Number of documents to retrieve
        chat_model:     OpenAI model for generation + evaluation
        temperature:    LLM temperature (0.0 recommended for evaluation steps)
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize retriever (embedder + vector store)
        self.retriever = RAGRetriever(embedding_model="text-embedding-3-small")

        # Initialize LLM for all evaluation + generation steps
        self.llm = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
            max_tokens=5000,
        )

        # Index the document
        self._index_file()

    def _index_file(self):
        """Index the document into the vector store."""
        if self.file_path.endswith(".pdf"):
            count = self.retriever.index_pdf(
                self.file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.file_path.endswith(".csv"):
            count = self.retriever.index_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")

        print(f"[Self-RAG] Indexed {count} chunks from {os.path.basename(self.file_path)}")



    def _decide_retrieval(self, query: str) -> bool:
        """
        Determine if retrieval is necessary for this query.
        
        Some queries can be answered directly (e.g., "What is 2+2?")
        while others need document context (e.g., "What does the report say about X?").
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval decision system. "
                    "Determine if the query requires retrieving external documents to answer accurately. "
                    "Respond with JSON: {\"needs_retrieval\": true} or {\"needs_retrieval\": false}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}",
            },
        ]
        result = self.llm.chat_json(messages)
        return result.get("needs_retrieval", True)


    def _retrieve_documents(self, query: str) -> List[RetrievalResult]:
        """Retrieve top-k documents from vector store."""
        return self.retriever.retrieve(query, k=self.top_k)


    def _evaluate_relevance(self, query: str, context: str) -> str:
        """
        Evaluate whether a retrieved document is relevant to the query.
        
        Returns: 'relevant' or 'irrelevant'
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance evaluator. "
                    "Determine if the given context is relevant to answering the query. "
                    "Respond with JSON: {\"relevance\": \"relevant\"} or {\"relevance\": \"irrelevant\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext: {context}",
            },
        ]
        result = self.llm.chat_json(messages)
        return result.get("relevance", "irrelevant").strip().lower()



    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response grounded in the provided context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question based on the provided context. "
                    "Be concise and accurate. If the context doesn't contain enough information, "
                    "say so rather than making up information."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]
        return self.llm.chat(messages)



    def _assess_support(self, response: str, context: str) -> str:
        """
        Evaluate how well the generated response is supported by the context.
        
        Returns: 'fully_supported', 'partially_supported', or 'no_support'
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a support assessment system. "
                    "Determine how well the response is supported by the given context. "
                    "Respond with JSON: {\"support\": \"fully_supported\"} or "
                    "{\"support\": \"partially_supported\"} or {\"support\": \"no_support\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nResponse: {response}",
            },
        ]
        result = self.llm.chat_json(messages)
        return result.get("support", "no_support").strip().lower()



    def _evaluate_utility(self, query: str, response: str) -> int:
        """
        Rate the utility of the response for answering the query.
        
        Returns: Integer score from 1 to 5
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a utility evaluator. "
                    "Rate how useful the response is for answering the query. "
                    "Respond with JSON: {\"utility\": <score>} where score is 1-5. "
                    "1=useless, 2=poor, 3=acceptable, 4=good, 5=excellent"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nResponse: {response}",
            },
        ]
        result = self.llm.chat_json(messages)
        score = result.get("utility", 3)
        # Clamp to valid range
        return max(1, min(5, int(score)))


    @staticmethod
    def _select_best(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best response based on support level and utility score.
        
        Priority: fully_supported > partially_supported > no_support
        Tiebreaker: higher utility score
        """
        support_priority = {
            "fully_supported": 3,
            "partially_supported": 2,
            "no_support": 1,
        }
        return max(
            candidates,
            key=lambda c: (support_priority.get(c["support"], 0), c["utility"]),
        )


    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        Run the Self-RAG pipeline.
        
        Args:
            question: User's query
            
        Returns:
            Tuple of (answer_string, context_list)
        """
        result = self.self_rag(question)
        return result.answer, result.context_used

    def self_rag(self, query: str) -> SelfRAGResponse:
        """
        Full Self-RAG pipeline with detailed response metadata.
        
        Args:
            query: User's question
            
        Returns:
            SelfRAGResponse with answer, support, utility, and evaluation details
        """
        print(f"\n{'='*60}")
        print(f"  Self-RAG Pipeline")
        print(f"  Query: {query}")
        print(f"{'='*60}")

        print("\n[Step 1] Retrieval Decision...")
        needs_retrieval = self._decide_retrieval(query)
        print(f"  → Needs retrieval: {needs_retrieval}")

        if not needs_retrieval:

            print("\n[Generate] Answering without retrieval...")
            answer = self._generate_response(query, "No retrieval necessary.")

            return SelfRAGResponse(
                answer=answer,
                support="no_context",
                utility=3,
                context_used=[],
                retrieval_needed=False,
                total_docs_retrieved=0,
                relevant_docs_count=0,
            )


        print(f"\n[Step 2] Retrieving top-{self.top_k} documents...")
        results = self._retrieve_documents(query)
        contexts = [(r.document.content, r.score) for r in results]
        print(f"  → Retrieved {len(contexts)} documents")


        print(f"\n[Step 3] Evaluating relevance...")
        relevant_contexts = []
        for i, (context, score) in enumerate(contexts):
            relevance = self._evaluate_relevance(query, context)
            print(f"  Doc {i+1} (sim={score:.3f}): {relevance}")
            if relevance == "relevant":
                relevant_contexts.append(context)

        print(f"  → {len(relevant_contexts)}/{len(contexts)} documents are relevant")

        if not relevant_contexts:
            print("\n[Generate] No relevant contexts found. Answering without context...")
            answer = self._generate_response(query, "No relevant context found.")
            return SelfRAGResponse(
                answer=answer,
                support="no_context",
                utility=3,
                context_used=[],
                retrieval_needed=True,
                total_docs_retrieved=len(contexts),
                relevant_docs_count=0,
            )

        print(f"\n[Steps 4-6] Generate, assess support, evaluate utility...")
        candidates = []

        for i, context in enumerate(relevant_contexts):
            print(f"\n  ── Candidate {i+1}/{len(relevant_contexts)} ──")

            response = self._generate_response(query, context)
            print(f"  [Generate] Response: {response[:80]}...")

            support = self._assess_support(response, context)
            print(f"  [Support]  {support}")

            utility = self._evaluate_utility(query, response)
            print(f"  [Utility]  {utility}/5")

            candidates.append({
                "response": response,
                "support": support,
                "utility": utility,
                "context": context,
            })

        print(f"\n[Step 7] Selecting best response from {len(candidates)} candidates...")
        best = self._select_best(candidates)
        print(f"  → Winner: support={best['support']}, utility={best['utility']}/5")

        return SelfRAGResponse(
            answer=best["response"],
            support=best["support"],
            utility=best["utility"],
            context_used=[best["context"]],
            retrieval_needed=True,
            total_docs_retrieved=len(contexts),
            relevant_docs_count=len(relevant_contexts),
            all_candidates=candidates,
        )


if __name__ == "__main__":
    pdf_file_path = r"data\Understanding_Climate_Change.pdf"

    self_rag_openai = SelfRAGOpenAI(pdf_file_path)

    question = "What are the main causes of climate change as per this document?"
    answer, context = self_rag_openai.query(question)
    print(f"Answer: {answer}")
    print(f"Context: {context}")