"""
Reranking RAG

Reranking adds a second-pass scoring step after initial vector search.
The initial retrieval (fast but imprecise) fetches many candidates, then
a more powerful model re-scores each candidate against the query to find
the truly most relevant chunks.

Why reranking matters — the "capital of France" problem:
    Vector search for "What is the capital of France?" returns:
        1. "The capital of France is great."      ← high similarity, WRONG answer
        2. "The capital of France is beautiful."   ← high similarity, WRONG answer
    After reranking with an LLM or cross-encoder:
        1. "I enjoyed my trip to Paris, France... great capital city." ← CORRECT
        2. "Have you visited Paris?... its capital with the Eiffel Tower" ← CORRECT
    
    Vector search matched keywords; reranking understood the INTENT.

Two reranking methods implemented:

    Method 1 — LLM Reranker:
        Uses OpenAI (gpt-4o-mini) to score each chunk's relevance 1-10.
        + Most accurate (LLM understands nuance)
        - Expensive (one LLM call per chunk)
        - Slow for many candidates

    Method 2 — Cross-Encoder Reranker:
        Uses a small local model (ms-marco-MiniLM) to score query-document pairs.
        + Fast (runs locally, no API calls)
        + Cheap (no token costs)
        - Less accurate than LLM for complex queries
        - Requires sentence-transformers dependency

Usage:
    from reranking_rag import RerankingRAG

    # LLM-based reranking (more accurate)
    rag = RerankingRAG(file_path="doc.pdf", rerank_method="llm")

    # Cross-encoder reranking (faster, cheaper)
    rag = RerankingRAG(file_path="doc.pdf", rerank_method="cross_encoder")

    answer, contexts = rag.query("What are the impacts on biodiversity?")
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass

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




# Method 1: LLM-Based Reranker
class LLMReranker:
    """
    Reranks chunks by asking an LLM to score each chunk's relevance (1-10)
    to the query.

    Args:
        model_name:   OpenAI model for scoring.
        temperature:  Should be 0 for consistent scoring.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=100,
        )



    def score_chunk(self, query: str, chunk_text: str) -> float:
        """
        Ask the LLM to rate a chunk's relevance to the query on a 1-10 scale.

        Args:
            query:       User's question.
            chunk_text:  The chunk to score.

        Returns:
            Relevance score (1.0 to 10.0).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance scoring system. Rate the relevance of a "
                    "document to a query on a scale of 1-10. Consider the specific "
                    "context and intent of the query, not just keyword matches. "
                    "Return JSON with key 'relevance_score' containing a number."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Document: {chunk_text}\n\n"
                    f"Rate relevance (1-10):"
                ),
            },
        ]

        try:

            result = self.llm.chat_json(messages)
            score = float(result.get("relevance_score", 0))
            return max(1.0, min(10.0, score))


        except (json.JSONDecodeError, ValueError, TypeError):

            try:
                raw = self.llm.chat(messages)
                import re
                numbers = re.findall(r'\d+\.?\d*', raw)
                if numbers:
                    return max(1.0, min(10.0, float(numbers[0])))
            except Exception:
                pass
            return 5.0 

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Score and rerank all chunks, return top_n.

        Args:
            query:   User's question.
            chunks:  List of chunk texts to rerank.
            top_n:   Number of top chunks to return.

        Returns:
            List of (chunk_text, score) tuples, sorted by score descending.
        """
        scored = []
        for i, chunk in enumerate(chunks):
            score = self.score_chunk(query, chunk)
            scored.append((chunk, score))
            print(f"    Chunk {i+1}/{len(chunks)}: score={score:.1f}")

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]




# Method 2: Cross-Encoder Reranker
class CrossEncoderReranker:
    """
    Reranks chunks using a local cross-encoder model.

    A cross-encoder takes a (query, document) pair as input and outputs
    a single relevance score. Unlike bi-encoders (which embed query and
    document separately), cross-encoders see both together — enabling
    deeper understanding of their relationship.

    The default model (ms-marco-MiniLM-L-6-v2) is small (~80MB),
    fast, and specifically trained for passage ranking.

    Args:
        model_name:  HuggingFace cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self._available = True
        except ImportError:
            print(
                "[CrossEncoder] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Score and rerank chunks using the cross-encoder.

        All (query, chunk) pairs are scored in a single batch call —
        much faster than N separate LLM calls.

        Args:
            query:   User's question.
            chunks:  List of chunk texts to rerank.
            top_n:   Number of top chunks to return.

        Returns:
            List of (chunk_text, score) tuples, sorted by score descending.
        """
        if not self._available:
            raise RuntimeError("Cross-encoder not available. Install sentence-transformers.")

        # Create query-document pairs
        pairs = [[query, chunk] for chunk in chunks]

        # Score all pairs in one batch
        scores = self.model.predict(pairs)

        # Pair chunks with scores and sort
        scored = list(zip(chunks, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]



# Reranking Retriever
class RerankingRetriever:
    """
    Retriever that performs initial vector search then reranks results.

    Pipeline:
        1. Index document → FAISS
        2. On query: vector search → top initial_k candidates
        3. Rerank candidates using LLM or cross-encoder
        4. Return top final_k reranked results

    Args:
        embedding_model:  OpenAI embedding model.
        rerank_method:    "llm" or "cross_encoder".
        reranker_model:   Model name for the reranker.
        chunk_size:       Characters per chunk.
        chunk_overlap:    Overlap between chunks.
        initial_k:        Number of candidates from vector search.
        final_k:          Number of results after reranking.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        rerank_method: Literal["llm", "cross_encoder"] = "llm",
        reranker_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        initial_k: int = 15,
        final_k: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.initial_k = initial_k
        self.final_k = final_k
        self.rerank_method = rerank_method

        # Core components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

        # Initialize reranker based on method
        if rerank_method == "llm":
            model = reranker_model or "gpt-4o-mini"
            self.reranker = LLMReranker(model_name=model)
        elif rerank_method == "cross_encoder":
            model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = CrossEncoderReranker(model_name=model)
        else:
            raise ValueError(f"Unknown rerank method: {rerank_method}")

    def index_document(self, text: str, doc_id: str = "doc_0") -> int:
        """Chunk and index a document in FAISS."""
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)
        return len(chunks)

    def index_pdf(self, file_path: str, doc_id: Optional[str] = None) -> int:
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        text = read_pdf(file_path)
        return self.index_document(text, doc_id)

    def index_text_file(self, file_path: str, doc_id: Optional[str] = None) -> int:
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_document(text, doc_id)

    def retrieve_without_rerank(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Standard vector search (no reranking). For comparison.

        Args:
            query:  Search query.
            k:      Number of results (defaults to final_k).

        Returns:
            List of chunk texts.
        """
        if k is None:
            k = self.final_k
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=k)
        return [r.document.content for r in results]

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve with reranking: vector search → rerank → top results.

        Args:
            query:  Search query.

        Returns:
            List of (chunk_text, rerank_score) tuples.
        """
        # Step 1: Initial broad vector search
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=self.initial_k)
        candidate_chunks = [r.document.content for r in results]

        if not candidate_chunks:
            return []

        # Step 2: Rerank
        print(f"  [Reranker] Reranking {len(candidate_chunks)} candidates ({self.rerank_method})...")
        reranked = self.reranker.rerank(query, candidate_chunks, top_n=self.final_k)
        print(f"  [Reranker] Top {len(reranked)} selected")

        return reranked

    def retrieve_context(self, query: str) -> List[str]:
        """Convenience: return just the reranked text strings."""
        reranked = self.retrieve(query)
        return [text for text, _ in reranked]




class RerankingRAG:
    """
    Complete RAG pipeline with reranking.

    Combines RerankingRetriever (vector search + reranking) with
    OpenAIChat (for answer generation).

    Replaces LangChain's:
        - CustomRetriever(BaseRetriever)
        - CrossEncoderRetriever(BaseRetriever)
        - RetrievalQA.from_chain_type()

    Usage:
        rag = RerankingRAG(file_path="report.pdf", rerank_method="llm")
        answer, contexts = rag.query("What are the impacts on biodiversity?")
    """

    def __init__(
        self,
        file_path: str,
        rerank_method: Literal["llm", "cross_encoder"] = "llm",
        reranker_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        final_k: int = 3,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the Reranking RAG pipeline.

        Args:
            file_path:        Path to document (PDF or text file).
            rerank_method:    "llm" (accurate, slow) or "cross_encoder" (fast, local).
            reranker_model:   Model name override for reranker.
            chunk_size:       Characters per chunk.
            chunk_overlap:    Overlap between chunks.
            initial_k:        Candidates from vector search (cast wide net).
            final_k:          Results after reranking (keep the best).
            embedding_model:  OpenAI embedding model.
            chat_model:       OpenAI model for answer generation.
            temperature:      LLM temperature for answers.
        """
        self.file_path = file_path
        self.initial_k=final_k*5
        # Initialize retriever with reranking
        self.retriever = RerankingRetriever(
            embedding_model=embedding_model,
            rerank_method=rerank_method,
            reranker_model=reranker_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            initial_k=self.initial_k,
            final_k=final_k,
        )

        # Initialize chat model
        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        # Index the document
        if file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(file_path)
        else:
            num_chunks = self.retriever.index_text_file(file_path)

        print(
            f"[Reranking] Indexed '{os.path.basename(file_path)}' → {num_chunks} chunks "
            f"(method={rerank_method}, initial_k={self.initial_k}, final_k={final_k})"
        )

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Query the RAG system with reranking.

        Flow:
            1. Vector search → initial_k candidates (cast wide net)
            2. Rerank → final_k best chunks (filter the best)
            3. Feed reranked chunks to answer LLM

        Args:
            question:        User's question.
            return_context:  Whether to return reranked contexts.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """
        contexts = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []

    def compare(self, question: str) -> None:
        """
        Debug helper: compare baseline vector search vs reranked results.

        Shows how reranking reorders chunks — the key "capital of France"
        demonstration from the notebook.

        Args:
            question:  Search query.
        """
        print(f"\nQuery: {question}")
        print("=" * 70)

        # Baseline: top final_k from vector search only
        baseline = self.retriever.retrieve_without_rerank(question)
        print(f"\n BASELINE (vector search, top {len(baseline)}):")
        print("-" * 50)
        for i, ctx in enumerate(baseline):
            preview = ctx[:200].replace('\n', ' ')
            print(f"  {i+1}. {preview}...")

        # Reranked
        reranked = self.retriever.retrieve(question)
        print(f"\n RERANKED ({self.retriever.rerank_method}, top {len(reranked)}):")
        print("-" * 50)
        for i, (ctx, score) in enumerate(reranked):
            preview = ctx[:200].replace('\n', ' ')
            print(f"  {i+1}. [score={score:.2f}] {preview}...")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    pdf_file_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"
    rag = RerankingRAG(file_path=pdf_file_path, rerank_method="llm")

    while True:
        user_question = input("Ask a question: ").strip().lower()

        if user_question == "exit":
            break

        answer, contexts = rag.query(user_question)
        print(f"\nAnswer: {answer}")
        print(f"\nContexts: {contexts}")
