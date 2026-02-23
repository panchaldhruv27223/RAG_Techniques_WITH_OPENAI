"""
Dartboard RAG

Balances RELEVANCE and DIVERSITY in retrieval to avoid redundant results.
Standard top-k retrieval in dense datasets returns near-duplicate chunks.
Dartboard selection picks chunks that are both relevant AND different from
each other, giving the LLM richer, more comprehensive context.

Based on the paper: "Better RAG using Relevant Information Gain"
(https://arxiv.org/abs/2407.12101) — vanilla approach with weighted balancing.

The problem:
    In a dense dataset (or one with overlapping content), top-3 retrieval
    returns the SAME chunk 3 times (or near-duplicates). The LLM gets
    redundant context and misses other relevant information.

    Standard top-3:
        1. "Greenhouse gases cause warming..."  ← same content
        2. "Greenhouse gases cause warming..."  ← same content  
        3. "Greenhouse gases cause warming..."  ← same content

    Dartboard top-3:
        1. "Greenhouse gases cause warming..."  ← most relevant
        2. "Modern observations show rapid temperature increase..."  ← diverse!
        3. "Fossil fuels release CO2 into the atmosphere..."  ← diverse!

How it works:
    1. Over-fetch candidates: retrieve 3x more chunks than needed

    2. Compute distance matrices:
       - query ↔ each candidate (relevance)
       - candidate ↔ candidate (diversity)
    
    3. Convert distances to log-normal probabilities
    
    4. Greedy selection:

       a. Pick the most relevant chunk first
    
       b. For remaining picks, score = relevance_weight * relevance + diversity_weight * (max distance from already selected)
    
       c. Pick highest combined score, repeat
    
    5. Return the selected diverse + relevant set

Usage:
    from dartboard_rag import DartboardRAG

    rag = DartboardRAG(file_path="document.pdf")
    answer, contexts = rag.query("What is the main cause of climate change?")
"""

import os
import sys
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

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


def lognorm(dist: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute log-normal probability for distances.

    Converts raw distances into log-space probabilities using a Gaussian kernel.
    Smaller distances → higher (less negative) log-probability.

    Args:
        dist:   Distance values (0 = identical, higher = more different).
        sigma:  Bandwidth parameter. Smaller sigma = sharper peaks = more
                sensitive to small distance differences.

    Returns:
        Log-probability values (always negative, closer to 0 = more probable).
    """
    if sigma < 1e-9:
        return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist ** 2 / (2 * sigma ** 2)


def logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute log(sum(exp(a))) in a numerically stable way.

    Replaces scipy.special.logsumexp — avoids the scipy dependency.

    The trick: log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
    This prevents overflow/underflow by factoring out the maximum.

    Args:
        a:     Input array.
        axis:  Axis along which to sum.

    Returns:
        Log-sum-exp values along the specified axis.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    # Handle -inf values (don't want -inf - (-inf) = nan)
    a_max_squeezed = np.squeeze(a_max, axis=axis)
    mask = np.isfinite(a_max_squeezed)
    result = np.where(
        mask,
        a_max_squeezed + np.log(np.sum(np.exp(a - a_max), axis=axis)),
        -np.inf,
    )
    return result

def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int,
    relevance_weight: float = 1.0,
    diversity_weight: float = 1.0,
    sigma: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """
    Select top-k documents balancing relevance to query AND diversity
    from already-selected documents.

    Algorithm:
        1. Convert all distances to log-normal probabilities
        2. Pick the single most relevant document
        3. For each remaining slot:
           a. For each candidate, compute:
              - diversity score = max log-prob distance from any selected doc
              - relevance score = log-prob distance from query
              - combined = diversity_weight * diversity + relevance_weight * relevance
           b. Normalize via logsumexp
           c. Pick the candidate with highest combined score
           d. Mask it out and repeat

    Args:
        query_distances:     Shape (1, N) — distance from query to each candidate.
        document_distances:  Shape (N, N) — pairwise distances between candidates.
        documents:           List of candidate document texts.
        num_results:         How many documents to select.
        relevance_weight:    Weight for query-relevance in combined score.
        diversity_weight:    Weight for diversity in combined score.
        sigma:               Gaussian bandwidth for log-normal conversion.

    Returns:
        Tuple of (selected_texts, selection_scores).
    """
    sigma = max(sigma, 1e-5)

    # Convert distances to log-probabilities
    query_probs = lognorm(query_distances, sigma)       # (1, N) or (N,)
    doc_probs = lognorm(document_distances, sigma)      # (N, N)

    # Flatten query_probs if needed
    if query_probs.ndim > 1:
        query_probs = query_probs.squeeze(0)

    n_candidates = len(documents)
    num_results = min(num_results, n_candidates)

    # Step 1: Pick the most relevant document
    most_relevant_idx = np.argmax(query_probs)
    selected_indices = [int(most_relevant_idx)]
    selection_scores = [1.0]  # Dummy score for first pick

    # Track max distance from any selected document (for diversity)
    max_distances = doc_probs[most_relevant_idx].copy()  # (N,)

    # Step 2: Iteratively pick remaining documents
    while len(selected_indices) < num_results:
        # Update diversity: for each candidate, what's the max log-prob
        # distance from ANY already-selected document?
        # This measures "how different is this candidate from what we already have"
        updated_distances = np.maximum(max_distances, doc_probs)  # (N, N) → element-wise max

        # Combined score: diversity + relevance
        # updated_distances shape: (N, N) — we need per-candidate scores
        # query_probs shape: (N,) — broadcast to (N, N) by adding
        combined = (
            updated_distances * diversity_weight
            + query_probs[np.newaxis, :] * relevance_weight
        )

        # Aggregate across dimensions via logsumexp
        normalized = logsumexp(combined, axis=1)  # (N,)

        # Mask already-selected documents
        for idx in selected_indices:
            normalized[idx] = -np.inf

        # Pick the best
        best_idx = int(np.argmax(normalized))
        best_score = float(normalized[best_idx])

        # Update tracking
        max_distances = updated_distances[best_idx]
        selected_indices.append(best_idx)
        selection_scores.append(best_score)

    # Return selected documents
    selected_texts = [documents[i] for i in selected_indices]
    return selected_texts, selection_scores


class DartboardRetriever:
    """
    Retriever that uses dartboard selection for relevance + diversity.

    Pipeline:
        1. Index document → FAISS with stored embeddings
        2. On query:
           a. Over-fetch candidates (3x more than needed)
           b. Compute distance matrices from embeddings
           c. Run greedy dartboard selection
           d. Return diverse + relevant subset

    The key difference from simple retrieval: we need access to the raw
    embedding vectors (not just similarity scores) to compute pairwise
    distances between candidates.

    Args:
        embedding_model:     OpenAI embedding model.
        chunk_size:          Characters per chunk.
        chunk_overlap:       Overlap between chunks.
        k:                   Final number of results to return.
        oversampling:        Over-fetch factor for initial candidates.
        relevance_weight:    Weight for relevance in combined score.
        diversity_weight:    Weight for diversity in combined score.
        sigma:               Gaussian bandwidth for probability conversion.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        oversampling: int = 3,
        relevance_weight: float = 1.0,
        diversity_weight: float = 1.0,
        sigma: float = 0.1,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.oversampling = oversampling
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.sigma = sigma

        # Core components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

    def index_document(self, text: str, doc_id: str = "doc_0", duplicate_factor: int = 1) -> int:
        """
        Chunk and index a document. Optionally duplicate to simulate dense dataset.

        Args:
            text:              Full document text.
            doc_id:            Document identifier.
            duplicate_factor:  Repeat chunks N times to simulate dense/overlapping data.

        Returns:
            Number of chunks indexed.
        """
        text = text.replace('\t', ' ')

        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if duplicate_factor > 1:
            chunks = chunks * duplicate_factor

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

    def index_pdf(self, file_path: str, doc_id: Optional[str] = None, duplicate_factor: int = 1) -> int:
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        text = read_pdf(file_path)
        return self.index_document(text, doc_id, duplicate_factor)

    def index_text_file(self, file_path: str, doc_id: Optional[str] = None, duplicate_factor: int = 1) -> int:
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_document(text, doc_id, duplicate_factor)

    def retrieve_standard(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Standard top-k retrieval (no diversity). For comparison.
        """
        if k is None:
            k = self.k
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=k)
        return [r.document.content for r in results]

    def retrieve_dartboard(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Retrieve with dartboard selection: relevance + diversity.

        Steps:
            1. Embed query
            2. Over-fetch candidates from FAISS
            3. Extract candidate embedding vectors
            4. Compute distance matrices:
               - query ↔ candidates (cosine distance)
               - candidates ↔ candidates (cosine distance)
            5. Run greedy dartboard selection
            6. Return diverse + relevant subset

        Args:
            query:  Search query.

        Returns:
            Tuple of (selected_texts, selection_scores).
        """
        query_emb = self.embedder.embed_text(query)
        query_vec = np.array([query_emb], dtype=np.float32)

        fetch_k = min(
            len(self.vector_store.documents),
            self.k * self.oversampling,
        )
        results = self.vector_store.search(query_emb, k=fetch_k)

        if not results:
            return [], []

        candidate_texts = []
        candidate_vecs = []
        for r in results:
            candidate_texts.append(r.document.content)
            candidate_vecs.append(r.document.embedding)

        candidate_matrix = np.array(candidate_vecs, dtype=np.float32)  # (N, dim)

        query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
        cand_norm = candidate_matrix / (np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-8)

        query_distances = 1.0 - np.dot(query_norm, cand_norm.T)        # (1, N)
        document_distances = 1.0 - np.dot(cand_norm, cand_norm.T)      # (N, N)

        selected_texts, scores = greedy_dartsearch(
            query_distances=query_distances,
            document_distances=document_distances,
            documents=candidate_texts,
            num_results=self.k,
            relevance_weight=self.relevance_weight,
            diversity_weight=self.diversity_weight,
            sigma=self.sigma,
        )

        return selected_texts, scores

    def retrieve_context(self, query: str) -> List[str]:
        """Convenience: return just the dartboard-selected texts."""
        texts, _ = self.retrieve_dartboard(query)
        return texts


class DartboardRAG:
    """
    Complete RAG pipeline with dartboard (relevance + diversity) retrieval.

    Combines DartboardRetriever with OpenAIChat for answer generation.

    Usage:
        rag = DartboardRAG(file_path="report.pdf")
        answer, contexts = rag.query("What causes climate change?")

        # Simulate dense dataset (duplicate chunks 5x)
        rag = DartboardRAG(file_path="report.pdf", duplicate_factor=5)
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        oversampling: int = 3,
        relevance_weight: float = 1.0,
        diversity_weight: float = 1.0,
        sigma: float = 0.1,
        duplicate_factor: int = 1,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the Dartboard RAG pipeline.

        Args:
            file_path:          Path to document (PDF or text file).
            chunk_size:         Characters per chunk.
            chunk_overlap:      Overlap between chunks.
            k:                  Final number of diverse results.
            oversampling:       Over-fetch factor (fetch k x oversampling candidates).
            relevance_weight:   Weight for relevance in scoring.
            diversity_weight:   Weight for diversity in scoring.
            sigma:              Gaussian bandwidth for probability conversion.
            duplicate_factor:   Repeat chunks Nx to simulate dense dataset.
            embedding_model:    OpenAI embedding model.
            chat_model:         OpenAI chat model.
            temperature:        LLM temperature.
        """
        self.file_path = file_path

        self.retriever = DartboardRetriever(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
            oversampling=oversampling,
            relevance_weight=relevance_weight,
            diversity_weight=diversity_weight,
            sigma=sigma,
        )

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        if file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(file_path, duplicate_factor=duplicate_factor)
        else:
            num_chunks = self.retriever.index_text_file(file_path, duplicate_factor=duplicate_factor)

        dup_note = f" (×{duplicate_factor} duplicated)" if duplicate_factor > 1 else ""
        print(
            f"[Dartboard] Indexed '{os.path.basename(file_path)}' → {num_chunks} chunks{dup_note} "
            f"(k={k}, relevance={relevance_weight}, diversity={diversity_weight})"
        )

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """Query with dartboard-selected diverse context."""
        contexts = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []

    def compare(self, question: str) -> None:
        """
        Debug helper: compare standard top-k vs dartboard selection.

        This demonstrates the core value — in dense datasets, standard
        retrieval returns duplicates while dartboard returns diverse results.

        Args:
            question:  Search query.
        """
        print(f"\nQuery: {question}")
        print("=" * 70)

        standard = self.retriever.retrieve_standard(question)
        print(f"\n STANDARD TOP-{len(standard)} (may have duplicates):")
        print("-" * 50)
        seen = set()
        for i, ctx in enumerate(standard):
            key = ctx[:80]
            is_dup = " DUPLICATE" if key in seen else ""
            seen.add(key)
            preview = ctx[:150].replace('\n', ' ')
            print(f"  {i+1}. {preview}... {is_dup}")

        dartboard, scores = self.retriever.retrieve_dartboard(question)
        print(f"\n DARTBOARD TOP-{len(dartboard)} (relevance + diversity):")
        print("-" * 50)
        seen = set()
        for i, (ctx, score) in enumerate(zip(dartboard, scores)):
            key = ctx[:80]
            is_dup = " DUPLICATE" if key in seen else ""
            seen.add(key)
            preview = ctx[:150].replace('\n', ' ')
            print(f"  {i+1}. [score={score:.3f}] {preview}... {is_dup}")

        standard_unique = len(set(ctx[:80] for ctx in standard))
        dartboard_unique = len(set(ctx[:80] for ctx in dartboard))
        print(f"\n Unique results: standard={standard_unique}/{len(standard)}, "
              f"dartboard={dartboard_unique}/{len(dartboard)}")


if __name__ == "__main__":
    pdf_file_path = r"data\Understanding_Climate_Change.pdf"
    rag = DartboardRAG(file_path=pdf_file_path)

    answer, context = rag.query(question="What are the main causes of climate change?")
    print(f"Answer: {answer}")
    print(f"Context: {context}")