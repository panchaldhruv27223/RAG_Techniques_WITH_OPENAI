"""
Relevant Segment Extraction (RSE)

RSE reconstructs multi-chunk segments of contiguous text from retrieved chunks.
Instead of returning isolated top-k chunks, RSE identifies clusters of relevant
chunks within their original documents and merges them into coherent segments.

Key insight: Relevant chunks tend to be clustered within their original documents.

How it works:
    1. Chunk documents with ZERO overlap (required for clean segment reconstruction)
    2. Store chunks in both a vector index AND a key-value store (doc_id, chunk_index)
    3. Retrieve candidate chunks via vector search
    4. Score and rerank chunks using OpenAI embeddings similarity
    5. Compute chunk values = fusion of absolute relevance + rank-based decay - threshold
    6. Solve constrained maximum-sum-subarray to find optimal segments
    7. Reconstruct segments by concatenating contiguous chunks from the KV store

Dependencies: openai, numpy, faiss-cpu, PyMuPDF (fitz)

Usage:
    from relevant_segment_extraction_rag import RSERetrievalRAG

    rag = RSERetrievalRAG(file_path="document.pdf")
    answer, segments = rag.query("What are the consolidated financial statements?")
"""

import os
import sys
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

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
    cosine_similarity,
)


@dataclass
class ChunkMeta:
    """
    Metadata for a single chunk stored in the key-value store.

    Attributes:
        doc_id:       Identifier for the source document.
        chunk_index:  Position of this chunk within the document (0-based).
        text:         Raw text content of the chunk.
    """
    doc_id: str
    chunk_index: int
    text: str


@dataclass
class Segment:
    """
    A contiguous segment of text reconstructed from multiple chunks.

    Attributes:
        doc_id:       Source document identifier.
        start_index:  Start chunk index (inclusive).
        end_index:    End chunk index (exclusive).
        text:         Concatenated text of all chunks in [start_index, end_index).
        score:        Segment-level relevance score.
    """
    doc_id: str
    start_index: int
    end_index: int
    text: str
    score: float


def _beta_cdf_approx(x: float, a: float = 0.4, b: float = 0.4) -> float:
    """
    Approximate Beta CDF using a simple numerical integration.

    The original notebook uses scipy.stats.beta.cdf to spread out
    the relevance values more uniformly. This implementation avoids
    the scipy dependency by using a basic trapezoidal integration
    over the Beta PDF: f(t) = t^(a-1) * (1-t)^(b-1) / B(a,b).

    Args:
        x:  Value in [0, 1].
        a:  Alpha parameter of the Beta distribution.
        b:  Beta parameter of the Beta distribution.

    Returns:
        Approximate CDF value at x.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Compute Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    beta_func = math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    # Trapezoidal integration from 0 to x
    n_steps = 200
    dt = x / n_steps
    total = 0.0
    for i in range(n_steps + 1):
        t = i * dt
        # Clamp t to avoid 0^negative
        t_clamped = max(t, 1e-12)
        one_minus_t = max(1.0 - t, 1e-12)
        val = (t_clamped ** (a - 1)) * (one_minus_t ** (b - 1))
        # Trapezoidal rule: half-weight at endpoints
        if i == 0 or i == n_steps:
            total += val * 0.5
        else:
            total += val
    total *= dt

    return total / beta_func

class ChunkKVStore:
    """
    In-memory key-value store for chunk text, keyed by (doc_id, chunk_index).

    RSE needs to retrieve chunks that weren't in the initial search results
    (e.g. non-relevant chunks sandwiched between relevant ones). This store
    enables O(1) lookup by position.
    """

    def __init__(self):
        self._store: Dict[Tuple[str, int], ChunkMeta] = {}
        self._doc_lengths: Dict[str, int] = {}  # doc_id → total chunks

    def add(self, doc_id: str, chunk_index: int, text: str) -> None:
        key = (doc_id, chunk_index)
        self._store[key] = ChunkMeta(doc_id=doc_id, chunk_index=chunk_index, text=text)
        # Track max chunk index per document
        self._doc_lengths[doc_id] = max(
            self._doc_lengths.get(doc_id, 0), chunk_index + 1
        )

    def get(self, doc_id: str, chunk_index: int) -> Optional[ChunkMeta]:
        return self._store.get((doc_id, chunk_index))

    def get_segment_text(self, doc_id: str, start: int, end: int) -> str:
        """
        Reconstruct contiguous text for chunks [start, end).

        Args:
            doc_id:  Document identifier.
            start:   Start chunk index (inclusive).
            end:     End chunk index (exclusive).

        Returns:
            Concatenated text of all chunks in the range.
        """
        parts = []
        for i in range(start, end):
            meta = self.get(doc_id, i)
            if meta:
                parts.append(meta.text)
        return "\n".join(parts)

    def doc_chunk_count(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    @property
    def all_doc_ids(self) -> List[str]:
        return list(self._doc_lengths.keys())



class OpenAIReranker:
    """
    Reranks chunks by computing cosine similarity between the query
    embedding and each chunk embedding via OpenAI's embedding model.
    """

    def __init__(self, embedder: OpenAIEmbedder):
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        chunks: List[str],
        decay_rate: float = 30.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute similarity scores and chunk values for all chunks.

        Args:
            query:       Search query string.
            chunks:      List of chunk texts to score.
            decay_rate:  Exponential decay rate applied to ranks.

        Returns:
            similarity_scores:  List of transformed absolute relevance (0–1), 
                                in original document order.
            chunk_values:       List of relevance values fusing rank + similarity,
                                in original document order.
        """
        if not chunks:
            return [], []

        # Embed query and all chunks
        query_emb = self.embedder.embed_text(query)
        chunk_embs = self.embedder.embed_texts(chunks)

        # Compute raw cosine similarities
        raw_scores = []
        for emb in chunk_embs:
            sim = cosine_similarity(query_emb, emb)
            # Clamp to [0, 1] for the beta transform
            sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # normalize from [-1,1] to [0,1]
            raw_scores.append(sim)

        # Sort by score descending to get ranks
        indexed_scores = list(enumerate(raw_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Build output arrays in original order
        similarity_scores = [0.0] * len(chunks)
        chunk_values = [0.0] * len(chunks)

        for rank, (orig_idx, raw_sim) in enumerate(indexed_scores):
            transformed = _beta_cdf_approx(raw_sim)
            similarity_scores[orig_idx] = transformed
            # Fuse rank-based decay with absolute relevance
            chunk_values[orig_idx] = math.exp(-rank / decay_rate) * transformed

        return similarity_scores, chunk_values

def get_best_segments(
    relevance_values: List[float],
    max_length: int,
    overall_max_length: int,
    minimum_value: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Find the best non-overlapping contiguous segments by solving a
    constrained version of the maximum-sum-subarray problem.

    After subtracting the irrelevant_chunk_penalty, irrelevant chunks have
    negative values and relevant chunks have positive values. Segment value
    is the sum of its constituent chunk values. The algorithm greedily picks
    the best segment, then repeats until constraints are met.

    Args:
        relevance_values:   Per-chunk values (already penalty-adjusted).
        max_length:         Max number of chunks in a single segment.
        overall_max_length: Max total chunks across all segments.
        minimum_value:      Minimum score a segment must have to be kept.

    Returns:
        best_segments:  List of (start, end) tuples (end is exclusive).
        scores:         Corresponding segment scores.
    """
    best_segments: List[Tuple[int, int]] = []
    scores: List[float] = []
    total_length = 0

    while total_length < overall_max_length:
        best_segment = None
        best_value = -1000.0

        for start in range(len(relevance_values)):
            # Skip negative starting points
            if relevance_values[start] < 0:
                continue

            for end in range(
                start + 1,
                min(start + max_length + 1, len(relevance_values) + 1),
            ):
                # Skip negative ending points
                if relevance_values[end - 1] < 0:
                    continue

                # Check overlap with existing segments
                if any(
                    start < seg_end and end > seg_start
                    for seg_start, seg_end in best_segments
                ):
                    continue

                # Check overall length constraint
                if total_length + (end - start) > overall_max_length:
                    continue

                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)

        # No valid segment found or below minimum
        if best_segment is None or best_value < minimum_value:
            break

        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores


class RSERetriever:
    """
    Retriever that uses Relevant Segment Extraction (RSE) to return
    contiguous, multi-chunk segments instead of isolated top-k chunks.

    Pipeline:
        1. Index document with zero-overlap chunking
        2. Store every chunk in both FAISS vector index + key-value store
        3. On query: retrieve top-k candidates via vector search
        4. Rerank ALL chunks of candidate documents using OpenAI embeddings
        5. Compute chunk values (similarity × rank decay − threshold)
        6. Run segment optimization per document
        7. Return reconstructed segment texts
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        irrelevant_chunk_penalty: float = 0.2,
        max_segment_length: int = 20,
        overall_max_length: int = 30,
        minimum_segment_value: float = 0.7,
        decay_rate: float = 30.0,
        initial_k: int = 40,
    ):
        """
        Args:
            embedding_model:          OpenAI embedding model.
            chunk_size:               Characters per chunk (NO overlap for RSE).
            irrelevant_chunk_penalty: Subtracted from every chunk value; controls
                                      how aggressively irrelevant chunks are penalized.
                                      ~0.2 works well empirically.
            max_segment_length:       Max chunks in one segment.
            overall_max_length:       Max total chunks across all returned segments.
            minimum_segment_value:    Min score for a segment to be returned.
            decay_rate:               Exponential decay rate for rank-based scoring.
            initial_k:                Number of candidates from initial vector search.
        """
        self.chunk_size = chunk_size
        self.irrelevant_chunk_penalty = irrelevant_chunk_penalty
        self.max_segment_length = max_segment_length
        self.overall_max_length = overall_max_length
        self.minimum_segment_value = minimum_segment_value
        self.decay_rate = decay_rate
        self.initial_k = initial_k

        # Core components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.kv_store = ChunkKVStore()
        self.reranker = OpenAIReranker(self.embedder)

    def index_document(self, text: str, doc_id: str = "doc_0") -> int:
        """
        Chunk and index a single document.

        IMPORTANT: RSE requires chunk_overlap=0 so that chunks can be
        cleanly concatenated to reconstruct document segments.

        Args:
            text:    Full document text.
            doc_id:  Unique identifier for this document.

        Returns:
            Number of chunks created.
        """
        # Split with ZERO overlap (RSE requirement)
        chunks = chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=0)

        documents = []
        for i, chunk in enumerate(chunks):
            # Store in key-value store
            self.kv_store.add(doc_id, i, chunk)

            # Prepare for vector indexing
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

        # Embed and add to FAISS
        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)

        return len(chunks)

    def index_pdf(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """
        Read a PDF and index its contents.

        Args:
            file_path:  Path to PDF file.
            doc_id:     Document ID (defaults to filename).

        Returns:
            Number of chunks created.
        """
        if doc_id is None:
            doc_id = os.path.basename(file_path)

        text = read_pdf(file_path)
        return self.index_document(text, doc_id)

    def index_text_file(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """
        Read a text file and index its contents.

        Args:
            file_path:  Path to text file.
            doc_id:     Document ID (defaults to filename).

        Returns:
            Number of chunks created.
        """
        if doc_id is None:
            doc_id = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_document(text, doc_id)

    def retrieve_segments(self, query: str) -> List[Segment]:
        """
        Core RSE retrieval pipeline.

        Steps:
            1. Vector search to find candidate chunks
            2. Identify which documents contain those candidates
            3. For each candidate document, rerank ALL its chunks
            4. Compute chunk values (fused relevance − penalty)
            5. Find optimal segments via constrained max-subarray
            6. Reconstruct and return segment texts

        Args:
            query: Search query string.

        Returns:
            List of Segment objects, sorted by score descending.
        """
        # Step 1: Initial vector search for candidates
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=self.initial_k)

        if not results:
            return []

        # Step 2: Identify candidate documents
        candidate_doc_ids = set()
        for r in results:
            doc_id = r.document.metadata.get("doc_id", "unknown")
            candidate_doc_ids.add(doc_id)

        # Step 3–5: Process each candidate document
        all_segments: List[Segment] = []

        for doc_id in candidate_doc_ids:
            num_chunks = self.kv_store.doc_chunk_count(doc_id)
            if num_chunks == 0:
                continue

            # Gather all chunk texts for this document
            doc_chunks = []
            for i in range(num_chunks):
                meta = self.kv_store.get(doc_id, i)
                doc_chunks.append(meta.text if meta else "")

            # Rerank all chunks against the query
            _, chunk_values = self.reranker.rerank(
                query, doc_chunks, decay_rate=self.decay_rate
            )

            # Subtract threshold to penalize irrelevant chunks
            adjusted_values = [
                v - self.irrelevant_chunk_penalty for v in chunk_values
            ]

            # Find best segments
            segments, scores = get_best_segments(
                adjusted_values,
                max_length=self.max_segment_length,
                overall_max_length=self.overall_max_length,
                minimum_value=self.minimum_segment_value,
            )

            # Reconstruct segment text
            for (start, end), score in zip(segments, scores):
                text = self.kv_store.get_segment_text(doc_id, start, end)
                all_segments.append(
                    Segment(
                        doc_id=doc_id,
                        start_index=start,
                        end_index=end,
                        text=text,
                        score=score,
                    )
                )

        # Sort all segments by score descending
        all_segments.sort(key=lambda s: s.score, reverse=True)
        return all_segments

    def retrieve_context(self, query: str) -> List[str]:
        """
        Convenience method: retrieve segment texts as a list of strings.

        Args:
            query: Search query.

        Returns:
            List of segment text strings.
        """
        segments = self.retrieve_segments(query)
        return [seg.text for seg in segments]

class RSERetrievalRAG:
    """
    Complete RAG pipeline using Relevant Segment Extraction.

    Combines RSERetriever (for intelligent multi-chunk retrieval) with
    OpenAIChat (for answer generation). Follows the same interface
    pattern as SimpleRAGOpenai and other RAG classes in this project.

    Usage:
        rag = RSERetrievalRAG(file_path="report.pdf")
        answer, segments = rag.query("What are the key financial metrics?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 800,
        irrelevant_chunk_penalty: float = 0.2,
        max_segment_length: int = 20,
        overall_max_length: int = 30,
        minimum_segment_value: float = 0.7,
        decay_rate: float = 30.0,
        initial_k: int = 40,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the RSE RAG pipeline.

        Args:
            file_path:                Path to document (PDF or text file).
            chunk_size:               Characters per chunk (0 overlap enforced).
            irrelevant_chunk_penalty: Threshold subtracted from chunk values (~0.2).
            max_segment_length:       Max chunks in one segment.
            overall_max_length:       Max total chunks across all segments.
            minimum_segment_value:    Min score for a segment to be returned.
            decay_rate:               Exponential decay for rank-based scoring.
            initial_k:                Vector search candidates.
            embedding_model:          OpenAI embedding model.
            chat_model:               OpenAI chat model.
            temperature:              LLM temperature.
        """
        self.file_path = file_path

        # Initialize retriever
        self.retriever = RSERetriever(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            irrelevant_chunk_penalty=irrelevant_chunk_penalty,
            max_segment_length=max_segment_length,
            overall_max_length=overall_max_length,
            minimum_segment_value=minimum_segment_value,
            decay_rate=decay_rate,
            initial_k=initial_k,
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

        print(f"[RSE] Indexed '{os.path.basename(file_path)}' → {num_chunks} chunks (0 overlap)")

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Query the RSE RAG system.

        Args:
            question:        User's question.
            return_context:  Whether to return retrieved segments.

        Returns:
            Tuple of (answer_string, list_of_segment_texts).
        """
        # Retrieve segments
        segments = self.retriever.retrieve_segments(question)
        context = [seg.text for seg in segments]

        if not context:
            return "No relevant information found in the document.", []

        # Generate answer
        answer = self.chat.chat_with_context(question, context)

        if return_context:
            return answer, context
        return answer, []

    def show_segments(self, question: str) -> None:
        """
        Debug helper: print retrieved segments with metadata.

        Args:
            question: Search query.
        """
        segments = self.retriever.retrieve_segments(question)

        print(f"\nQuery: {question}")
        print("=" * 70)

        if not segments:
            print("  No segments found.")
            return

        for i, seg in enumerate(segments):
            print(f"\n  Segment {i + 1}:")
            print(f"    Document:  {seg.doc_id}")
            print(f"    Chunks:    [{seg.start_index}, {seg.end_index})")
            print(f"    Length:    {seg.end_index - seg.start_index} chunks")
            print(f"    Score:     {seg.score:.4f}")
            print(f"    Preview:   {seg.text[:200]}...")
            print("-" * 70)


if __name__ == "__main__":

    pdf_path = r"data\Understanding_Climate_Change.pdf"
    rag = RSERetrievalRAG(
        file_path=pdf_path,
        chunk_size=800,
        irrelevant_chunk_penalty=0.2,
        max_segment_length=20,
        overall_max_length=30,
        minimum_segment_value=0.7,
    )

    # Interactive loop
    print("\n[RSE RAG] Ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() == "exit":
            break

        # Show segments for debugging
        rag.show_segments(question)

        # Get answer
        answer, context = rag.query(question)
        print(f"\nAnswer: {answer}")
        print(f"Segments used: {len(context)}")
        print()