"""
Fusion Retrieval RAG

Combines two fundamentally different search strategies:
    - Vector search (semantic): finds conceptually similar text
    - BM25 (keyword): finds exact keyword matches using TF-IDF statistics

Each method has blind spots the other covers:
    - Vector search: great at "What causes global warming?" matching
      "Greenhouse gases trap heat" but misses exact terms like "UNFCCC"
    - BM25: great at exact keyword matching like "UNFCCC" or "Paris Agreement"
      but misses paraphrased concepts

Fusion retrieval scores every chunk with BOTH methods, normalizes the scores
to [0,1], then combines them with a weighted alpha parameter.

How it works:
    1. Chunk document → store in FAISS (vector) + BM25 index (keyword)
    2. On query: run BOTH searches over ALL chunks
    3. Normalize vector scores and BM25 scores to [0,1]
    4. Combined score = alpha x vector_score + (1-alpha) x bm25_score
    5. Rank by combined score → return top-k
    6. Feed top-k to LLM for answer generation

Alpha tuning:
    - alpha=1.0 → pure vector search (semantic only)
    - alpha=0.0 → pure BM25 (keyword only)
    - alpha=0.5 → equal weight (good default)
    - alpha=0.7 → favor semantic, with keyword boost

Usage:
    from fusion_retrieval_rag import FusionRetrievalRAG

    rag = FusionRetrievalRAG(file_path="document.pdf", alpha=0.5)
    answer, contexts = rag.query("What are the impacts of climate change?")
"""



import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from rank_bm25 import BM25Okapi

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    chunk_text,
)


class BM25Index:
    """
    BM25 (Best Matching 25) keyword-based retrieval index.

    BM25 is a probabilistic ranking function based on TF-IDF. It scores
    documents by how frequently query terms appear, with diminishing
    returns for repeated terms and length normalization.

    Unlike vector search, BM25:
        - Finds EXACT keyword matches (acronyms, names, codes)
        - Doesn't understand paraphrasing or synonyms
        - Is very fast (no embedding computation needed)
        - Works well for specific/technical queries

    Args:
        documents:  List of Document objects to index.
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from documents.

        Tokenization is simple whitespace splitting — could be improved
        with stemming/lemmatization for production use.

        Args:
            documents:  List of Document objects.
        """
        self._documents = documents

        # Tokenize: split each document's content on whitespace
        tokenized_docs = [doc.content.lower().split() for doc in documents]
        self._index = BM25Okapi(tokenized_docs)

    def score_all(self, query: str) -> np.ndarray:
        """
        Score ALL documents against a query using BM25.

        Returns raw BM25 scores for every document in the index.
        Higher score = more keyword overlap with the query.

        Args:
            query:  Search query string.

        Returns:
            numpy array of BM25 scores, one per document.
        """
        if self._index is None:
            raise ValueError("BM25 index not built. Call build() first.")

        tokenized_query = query.lower().split()
        return self._index.get_scores(tokenized_query)

    @property
    def documents(self) -> List[Document]:
        return self._documents



class FusionRetriever:
    """
    Retriever that fuses vector-based (semantic) and BM25 (keyword) search.

    Pipeline:
        1. Index document → both FAISS vector store AND BM25 index
        2. On query:
           a. Vector search → score all chunks by semantic similarity
           b. BM25 search → score all chunks by keyword relevance
           c. Normalize both score sets to [0, 1]
           d. Combine: alpha × vector + (1-alpha) × bm25
           e. Rank by combined score → return top-k

    The notebook retrieves ALL docs from the vectorstore to get scores
    for every chunk. We do the same via FAISS's search with k=total.

    Args:
        embedding_model:  OpenAI embedding model.
        chunk_size:       Characters per chunk.
        chunk_overlap:    Overlap between chunks.
        k:                Number of top results to return.
        alpha:            Weight for vector scores (0=pure BM25, 1=pure vector).
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        alpha: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.alpha = alpha

        # Core components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.bm25_index = BM25Index()

        # Keep reference to all chunks for scoring
        self._all_chunks: List[Document] = []

    def index_document(self, text: str, doc_id: str = "doc_0") -> int:
        """
        Chunk and index a document in BOTH vector store and BM25 index.

        Args:
            text:    Full document text.
            doc_id:  Document identifier.

        Returns:
            Number of chunks created.
        """
        # Clean text: replace tabs with spaces (matching notebook's replace_t_with_space)
        text = text.replace('\t', ' ')

        # Chunk the text
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Create Document objects
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

        self._all_chunks = documents

        # Index in FAISS (vector search)
        embedded_docs = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(embedded_docs)

        # Index in BM25 (keyword search)
        self.bm25_index.build(documents)

        return len(chunks)

    def index_pdf(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """Read and index a PDF file."""
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        text = read_pdf(file_path)
        return self.index_document(text, doc_id)

    def index_text_file(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """Read and index a text file."""
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_document(text, doc_id)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores:  Raw score array.

        Returns:
            Normalized scores in [0, 1].
        """
        epsilon = 1e-8
        min_s = np.min(scores)
        max_s = np.max(scores)
        return (scores - min_s) / (max_s - min_s + epsilon)

    def retrieve(self, query: str, alpha: Optional[float] = None) -> List[Document]:
        """
        Perform fusion retrieval combining vector and BM25 search.

        This is the core method. Both search methods score ALL chunks,
        scores are normalized to [0,1], then combined with alpha weighting.

        Args:
            query:  Search query.
            alpha:  Optional override for vector weight. If None, uses self.alpha.

        Returns:
            List of Document objects, ranked by combined score.
        """
        if alpha is None:
            alpha = self.alpha

        total_docs = len(self._all_chunks)
        if total_docs == 0:
            return []

        # Step 1: Vector search — score ALL chunks
        query_emb = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(query_emb, k=total_docs)

        # Build a map: chunk_index → vector score
        # FAISS with IndexFlatIP returns inner product (higher = better)
        vector_score_map = {}
        for r in vector_results:
            idx = r.document.metadata.get("chunk_index", -1)
            vector_score_map[idx] = r.score

        # Create ordered vector scores array
        vector_scores = np.array([
            vector_score_map.get(i, 0.0) for i in range(total_docs)
        ])

        # Step 2: BM25 search — score ALL chunks
        bm25_scores = self.bm25_index.score_all(query)

        # Step 3: Normalize both to [0, 1]
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Step 4: Combine with alpha weighting
        combined_scores = alpha * vector_scores_norm + (1 - alpha) * bm25_scores_norm

        # Step 5: Rank and return top-k
        sorted_indices = np.argsort(combined_scores)[::-1]
        top_indices = sorted_indices[:self.k]

        return [self._all_chunks[i] for i in top_indices]

    def retrieve_context(self, query: str, alpha: Optional[float] = None) -> List[str]:
        """
        Convenience method: return just the text strings.

        Args:
            query:  Search query.
            alpha:  Optional vector weight override.

        Returns:
            List of chunk text strings.
        """
        docs = self.retrieve(query, alpha)
        return [doc.content for doc in docs]

    def retrieve_with_scores(self, query: str, alpha: Optional[float] = None) -> List[Tuple[Document, float, float, float]]:
        """
        Retrieve with detailed score breakdown for debugging.

        Args:
            query:  Search query.
            alpha:  Optional vector weight override.

        Returns:
            List of (document, vector_score, bm25_score, combined_score) tuples.
        """
        if alpha is None:
            alpha = self.alpha

        total_docs = len(self._all_chunks)
        if total_docs == 0:
            return []

        # Vector scores
        query_emb = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(query_emb, k=total_docs)
        vector_score_map = {}
        for r in vector_results:
            idx = r.document.metadata.get("chunk_index", -1)
            vector_score_map[idx] = r.score
        vector_scores = np.array([vector_score_map.get(i, 0.0) for i in range(total_docs)])

        # BM25 scores
        bm25_scores = self.bm25_index.score_all(query)

        # Normalize
        v_norm = self._normalize_scores(vector_scores)
        b_norm = self._normalize_scores(bm25_scores)
        combined = alpha * v_norm + (1 - alpha) * b_norm
        # Sort and return top-k with score breakdown
        sorted_indices = np.argsort(combined)[::-1][:self.k]

        results = []
        for i in sorted_indices:
            results.append((
                self._all_chunks[i],
                float(v_norm[i]),
                float(b_norm[i]),
                float(combined[i]),
            ))
        return results



class FusionRetrievalRAG:
    """
    Complete RAG pipeline using fusion retrieval (vector + BM25).

    Usage:
        rag = FusionRetrievalRAG(file_path="report.pdf", alpha=0.5)
        answer, contexts = rag.query("What are the impacts of climate change?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        alpha: float = 0.5,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the Fusion Retrieval RAG pipeline.

        Args:
            file_path:        Path to document (PDF or text file).
            chunk_size:        Characters per chunk.
            chunk_overlap:     Overlap between chunks.
            k:                 Number of top results to return.
            alpha:             Vector weight (0=pure BM25, 1=pure vector, 0.5=equal).
            embedding_model:   OpenAI embedding model.
            chat_model:        OpenAI chat model.
            temperature:       LLM temperature.
        """
        self.file_path = file_path

        # Initialize retriever
        self.retriever = FusionRetriever(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
            alpha=alpha,
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
            f"[FusionRetrieval] Indexed '{os.path.basename(file_path)}' "
            f"→ {num_chunks} chunks (alpha={alpha}, k={k})"
        )
        print(f"[FusionRetrieval] Both FAISS (vector) and BM25 (keyword) indexes built")

    def query(
        self,
        question: str,
        return_context: bool = True,
        alpha: Optional[float] = None,
    ) -> Tuple[str, List[str]]:
        """
        Query the fusion RAG system.

        Args:
            question:        User's question.
            return_context:  Whether to return retrieved contexts.
            alpha:           Optional override for vector/BM25 weight.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """
        contexts = self.retriever.retrieve_context(question, alpha)

        if not contexts:
            return "No relevant information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []



if __name__ == "__main__":
    pdf_path = r"data\Understanding_Climate_Change.pdf"

    rag = FusionRetrievalRAG(
        file_path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        k=5,
        alpha=0.5,
    )


    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        answer, context = rag.query(question)
        print(f"\nAnswer: {answer}")
        print(f"Chunks used: {len(context)}")
        print()