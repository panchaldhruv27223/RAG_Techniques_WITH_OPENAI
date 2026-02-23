"""
Hierarchical Indices RAG

Uses a two-tier index structure: summaries for broad topic matching,
then detailed chunks within matched sections for precise retrieval.

Key insight: Searching summaries first acts as a coarse filter — it
identifies WHICH sections of the document are relevant. Then searching
detailed chunks WITHIN those sections gives precise answers without
noise from unrelated parts.

How it works:
    1. Split PDF into pages (or large sections)
    2. Summarize each page/section using LLM → store in SUMMARY index
    3. Split each page into small chunks → store in DETAIL index
       (each chunk tagged with its page number)
    4. On query:
       a. Search SUMMARY index → find top-k relevant pages
       b. For each relevant page, search DETAIL index (filtered by page)
       c. Return the matching detailed chunks
    5. Feed detailed chunks to answer LLM

Why this beats flat search:
    - A 500-page document has ~2000 chunks. Flat search scores all 2000.
    - Hierarchical: first find 3 relevant pages from 500 summaries,
      then search ~12 chunks within those 3 pages. Much more focused.
    - Summaries capture the "big picture" that individual chunks miss.

Usage:
    from hierarchical_indices_rag import HierarchicalIndicesRAG

    rag = HierarchicalIndicesRAG(file_path="document.pdf")
    answer, chunks = rag.query("What is the greenhouse effect?")
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    read_pdf_with_metadata,
    chunk_text,
)



# Page Summarizer

class PageSummarizer:
    """
    Summarizes document pages using OpenAI chat API.

    Uses ThreadPoolExecutor for parallel summarization instead of asyncio,
    which is simpler and doesn't require an event loop.

    Args:
        model_name:   OpenAI model for summarization.
        temperature:  Should be 0 for consistent summaries.
        max_workers:  Parallel summarization threads.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_workers: int = 5,
    ):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=500,
        )
        self.max_workers = max_workers

    def summarize_page(self, page_text: str) -> str:
        """
        Summarize a single page/section.

        Args:
            page_text:  Full text of the page.

        Returns:
            Summary string (2-4 sentences).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document summarizer. Write a concise 2-4 sentence "
                    "summary of the provided text. Focus on the main topics and "
                    "key information. Do not include preamble like 'This text discusses'."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this text:\n\n{page_text}",
            },
        ]

        return self.llm.chat(messages)

    def summarize_pages(self, pages: List[Document]) -> List[Document]:
        """
        Summarize multiple pages in parallel using ThreadPoolExecutor.

        Replaces the notebook's asyncio + batch + exponential backoff pattern
        with a simpler threading approach.

        Args:
            pages:  List of Document objects (one per page).

        Returns:
            List of Document objects containing summaries with page metadata.
        """
        summaries = [None] * len(pages)

        def _summarize(idx: int, page: Document) -> Tuple[int, str]:
            summary = self.summarize_page(page.content)
            return idx, summary

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(_summarize, i, page)
                for i, page in enumerate(pages)
            ]

            for future in as_completed(futures):
                idx, summary = future.result()
                page = pages[idx]
                summaries[idx] = Document(
                    content=summary,
                    metadata={
                        "source": page.metadata.get("source", ""),
                        "page": page.metadata.get("page", idx),
                        "is_summary": True,
                    },
                )
                print(f"    Summarized page {page.metadata.get('page', idx)}")

        return summaries

    

# Hierarchical Retriever
class HierarchicalRetriever:
    """
    Two-tier retriever: summaries (coarse) → detailed chunks (fine).

    Architecture:
        SUMMARY INDEX (FAISS):  One entry per page/section summary
        DETAIL INDEX (FAISS):   Many entries per page (small chunks)
        
        Both share the same embedder. Each detail chunk stores its
        page number in metadata for filtering.

    Retrieval flow:
        Query → search summaries → get relevant page numbers
              → search detail chunks filtered by those pages
              → return matched chunks

    Args:
        embedding_model:   OpenAI embedding model.
        summary_model:     OpenAI model for page summarization.
        chunk_size:        Characters per detail chunk.
        chunk_overlap:     Overlap between detail chunks.
        k_summaries:       Number of summary matches (pages to drill into).
        k_chunks:          Number of detail chunks per matched page.
        max_workers:       Parallel threads for summarization.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        summary_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_summaries: int = 3,
        k_chunks: int = 5,
        max_workers: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_summaries = k_summaries
        self.k_chunks = k_chunks

        # Shared embedder for both indexes
        self.embedder = OpenAIEmbedder(model=embedding_model)

        # Two separate FAISS indexes
        self.summary_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.detail_store = FAISSVectorStore(dimension=self.embedder.dimension)

        # Summarizer
        self.summarizer = PageSummarizer(
            model_name=summary_model,
            max_workers=max_workers,
        )

        # Track page metadata for detail chunks (used for filtering)
        self._detail_page_map: List[int] = []  # index → page number

        # Stats
        self.stats = {
            "pages": 0,
            "summaries": 0,
            "detail_chunks": 0,
        }

    def index_pdf(self, file_path: str) -> Tuple[int, int]:
        """
        Index a PDF with hierarchical (summary + detail) indexing.

        Full pipeline:
            1. Read PDF → one Document per page
            2. Summarize each page → store in summary index
            3. Chunk each page → store in detail index (tagged with page #)

        Args:
            file_path:  Path to PDF file.

        Returns:
            Tuple of (num_summaries, num_detail_chunks).
        """
        pages = read_pdf_with_metadata(file_path)
        self.stats["pages"] = len(pages)
        print(f"  [Hierarchical] {len(pages)} pages loaded")

        print(f"  [Hierarchical] Summarizing {len(pages)} pages...")
        summary_docs = self.summarizer.summarize_pages(pages)
        
        summary_docs = self.embedder.embed_documents(summary_docs)
        self.summary_store.add_documents(summary_docs)
        self.stats["summaries"] = len(summary_docs)
        print(f"  [Hierarchical] {len(summary_docs)} summaries indexed")

        detail_docs = []
        for page in pages:
            page_num = page.metadata.get("page", 0)
            chunks = chunk_text(
                page.content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            for j, chunk_content in enumerate(chunks):
                detail_docs.append(
                    Document(
                        content=chunk_content,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "chunk_index": j,
                            "is_summary": False,
                        },
                    )
                )

                self._detail_page_map.append(page_num)

        detail_docs = self.embedder.embed_documents(detail_docs)
        self.detail_store.add_documents(detail_docs)
        self.stats["detail_chunks"] = len(detail_docs)
        print(f"  [Hierarchical] {len(detail_docs)} detail chunks indexed")

        return len(summary_docs), len(detail_docs)

    def index_text(self, text: str, doc_id: str = "doc_0") -> Tuple[int, int]:
        """
        Index raw text by splitting into sections, summarizing, and chunking.

        For non-PDF text, we split into large sections first (acting as "pages"),
        then summarize and chunk each.

        Args:
            text:    Full document text.
            doc_id:  Document identifier.

        Returns:

            Tuple of (num_summaries, num_detail_chunks).
        """
        section_size = 3000
        sections = chunk_text(text, chunk_size=section_size, chunk_overlap=200)

        pages = []
        for i, section in enumerate(sections):
            pages.append(
                Document(
                    content=section,
                    metadata={"source": doc_id, "page": i, "total_pages": len(sections)},
                )
            )

        self.stats["pages"] = len(pages)
        print(f"  [Hierarchical] {len(pages)} sections created")

        print(f"  [Hierarchical] Summarizing {len(pages)} sections...")
        summary_docs = self.summarizer.summarize_pages(pages)
        summary_docs = self.embedder.embed_documents(summary_docs)
        self.summary_store.add_documents(summary_docs)
        self.stats["summaries"] = len(summary_docs)

        detail_docs = []
        for page in pages:
            page_num = page.metadata.get("page", 0)
            chunks = chunk_text(
                page.content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            for j, chunk_content in enumerate(chunks):
                detail_docs.append(
                    Document(
                        content=chunk_content,
                        metadata={"source": doc_id, "page": page_num, "chunk_index": j, "is_summary": False},
                    )
                )
                self._detail_page_map.append(page_num)

        detail_docs = self.embedder.embed_documents(detail_docs)
        self.detail_store.add_documents(detail_docs)
        self.stats["detail_chunks"] = len(detail_docs)

        return len(summary_docs), len(detail_docs)

    def retrieve(self, query: str) -> List[Document]:
        """
        Hierarchical retrieval: summaries first, then detail chunks.

        Flow:
            1. Search SUMMARY index → top k_summaries pages
            2. For each matched page, search DETAIL index
               (manually filter by page number since FAISS doesn't
               support metadata filtering natively)
            3. Return all matched detail chunks

        This replaces the notebook's retrieve_hierarchical() function
        and its use of FAISS similarity_search(filter=page_filter).

        Args:
            query:  Search query.

        Returns:
            List of relevant detail chunk Documents.
        """
        query_emb = self.embedder.embed_text(query)
        summary_results = self.summary_store.search(query_emb, k=self.k_summaries)

        relevant_pages = set()
        for r in summary_results:
            page = r.document.metadata.get("page")
            if page is not None:
                relevant_pages.add(page)

        if not relevant_pages:
            return []

        print(
            f"  [Hierarchical] Summary search → pages: {sorted(relevant_pages)}"
        )

        search_k = min(
            len(self.detail_store.documents),
            self.k_chunks * len(relevant_pages) * 3,
        )
        detail_results = self.detail_store.search(query_emb, k=search_k)

        filtered_chunks = []
        page_counts = {p: 0 for p in relevant_pages}

        for r in detail_results:
            page = r.document.metadata.get("page")
            if page in relevant_pages and page_counts[page] < self.k_chunks:
                filtered_chunks.append(r.document)
                page_counts[page] += 1

            if all(c >= self.k_chunks for c in page_counts.values()):
                break

        print(
            f"  [Hierarchical] Detail search → {len(filtered_chunks)} chunks "
            f"from {len(relevant_pages)} pages"
        )

        return filtered_chunks

    def retrieve_context(self, query: str) -> List[str]:
        """Convenience: return just the chunk texts."""
        chunks = self.retrieve(query)
        return [c.content for c in chunks]



# Complete RAG Pipeline
class HierarchicalIndicesRAG:
    """
    Complete RAG pipeline using hierarchical (summary + detail) indexing.

    Combines HierarchicalRetriever with OpenAIChat for answer generation.

    Usage:
        rag = HierarchicalIndicesRAG(file_path="report.pdf")
        answer, chunks = rag.query("What is the greenhouse effect?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_summaries: int = 3,
        k_chunks: int = 5,
        embedding_model: str = "text-embedding-3-small",
        summary_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_workers: int = 5,
    ):
        """
        Initialize the Hierarchical Indices RAG pipeline.

        Args:
            file_path:         Path to document (PDF or text file).
            chunk_size:        Characters per detail chunk.
            chunk_overlap:     Overlap between detail chunks.
            k_summaries:       Pages to match via summary search.
            k_chunks:          Detail chunks per matched page.
            embedding_model:   OpenAI embedding model.
            summary_model:     OpenAI model for summarization.
            chat_model:        OpenAI model for answer generation.
            temperature:       LLM temperature for answers.
            max_workers:       Parallel summarization threads.
        """
        self.file_path = file_path

        self.retriever = HierarchicalRetriever(
            embedding_model=embedding_model,
            summary_model=summary_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k_summaries=k_summaries,
            k_chunks=k_chunks,
            max_workers=max_workers,
        )

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        print(
            f"[Hierarchical] Indexing '{os.path.basename(file_path)}' "
            f"(k_summaries={k_summaries}, k_chunks={k_chunks})..."
        )

        if file_path.endswith(".pdf"):
            n_sum, n_det = self.retriever.index_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            n_sum, n_det = self.retriever.index_text(text)

        stats = self.retriever.stats
        print(
            f"[Hierarchical] Done → {stats['pages']} pages, "
            f"{n_sum} summaries, {n_det} detail chunks"
        )

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Query the hierarchical RAG system.

        Flow:
            1. Search summaries → identify relevant pages
            2. Search detail chunks within those pages
            3. Feed detail chunks to answer LLM

        Args:
            question:        User's question.
            return_context:  Whether to return retrieved chunks.

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

    def show_hierarchy(self, question: str) -> None:
        """
        Debug helper: show the full hierarchical retrieval flow.

        Displays which summaries matched, which pages were selected,
        and which detail chunks were returned.

        Args:
            question:  Search query.
        """
        print(f"\nQuery: {question}")
        print("=" * 70)

        # Step 1: Show summary matches
        query_emb = self.retriever.embedder.embed_text(question)
        summary_results = self.retriever.summary_store.search(
            query_emb, k=self.retriever.k_summaries
        )

        print(f"\n TIER 1 — Summary Matches ({len(summary_results)}):")
        print("-" * 50)
        for i, r in enumerate(summary_results):
            page = r.document.metadata.get("page", "?")
            print(f"  {i+1}. [Page {page}, score={r.score:.4f}]")
            print(f"     {r.document.content[:200]}...")

        # Step 2: Show detail matches
        detail_chunks = self.retriever.retrieve(question)

        print(f"\n TIER 2 — Detail Chunks ({len(detail_chunks)}):")
        print("-" * 50)
        for i, chunk in enumerate(detail_chunks):
            page = chunk.metadata.get("page", "?")
            preview = chunk.content[:200].replace('\n', ' ')
            print(f"  {i+1}. [Page {page}] {preview}...")

        print("\n" + "=" * 70)
