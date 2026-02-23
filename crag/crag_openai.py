"""
Corrective RAG (CRAG): Retrieval-Augmented Generation with Dynamic Correction

CRAG extends standard RAG by evaluating retrieved document relevance and
dynamically correcting the knowledge source when local retrieval is insufficient.

Pipeline:
    1. Retrieve top-k documents from vector store
    2. Evaluate relevance of each document (0.0 - 1.0 score via LLM)
    3. Based on max relevance score, take one of three actions:
        - CORRECT  (score > 0.7): Use best retrieved document directly
        - INCORRECT (score < 0.3): Discard retrieval, perform web search instead
        - AMBIGUOUS  (0.3 - 0.7): Combine best document + web search results
    4. Refine knowledge (extract key points from web results)
    5. Generate final response with source attribution

Uses:
    - OpenAIChat.chat_json() for structured relevance scoring
    - RAGRetriever for vector search
    - DuckDuckGo for web search fallback (no API key needed)
"""



import os
import sys
import json
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
class CRAGResponse:
    """Complete CRAG response with evaluation metadata."""
    answer: str
    action: str
    sources: List[Tuple[str, str]]
    relevance_scores: List[float]
    max_score: float
    context_used: List[str]
    retrieval_needed: bool = True


def web_search_duckduckgo(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo.
    No API key required — uses the duckduckgo_search package.
    
    Returns: List of {"title": ..., "link": ..., "snippet": ...}
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "Untitled"),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    except ImportError:
        print("[CRAG] duckduckgo_search not installed. Run: pip install duckduckgo-search")
        return []

    except Exception as e:
        print(f"[CRAG] Web search error: {e}")
        return []




class CRAGOpenAI:
    """
    Corrective RAG.
    
    Evaluates retrieval quality and dynamically corrects by:
    - Using local docs when highly relevant (score > 0.7)
    - Falling back to web search when irrelevant (score < 0.3)
    - Combining both sources when ambiguous (0.3 - 0.7)
    
    Args:
        file_path:          Path to PDF or CSV to index
        chunk_size:         Characters per chunk
        chunk_overlap:      Overlap between chunks
        top_k:              Number of documents to retrieve
        chat_model:         OpenAI model for evaluation + generation
        temperature:        LLM temperature
        high_threshold:     Score above which retrieval is trusted (default 0.7)
        low_threshold:      Score below which retrieval is discarded (default 0.3)
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        self.file_path = file_path
        self.top_k = top_k
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # Initialize retriever
        self.retriever = RAGRetriever(embedding_model="text-embedding-3-small")

        # Initialize LLM
        self.llm = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
            max_tokens=1000,
        )

        # Index document
        self._index_file()

    def _index_file(self):
        """Index the document into the vector store."""
        if self.file_path.endswith(".pdf"):
            count = self.retriever.index_pdf(
                self.file_path,
                chunk_size=1000,
                chunk_overlap=200,
            )
        elif self.file_path.endswith(".csv"):
            count = self.retriever.index_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")
        print(f"[CRAG] Indexed {count} chunks from {os.path.basename(self.file_path)}")


    def _retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve top-k documents from vector store."""
        return self.retriever.retrieve(query, k=self.top_k)


    def _evaluate_relevance(self, query: str, document: str) -> float:
        """
        Score the relevance of a document to the query on a 0.0-1.0 scale.
        Uses structured JSON output for reliable parsing.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance evaluator. "
                    "Score how relevant the document is to answering the query. "
                    'Respond with JSON: {"relevance_score": <float between 0.0 and 1.0>}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument: {document}",
            },
        ]
        result = self.llm.chat_json(messages)
        score = result.get("relevance_score", 0.5)
        return max(0.0, min(1.0, float(score)))


    def _rewrite_query_for_web(self, query: str) -> str:
        """Rewrite the query to be more suitable for web search."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. "
                    "Rewrite the query to get better web search results. "
                    'Respond with JSON: {"rewritten_query": "<optimized query>"}'
                ),
            },
            {
                "role": "user",
                "content": f"Original query: {query}",
            },
        ]
        result = self.llm.chat_json(messages)
        return result.get("rewritten_query", query)

    def _web_search(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Perform web search and return refined knowledge + sources.
        
        Returns:
            Tuple of (refined_knowledge_string, list of (title, link) tuples)
        """

        rewritten = self._rewrite_query_for_web(query)
        print(f"  [Web] Rewritten query: {rewritten}")


        web_results = web_search_duckduckgo(rewritten, max_results=3)

        if not web_results:
            return "No web results found.", []

 
        sources = [(r["title"], r["link"]) for r in web_results]

        raw_text = "\n\n".join(
            f"Source: {r['title']}\n{r['snippet']}" for r in web_results
        )

        # Refine: extract key points via LLM
        refined = self._refine_knowledge(raw_text)

        return refined, sources

    
    def _refine_knowledge(self, raw_text: str) -> str:
        """Extract key information from raw text into concise bullet points."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract the key information from the following text as concise bullet points. "
                    "Focus on facts that would help answer a question. "
                    'Respond with JSON: {"key_points": ["point1", "point2", ...]}'
                ),
            },
            {
                "role": "user",
                "content": raw_text,
            },
        ]
        result = self.llm.chat_json(messages)
        points = result.get("key_points", [])
        return "\n".join(f"• {p}" for p in points)

    def _generate_response(
        self, query: str, knowledge: str, sources: List[Tuple[str, str]]
    ) -> str:
        """Generate final response with source attribution."""
        source_text = "\n".join(
            f"- {title}: {link}" if link else f"- {title}"
            for title, link in sources
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question based on the provided knowledge. "
                    "Be concise and accurate. Include source references at the end of your answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Knowledge:\n{knowledge}\n\n"
                    f"Sources:\n{source_text}\n\n"
                    f"Question: {query}\n\nAnswer:"
                ),
            },
        ]
        return self.llm.chat(messages)

    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        Standard interface: returns (answer, context_list).
        Compatible with evaluation framework.
        """
        result = self.crag(question)
        return result.answer, result.context_used

    def crag(self, query: str) -> CRAGResponse:
        """
        Full CRAG pipeline with detailed metadata.
        
        Flow:
            1. Retrieve documents
            2. Score relevance per document
            3. Decide action: CORRECT / INCORRECT / AMBIGUOUS
            4. Acquire + refine knowledge accordingly
            5. Generate response with sources
        """
        print(f"\n{'='*60}")
        print(f"  CRAG Pipeline")
        print(f"  Query: {query}")
        print(f"{'='*60}")

        # ── Step 1: Retrieve ──
        print(f"\n[Step 1] Retrieving top-{self.top_k} documents...")
        results = self._retrieve(query)
        docs = [(r.document.content, r.score) for r in results]
        print(f"  → Retrieved {len(docs)} documents")

        # ── Step 2: Evaluate Relevance ──
        print(f"\n[Step 2] Evaluating relevance (LLM scoring)...")
        relevance_scores = []
        for i, (content, sim_score) in enumerate(docs):
            rel_score = self._evaluate_relevance(query, content)
            relevance_scores.append(rel_score)
            print(f"  Doc {i+1} (sim={sim_score:.3f}): relevance={rel_score:.2f}")

        max_score = max(relevance_scores)
        best_idx = relevance_scores.index(max_score)
        best_doc = docs[best_idx][0]

        # ── Step 3: Decide Action ──
        print(f"\n[Step 3] Max relevance: {max_score:.2f}")

        if max_score > self.high_threshold:
            # ── CORRECT: Use retrieved document directly ──
            action = "correct"
            print(f"  → Action: CORRECT (score > {self.high_threshold})")
            print(f"    Using best retrieved document (Doc {best_idx+1})")
            final_knowledge = best_doc
            sources = [("Retrieved document", "")]
            context_used = [best_doc]

        elif max_score < self.low_threshold:
            # ── INCORRECT: Discard retrieval, use web search ──
            action = "incorrect"
            print(f"  → Action: INCORRECT (score < {self.low_threshold})")
            print(f"    Discarding local retrieval. Performing web search...")
            final_knowledge, sources = self._web_search(query)
            context_used = [final_knowledge]

        else:
            # ── AMBIGUOUS: Combine best document + web search ──
            action = "ambiguous"
            print(f"  → Action: AMBIGUOUS ({self.low_threshold} ≤ score ≤ {self.high_threshold})")
            print(f"    Combining best document (Doc {best_idx+1}) with web search...")

            # Refine the retrieved doc
            refined_local = self._refine_knowledge(best_doc)

            # Also get web knowledge
            web_knowledge, web_sources = self._web_search(query)

            # Combine both
            final_knowledge = f"Local Knowledge:\n{refined_local}\n\nWeb Knowledge:\n{web_knowledge}"
            sources = [("Retrieved document", "")] + web_sources
            context_used = [best_doc, web_knowledge]

        print(f"\n[Step 4] Generating response...")
        answer = self._generate_response(query, final_knowledge, sources)

        print(f"\n  → Action taken: {action.upper()}")
        print(f"  → Sources: {len(sources)}")

        return CRAGResponse(
            answer=answer,
            action=action,
            sources=sources,
            relevance_scores=relevance_scores,
            max_score=max_score,
            context_used=context_used,
        )

    
if __name__ == "__main__":

    pdf_file_path = r"data\Understanding_Climate_Change.pdf"

    rag = CRAGOpenAI(file_path=pdf_file_path)

    question = "What is the impact of ai summit at india 2026?"

    response, context_used = rag.query(question)

    print(f"Answer: {response}")
    print(f"Context Used: {context_used}")