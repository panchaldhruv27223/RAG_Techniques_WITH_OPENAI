"""
Explainable Retrieval RAG

A RAG system that doesn't just retrieve and answer — it EXPLAINS WHY
each chunk was retrieved and how it relates to the query. This turns
the retrieval process from a black box into a transparent, debuggable,
and trustworthy system.

Standard RAG:
    Query → chunks → answer (you never know WHY those chunks were picked)

Explainable RAG:
    Query → chunks + explanation per chunk → answer
    "This chunk was retrieved because it discusses greenhouse gas
     mechanisms, which directly answers the query about warming causes."

Why this matters:
    - Trust: Users see WHY the system chose specific context
    - Debugging: Developers spot bad retrievals immediately
    - Learning: Users discover connections they didn't expect
    - Auditing: Critical for legal, medical, and compliance use cases

How it works:
    1. Standard vector search → retrieve top-k chunks
    2. For EACH chunk: LLM explains "why is this relevant to the query?"
    3. Optionally: use explanations to filter out low-relevance chunks
    4. Feed remaining chunks + explanations to answer LLM

Usage:
    from explainable_retrieval_rag import ExplainableRetrievalRAG

    rag = ExplainableRetrievalRAG(file_path="document.pdf")
    answer, explained = rag.query("What causes global warming?")
    rag.show_explanations("What causes global warming?")
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

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


# Data class for explained results
@dataclass
class ExplainedChunk:
    """A retrieved chunk with its relevance explanation."""
    content: str
    explanation: str
    score: float
    metadata: Dict[str, Any]

class ChunkExplainer:
    """
    Generates natural language explanations for why a chunk is relevant.

    With a direct OpenAI chat call that produces structured explanations.

    Args:
        model_name:   OpenAI model for explanation generation.
        temperature:  Low for consistent explanations.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=300,
        )

    def explain(self, query: str, chunk_content: str) -> str:
        """
        Explain why a chunk is relevant to the query.

        Args:
            query:          User's search query.
            chunk_content:  Content of the retrieved chunk.

        Returns:
            Natural language explanation string.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze why a retrieved document chunk is relevant to a query. "
                    "Give a concise 2-3 sentence explanation covering: "
                    "(1) what key information in the chunk relates to the query, "
                    "(2) how it helps answer the query. "
                    "Be specific - reference actual content from the chunk."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Retrieved chunk:\n{chunk_content}\n\n"
                    f"Why is this chunk relevant?"
                ),
            },
        ]

        return self.llm.chat(messages)

    def explain_batch(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[ExplainedChunk]:
        """
        Generate explanations for multiple retrieved chunks.

        Args:
            query:    User's search query.
            results:  Retrieved chunks with scores.

        Returns:
            List of ExplainedChunk objects.
        """
        explained = []
        for i, r in enumerate(results):
            print(f"    Explaining chunk {i+1}/{len(results)}...")
            explanation = self.explain(query, r.document.content)
            explained.append(
                ExplainedChunk(
                    content=r.document.content,
                    explanation=explanation,
                    score=r.score,
                    metadata=r.document.metadata,
                )
            )
        return explained


class ExplainableRetriever:
    """
    Retriever that provides explanations alongside each retrieved chunk.

    Pipeline:
        1. Index documents → FAISS
        2. On query:
           a. Vector search → top-k chunks
           b. For each chunk: LLM explains relevance
           c. Return chunks + explanations

    Args:
        embedding_model:  OpenAI embedding model.
        explain_model:    OpenAI model for generating explanations.
        chunk_size:       Characters per chunk.
        chunk_overlap:    Overlap between chunks.
        k:                Number of chunks to retrieve.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        explain_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.explainer = ChunkExplainer(model_name=explain_model)

    def index_texts(self, texts: List[str]) -> int:
        """Index a list of text strings directly (like the notebook's usage)."""
        documents = []
        for i, text in enumerate(texts):
            documents.append(
                Document(
                    content=text,
                    metadata={"text_index": i},
                )
            )
        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)
        return len(documents)

    def index_document(self, text: str, doc_id: str = "doc_0") -> int:
        """Chunk and index a full document."""
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        documents = []
        for i, c in enumerate(chunks):
            documents.append(
                Document(
                    content=c,
                    metadata={"doc_id": doc_id, "chunk_index": i},
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

    def retrieve_and_explain(self, query: str) -> List[ExplainedChunk]:
        """
        Retrieve chunks AND generate explanations for each.

        Args:
            query:  Search query.

        Returns:
            List of ExplainedChunk objects (content + explanation + score).
        """
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=self.k)

        print(f"  [Explainable] Retrieved {len(results)} chunks, generating explanations...")
        return self.explainer.explain_batch(query, results)

    def retrieve_context(self, query: str) -> Tuple[List[str], List[ExplainedChunk]]:
        """Convenience: return (context_texts, explained_chunks)."""
        explained = self.retrieve_and_explain(query)
        contexts = [e.content for e in explained]
        return contexts, explained




class ExplainableRetrievalRAG:
    """
    Complete RAG pipeline with transparent, explainable retrieval.

    Every retrieval is accompanied by natural language explanations
    of WHY each chunk was selected.

    Replaces the notebook's:
        - ExplainableRetriever class with LangChain dependencies
        - FAISS.from_texts() + as_retriever()
        - PromptTemplate | llm explanation chain
        - Manual result formatting

    Usage:
        rag = ExplainableRetrievalRAG(file_path="report.pdf")
        answer, explained = rag.query("What causes global warming?")
        rag.show_explanations("What causes global warming?")
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        texts: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        embedding_model: str = "text-embedding-3-small",
        explain_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        explain_pass_threshold: float = 0.7,
    ):
        """
        Initialize the Explainable RAG pipeline.

        Provide EITHER file_path (PDF/text file) OR texts (list of strings).

        Args:
            file_path:       Path to document file.
            texts:           List of text strings to index directly.
            chunk_size:      Characters per chunk (for file indexing).
            chunk_overlap:   Overlap between chunks.
            k:               Number of chunks to retrieve.
            embedding_model: OpenAI embedding model.
            explain_model:   OpenAI model for explanations.
            chat_model:      OpenAI model for answer generation.
            temperature:     LLM temperature.
        """
        self.retriever = ExplainableRetriever(
            embedding_model=embedding_model,
            explain_model=explain_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        self.explain_pass_threshold = explain_pass_threshold

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        # Index from texts or file
        if texts is not None:
            num = self.retriever.index_texts(texts)
            print(f"[Explainable] Indexed {num} text entries (k={k})")
        elif file_path is not None:
            if file_path.endswith(".pdf"):
                num = self.retriever.index_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    num = self.retriever.index_document(f.read())
            print(f"[Explainable] Indexed '{os.path.basename(file_path)}' → {num} chunks (k={k})")
        else:
            raise ValueError("Provide either file_path or texts")

    def query(
        self,
        question: str,
        return_explanations: bool = False,
    ) -> Tuple[str, List[ExplainedChunk]]:
        """
        Query with explainable retrieval.

        Returns both the answer AND the explained chunks so users
        can see exactly why each piece of context was used.

        Args:
            question:              User's question.
            return_explanations:   Whether to return explained chunks.

        Returns:
            Tuple of (answer_string, list_of_ExplainedChunk).
        """
        contexts, explained = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found.", []

        used_context = []

        for idx, e in enumerate(explained):
            if e.score > self.explain_pass_threshold:
                used_context.append(e.content)

        answer = self.chat.chat_with_context(question, used_context)

        if return_explanations:
            return answer, used_context, explained

        return answer, used_context

    def show_explanations(self, question: str) -> None:
        """
        Debug helper: show each retrieved chunk with its explanation.

        This is the key feature — making retrieval transparent.

        Args:
            question:  Search query.
        """
        explained = self.retriever.retrieve_and_explain(question)
        print(f"Total explained chunks: {len(explained)}")

        ## lets find usefull contexts only.

        used_explained = []
        for idx, e in enumerate(explained):
            if e.score > self.explain_pass_threshold:
                used_explained.append(e)

        print(f"Total used contexts: {len(used_explained)}")

        print(f"\nQuery: {question}")
        print("=" * 70)

        for i, e in enumerate(used_explained):
            print(f"\n  Chunk {i+1} (score: {e.score:.4f}):")
            print(f"  {'─' * 50}")

            # Content preview
            preview = e.content[:300].replace('\n', ' ')
            print(f"  Content: {preview}...")

            # Explanation
            print(f"\n  Why relevant:")
            for line in e.explanation.split('. '):
                line = line.strip()
                if line:
                    print(f"     • {line}{'.' if not line.endswith('.') else ''}")

            print()

        print("=" * 70)


if __name__ == "__main__":

    pdf_file_path = r"data\Understanding_Climate_Change.pdf"

    rag = ExplainableRetrievalRAG(file_path=pdf_file_path, explain_pass_threshold=0.5)

    user_question = "What are the main causes of climate change?"

    rag.show_explanations(user_question)

    answer, used_context = rag.query(user_question)

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    