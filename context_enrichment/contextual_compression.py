"""
Contextual Compression RAG

Standard RAG retrieves full chunks, but often only a small portion of each
chunk is actually relevant to the query. Contextual compression uses an LLM
to extract/compress only the relevant parts BEFORE passing them to the
answer-generation LLM.

Key insight: Retrieved chunks are like full paragraphs — the answer might be
in one sentence. The compressor acts as a filter, distilling each chunk down
to just the parts that matter for the specific query.


How it works:
    1. Chunk and index document normally (standard RAG indexing)
    2. On query: retrieve top-k chunks via vector search
    3. For EACH retrieved chunk, ask an LLM to extract only the parts
       relevant to the query (this is the "compression" step)
    4. Filter out chunks where the LLM found nothing relevant
    5. Feed the compressed extracts to the answer-generation LLM

Usage:
    from contextual_compression_rag import ContextualCompressionRAG

    rag = ContextualCompressionRAG(file_path="document.pdf")
    answer, compressed = rag.query("What is the main topic?")
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


class ContextualCompressor:
    """
    Uses an LLM to extract only the query-relevant portions from a chunk.


    Args:
        model_name:   OpenAI model for compression.
        temperature:  Should be 0 for deterministic extraction.
        max_tokens:   Max tokens for compressed output.
    """

    def __init__(
        self,
        model_name:str = "gpt-4o-mini",
        temperature:float=0.0,
        max_tokens:int=5000
        ):

        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
            )

    
    def compress(self, chunk_text:str, query:str)->str:
        """
        Extract only the query-relevant portions from a chunk.

        This is the core operation. The LLM reads the chunk in the context
        of the query and returns only the relevant parts. If nothing is
        relevant, it returns "NO_RELEVANT_CONTENT".

        Args:
            chunk_text:  The full retrieved chunk text.
            query:       The user's question.

        Returns:
            Compressed/extracted text, or empty string if nothing relevant.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert text extractor. Given a user query and a text chunk, "
                    "extract ONLY the parts of the text that are directly relevant to answering "
                    "the query. Preserve the original wording — do not paraphrase or summarize. "
                    "If no part of the text is relevant to the query, respond with exactly: "
                    "NO_RELEVANT_CONTENT"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Text chunk:\n{chunk_text}\n\n"
                    "Extract only the relevant parts:"
                ),
            },
        ]

        result = self.llm.chat(messages)

        # Check if the LLM found nothing relevant
        if "NO_RELEVANT_CONTENT" in result.strip():
            return ""

        return result.strip()


    
    def compress_document(
        self,
        chunks:List[str],
        query:str,
        )->List[str]:

        """
        Compress multiple chunks, filtering out irrelevant ones.

        Args:
            chunks:  List of retrieved chunk texts.
            query:   The user's question.

        Returns:
            List of compressed texts (empty/irrelevant chunks removed).
        """
        compressed = []
        for i, chunk in enumerate(chunks):
            result = self.compress(chunk, query)

            if result:
                compressed.append(result)

        return compressed


class ContextualCompressorRetriever:
    """
    Retriever that compresses each retrieved chunk to only the relevant parts.

    Pipeline:
        1. Index document with standard chunking → FAISS
        2. On query: vector search → top-k chunks
        3. For each chunk: LLM extracts only relevant portions
        4. Filter out chunks where nothing was relevant
        5. Return compressed extracts
    

    Args:
        embedding_model:   OpenAI embedding model.
        compressor_model:  OpenAI model for chunk compression.
        chunk_size:        Characters per chunk.
        chunk_overlap:     Overlap between chunks.
        k:                 Number of chunks to retrieve before compression.
    """

    def __init__(
        self,
        embedding_model:str="text-embedding-3-small",
        compressor_model:str="gpt-4o-mini",
        chunk_size:int=1000,
        chunk_overlap:int=200,
        k:int=5
        ):

        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.k=k

        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.compressor = ContextualCompressor(model_name=compressor_model)

    
    def index_document(self, text:str, doc_id:str="doc_0")->int:
        """
        Chunk and index a document. Standard RAG indexing — nothing special here.

        Args:
            text:    Full document text.
            doc_id:  Document identifier.

        Returns:
            Number of chunks created.
        """
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
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



    def retrieve_raw(self, query: str) -> List[RetrievalResult]:
        """
        Standard vector search (no compression). For comparison.

        Args:
            query:  Search query.

        Returns:
            List of RetrievalResult objects (full chunks).
        """
        query_emb = self.embedder.embed_text(query)
        return self.vector_store.search(query_emb, k=self.k)


    
    def retrieve_compressed(self, query:str)->Tuple[List[str],List[str]]:
        """
        Retrieve chunks, then compress each to only the relevant parts.

        This is the core method. The flow is:
            1. Vector search → top-k full chunks
            2. For each chunk → LLM extracts relevant portions
            3. Filter out empty results

        Args:
            query:  Search query.

        Returns:
            Tuple of (compressed_texts, original_texts).
            compressed_texts has irrelevant chunks filtered out.
        """

        results = self.retrieve_raw(query)
        original_texts = [r.document.content for r in results]

        if not original_texts:
            return [], []

        
        ## Now we compress each chunk

        compressed_texts = self.compressor.compress_document(original_texts, query)

        return compressed_texts, original_texts

    def retrieve_context(self, query:str) -> List[str]:
        """
        Convenience method: return just the compressed texts.

        Args:
            query:  Search query.

        Returns:
            List of compressed context strings.
        """
        compressed, _ = self.retrieve_compressed(query)
        return compressed



class ContextualCompressionRAG:
    """
    Complete RAG pipeline with contextual compression.


    Usage:
        rag = ContextualCompressionRAG(file_path="report.pdf")
        answer, compressed = rag.query("What is the main topic?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        embedding_model: str = "text-embedding-3-small",
        compressor_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        ):

        """
        Initialize the Contextual Compression RAG pipeline.

        Args:
            file_path:         Path to document (PDF or text file).
            chunk_size:        Characters per chunk.
            chunk_overlap:     Overlap between chunks.
            k:                 Number of chunks to retrieve before compression.
            embedding_model:   OpenAI embedding model.
            compressor_model:  OpenAI model for compressing chunks (fast/cheap).
            chat_model:        OpenAI model for final answer generation.
            temperature:       LLM temperature for answer generation.
        """
        self.file_path = file_path

        self.retriever = ContextualCompressorRetriever(
            embedding_model=embedding_model,
            compressor_model=compressor_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k
        )

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )


        # Index the document
        if file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(file_path)
        else:
            num_chunks = self.retriever.index_text_file(file_path)


    def query(
        self,
        question: str,
        return_context: bool = True,
        ) -> Tuple[str, List[str]]:
        
        """
        Query the RAG system with contextual compression.

        Flow:
            1. Vector search → top-k full chunks
            2. LLM compresses each chunk to relevant parts only
            3. Compressed extracts → answer-generation LLM
            4. Return answer + compressed contexts

        Args:
            question:        User's question.
            return_context:  Whether to return compressed contexts.

        Returns:
            Tuple of (answer_string, list_of_compressed_context_strings).
        """


        compressed_contexts = self.retriever.retrieve_context(question)
        
        if not compressed_contexts:
            return "No Relevant Information found in the document.", []
        
        answer = self.chat.chat_with_context(question, compressed_contexts)

        if return_context:
            return answer, compressed_contexts

        return answer, []

    
    def compare(self, question:str)->None:
        """
        Debug helper: compare raw chunks vs compressed extracts side by side.

        Shows how much irrelevant content the compressor removes.

        Args:
            question:  Search query.
        """
        compressed, originals = self.retriever.retrieve_compressed(question)

        total_raw_chars = 0
        for i, chunk in enumerate(originals):
            total_raw_chars += len(chunk)
            print(f"\n  Chunk {i + 1} ({len(chunk)} chars):")
            preview = chunk[:200].replace('\n', ' ')
            print(f"    {preview}...")


        total_compressed_chars = 0
        for i, extract in enumerate(compressed):
            total_compressed_chars += len(extract)
            print(f"\n  Extract {i + 1} ({len(extract)} chars):")
            preview = extract[:300].replace('\n', ' ')
            print(f"    {preview}...")

        if total_raw_chars > 0:

            ratio = (1 - total_compressed_chars / total_raw_chars) * 100
            print(f"    Raw:        {total_raw_chars:,} chars across {len(originals)} chunks")
            print(f"    Compressed: {total_compressed_chars:,} chars across {len(compressed)} extracts")
            print(f"    Reduction:  {ratio:.1f}%")
            print(f"    Chunks filtered out: {len(originals) - len(compressed)}")

if __name__ == "__main__":
    
    pdf_file_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"

    rag = ContextualCompressionRAG(
        file_path=pdf_file_path,
        chunk_size=1000,
        chunk_overlap=200,
        k=5,
        compressor_model="gpt-4o-mini",
        chat_model="gpt-4o-mini"
    )


    user_query = "What is the main topic of the document?"

    answer, compressed = rag.query(user_query)

    print("\nAnswer:")
    print(answer)

    print("\nCompressed Extracts:")
    for i, extract in enumerate(compressed):
        print(f"\nExtract {i+1}:")
        print(extract)
        