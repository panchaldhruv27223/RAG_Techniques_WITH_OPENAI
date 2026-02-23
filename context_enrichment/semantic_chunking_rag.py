import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np 
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    cosine_similarity,
    chunk_text
)

def split_into_sentences(text:str)->List[str]:
    """
    Split text into sentences using regex-based rules.

    Handles common abbreviations, decimal numbers, and other edge cases
    to avoid false splits. Not perfect, but good enough for chunking.

    Args:
        text:  Raw document text.

    Returns:
        List of sentence strings (whitespace-stripped, non-empty).
    """

    ## normalize whitespace
    text = re.sub(r"\s+", ' ', text).strip()

    # Split on sentence-ending punctuation followed by space + uppercase
    # or newline patterns that indicate paragraph breaks
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+',
        text
    )

    ## Clean up and filter empty strings
    cleaned = []

    for s in sentences:
        s = s.strip()
        if s and len(s) > 5:
            cleaned.append(s)

    return cleaned


class SemanticChunker:
    """
    Splits text into semantically coherent chunks by detecting topic shifts
    between consecutive sentences using embedding similarity.

        1. Embed every sentence
        2. Compute similarity between consecutive sentence pairs
        3. Find breakpoints where similarity drops significantly
        4. Group sentences between breakpoints into chunks

    Three breakpoint detection methods are supported:
        - 'percentile':          Split where similarity < Nth percentile
        - 'standard_deviation':  Split where similarity < mean - N*std
        - 'interquartile':       Split where similarity < Q1 - N*IQR


    Args:
        embedder:          OpenAIEmbedder instance for sentence embeddings.
        method:            Breakpoint detection method.
        threshold:         Threshold value (percentile number, std multiplier,
                           or IQR multiplier depending on method).
        min_chunk_size:    Minimum characters per chunk. Tiny chunks get
                           merged with the next one.

    """
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        method:Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        threshold:float = 90.0,
        min_chunk_size:int = 1000,
        max_chunk_size: int = 20000
    ):
        self.embedder = embedder
        self.method = method
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    
    def _compute_similarity(self, embeddings:List[List[float]])->List[float]:
        """
        Compute cosine similarity between each pair of consecutive sentences.

        Given N sentence embeddings, returns N-1 similarity scores.
        similarity[i] = cosine_sim(embedding[i], embedding[i+1])

        Args:
            embeddings:  List of sentence embedding vectors.

        Returns:
            List of similarity scores (length = len(embeddings) - 1).
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        return similarities

    def _find_breakpoints(self, similarities: List[float])->List[int]:
        """
        Detect breakpoint indices where the topic shifts.

        A breakpoint at index i means: split AFTER sentence i.
        The method determines what counts as a "significant" drop.

        Args:
            similarities:  Consecutive sentence-pair similarity scores.

        Returns:
            List of indices where splits should occur.
        """
        if not similarities:
            return []

        if self.method == "percentile":
            breakpoint_indices = [
                i for i, sim in enumerate(similarities)
                if sim < np.percentile(similarities, 100-self.threshold)
            ]
        elif self.method == "standard_deviation":
            mean = np.mean(similarities)
            std = np.std(similarities)
            breakpoint_indices = [
                i for i, sim in enumerate(similarities)
                if sim < (mean - (self.threshold * std))
            ]
        elif self.method == "interquartile":
            q1 = np.percentile(similarities, 25)
            iqr = np.percentile(similarities, 75) - np.percentile(similarities, 25)
            breakpoint_indices = [
                i for i, sim in enumerate(similarities)
                if sim < (q1 - (self.threshold * iqr))
            ]
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return breakpoint_indices

    
    def chunk(self, text:str)->List[str]:
        """
        Split text into semantic chunks.

        Full pipeline:
            1. Split text → sentences
            2. Embed all sentences (batched API call)
            3. Compute consecutive similarities
            4. Find breakpoints
            5. Group sentences between breakpoints
            6. Merge tiny chunks with neighbors

        Args:
            text:  Full document text.

        Returns:
            List of chunk strings, each containing one or more sentences.
        """

        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            return [text.split()] if text.split() else []

        print(f"[SemanticChunker] {len(sentences)} sentences found.")

        ## Embed sentences into one batch
        embeddings = self.embedder.embed_texts(sentences)

        similarities = self._compute_similarity(embeddings=embeddings)

        breakpoints = self._find_breakpoints(similarities=similarities)

        print(f"[SemanticChunker] {len(breakpoints)} breakpoints found.")

        chunks = []

        start = 0

        for bp in breakpoints:

            chunk_sentences = sentences[start:bp+1]
            chunk_text_raw = " ".join(chunk_sentences)
            chunks.append(chunk_text_raw)
            start = bp + 1

        if start < len(sentences):
            last_chunk = " ".join(sentences[start:])
            chunks.append(last_chunk)

        merged = []
        buffer = ""

        for chunk in chunks:

            if buffer:
                chunk = buffer + " " + chunk
                buffer = ""

            if len(chunk) < self.min_chunk_size:
                buffer = chunk
            
            else:
                merged.append(chunk)

        if buffer:
            if merged:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        print(f"  [SemanticChunker] {len(merged)} chunks created.    ")

        final_chunks = []

        for c in merged:
            if len(c) > self.max_chunk_size:
                sub_chunks = chunk_text(c, chunk_size=self.max_chunk_size, chunk_overlap=0)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(c)

        return final_chunks

class SemanticChunkingRetriever:

    """
    Retriever that uses semantic chunking instead of fixed-size chunking.

    Pipeline:
        1. Read document → split into semantic chunks (topic-aware)
        2. Embed chunks → store in FAISS
        3. On query: standard vector search over semantic chunks

    The key difference from other retrievers is the CHUNKING strategy —
    retrieval itself is standard top-k vector search.

    Args:
        embedding_model:  OpenAI embedding model name.
        method:           Breakpoint detection method for semantic chunking.
        threshold:        Threshold value for breakpoint detection.
        min_chunk_size:   Minimum characters per semantic chunk.
        k:                Number of results to return from vector search.
    """


    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        method: Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        threshold: float = 90.0,
        min_chunk_size: int = 1000,
        max_chunk_size: int = 20000,
        k: int = 3,
        ):
        
        self.embedder = OpenAIEmbedder(model=embedding_model)

        self.chunker = SemanticChunker(
            embedder=self.embedder,
            method=method,
            threshold=threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )

        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        
        self.k = k

        self._chunks : List[str] = []
    
    def index_document(self, text:str, doc_id:str ="doc_0") -> int:

        """
        Semantically chunk and index a document.

        Args:
            text:    Full document text.
            doc_id:  Document identifier.

        Returns:
            Number of semantic chunks created.
        """

        chunks = self.chunker.chunk(text)
        self._chunks = chunks

        documents = []

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    content=chunk,
                    metadata={
                        "doc_id":doc_id,
                        "chunk_index":i,
                        "total_chunks":len(chunks),
                        "chunk_length":len(chunk)
                    }
                )
            )
        
        documents = self.embedder.embed_documents(documents)

        self.vector_store.add_documents(documents)

        return len(chunks)


    def index_pdf(self, file_path:str, doc_id:Optional[str]=None)->int:
        """
        Read a PDF and index its contents with semantic chunking.

        Args:
            file_path:  Path to PDF file.
            doc_id:     Document ID (defaults to filename).

        Returns:
            Number of semantic chunks created.
        """

        if doc_id is None:
            doc_id = os.path.basename(file_path)

        text = read_pdf(file_path)
        return self.index_document(text, doc_id)

    
    def index_text_file(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """
        Read a text file and index its contents with semantic chunking.

        Args:
            file_path:  Path to text file.
            doc_id:     Document ID (defaults to filename).

        Returns:
            Number of semantic chunks created.
        """
        if doc_id is None:
            doc_id = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_document(text, doc_id)


    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant semantic chunks.

        Args:
            query:  Search query.

        Returns:
            List of RetrievalResult objects.
        """
        query_emb = self.embedder.embed_text(query)
        return self.vector_store.search(query_emb, k=self.k)


    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve context strings for a query.

        Args:
            query:  Search query.

        Returns:
            List of chunk text strings.
        """
        results = self.retrieve(query)
        return [r.document.content for r in results]

class SemanticChunkingRAG:
    """
    Complete RAG pipeline using semantic chunking.

    Combines SemanticChunkingRetriever (topic-aware chunking + vector search)
    with OpenAIChat (for answer generation). Follows the same interface
    pattern as SimpleRAGOpenai, RSERetrievalRAG, ContextEnrichmentRAG, etc.

    Usage:
        rag = SemanticChunkingRAG(file_path="report.pdf")
        answer, contexts = rag.query("What is the main cause of climate change?")
    """

    def __init__(
        self,
        file_path:str,
        method:Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        threshold:float = 90.0,
        min_chunk_size:int = 200,
        max_chunk_size: int = 20000,
        k:int = 3,
        embedding_model:str = "text-embedding-3-small",
        chat_model:str ="gpt-4o-mini",
        temperature:float=0.0,
        ):

        """
        Initialize the Semantic Chunking RAG pipeline.

        Args:
            file_path:        Path to document (PDF or text file).
            method:           Breakpoint detection method:
                              'percentile' — split at bottom (100-threshold)% similarities
                              'standard_deviation' — split below mean - threshold*std
                              'interquartile' — split below Q1 - threshold*IQR
            threshold:        Threshold value for breakpoint detection.
                              For percentile: 90 = split at lowest 10% similarities.
                              For std_dev: 1.0 = split at 1 std below mean.
                              For IQR: 1.5 = standard outlier detection.
            min_chunk_size:   Minimum characters per chunk (tiny chunks get merged).
            k:                Number of chunks to retrieve per query.
            embedding_model:  OpenAI embedding model.
            chat_model:       OpenAI chat model.
            temperature:      LLM temperature.
        """

        self.file_path = file_path

        self.retriever = SemanticChunkingRetriever(
            embedding_model=embedding_model,
            method=method,
            threshold=threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            k=k
        )
        
        self.chat = OpenAIChat(model_name=chat_model, temperature=temperature)

        if self.file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(self.file_path)

        else:
            num_chunks = self.retriever.index_text_file(self.file_path)


        print(
            f"[SemanticChunking] Done -> {num_chunks} semantic chunks"
            f"(method={method}, threshold={threshold})"
        )

    
    def query(
        self,
        question:str,
        return_context:bool=True
    )-> Tuple[str, List[str]]:

        """
        Query the RAG system.

        Args:
            question:        User's question.
            return_context:  Whether to return retrieved chunks.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """

        contexts = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found in the document.", []

        # Generate answer
        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []


    def show_chunks(self)->None:
        """
        Debug helper: print all semantic chunks with their sizes.
        Useful for inspecting how the document was split.
        """
        chunks = self.retriever._chunks

        print(f"\n{'=' * 70}")
        print(f"SEMANTIC CHUNKS ({len(chunks)} total)")
        print(f"{'=' * 70}")

        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i + 1} ({len(chunk)} chars):")
            print(f"  {'-' * 50}")
            preview = chunk[:250].replace('\n', ' ')
            print(f"  {preview}...")
            print()


    def show_context(self, question: str) -> None:
        """
        Debug helper: show retrieved chunks for a query.

        Args:
            question:  Search query.
        """
        results = self.retriever.retrieve(question)

        print(f"\nQuery: {question}")
        print("=" * 70)

        for i, result in enumerate(results):
            print(f"\n  Result {i + 1}:")
            print(f"    Score:    {result.score:.4f}")
            print(f"    Length:   {len(result.document.content)} chars")
            print(f"    Chunk #:  {result.document.metadata.get('chunk_index', '?')}")
            preview = result.document.content[:300].replace('\n', ' ')
            print(f"    Preview:  {preview}...")
            print(f"  {'-' * 50}")



if __name__ == "__main__":
    pdf_path = r"data\IntermediaryGuidelinesandDigitalMediaEthicsCode.pdf"
    
    rag = SemanticChunkingRAG(
        file_path=pdf_path,
        method="percentile",
        threshold=90.0,
        min_chunk_size=50,
        k=2,
    )

    glod_standard_data = json.load(open("data\gold_standared_q_a.json", "rb"))

    # print(glod_standard_data)
    for key, value in glod_standard_data.items():
        # print(key)
        # print(value)
        # print("\n")
        answer, context = rag.query(value["question"])
        print("\nAnswer:")
        print(answer)
        print("\nGold Standard Answer:")
        print(value["ground_truth"])
        print("\nContext:")
        print(context)


    # answer, context = rag.query("What is climate change ??")

    # print("\nAnswer:")
    # print(answer)

    # print("\nContext:")
    # print(context)