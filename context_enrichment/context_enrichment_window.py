import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    chunk_text,
)
from dotenv import load_dotenv
load_dotenv()


class ChunkStore:

    def __init__(self):
        self._chunks: List[str] = []
        self._doc_id: str = ""

    def add_chunks(self, chunks: List[str], doc_id: str = "doc_0") -> None:
        """
        Store all chunks in order.

        Args:
            chunks:  List of chunk texts, in document order.
            doc_id:  Source document identifier.
        """
        self._chunks = chunks
        self._doc_id = doc_id

    def get(self, index: int) -> Optional[str]:
        """
        Retrieve a chunk by its position index. O(1) lookup.

        Args:
            index:  Chunk position (0-based).

        Returns:
            Chunk text, or None if index is out of range.
        """
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None

    def get_window(self, center: int, num_neighbors: int) -> List[str]:
        """
        Retrieve a window of chunks centered on the given index.

        Args:
            center:         Index of the retrieved (relevant) chunk.
            num_neighbors:  Number of chunks to include before and after.

        Returns:
            Ordered list of chunk texts in [center - N, center + N] range.
        """
        start = max(0, center - num_neighbors)
        end = min(len(self._chunks), center + num_neighbors + 1)
        return [self._chunks[i] for i in range(start, end)]

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


class ContextEnrichmentRetriever:
    """
    Retriever that expands each vector search hit with neighboring chunks.

    Pipeline:
        1. Chunk document with overlap → store in FAISS + ChunkStore
        2. Query → vector search → top-k chunk indices
        3. For each hit, grab ±num_neighbors from ChunkStore
        4. Concatenate, removing overlap to avoid text duplication
        5. Return expanded context strings

    Args:
        embedding_model:  OpenAI embedding model name.
        chunk_size:       Characters per chunk.
        chunk_overlap:    Overlap between consecutive chunks.
        num_neighbors:    How many chunks to pad before/after each hit.
        k:                Number of top results from vector search.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        num_neighbors: int = 1,
        k: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_neighbors = num_neighbors
        self.k = k
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.chunk_store = ChunkStore()

    
    def index_document(self, text:str, doc_id:str="doc_0")->int:
        """
        Chunk and index a document into both FAISS and the ChunkStore.

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

        self.chunk_store.add_chunks(chunks=chunks, doc_id=doc_id)

        documents = []

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    content=chunk,
                    metadata={
                        "doc_id":doc_id,
                        "chunk_index":i, 
                        "total_chunks":len(chunks)
                    }
                )
            )

        documents = self.embedder.embed_documents(documents)
        
        self.vector_store.add_documents(documents)

        return len(chunks)


    def index_pdf(self, file_path:str, doc_id:Optional[str]=None)->int:
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

        return self.index_document(text=text, doc_id=doc_id)

    def index_text_file(self, file_path:str, doc_id:Optional[str]=None)->int:
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

        return self.index_document(text=text, doc_id=doc_id)

    def _merge_chunks_with_overlap_removal(self, chunks:List[str])->str:
        """
        Concatenate neighboring chunks, trimming overlapping text.

        When chunks were created with overlap, adjacent chunks share
        `chunk_overlap` characters. This method removes that duplication
        so the merged text reads naturally.

        Example with chunk_overlap=200:
            Chunk A: [AAAAAAA|BBBBB]     ← last 200 chars = "BBBBB"
            Chunk B: [BBBBB|CCCCCCC]     ← first ~200 chars = "BBBBB"
            
            Merged:  [AAAAAAA|BBBBB|CCCCCCC]   ← overlap trimmed from A's tail

        Args:
            chunks:  Ordered list of chunk texts to merge.

        Returns:
            Merged text with overlaps removed.
        """
        
        if not chunks:
            return ""

        merged = chunks[0]

        for i in range(1, len(chunks)):

            current_chunk = chunks[i]

            if self.chunk_overlap > 0:
                trim_point = max(0, len(merged) - self.chunk_overlap)
                merged = merged[:trim_point] + current_chunk

            else:
                merged = merger + "\n" + current_chunk

        return merged

    def retrieve_standard(self, query:str) -> List[str]:
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=self.k)
        return [r.document.content for r in results]

    def retrieve_with_context_window(self, query:str)->List[str]:
        """
        Retrieve chunks and expand each with neighboring chunks.

        This is the core method. For each vector search hit:
            1. Get the chunk's position index from metadata
            2. Fetch ±num_neighbors chunks from ChunkStore (O(1) each)
            3. Merge them with overlap removal
            4. Return the expanded text

        Args:
            query:  Search query.

        Returns:
            List of expanded context strings (one per search hit).
        """

        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=self.k)

        expanded_contexts = []
        seen_ranges = set()

        for result in results:
            chunk_index = result.document.metadata.get("chunk_index")

            if chunk_index is None:
                expanded_contexts.append(result.document.content)
                continue 

            start = max(0, chunk_index - self.num_neighbors)

            end = min(
                self.chunk_store.total_chunks,
                chunk_index + self.num_neighbors + 1
            )

            range_key = (start, end)

            if range_key in seen_ranges:
                continue

            seen_ranges.add(range_key)

            window_chunks = self.chunk_store.get_window(
                center=chunk_index, 
                num_neighbors=self.num_neighbors
            )


            merged_text = self._merge_chunks_with_overlap_removal(window_chunks)
            expanded_contexts.append(merged_text)

        return expanded_contexts
    
    def compare_retrieval(self, query:str)->List[str]:
        standard = self.retrieve_standard(query)
        context_window = self.retrieve_with_context_window(query)
        return {
            "standard": standard,
            "context_window": context_window
        }


        
class ContextEnrichmentRAG:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 400,
        chunk_overlap: int = 200,
        num_neighbors: int = 1,
        k: int = 3,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):

        self.file_path = file_path

        self.retriever = ContextEnrichmentRetriever(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_neighbors=num_neighbors,
            k=k
        )

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature
        )

        if file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(file_path=file_path)
        elif file_path.endswith(".txt"):
            num_chunks = self.retriever.index_text_file(file_path=file_path)
        

    def query(self, question:str, return_context:bool=True)->str:
        """
        Query the RAG system with context-enriched retrieval.

        Args:
            question:        User's question.
            return_context:  Whether to return the expanded contexts.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """
        contexts = self.retriever.retrieve_with_context_window(question)

        if not contexts:
            return "No relevant information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []

    def compare(self, question:str)->None:
        """
        Compare standard VS enriched retrieval side by side.

        Args:
            question: search query
        """ 

        comparison = self.retriever.compare_retrieval(question)
        
        print(f"\nQuery: {question}")
        print("=" * 70)

        print("\n📦 STANDARD RETRIEVAL (isolated chunks):")
        print("-" * 50)
        for i, ctx in enumerate(comparison["standard"]):
            print(f"\n  Chunk {i + 1} ({len(ctx)} chars):")
            print(f"    {ctx[:200]}...")

        print(f"\n🔍 ENRICHED RETRIEVAL (with ±{self.retriever.num_neighbors} neighbors):")
        print("-" * 50)
        for i, ctx in enumerate(comparison["context_window"]):
            print(f"\n  Window {i + 1} ({len(ctx)} chars):")
            print(f"    {ctx[:300]}...")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    pdf_file_path = r"data\Understanding_Climate_Change.pdf"

    rag_cw = ContextEnrichmentRAG(
        file_path=pdf_file_path,
        chunk_size=1000,
        chunk_overlap=200,
        num_neighbors=2,
        k=3
    )

    print("\n[Context Enrichment RAG] Ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() == "exit":
            break

        answer, context = rag_cw.query(question)
        print(f"\nAnswer: {answer}")
        print(f"Context windows used: {len(context)}")
        print()
