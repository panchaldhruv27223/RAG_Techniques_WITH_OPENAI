"""
RAG Helper Functions - Pure OpenAI SDK Implementation.
"""

import os 
import fitz
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import tiktoken
import faiss

## Data Classes
@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from vector search"""
    document: Document
    score: float
    rank: int


## pdf processing.

def read_pdf(file_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        path: Path to PDF file
        
    Returns:
        Extracted text as string
    """

    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

## read pdf with metadata.
def read_pdf_with_metadata(file_path:str) -> List[Document]:
    """
    Extract text from PDF with page-level metadata.
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of Document objects, one per page
    """
    docs = fitz.open(file_path)
    documents = []

    for page_num in range(len(docs)):
        page = docs[page_num]
        text = page.get_text()

        if text.split():
            documents.append(
                Document(
                    content = text,
                    metadata = {
                        "source" : file_path,
                        "page" : page_num + 1,
                        "total_pages" : len(docs)
                    }
                )
            )

    docs.close()
    return documents

## text chunking
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens in
        model: Model name
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

## chunking.
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None) -> List[str]:
    """
    Split text into overlapping chunks using recursive character splitting.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        separators: List of separators to try (in order of preference)
        
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = []

    def split_recursive(text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        separator = separators[0] if separators else ""

        if separator:
            parts = text.split(separator)

        else:
            parts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
            return parts

        result =[]

        current_chunk = ""

        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk

            else:
                if current_chunk:
                    result.append(current_chunk)

                if len(part) > chunk_size and len(separators) > 1:
                    result.extend(split_recursive(part, separators[1:]))

                else:
                    current_chunk = part
            
        if current_chunk:
            result.append(current_chunk)

        return result

    raw_chunk = split_recursive(text, separators)

    for i, chunk in enumerate(raw_chunk):

        if i > 0 and chunk_overlap > 0:
            prev_chunk = raw_chunk[i-1]
            overlap_text = prev_chunk[-chunk_overlap:]
            chunk = overlap_text + chunk

        ## clean whitespace.
        chunk = chunk.strip()
        chunk = chunk.replace("\t", ' ')

        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunk a list of documents while preserving metadata.
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    chunked_documents = []

    for doc in documents:
        chunks = chunk_text(doc.content, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            chunked_documents.append(
                Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks)
                        }
                )
            )

    return chunked_documents


# Openai Embeddings 
class OpenAIEmbedder:
    """
    Handles embeddings using OpenAI SDK directly.
    """
    def __init__(
        self,
        model:str = "text-embedding-3-small",
        api_key: Optional[str] = None
        ):

        """
        Initialize embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self._dimension = None

    @property
    def dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        if self._dimension is None:
            test_embedding = self.embed_text("test")
            self._dimension = len(test_embedding)

        return self._dimension


    def embed_text(self, text: List[str]) -> List[float]:
        """
        Create embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )

        return response.data[0].embedding

    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Create embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )

            batch_embeddings  = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Add embeddings to Document objects.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Documents with embeddings added
        """
        texts = [doc.content for doc in documents]
        embeddings = self.embed_texts(texts)

        for doc, embed in zip(documents, embeddings):
            doc.embedding = embed
        # print(f"Embeddings added to {len(documents)} documents")
        # print(f"Embeddings: {documents[0].embedding}")
        return documents


## vector store
class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    """
    def __init__(
        self,
        dimension:int):

        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension

        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents with embeddings to the store.
        
        Args:
            documents: Documents with embeddings
        """

        embeddings = []

        for doc in documents:

            if doc.embedding is None:
                raise ValueError("Document must have embeddings")

            embeddings.append(doc.embedding)
            self.documents.append(doc)

        ## normalize for cosine similarity 
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self,
        query_embeddings: List[float],
        k:int = 5
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        ## normalize query 
        query_vector = np.array([query_embeddings], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        ## seach 
        scores, indices = self.index.search(query_vector, k)

        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=float(score),
                        rank=rank
                    )
                )

        return results

    def save(self, path:str) -> None:
        """Save index and documents to disk."""
        faiss.write_index(self.index, f"{path}.index")

        docs_data = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            }
            for doc in self.documents
        ]

        with open(f"{path}.json","w") as f:
            json.dump(docs_data, f, indent=4)

    @classmethod
    def load(cls, path:str) -> "FAISSVectorStore":
        """Load index and documents from disk."""
        index = faiss.read_index(f"{path}.index")
        with open(f"{path}.json", "r") as f:
            docs_data = json.load(f)

        store = cls(index.d)
        store.index = index
        store.documents = [
            Document(
                content=d['content'],
                metadata=d['metadata'],
                embedding= d['embedding']
            )
            for d in docs_data
        ]

        return store


### RAG retriever

class RAGRetriever:
    """
    Complete RAG retriever combining embedder and vector store.
    """

    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize retriever.
        
        Args:
            embedder: Optional pre-configured embedder
            embedding_model: Model name if creating new embedder
        """

        self.embedder = embedder or OpenAIEmbedder(model = embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: Documents to index
        """
        # Add embeddings
        documents = self.embedder.embed_documents(documents)

        # Create Vectore store
        # print(f"Dimension: {self.embedder.dimension}")
        self.vector_store.add_documents(documents)

    def index_pdf(
        self,
        path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> int:
    
        """
        Index a PDF file.
        
        Args:
            path: Path to PDF
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks indexed
        """
        # Load and chunk
        documents = read_pdf_with_metadata(path)
        chunked = chunk_documents(documents, chunk_size, chunk_overlap)

        # print(f"Chunked {len(chunked)} documents")
        # print(f"Chunked documents: {chunked[0]}")
        # index
        self.index_documents(chunked)

        return len(chunked)

    
    def retrieve(self, query:str, k=5) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of retrieval results
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        query_embedding = self.embedder.embed_text(query)
        
        return self.vector_store.search(query_embedding, k)

    def retrieve_context(self, query:str, k:int=5) -> List[str]:
        """
        Retrieve context strings for a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of context strings
        """
        results = self.retrieve(query, k)
        return [result.document.content for result in results]


## Openai chat completion

class OpenAIChat:
    """
    OpenAI chat completion wrapper.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1000, api_key: Optional[str] = None):
        """
        Initialize chat client.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: API key (defaults to env var)
        """

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for API
            
        Returns:
            Assistant's response text
        """

        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = kwargs.get("temperature", self.temperature),
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
        )

        # print("Response: ", response)

        return response.choices[0].message.content
    
    def chat_with_context(
        self, 
        query : str,
        context: List[str],
        system_prompt : Optional[str] = None
    ) -> str:

        """
        Answer question using provided context (RAG).
        
        Args:
            query: User's question
            context: List of context strings
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
                    Answer ONLY based on the context provided. If the answer cannot be found in the context, say so.
                    Be concise but complete in your answers.""".strip()

        
        context_text = "\n\n --- \n\n".join(context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context: {context_text} \n\n Question: {query}  Answer based on the context above:"""}
        ]

        return self.chat(messages)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Get structured JSON response.
        
        Args:
            messages: Chat messages
            schema: Optional JSON schema hint
            
        Returns:
            Parsed JSON response
        """
        json_instruction = "\nRespond with valid JSON only. No markdown, no explanation."

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += json_instruction
        else:
            messages.insert(0, {"role": "system", "content": json_instruction})

        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
            response_format = {"type": "json_object"}
            )
        
        return json.loads(response.choices[0].message.content)


## Complete RAG Pipeline


class SimpleRAG:
    """
    Complete RAG pipeline using pure OpenAI SDK.
    """
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Model for embeddings
            chat_model: Model for chat completion
            temperature: LLM temperature
        """
        self.retriever = RAGRetriever(
            embedding_model=embedding_model
        )
        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature
        )


    def index_pdf(
        self,
        path:str,
        chunk_size:int = 1000,
        chunk_overlap:int = 200
    ) -> int:
        """Index a PDF file."""
        return self.retriever.index_pdf(path, chunk_size, chunk_overlap)

    def index_text(
        self,
        text:str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict] = None
    ) -> int:
        """Index raw text."""
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        documents = [Document(content=chunk, metadata= metadata or {}) for chunk in chunks]

        self.retriever.index_documents(documents=documents)
        return len(documents)

    def index_csv(
        self,
        file_path:str,
        text_columns: Optional[List[str]] = None
    ) -> int:
        """
        Index a CSV file into an existing SimpleRAG instance.
        
        Args:
            rag: Initialized SimpleRAG instance
            file_path: Path to CSV file
            text_columns: Columns to include in searchable content
        
        Returns:
            Number of documents indexed
        """
        documents = read_csv_as_documents(file_path=file_path, text_columns=text_columns)
        self.retriever.index_documents(documents=documents)
        return len(documents)

    def query(self, query:str, k:int=5, return_context:bool=False) -> Dict[str,Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            k: Number of context chunks to retrieve
            return_context: Whether to include context in response
            
        Returns:
            Dict with 'answer' and optionally 'context'
        """

        ## Retrieve 
        context = self.retriever.retrieve_context(query, k)

        ## Generate Answer
        answer = self.chat.chat_with_context(query, context)

        ## Prepare Response
        response = {
            "answer": answer,
            "question": query
        }

        if return_context:
            response["context"] = context

        return response

    def show_context(self, query:str, k:int=5) -> List[str]:
        """Display retrieved context for debugging."""
        context = self.retriever.retrieve_context(query, k)

        print(f"Query: {query}\n")
        print("=" * 60)
        
        for i, ctx in enumerate(context, 1):
            print(f"\nContext {i}:")
            print("-" * 40)
            print(ctx[:500] + "..." if len(ctx) > 500 else ctx)
            print()


## Utility Functions 

def show_context(context: List[str])->None:
    """Display context chunks nicely formatted."""
    for i, c in enumerate(context, 1):
        print(f"\nContext {i}:")
        print(c)
        print("\n")
    
def cosine_similarity(a:List[float], b:List[float])->float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


### helper function for csv
import pandas as pd 
import numpy as np 
from typing import List, Dict, Any, Optional

def read_csv_as_documents(
    file_path:str,
    text_columns:Optional[List[str]]= None,
    metadata_columns:Optional[List[str]]= None
) -> List[Document]:

    """
    Convert a CSV file into Document objects for RAG indexing.
    
    Each row becomes one Document with structured "Column: Value" content,
    making it ideal for semantic search over tabular data.
    
    Args:
        file_path: Path to the CSV file
        text_columns: Columns to include in searchable content (default: all)
        metadata_columns: Extra columns to store in metadata (default: all)
    
    Returns:
        List of Document objects ready for embedding
    """
    df = pd.read_csv(file_path)
    columns = text_columns or df.columns.tolist()
    meta_cols =  metadata_columns or df.columns.tolist() 
    
    documents = []
    for idx, row in df.iterrows():
        text_parts = [
            f"{col}: {row[col]}"
            for col in columns
            if pd.notna(row[col])
        ]

        content = "\n".join(text_parts)

        metadata = {
            "row_index": idx,
            "source":file_path
        }

        for col in meta_cols:
            if pd.notna(row[col]):
                metadata[col] = str(row[col])

        documents.append(Document(content=content, metadata=metadata))

    return documents

def read_csv_grouped(
    file_path:str,
    group_by:str,
    text_columns: Optional[List[str]] = None
) -> List[Document]:

    """
    Group CSV rows by a column and create one Document per group.
    
    Useful when multiple rows relate to the same entity (e.g., orders per customer).
    
    Args:
        file_path: Path to the CSV file
        group_by: Column name to group rows by
        text_columns: Columns to include in content (default: all)
    
    Returns:
        List of Document objects (one per group)
    """

    df = pd.read_csv(file_path)
    columns = text_columns or [c for c in df.columns if c!= group_by]

    documents = []

    for group_key, group_df in df.groupby(group_by):

        text_parts = [f"{group_by}: {group_key}\n"]

        for _, row in group_df.iterrows():
            row_text = " | ".join(
                f"{col}: {row[col]}"
                for col in columns
                if pd.notna(row[col])
            )

            text_parts.append(row_text)

        content = "\n".join(text_parts)

        documents.append(
            Document(
                content = content,
                metadata = {
                    "group_key": str(group_key),
                    "group_by":group_by,
                    "row_count":len(group_df),
                    "source":file_path
                }
            )
        )

    
    return documents