import os 
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from helper_function_openai import (
    Document,
    RAGRetriever,
    OpenAIChat,
    chunk_text,
    read_pdf,
    RetrievalResult
)

from typing import List, Dict, Any, Optional, Tuple
import json


class HyDERAGRetriever(RAGRetriever):

    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200, k=3, model_name: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1000, api_key: Optional[str] = None):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        super().__init__()

        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

        is_pdf = self.file_path.endswith(".pdf")
        is_csv = self.file_path.endswith(".csv")

        if is_pdf:
            self.index_pdf(self.file_path)
        elif is_csv:
            self.index_csv(self.file_path)
        else:
            raise ValueError("Unsupported file type")

    def generate_hypothetical_answer(self, query:str) -> str:
        """
        Generate a hypothetical answer document for the given query.
        """

        messages = [
            {
                "role":"system",
                "content": (
                    "You are an expert at generating detailed, in-depth documents "
                    "that directly answer questions. Generate a document that would "
                    "be found in a knowledge base as the perfect answer."
                )
            },
            {
                "role":"user",
                "content":(
                    f"Given the question '{query}', generate a hypothetical document "
                    f"that directly answers this question. The document should be "
                    f"detailed and in-depth. The document size should be exactly "
                    f"{self.chunk_size} characters."
                )
            }
        ]

        return self.llm.chat(messages=messages)

    def retrieve(self, query:str, k=3) -> List[RetrievalResult]:
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
        
        hypothetical_doc = self.generate_hypothetical_answer(query=query)

        hypothetical_embedding = self.embedder.embed_text(hypothetical_doc)
        
        return self.vector_store.search(hypothetical_embedding, k), hypothetical_doc

    def retrieve_context(self, query:str, k:int=3) -> List[str]:
        """
        Retrieve context strings for a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of context strings
        """

        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        results, hypothetical_doc = self.retrieve(query, k)
        return [result.document.content for result in results], hypothetical_doc

    def query(self, question:str)-> Tuple[str, List[str], str]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer to the question
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        context, hypothetical_doc = self.retrieve_context(question, self.k)
        
        context_message = "\n\n".join(context)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the context below to answer the question.\n\n"
                    f"Context:\n{context_message}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        return self.llm.chat(messages=messages), context



if __name__ == "__main__":
    hyde_rag = HyDERAGRetriever(file_path=r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf", chunk_size=1000, chunk_overlap=200, k=3)
    query = "what are the main causes of climate change?"
    answer, context, hypothetical_doc = hyde_rag.query(query)
    print("Hypothetical Document:", hypothetical_doc)
    print("Retrieved Documents:", context)
    print("Answer:", answer)