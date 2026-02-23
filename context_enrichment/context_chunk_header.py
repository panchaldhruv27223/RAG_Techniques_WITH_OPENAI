import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import json

load_dotenv()

import numpy as np
from helper_function_openai import (
    Document,
    read_pdf,
    OpenAIChat,
    chunk_text,
    RAGRetriever
)


class ContextChunkHeaderRAGRetriever(RAGRetriever):

    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200, k=3, model_name: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1000, api_key: Optional[str] = None, max_workers: int = 10, add_title: bool = True, add_summary: bool = True):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.max_workers = max_workers
        self.add_title = add_title
        self.add_summary = add_summary
        super().__init__()

        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

        self.is_pdf = self.file_path.endswith(".pdf")
        self.is_csv = self.file_path.endswith(".csv")

        self.embed_file()

    def get_document_title(self, document_text:str, guidance:str="")->str:
        """
        Use LLM to generate a descriptive document title.
        
        Replaces: get_document_title() with manual tiktoken truncation + OpenAI client
        """

        messages = [
            {
                "role":"user",
                "content": (
                    "What is the title of the following document?\n\n"
                    "Your response MUST be the title of the document, and nothing else. "
                    "DO NOT respond with anything else.\n\n"
                    f"{guidance}\n\n"
                    f"DOCUMENT\n{document_text}"
                )
            }
        ]

        return self.llm.chat(messages)
 
    def get_document_summary(self, document_text: str) -> str:
        """Generate a concise summary of the document for use in chunk headers."""
        
        messages = [
            {
                "role": "system",
                "content": "You generate concise document summaries for use as context in RAG systems."
            },
            {
                "role": "user",
                "content": (
                    "Write a 2-3 sentence summary of this document. "
                    "Focus on what the document is about and who it pertains to.\n\n"
                    f"Document:\n{document_text}"
                )
            }
        ]
        
        return self.llm.chat(messages)
 
 
    def embed_file(self):
        """
        Embed the file.
        """
        if self.is_pdf:
            text = read_pdf(file_path=self.file_path)
            chunks = chunk_text(text=text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

            chunk_docs = []
            for i, chunk in enumerate(chunks):
                content = chunk

                if self.add_summary:
                    summary = self.get_document_summary(chunk)
                    content = summary +"\n\n" + content
                
                if self.add_title:
                    title = self.get_document_title(chunk)
                    content = title +"\n\n" + content


                chunk_docs.append(
                    Document(
                        content = chunk,
                        metadata = {
                            "source": self.file_path,
                            "chunk_id": i
                        },
                        embedding=self.embedder.embed_text(content)
                    )
                )
            self.vector_store.add_documents(chunk_docs)
        
        elif self.is_csv:
            pass



    def query(self, question:str)-> Tuple[str, List[str]]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer to the question
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        
        # Search for similar question embeddings, return the original chunks.
        # Deduplicates chunks (since one chunk may match via multiple questions).
        query_embedding = self.embedder.embed_text(question)

        query_vec = np.array([query_embedding], dtype=np.float32)
        
        context = self.retrieve_context(question, k=self.k)

        context_text = "\n\n".join(context)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "You are given a question and a context. "
                    "Answer the question based on the context."
                )
            },
            {
                "role": "user",
                "content": (
                    "Question: \n"
                    f"{question}\n\n"
                    "Context: \n"
                    f"{context_text}\n\n"
                    "Answer:"
                )
            }
        ]
        
        return self.llm.chat(messages=messages), context



if __name__ == "__main__":
    context_chunk_header_rag = ContextChunkHeaderRAGRetriever(file_path=r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf", chunk_size=1000, chunk_overlap=200, k=3)

    while True:
        query = input("User: ")


        if query.lower() == "exit":
            break

        answer, context = context_chunk_header_rag.query(query)
        print("Context:", context)
        print("Answer:", answer)
