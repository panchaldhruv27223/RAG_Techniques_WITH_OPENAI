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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import json


class HyPERAGRetriever(RAGRetriever):

    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200, k=3, model_name: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1000, api_key: Optional[str] = None, max_workers: int = 10):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.max_workers = max_workers
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

    def embed_file(self):
        """
        Embed the file.
        """
        if self.is_pdf:
            text = read_pdf(file_path=self.file_path)
            chunks = chunk_text(text=text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunk_docs = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [
                    pool.submit(
                        self.generate_hypothetical_prompt_embeddings,
                        chunk
                    )

                    for chunk in chunks
                ]
                
                for i, future in enumerate(tqdm(as_completed(futures), total=len(chunks))):
                    chunk_content, question_embeddings = future.result()
                    # print(f"Length of question embeddings: {len(question_embeddings)}")
                    
                    for qe in question_embeddings:
                        # print(f"length of qe: {len(qe)}")
                        chunk_doc = Document(
                            content = chunk_content,
                            metadata = {
                                "source": self.file_path,
                                "chunk_id": i
                            },
                            embedding=qe
                        )
                        chunk_docs.append(chunk_doc)

            self.vector_store.add_documents(chunk_docs)
        
        elif self.is_csv:
            pass


    def generate_hypothetical_prompt_embeddings(self, chunk_text_str: str) -> Tuple[str, List[List[float]]]:
        """
        Generate hypothetical questions for a chunk, then embed them.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate essential questions from text. "
                    "Each question should be on one line, without numbering or prefixes."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the input text and generate essential questions that, "
                    "when answered, capture the main points of the text. "
                    "Each question should be one line, without numbering or prefixes.\n\n"
                    f"Text:\n{chunk_text_str}\n\nQuestions:"
                )
            }
        ]

        response = self.llm.chat(messages=messages)

        questions = [
            q.strip()

            for q in response.replace("\n\n","\n").split("\n")
            if q.strip() and len(q.strip()) > 10
        ]

        questions_embeddings = self.embedder.embed_texts(questions)

        return chunk_text_str, questions_embeddings

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
        
        ## Search more than k since we need to deduplicate chunks
        search_k = min(self.vector_store.index.ntotal, self.k * 5)
        
        distances, indices = self.vector_store.index.search(query_vec, search_k)

        seen_chunks = {}

        for dist, idx in zip(distances[0], indices[0]):

            if idx < len(self.vector_store.documents):
            
                doc = self.vector_store.documents[idx]
                chunk_key = doc.content[:100]

                if chunk_key not in seen_chunks or dist < seen_chunks[chunk_key][0]:
                    seen_chunks[chunk_key] = (dist, doc)

        
        ## sort by score and take top k
        sorted_results = sorted(seen_chunks.values(), key=lambda x: -x[0])[:self.k]

        context = [
            doc.content
            for _, (_, doc) in enumerate(sorted_results)
        ]

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
    hype_rag = HyPERAGRetriever(file_path=r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf", chunk_size=1000, chunk_overlap=200, k=3)

    while True:
        query = input("User: ")


        if query.lower() == "exit":
            break

        answer, context = hype_rag.query(query)
        print("Context:", context)
        print("Answer:", answer)
