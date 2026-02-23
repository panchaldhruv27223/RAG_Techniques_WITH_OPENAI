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
    read_pdf
)

from typing import List, Dict, Any, Optional, Tuple
import json


#    Helper function for proposition chunking

def generate_step_back_query(llm:OpenAIChat, original_query: str) -> str:

    messages = [
        {
            "role":"system",
            "content":(
                "You are an AI assistant tasked with generating broader, more general "
                "queries to improve context retrieval in a RAG system. Given the original "
                "query, generate a step back query that is more general and can help "
                "retrieve relevant background information."
            )
        },
        {
            "role":"user",
            "content":f"Original query: {original_query} \n\n Step-Back query:"
        }
    ]

    return llm.chat(messages=messages)

def rewrite_query(llm:OpenAIChat, original_query: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with reformulating user queries "
                "to improve retrieval in a RAG system. Given the original query, "
                "rewrite it to be more specific, detailed, and likely to retrieve "
                "relevant information."
            )
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}\n\nRewritten query:"
        }
    ]
    
    return llm.chat(messages)

def decompose_query(llm:OpenAIChat, original_query:str)-> List:
    """
    Decompose a complex query into 2-4 simpler sub-queries.

    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with breaking down complex queries into "
                "simpler sub-queries for a RAG system. Given the original query, decompose "
                "it into 2-4 simpler sub-queries that, when answered together, would provide "
                "a comprehensive response to the original query.\n\n"
                "Example:\n"
                "Original: What are the impacts of climate change on the environment?\n"
                '{"sub_queries": ['
                '"What are the impacts of climate change on biodiversity?", '
                '"How does climate change affect the oceans?", '
                '"What are the effects of climate change on agriculture?", '
                '"What are the impacts of climate change on human health?"]}\n\n'
                'Respond with JSON: {"sub_queries": ["query1", "query2", ...]}'
            )
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}"
        }
    ]

    result = llm.chat_json(messages)
    # print(f"Generated output: {result}")
    return result.get("sub_queries", [])


class QueryTransformRagOpenai:

    def __init__(self, file_path:str,
        chunk_size:int = 1000,
        chunk_overlap:int = 200,
        k:int = 3):
    
        self.is_pdf = file_path.endswith(".pdf")
        self.is_csv = file_path.endswith(".csv")
        
        self.rag_retriever = RAGRetriever(
                embedding_model="text-embedding-3-small",
            )
        self.llm = OpenAIChat()

        self.methods = {
            "rewrite": rewrite_query,
            "step_back": generate_step_back_query,
            "decompose": decompose_query
        }

        if self.is_pdf:
            self.rag_retriever.index_pdf(path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if self.is_csv:
            self.rag_retriever.index_csv(path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        

    def query(self, question:str, method:str = "rewrite") -> str:

        if method not in self.methods:
            raise ValueError(f"Invalid method. Choose from {list(self.methods.keys())}")
        
        if method == "rewrite":
            new_query = self.methods[method](self.llm, question)
            context_retrieved = self.rag_retriever.retrieve_context(new_query)
            return self.llm.chat_with_context(question, context_retrieved), context_retrieved

        if method == "step_back":
            new_query = self.methods[method](self.llm, question)
            context_retrieved = self.rag_retriever.retrieve_context(new_query)
            return self.llm.chat_with_context(question, context_retrieved), context_retrieved

        new_queries = self.methods[method](self.llm, question)
        
        results = []
        context_retrieved = []
        for sq in new_queries:
            context = self.rag_retriever.retrieve_context(sq)
            answer = self.llm.chat_with_context(sq, context)
            results.append({"query": sq, "answer": answer})
            context_retrieved.append(context)
        
        # Combine answers
        combined_answer = self.llm.chat_with_context(
            question,
            "\n\n".join([f"Query: {r['query']}\nAnswer: {r['answer']}" for r in results])
        )
        
        return combined_answer, context_retrieved

        


if __name__ == "__main__":
    
    rag = QueryTransformRagOpenai(r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf")
    print(rag.query("What are the main causes of climate change?", method="decompose"))

