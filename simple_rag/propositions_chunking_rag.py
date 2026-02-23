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

#    Generate Propositions

def generate_propositions(llm, chunk_text_var:str) -> List[str]:
    """
    Break a text chunk into atomic, factual, self-contained propositions.
    
    Replaces: ChatGroq + with_structured_output(GeneratePropositions) + FewShotChatMessagePromptTemplate
    Uses: OpenAI JSON mode with few-shot examples inline
    
    Returns:
        List of proposition strings
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Please break down the following text into simple, self-contained propositions. "
                "Ensure that each proposition meets the following criteria:\n\n"
                "1. Express a Single Fact: Each proposition should state one specific fact or claim.\n"
                "2. Be Understandable Without Context: The proposition should be self-contained.\n"
                "3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.\n"
                "4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers.\n"
                "5. Contain One Subject-Predicate Relationship: Focus on a single subject and its action/attribute.\n\n"
                "Respond with JSON: {\"propositions\": [\"prop1\", \"prop2\", ...]}"
            )
        },
        {
            "role": "user",
            "content": "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission."
        },
        {
            "role": "assistant",
            "content": json.dumps({"propositions": [
                "Neil Armstrong was an astronaut.",
                "Neil Armstrong walked on the Moon in 1969.",
                "Neil Armstrong was the first person to walk on the Moon.",
                "Neil Armstrong walked on the Moon during the Apollo 11 mission.",
                "The Apollo 11 mission occurred in 1969."
            ]})
        },
        {
            "role": "user",
            "content": chunk_text_var
        }
    ]
    
    result = llm.chat_json(messages)
    return result.get("propositions", [])

# Evaluate propositions
def evaluate_propositions(llm, proposition:str, original_text: str) -> Dict[str,int]:
    """
    Grade a proposition on accuracy, clarity, completeness, and conciseness (1-10 each).
    
    Replaces: ChatGroq + with_structured_output(GradePropositions)
    
    Returns:
        Dict with scores: {"accuracy": int, "clarity": int, "completeness": int, "conciseness": int}
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate propositions extracted from documents. Rate each on a 1-10 scale:\n"
                "- accuracy: How well the proposition reflects the original text\n"
                "- clarity: How easy it is to understand without additional context\n"
                "- completeness: Whether it includes necessary details (dates, qualifiers)\n"
                "- conciseness: Whether it is concise without losing important information\n\n"
                "Example:\n"
                'Docs: "In 1969, Neil Armstrong became the first person to walk on the Moon during Apollo 11."\n'
                'Proposition: "Neil Armstrong walked on the Moon in 1969."\n'
                'Evaluation: {"accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10}\n\n'
                'Respond with JSON: {"accuracy": N, "clarity": N, "completeness": N, "conciseness": N}'
            )
        },
        {
            "role": "user",
            "content": f'Proposition: "{proposition}"\nOriginal Text: "{original_text}"'
        }
    ]
    
    return llm.chat_json(messages)


def passes_quality_check(scores:Dict[str, int], threshold:int=7) -> bool:
    """Check if all scores meet the threshold."""
    for category in ["accuracy", "clarity", "completeness", "conciseness"]:
        if scores.get(category,0) < threshold:
            return False
    return True





class PropositionChunkRagOpenai:

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

        if self.is_pdf:
            
            pdf_text = read_pdf(file_path)
            chunks = chunk_text(pdf_text, chunk_size, chunk_overlap)

            doc_splits = []
            for i, chunk in enumerate(chunks):
                doc_splits.append(Document(
                    content=chunk,
                    metadata={
                        "title": file_path,
                        "source": file_path,
                        "chunk_id": i + 1
                    }
                ))

            propositions = []

            for doc in doc_splits:
                chunk_id = doc.metadata["chunk_id"]
                props = generate_propositions(self.llm, doc.content)
                
                # print(f"\n Chunk {chunk_id}: generated {len(props)} Propositions")
                
                for prop in props:
                    propositions.append(Document(
                        content=prop,
                        metadata={
                            "title": doc.metadata["title"],
                            "source": doc.metadata["source"],
                            "chunk_id": chunk_id
                        }
                    ))

            evaluated_propositions = []

            for idx, prop_doc in enumerate(propositions):

                chunk_id = prop_doc.metadata["chunk_id"]
                original_text = doc_splits[chunk_id - 1].content

                scores = evaluate_propositions(self.llm, prop_doc.content, original_text)

                if passes_quality_check(scores):
                    evaluated_propositions.append(prop_doc)
                else:
                    print(f"   X [{idx+1}] FAIL - {prop_doc.content}")
                    print(f"   Scores: {scores}")
                    
            print(f"\n✓ {len(evaluated_propositions)}/{len(propositions)} propositions passed quality check")

            self.rag_retriever.index_documents(evaluated_propositions)


    def query(self, question:str) -> str:
        
        retrieved_docs = self.rag_retriever.retrieve_context(question)
        
        return self.llm.chat_with_context(question, retrieved_docs), retrieved_docs


if __name__ == "__main__":
    
    rag = PropositionChunkRagOpenai(r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf")
    print(rag.query("What are the main causes of climate change?"))

