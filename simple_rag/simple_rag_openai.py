import os 
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
from helper_function_openai import SimpleRAG


class SimpleRAGOpenai:
    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200, k:int = 3):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        is_pdf = self.file_path.endswith(".pdf")
        is_csv = self.file_path.endswith(".csv")

        self.simple_rag = SimpleRAG(
            embedding_model="text-embedding-3-small",
            chat_model = "gpt-4o-mini",
            temperature=0.0
        )

        if is_pdf:
            self.simple_rag.index_pdf(
                self.file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        elif is_csv:
            self.simple_rag.index_csv(self.file_path)

    def query(self, question):  
        response = self.simple_rag.query(question, k=self.k, return_context=True)

        return response["answer"], response["context"]


if __name__ == "__main__":
    pdf_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"
    question = "What is the main cause of climate change?"
    simple_rag = SimpleRAGOpenai(pdf_path)
    answer, context = simple_rag.query(question)
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print(f"length of context: {len(context)}")

    csv_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\customers-100.csv"
    question = "Which company does Sheryl Baxter work for?"
    simple_rag = SimpleRAGOpenai(csv_path)
    answer, context = simple_rag.query(question)
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print(f"length of context: {len(context)}")
