import os 
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
from typing import List, Dict, Any, Optional, Tuple

from helper_function_openai import (
    RAGRetriever,
    OpenAIEmbedder,
    OpenAIChat,
    FAISSVectorStore,
    chunk_text,
    chunk_documents,
    show_context,
    Document,
    RetrievalResult,
    cosine_similarity
)


chat = OpenAIChat(
    model_name="gpt-4o-mini",
    temperature=0.0
)


## Helper functions

## Grade documents
def grade_document(rag_retriever: RAGRetriever, question: str, document_content: str)-> str:
    """
    Grade whether a retrieved document is relevant to the question.
    
    Uses: OpenAI JSON mode for structured output
    
    Returns:
        'yes' or 'no'
    """

    query_embedding = rag_retriever.embedder.embed_text([question])
    document_embedding = rag_retriever.embedder.embed_text([document_content])

    similarity = cosine_similarity(query_embedding, document_embedding)

    # print(f"similarity: {similarity}")
    return "yes" if similarity > 0.5 else "no"


## format documents for contexts.
def format_docs_for_contexts(docs: List[Document]) -> str:
    """
    Format documents into structured context strings.
    Includes doc ID, title, and source for attribution.
    """

    formatted = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "Unkown")
        formatted.append(
            f"""<doc{i+1}>
            <title>{title}</title>
            <source>{source}</source>
            <content>{doc.content}</content>
            </doc{i+1}>
            """
        )
    
    return formatted

class ReliableRAGOpenAI(RAGRetriever):

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

    def embed_file(self):
        if self.is_pdf:
            self.index_pdf(path=self.file_path)

        elif self.is_csv:
            self.index_csv(file_path=self.file_path)

    def generate_answer(self, query:str, docs:List[Document]) -> str:
        """
        Generate a RAG answer using filtered documents.
        """
        if docs:
            context = format_docs_for_contexts(docs)
        
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Answer the question based upon the provided documents. "
                "Use three-to-five sentences maximum and keep the answer concise."
                )

            return self.llm.chat_with_context(query=query, context=context, system_prompt=system_prompt)
        else:
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                )
            messages = [
                {
                    "role":"system",
                    "content":system_prompt
                },
                {
                    "role":"user",
                    "content":query
                }
            ]
            return self.llm.chat(messages=messages)

    def query(self, question):
        
        results = self.retrieve(question, k=self.k)

        docs_to_use = []
        context_used = []

        for re in results:
            # content = re.document.content
            # print(content)
            # score = grade_document(rag_retriever, question, content)
            # print(score)

            # if score == "yes":
            if re.score > 0.5:
                docs_to_use.append(re.document)
                context_used.append(re.document.content)

        # print(f"\n {len(docs_to_use)} / {len(results)} documents passed relevancy filter.")

        answer = self.generate_answer(question, docs_to_use)

        return answer, context_used





if __name__ == "__main__":
    pdf_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"
    question = "What are the main causes of climate change?"
    rag_retriever = ReliableRAGOpenAI(pdf_path)
    answer, context = rag_retriever.query(question)
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print(f"length of context: {len(context)}")

    csv_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\customers-100.csv"
    question = "Which company does Sheryl Baxter work for?"
    rag_retriever = ReliableRAGOpenAI(csv_path)
    answer, context = rag_retriever.query(question)
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print(f"length of context: {len(context)}")
