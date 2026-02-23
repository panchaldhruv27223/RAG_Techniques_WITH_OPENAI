"""
RAG with Feedback Loop

A self-improving RAG system that learns from user feedback over time.
After each query-answer cycle, the user rates relevance and quality.
This feedback is stored and used in two ways:

    1. Score Adjustment (per-query): Past feedback on similar topics
       boosts/penalizes chunk relevance scores for future queries.
    2. Index Fine-Tuning (periodic): High-quality Q&A pairs are added
       to the vector store as new documents, enriching the knowledge base.

How it works:

    Standard RAG loop:
        Query → retrieve chunks → generate answer → show to user

    Feedback enhancement (after user rates the answer):
        Store feedback → next query checks past feedback
        → LLM evaluates: "is old feedback relevant to new query?"
        → if yes, adjust chunk scores based on old feedback ratings
        → re-rank chunks → better answer

    Periodic fine-tuning:
        Load all feedback → filter high-quality (relevance≥4 AND quality≥4)
        → combine query+answer as new document text
        → re-index the entire store with original + new docs
        → future searches benefit from proven good answers

Why this matters:
    - Standard RAG is static — same query always returns same results
    - Feedback loop is adaptive — system improves with each interaction
    - Bad answers get penalized, good answers get reinforced
    - Over time, the system learns what users actually find useful

Usage:
    from feedback_loop_rag import FeedbackLoopRAG

    rag = FeedbackLoopRAG(file_path="document.pdf")
    answer, contexts = rag.query("What is the greenhouse effect?")
    rag.submit_feedback(relevance=5, quality=5, comments="Great answer")
    rag.fine_tune_index()  # periodic call to absorb good feedback
"""


import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    read_pdf,
    chunk_text,
)

# Feedback Store — JSON-based persistence

class FeedbackStore:
    """
    Stores and loads user feedback as JSON lines (one JSON object per line).

    File format (JSONL — one object per line):
        {"query": "...", "response": "...", "relevance": 5, "quality": 4, ...}
        {"query": "...", "response": "...", "relevance": 2, "quality": 3, ...}

    Args:
        file_path:  Path to the feedback JSONL file.
    """

    def __init__(self, file_path:str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def store(self, feedback: Dict[str, Any]) -> None:
        """ Append a feedback entry to the file. """

        feedback["timestamp"] = datetime.now().isoformat()

        with open(self.file_path, "a", encoding="utf-8") as f:
            json.dump(feedback,f)
            f.write("\n")

    def load_all(self) -> List[Dict[str, Any]]:
        """ Load all feedback entries from the file. """
        feedback_data = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:

                        if not line.strip():
                            continue
                        line =  line.strip()

                        feedback_data.append(json.loads(line))

                    except json.JSONDecodeError:
                        continue

        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []

        return feedback_data

    def load_high_quality(self, min_relevance:int=4, min_quality:int=4) -> List[Dict]:
        """ Load only feedback that meets the minimum thresholds. """
        all_feedback = self.load_all()
        return [f for f in all_feedback if f.get("relevance",0) >= min_relevance and f.get("quality",0) >= min_quality]

    @property
    def count(self)-> int:
        """ Return the total number of feedback entries. """
        return len(self.load_all())




# Feedback-Aware Relevance Adjuster
class RelevanceAdjuster:
    """
    Adjusts chunk relevance scores based on past user feedback.

    For each retrieved chunk, the adjuster:
        1. Loads all past feedback
        2. Asks LLM: "Is this old feedback relevant to the current query + chunk?"
        3. If yes, adjusts the chunk's score based on feedback ratings

    Args:
        model_name:   OpenAI model for relevance checking.
        temperature:  Should be 0 for consistent evaluation.
    """

    def __init__(self, model_name:str = "gpt-4o-mini", temperature:float=0.0):


        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=50
        )

    def is_feedback_relevant(
        self,
        query:str,
        chunk_content:str,
        feedback:Dict[str,Any]
    ) -> bool:

        """
        Ask LLM whether past feedback is relevant to the current query+chunk.

        Args:
            query:          Current user query.
            chunk_content:  Content of the chunk being evaluated.
            feedback:       Past feedback entry.

        Returns:
            True if the feedback is relevant.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You determine if past feedback is relevant to a current query."
                    "Return Json with key 'relevant' set to true or false"
                )
            },

            {
                "role":"user",
                "content":(
                    f"Current query: {query}\n"
                    f"Feedback's original query: {feedback['query']}\n"
                    f"Document Content (First 100 characters): {chunk_content[:1000]}\n"
                    f"Feedback Response: {feedback['response']}\n\n"
                    f"Is this feedback relevant to the current query and document content?"
                )
            }
        ]

        try :

            result = self.llm.chat_json(messages)
            return result.get("relevant", False) is True

        except Exception as e:
            print(f"Error checking relevance: {e}")
            return False

    
    def adjust_score(
        self,
        query:str,
        results:List[RetrievalResult],
        feedback_data:List[Dict[str,Any]],
        neutral_score:float= 3.0,
    ) -> List[RetrievalResult]:
        
        """
        Adjust retrieval scores based on relevant past feedback.

        For each chunk:
            - Find feedback entries relevant to this query+chunk
            - Compute average relevance rating from those entries
            - Multiply the chunk's score by (avg_rating / neutral_score)
            - Scores above neutral get boosted, below get penalized

        Example:
            chunk_score = 0.85, relevant feedback avg = 4.5, neutral = 3.0
            adjusted = 0.85 × (4.5 / 3.0) = 1.275  → boosted!

            chunk_score = 0.85, relevant feedback avg = 1.5, neutral = 3.0
            adjusted = 0.85 × (1.5 / 3.0) = 0.425  → penalized!

        Args:
            query:          Current user query.
            results:        Retrieved chunks with scores.
            feedback_data:  All past feedback entries.
            neutral_score:  Rating that means "no adjustment" (default 3 on 1-5 scale).

        Returns:
            Re-ranked list of RetrievalResult objects.
        """
        if not feedback_data:
            return results

        print(f"   [Feedback] Checking {(len(feedback_data))} feedback entires against {len(results)} chunks...")

        for result in results:

            relevent_feedback = []

            for fb in feedback_data:
                if self.is_feedback_relevant(query, result.document.content, fb):
                    relevent_feedback.append(fb)

            if relevent_feedback:
                avg_relevance = sum(f['relevance'] for f in relevent_feedback) / len(relevent_feedback)

                adjustment = avg_relevance / neutral_score
                original_score = result.score
                result.score = original_score * adjustment

        # Re-sort by adjusted score
        results.sort(key=lambda r: r.score, reverse=True)
        return results



# Feedback Loop Retriever

class FeedbackLoopRetriever:
    """
    Retriever with feedback-based score adjustment and index fine-tuning.

    Two enhancement mechanisms:

    1. Per-query adjustment: Past feedback adjusts chunk relevance scores
       in real-time for each new query.

    2. Periodic fine-tuning: High-quality Q&A pairs are added to the index
       as new documents, enriching future searches.

    Args:
        embedding_model:    OpenAI embedding model.
        adjuster_model:     OpenAI model for feedback relevance checking.
        chunk_size:         Characters per chunk.
        chunk_overlap:      Overlap between chunks.
        k:                  Number of retrieved chunks.
        feedback_file:      Path to feedback JSONL file.
        use_feedback:       Whether to apply feedback adjustment per query.
    """

    def __init__(self, embedding_model:str = "text-embedding-3-small", adjuster_model:str="gpt-4o-mini", chunk_size:int=1000, chunk_overlap:int=200, k:int=3, feedback_file:str="data/feedback_data.json", use_feedback:bool=True):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.use_feedback = use_feedback

        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.feedback_store = FeedbackStore(file_path=feedback_file)
        self.adjuster = RelevanceAdjuster(model_name=adjuster_model)

        self._original_text: str = ""

    def index_document(self, text:str, doc_id:str="doc_0")->int:
        """Chunk and index Document"""

        self._original_text = text
        
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        documents = []

        for i, chunks in enumerate(chunks):
            documents.append(
                Document(
                    content=chunks,
                    metadata={
                        "doc_id":doc_id,
                        "chunk_index":i,
                        "relevence_score":1.0
                    }
                )
            )

        documents = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents)

        return len(chunks)


    def index_pdf(self, file_path:str, doc_id:Optional[str]=None)->int:

        if doc_id is None:
            doc_id = os.path.basename(file_path)

        text = read_pdf(file_path)

        return self.index_document(text=text, doc_id=doc_id)

    def retrieve(self, query:str)->List[str]:
        """
        Retrieve chunks with optional feedback-based score adjustment.

        Args:
            query:  Search query.

        Returns:
            List of chunk texts, potentially re-ranked by feedback.
        """

        query_embed = self.embedder.embed_text(query)

        results = self.vector_store.search(query_embed, k=self.k)

        if self.use_feedback:
            feedback_data = self.feedback_store.load_all()

            if feedback_data:
                results = self.adjuster.adjust_score(
                    query=query,
                    results=results,
                    feedback_data=feedback_data
                )

        return [r.document.content for r in results]

    def store_feedback(self, query:str, response:str, relevance:int, quality:int, comments:str ="")->None:

        """
        Store user feedback for a query-response pair.

        Args:
            query:      The user's original question.
            response:   The RAG system's answer.
            relevance:  User's relevance rating (1-5).
            quality:    User's quality rating (1-5).
            comments:   Optional free-text feedback.
        """

        feedback = {
            "query":query,
            "response":response,
            "relevance":max(1, min(5,int(relevance))),
            "quality":max(1, min(5,int(quality))),
            "comments":comments
        }

        self.feedback_store.store(feedback)

        print(f"    [FEEDBACK]  Stored feedback for query: {query}")

    def fine_tune_index(self)->int:
        """
        Rebuild the index including high-quality Q&A pairs as new documents.

        This is the "periodic fine-tuning" step. It:
            1. Loads high-quality feedback (relevance≥4 AND quality≥4)
            2. Combines query+response into new document text
            3. Appends to original text
            4. Rebuilds the entire vector store

        Should be called periodically (daily/weekly), not per-query.

        Returns:
            New total number of indexed chunks.
        """

        good_feedback = self.feedback_store.load_high_quality()

        if not good_feedback:
            print(f"  [FineTune]  No high-quality feedback to incorporate.")

        additional_text = "\n\n".join(
            f"Question: {f['query']}\nAnswer: {f['response']}\n"
            for f in good_feedback
        )

        combined_text = self._original_text + "\n\n" + additional_text
        
        # Reset and re-index

        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

        num_chunks = self.index_document(combined_text)

        print(
            f"  [Finetune] Re-Indexed with {len(good_feedback)} high-quality Q&A pairs "
            f" -> {num_chunks} total chunks"
        )

        return num_chunks


class FeedbackLoopRAG:
    """
    Complete RAG pipeline with user feedback loop.

    Workflow:
        1. rag.query("question") → answer + contexts
        2. User evaluates → rag.submit_feedback(relevance=5, quality=4)
        3. Next query benefits from past feedback (score adjustment)
        4. Periodically call rag.fine_tune_index() to absorb good Q&A pairs

    Usage:
        rag = FeedbackLoopRAG(file_path="report.pdf")

        # Query
        answer, ctx = rag.query("What is the greenhouse effect?")

        # Feedback
        rag.submit_feedback(relevance=5, quality=5, comments="Great!")

        # Fine-tune (periodic)
        rag.fine_tune_index()
    """

    def __init__(self, file_path:str, chunk_size:int=1000, chunk_overlap:int=200, k:int=3, feedback_file:str="data/feedback_data.json", use_feedback:bool=True, embedding_model:str="text-embedding-3-small", adjuster_model:str="gpt-4o-mini", chat_model:str="gpt-4o-mini", temperature:float=0.1):

        self.file_path = file_path
        self._last_query:str = ""
        self._last_response:str = ""

        self.retriever = FeedbackLoopRetriever(
            embedding_model=embedding_model,
            adjuster_model=adjuster_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
            feedback_file=feedback_file,
            use_feedback=use_feedback
        )

        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature
        )

        if file_path.endswith(".pdf"):
            num_chunks = self.retriever.index_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            num_chunks = self.retriever.index_document(text)

        fb_count = self.retriever.feedback_store.count  
        print(
            f"[FeedbackLoop] Indexed '{fb_count}' existing feedback entries loaded"
        )

        if fb_count > 0:
            print(f"[FeedbackLoop] {fb_count} existing feedback entries loaded")
        
        print(f"[FeedbackLoop] feedback adjustment: {"ON" if use_feedback else "OFF"}")
    
    def query(self, question:str, return_context: bool = True)->Tuple[str, List[str]]:

        """
        Query the RAG system. Results may be adjusted by past feedback.
        args:
        questions:  user's question.
        return_context: whatever to return retrieved contexts.

        returns:
        Tuple of (answer_string, list_of_context_string).
        """

        contexts = self.retriever.retrieve(question)

        if not contexts:
            return "No relevent information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        self._last_query = question
        self._last_response = answer

        if return_context:
            return answer, contexts
        else:
            return answer, []

    def submit_feedback(self, relevance:int, quality:int, comments:str ="", query:Optional[str]=None, response:Optional[str]=None)->None:
        
        """
        Submit feedback for a query-response pair.

        If query/response not provided, uses the last query/response.

        Args:
            relevance:  Relevance rating 1-5 (5=highly relevant).
            quality:    Quality rating 1-5 (5=excellent).
            comments:   Optional free-text feedback.
            query:      Override query (defaults to last query).
            response:   Override response (defaults to last response).
        """

        q = query or self._last_query
        r = response or self._last_response

        if not q or not r:
            print("   [FeedBack] No Query/response to provide feedback on.  ")
            return 


        self.retriever.store_feedback(q, r, relevance, quality, comments)

    def fine_tune_index(self)->int:
        """
        Rebuild the index with high-quality feedback incorporated.

        Call this periodically (daily/weekly), not after every query.

        Returns:
            New total number of indexed chunks.
        """

        print("\n [FeedbackLoop]  Fine-Tuning Index with high-quality feedback...")
        return self.retriever.fine_tune_index()

    def show_feedback_stats(self)->None:
        """Show Feedback statistics. """
        all_fb = self.retriever.feedback_store.load_all()
        good_fb = self.retriever.feedback_store.load_high_quality()

        if not all_fb:
            print("\n NO FEEDBACK COLLECTED YET.  ")
            return 

        avg_rel = sum(f['relevance'] for f in all_fb) / len(all_fb)
        avg_qual = sum(f['quality'] for f in all_fb) / len(all_fb)
        
        print(f"Feedback statistics:")
        print(f" Total entries:   {len(all_fb)}")
        print(f" HIgh-quality entries: {len(all_fb)} (for fine-tuning)")
        print(f" AVG relevance: {avg_rel:.1f}/5")
        print(f" AVG quality: {avg_qual:.1f}/5")


        if all_fb:
            print(f"\n  Recent Feedback:")
            for f in all_fb[-3:]:
                print(
                    f"  [{f.get('timestamp', '?')[:10]}]"
                    f"  rel={f["relevance"]} qual={f["quality"]}"
                    f"  q = \"{f["query"][:50]}...\""
                )



if __name__ == "__main__":
    
    pdf_path = r"data\Understanding_Climate_Change.pdf"

    rag = FeedbackLoopRAG(
        file_path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        k=3,
        use_feedback=True
    )

    print("\n [Feedback RAG] Ready. After each answer, you can rate it.")
    print("Commands: 'exit', 'stats', 'finetune' \n")

    while True:
        question = input("User: ").strip()

        if question.lower() == "exit":
            break 

        elif question.lower() == "stats":
            rag.show_feedback_stats()
            continue

        elif question.lower() == "finetune":
            rag.fine_tune_index()
            continue

        answer, contexts = rag.query(question)
        print(f"\n RAG: {answer}")
        print(f"\n Context Used: {len(contexts)}")
        print(f"\n Used Contexts: {contexts}")



        ## lets collecting the feedback
        try:
            print("\n Rate this answer (1-5, 5=best): ")
            rel = int(input("Relevance (1-5): ").strip())
            qual = int(input("Quality (1-5): ").strip())
            comments = input("Comments (optional): ").strip()

            rag.submit_feedback(rel, qual, comments)
            print("Feedback submitted. Thank you!\n")
 
        except (ValueError, EOFError):
            print("Invalid input. Please enter numbers between 1 and 5. or press Ctrl+C to exit")
            pass

        print("")

