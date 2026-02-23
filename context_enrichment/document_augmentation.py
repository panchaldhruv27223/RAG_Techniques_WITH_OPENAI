"""
Document Augmentation via Question Generation.


Enhances retrieval by generating hypothetical questions for each chunk at
indexing time and embedding those questions alongside the original text.
When a user asks a question, vector search is more likely to match because
the query is compared against both original text AND pre-generated questions
that are phrased similarly to how a user would ask.

Key insight: Users ask questions, but documents contain statements. There's
a semantic gap between "What causes global warming?" (query) and "Greenhouse
gases trap heat in the atmosphere" (document). By generating questions like
"What role do greenhouse gases play?" at index time, we bridge that gap.

How it works:
    1. Split document into large "documents" (for context) and small "fragments"
       (for retrieval precision)
    2. For each document/fragment, use LLM to generate N related questions
    3. Index BOTH original fragments AND generated questions in FAISS
       (each question stores a reference back to its parent document text)
    4. On query: vector search finds the most similar item (often a question)
    5. Use the parent document text as context for answer generation


Usage:
    from document_augmentation_rag import DocumentAugmentationRAG

    rag = DocumentAugmentationRAG(file_path="document.pdf")
    answer, context = rag.query("What causes climate change?")
"""

import os
import sys
import re
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

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
)


class QuestionGeneration(Enum):
    """Level at which questions are generated."""
    DOCUMENT_LEVEL = 1   # Generate from large document chunks (faster)
    FRAGMENT_LEVEL = 2   # Generate from small fragments (more targeted)



class QuestionGenerator:
    """
    Generates hypothetical questions from text using OpenAI's chat API.

    Args:
        model_name:   OpenAI model for question generation.
        temperature:  Should be 0 for consistent generation.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.llm = OpenAIChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=2000,
        )

    def generate(self, text: str, num_questions: int = 10) -> List[str]:
        """
        Generate questions that could be answered by the given text.

        Args:
            text:           Source text to generate questions from.
            num_questions:  Minimum number of questions to generate.

        Returns:
            List of unique, cleaned question strings.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate questions from text for a retrieval system. "
                    "Return a JSON object with a single key 'questions' containing "
                    "a list of question strings. Every question must end with '?' "
                    "and be directly answerable from the provided text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Using the following context, generate at least {num_questions} "
                    f"questions that can be answered from this text.\n\n"
                    f"Context:\n{text}\n\n"
                    f"Return JSON with key 'questions' containing the list."
                ),
            },
        ]

        try:
            result = self.llm.chat_json(messages)
            questions = result.get("questions", [])
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [QuestionGen] JSON parse failed, falling back to text: {e}")
            # Fallback: use regular chat and parse line by line
            raw = self.llm.chat(messages)
            questions = [
                line.strip() for line in raw.split("\n")
                if line.strip().endswith("?")
            ]

        # Clean and deduplicate
        cleaned = self._clean_questions(questions)
        return list(set(cleaned))

    def _clean_questions(self, questions: List[str]) -> List[str]:
        """
        Remove numbering prefixes and filter to valid questions.

        "1. What is climate change?" → "What is climate change?"
        "This is not a question."    → filtered out
        """
        cleaned = []
        for q in questions:
            # Remove leading numbers like "1. " or "1) "
            q = re.sub(r'^\d+[\.\)]\s*', '', q.strip())
            if q.endswith('?') and len(q) > 10:
                cleaned.append(q)
        return cleaned

def split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into chunks based on word/token count (not character count).

    The original notebook uses token-based splitting (re.findall for words)
    rather than character-based. We replicate that here.

    Args:
        text:           Full text to split.
        chunk_size:     Max tokens (words) per chunk.
        chunk_overlap:  Overlapping tokens between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    tokens = re.findall(r'\b\w+\b', text)
    chunks = []

    step = chunk_size - chunk_overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break

    return chunks



class DocumentAugmentationRetriever:

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        question_model: str = "gpt-4o-mini",
        generation_level: QuestionGeneration = QuestionGeneration.DOCUMENT_LEVEL,
        questions_per_chunk: int = 40,
        document_max_tokens: int = 4000,
        document_overlap: int = 100,
        fragment_max_tokens: int = 128,
        fragment_overlap: int = 16,
        k: int = 1,
    ):
        self.generation_level = generation_level
        self.questions_per_chunk = questions_per_chunk
        self.document_max_tokens = document_max_tokens
        self.document_overlap = document_overlap
        self.fragment_max_tokens = fragment_max_tokens
        self.fragment_overlap = fragment_overlap
        self.k = k


        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.question_gen = QuestionGenerator(model_name=question_model)


        self.stats = {
            "documents": 0,
            "fragments": 0,
            "questions": 0,
            "total_indexed": 0,
        }



    def index_text(self, text: str) -> int:
        """
        Process and index a document with question augmentation.

        Full pipeline:
            1. Split into large document chunks
            2. Split each document into small fragments
            3. Generate questions (at document or fragment level)
            4. Index everything in FAISS with parent doc references

        Args:
            text:  Full document text.

        Returns:
            Total number of items indexed (fragments + questions).
        """
        # Step 1: Split into large document chunks
        text_documents = split_by_tokens(
            text, self.document_max_tokens, self.document_overlap
        )
        self.stats["documents"] = len(text_documents)
        print(f"  [Augmentation] Split into {len(text_documents)} document chunks")

        all_docs: List[Document] = []
        total_questions = 0

        for i, text_document in enumerate(text_documents):
            # Step 2: Split document into small fragments
            fragments = split_by_tokens(
                text_document, self.fragment_max_tokens, self.fragment_overlap
            )
            print(f"  [Augmentation] Document {i} → {len(fragments)} fragments")

            # Index each fragment (with parent doc reference in metadata)
            for j, fragment in enumerate(fragments):
                all_docs.append(
                    Document(
                        content=fragment,
                        metadata={
                            "type": "ORIGINAL",
                            "doc_index": i,
                            "fragment_index": j,
                            "parent_text": text_document,  # Context for answers
                        },
                    )
                )

                # FRAGMENT_LEVEL: generate questions per fragment
                if self.generation_level == QuestionGeneration.FRAGMENT_LEVEL:
                    questions = self.question_gen.generate(
                        fragment, self.questions_per_chunk
                    )
                    for q in questions:
                        all_docs.append(
                            Document(
                                content=q,
                                metadata={
                                    "type": "AUGMENTED",
                                    "doc_index": i,
                                    "fragment_index": j,
                                    "parent_text": text_document,
                                },
                            )
                        )
                    total_questions += len(questions)
                    print(
                        f"    Fragment {j} → {len(questions)} questions generated"
                    )

            # DOCUMENT_LEVEL: generate questions per document chunk
            if self.generation_level == QuestionGeneration.DOCUMENT_LEVEL:
                questions = self.question_gen.generate(
                    text_document, self.questions_per_chunk
                )
                for q in questions:
                    all_docs.append(
                        Document(
                            content=q,
                            metadata={
                                "type": "AUGMENTED",
                                "doc_index": i,
                                "fragment_index": -1,
                                "parent_text": text_document,
                            },
                        )
                    )
                total_questions += len(questions)
                print(f"  [Augmentation] Document {i} → {len(questions)} questions generated")

        # Step 4: Embed everything and add to FAISS
        self.stats["fragments"] = len(
            [d for d in all_docs if d.metadata["type"] == "ORIGINAL"]
        )
        self.stats["questions"] = total_questions
        self.stats["total_indexed"] = len(all_docs)

        print(
            f"  [Augmentation] Indexing {len(all_docs)} items "
            f"({self.stats['fragments']} fragments + {total_questions} questions)"
        )

        all_docs = self.embedder.embed_documents(all_docs)
        self.vector_store.add_documents(all_docs)

        return len(all_docs)

    def index_pdf(self, file_path: str) -> int:
        """Read and index a PDF with question augmentation."""
        text = read_pdf(file_path)
        return self.index_text(text)

    def index_text_file(self, file_path: str) -> int:
        """Read and index a text file with question augmentation."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.index_text(text)

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Find the most relevant item (fragment or question) via vector search.

        Args:
            query:  User's question.

        Returns:
            List of RetrievalResult objects.
        """
        query_emb = self.embedder.embed_text(query)
        return self.vector_store.search(query_emb, k=self.k)

    def retrieve_context(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve parent document context for a query.

        The key trick: vector search might match a generated QUESTION,
        but we return the PARENT DOCUMENT text as context (not the question).
        This gives the answer LLM a much richer context to work with.

        Args:
            query:  User's question.

        Returns:
            Tuple of (context_texts, match_info).
            context_texts: Parent document texts for answer generation.
            match_info: What was actually matched (for debugging).
        """
        results = self.retrieve(query)

        contexts = []
        match_info = []
        seen_parents = set()

        for r in results:
            parent_text = r.document.metadata.get("parent_text", "")
            match_type = r.document.metadata.get("type", "UNKNOWN")

            # Deduplicate: don't include the same parent document twice
            parent_key = parent_text[:100]  # Use first 100 chars as key
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)

            contexts.append(parent_text)
            match_info.append({
                "matched_content": r.document.content[:200],
                "matched_type": match_type,
                "score": r.score,
                "doc_index": r.document.metadata.get("doc_index"),
            })

        return contexts, match_info



class DocumentAugmentationRAG:
    def __init__(
        self,
        file_path: str,
        generation_level: QuestionGeneration = QuestionGeneration.DOCUMENT_LEVEL,
        questions_per_chunk: int = 40,
        document_max_tokens: int = 4000,
        document_overlap: int = 100,
        fragment_max_tokens: int = 128,
        fragment_overlap: int = 16,
        k: int = 1,
        embedding_model: str = "text-embedding-3-small",
        question_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the Document Augmentation RAG pipeline.

        Args:
            file_path:             Path to document (PDF or text file).
            generation_level:      DOCUMENT_LEVEL or FRAGMENT_LEVEL.
            questions_per_chunk:   Questions to generate per chunk.
            document_max_tokens:   Large chunk size (for context).
            document_overlap:      Overlap for large chunks.
            fragment_max_tokens:   Small chunk size (for retrieval).
            fragment_overlap:      Overlap for small chunks.
            k:                     Vector search results count.
            embedding_model:       OpenAI embedding model.
            question_model:        OpenAI model for question generation.
            chat_model:            OpenAI model for answer generation.
            temperature:           LLM temperature for answers.
        """
        self.file_path = file_path

        # Initialize retriever
        self.retriever = DocumentAugmentationRetriever(
            embedding_model=embedding_model,
            question_model=question_model,
            generation_level=generation_level,
            questions_per_chunk=questions_per_chunk,
            document_max_tokens=document_max_tokens,
            document_overlap=document_overlap,
            fragment_max_tokens=fragment_max_tokens,
            fragment_overlap=fragment_overlap,
            k=k,
        )

        # Initialize chat model
        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        # Index the document (this takes a while due to question generation)
        level_name = generation_level.name
        print(
            f"[DocAugmentation] Indexing '{os.path.basename(file_path)}' "
            f"(level={level_name}, questions_per_chunk={questions_per_chunk})..."
        )
        print(f"[DocAugmentation] This may take a few minutes due to question generation.\n")

        if file_path.endswith(".pdf"):
            total = self.retriever.index_pdf(file_path)
        else:
            total = self.retriever.index_text_file(file_path)

        stats = self.retriever.stats
        print(
            f"\n[DocAugmentation] Done → {total} items indexed "
            f"({stats['documents']} docs, {stats['fragments']} fragments, "
            f"{stats['questions']} questions)"
        )

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Query the augmented RAG system.

        Flow:
            1. Vector search → match (often a generated question)
            2. Retrieve the PARENT DOCUMENT text as context
            3. Feed parent doc + question to answer LLM
            4. Return answer + context

        Args:
            question:        User's question.
            return_context:  Whether to return context texts.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """
        contexts, match_info = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found in the document.", []

        # Generate answer using parent document context
        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []


    def show_match(self, question: str) -> None:
        """
        Debug helper: show what was actually matched and what context is used.

        This reveals the key insight — the vector search often matches a
        generated QUESTION (not the original text), but the PARENT DOCUMENT
        text is used as context for the answer.

        Args:
            question:  User's question.
        """
        contexts, match_info = self.retriever.retrieve_context(question)

        print(f"\nQuery: {question}")
        print("=" * 70)

        for i, info in enumerate(match_info):
            match_type = info["matched_type"]

            print(f"    Score:   {info['score']:.4f}")
            print(f"    Doc #:   {info['doc_index']}")
            print(f"    Matched: {info['matched_content']}...")

        print(f"\n  Context used for answer ({len(contexts)} parent docs):")
        for i, ctx in enumerate(contexts):
            print(f"    Doc {i + 1}: {ctx[:200]}...")

        print("\n" + "=" * 70)



if __name__ == "__main__":
    pdf_path = r"C:\Users\TempAccess\Documents\Dhruv\RAG\data\Understanding_Climate_Change.pdf"

    rag = DocumentAugmentationRAG(
        file_path=pdf_path,
        generation_level=QuestionGeneration.DOCUMENT_LEVEL,
        questions_per_chunk=40,
        document_max_tokens=4000,
        fragment_max_tokens=128,
        k=1,
    )

    question = "What causes global warming?"

    answer, context = rag.query(question)
    print(f"\nAnswer: {answer}")
    print(f"Context docs used: {len(context)}")