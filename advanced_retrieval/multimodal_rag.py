"""
Multi-Modal RAG with Image Captioning — Pure OpenAI SDK Implementation

Processes BOTH text and images from PDFs. Images are captioned by an
LLM (GPT-4o), and those captions are embedded alongside text chunks
in the same vector store. This means queries can match against image
descriptions too — not just text.

Why multi-modal matters:
    Many documents (research papers, reports, manuals) contain critical
    information in figures, tables, and diagrams that text-only RAG
    completely misses. For example, in "Attention Is All You Need":
    - BLEU scores are in a TABLE (image)
    - The Transformer architecture is a FIGURE
    - Training details are in TEXT
    
    Text-only RAG can't answer "What does the Transformer architecture
    look like?" because that info is in an image.

How it works:
    1. Extract text from each PDF page (PyMuPDF)
    2. Extract images from each PDF page (PyMuPDF)
    3. Caption each image using GPT-4o (vision model)
    4. Chunk text + image captions together
    5. Embed and index everything in one FAISS store
    6. On query: search finds text chunks OR image captions
    7. Answer LLM uses matched context (text or caption) to respond

Usage:
    from multi_modal_rag import MultiModalRAG

    rag = MultiModalRAG(file_path="paper.pdf")
    answer, contexts = rag.query("What is the BLEU score of the base model?")
"""

import os
import sys
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

import fitz
from PIL import Image

from openai import OpenAI


from helper_function_openai import (
    Document,
    RetrievalResult,
    OpenAIEmbedder,
    FAISSVectorStore,
    OpenAIChat,
    chunk_text,
)


class PDFContentExtractor:
    """
    Extracts both text and images from PDF files using PyMuPDF.

    Args:
        image_output_dir:  Directory to save extracted images (optional).
        min_image_size:    Minimum pixel dimension to keep an image.
                           Filters out tiny icons/logos.
    """

    def __init__(
        self,
        image_output_dir: Optional[str] = None,
        min_image_size: int = 100,
    ):
        self.image_output_dir = image_output_dir
        self.min_image_size = min_image_size

        if image_output_dir and not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir, exist_ok=True)

    def extract(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract all text and images from a PDF.

        Args:
            file_path:  Path to PDF file.

        Returns:
            Tuple of (text_pages, images).
            text_pages: List of {"text": str, "page": int}
            images: List of {"image_bytes": bytes, "ext": str, "page": int,
                             "index": int, "path": str|None}
        """
        text_pages = []
        images = []

        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # Extract text
                text = page.get_text().strip()
                if text:
                    text_pages.append({
                        "text": text,
                        "page": page_num + 1,
                    })

                # Extract images
                page_images = page.get_images(full=True)
                for img_idx, img in enumerate(page_images):
                    xref = img[0]
                    try:
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Filter tiny images
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        w, h = pil_image.size
                        if w < self.min_image_size or h < self.min_image_size:
                            continue

                        # Save to disk if output dir specified
                        save_path = None
                        if self.image_output_dir:
                            filename = f"image_p{page_num+1}_{img_idx+1}.{image_ext}"
                            save_path = os.path.join(self.image_output_dir, filename)
                            pil_image.save(save_path)

                        images.append({
                            "image_bytes": image_bytes,
                            "ext": image_ext,
                            "page": page_num + 1,
                            "index": img_idx + 1,
                            "width": w,
                            "height": h,
                            "path": save_path,
                        })
                    except Exception as e:
                        print(f"  Warning: Failed to extract image on page {page_num+1}: {e}")
                        continue

        return text_pages, images




class ImageCaptioner:
    """
    Generates text descriptions of images using OpenAI's vision model.

    Args:
        model_name:   OpenAI vision model (must support image input).
        temperature:  Low for consistent captions.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def caption_image(self, image_bytes: bytes, image_ext: str = "png") -> str:
        """
        Generate a retrieval-optimized caption for an image.

        Args:
            image_bytes:  Raw image bytes.
            image_ext:    Image format extension (png, jpg, etc).

        Returns:
            Caption string describing the image content.
        """
        # Convert to base64 for OpenAI API
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Map extensions to MIME types
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(image_ext.lower(), "image/png")

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that describes images, tables, and figures "
                        "for a document retrieval system. Your descriptions will be embedded "
                        "and used to find relevant content. Give a concise but thorough "
                        "description that captures all data, numbers, labels, and key "
                        "information visible in the image. Optimize for retrieval."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image/table/figure for retrieval purposes:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )

        return response.choices[0].message.content.strip()

    def caption_images(self, images: List[Dict]) -> List[Dict]:
        """
        Caption multiple images, returning augmented dicts with captions.

        Args:
            images:  List of image dicts from PDFContentExtractor.

        Returns:
            Same list with "caption" key added to each dict.
        """
        for i, img in enumerate(images):
            print(f"    Captioning image {i+1}/{len(images)} (page {img['page']})...")
            caption = self.caption_image(img["image_bytes"], img["ext"])
            img["caption"] = caption

        return images


class MultiModalRetriever:
    """
    Retriever that indexes both text chunks and image captions in one store.

    Pipeline:
        1. Extract text + images from PDF
        2. Caption each image with vision LLM
        3. Chunk text pages into small pieces
        4. Create Documents for text chunks AND image captions
        5. Embed everything → single FAISS index
        6. On query: search finds either text or caption matches

    Replaces:
        - Notebook's Chroma vectorstore with FAISS
        - Cohere embeddings with OpenAI embeddings
        - Gemini captioning with OpenAI GPT-4o vision
        - LangChain's RecursiveCharacterTextSplitter with custom chunk_text()

    Args:
        embedding_model:   OpenAI embedding model.
        caption_model:     OpenAI vision model for image captioning.
        chunk_size:        Characters per text chunk.
        chunk_overlap:     Overlap between text chunks.
        k:                 Number of results from vector search.
        min_image_size:    Minimum pixel size to keep an image.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        caption_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 3,
        min_image_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        # Core components
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.extractor = PDFContentExtractor(min_image_size=min_image_size)
        self.captioner = ImageCaptioner(model_name=caption_model)

        # Stats
        self.stats = {
            "text_pages": 0,
            "text_chunks": 0,
            "images_found": 0,
            "images_captioned": 0,
            "total_indexed": 0,
        }

    def index_pdf(self, file_path: str) -> int:
        """
        Extract, caption, chunk, and index a PDF.

        Full pipeline:
            1. Extract text pages + images from PDF
            2. Caption each image with vision model
            3. Chunk text pages
            4. Create Documents (text chunks + image captions)
            5. Embed and store in FAISS

        Args:
            file_path:  Path to PDF file.

        Returns:
            Total number of items indexed.
        """
        basename = os.path.basename(file_path)

        # Step 1: Extract text and images
        print(f"  [MultiModal] Extracting content from '{basename}'...")
        text_pages, images = self.extractor.extract(file_path)
        self.stats["text_pages"] = len(text_pages)
        self.stats["images_found"] = len(images)
        print(f"  [MultiModal] Found {len(text_pages)} text pages, {len(images)} images")

        # Step 2: Caption images
        if images:
            print(f"  [MultiModal] Captioning {len(images)} images...")
            images = self.captioner.caption_images(images)
            self.stats["images_captioned"] = len(images)

        # Step 3: Chunk text pages
        all_docs: List[Document] = []

        for page_data in text_pages:
            chunks = chunk_text(
                page_data["text"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            for j, chunk_content in enumerate(chunks):
                all_docs.append(
                    Document(
                        content=chunk_content,
                        metadata={
                            "source": file_path,
                            "page": page_data["page"],
                            "type": "text",
                            "chunk_index": j,
                        },
                    )
                )

        self.stats["text_chunks"] = len(all_docs)

        # Step 4: Create Documents for image captions
        for img in images:
            caption = img.get("caption", "")
            if caption:
                all_docs.append(
                    Document(
                        content=caption,
                        metadata={
                            "source": file_path,
                            "page": img["page"],
                            "type": "image_caption",
                            "image_index": img["index"],
                            "image_path": img.get("path"),
                            "image_size": f"{img['width']}x{img['height']}",
                        },
                    )
                )

        # Step 5: Embed and index
        self.stats["total_indexed"] = len(all_docs)
        print(
            f"  [MultiModal] Indexing {len(all_docs)} items "
            f"({self.stats['text_chunks']} text + {self.stats['images_captioned']} captions)"
        )

        all_docs = self.embedder.embed_documents(all_docs)
        self.vector_store.add_documents(all_docs)

        return len(all_docs)

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Search the unified index (text + image captions)."""
        query_emb = self.embedder.embed_text(query)
        return self.vector_store.search(query_emb, k=self.k)

    def retrieve_context(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve context with metadata about what type matched.

        Returns:
            Tuple of (context_texts, match_info).
        """
        results = self.retrieve(query)
        contexts = []
        match_info = []

        for r in results:
            contexts.append(r.document.content)
            match_info.append({
                "type": r.document.metadata.get("type", "unknown"),
                "page": r.document.metadata.get("page"),
                "score": r.score,
                "image_path": r.document.metadata.get("image_path"),
            })

        return contexts, match_info

class MultiModalRAG:
    """
    Complete multi-modal RAG pipeline: text + image captioning + retrieval.

    Replaces the notebook's:
        - Gemini 1.5-flash → OpenAI GPT-4o (captioning)
        - Cohere embeddings → OpenAI text-embedding-3-small
        - Chroma vectorstore → FAISS
        - LangChain ChatPromptTemplate + ChatCohere + StrOutputParser → OpenAIChat
        - LangChain RecursiveCharacterTextSplitter → custom chunk_text()

    Usage:
        rag = MultiModalRAG(file_path="paper.pdf")
        answer, contexts = rag.query("What is the BLEU score of the base model?")
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 3,
        min_image_size: int = 100,
        embedding_model: str = "text-embedding-3-small",
        caption_model: str = "gpt-4o-mini",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the Multi-Modal RAG pipeline.

        Args:
            file_path:        Path to PDF file.
            chunk_size:        Characters per text chunk.
            chunk_overlap:     Overlap between chunks.
            k:                 Number of retrieval results.
            min_image_size:    Min pixel size to keep an image.
            embedding_model:   OpenAI embedding model.
            caption_model:     OpenAI vision model for captioning.
            chat_model:        OpenAI model for answer generation.
            temperature:       LLM temperature.
        """
        self.file_path = file_path

        # Initialize retriever
        self.retriever = MultiModalRetriever(
            embedding_model=embedding_model,
            caption_model=caption_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
            min_image_size=min_image_size,
        )

        # Initialize chat model
        self.chat = OpenAIChat(
            model_name=chat_model,
            temperature=temperature,
        )

        # Index
        print(
            f"[MultiModal] Processing '{os.path.basename(file_path)}' "
            f"(text + images)...\n"
        )
        total = self.retriever.index_pdf(file_path)

        stats = self.retriever.stats
        print(
            f"\n[MultiModal] Done → {total} items indexed "
            f"({stats['text_chunks']} text chunks, "
            f"{stats['images_captioned']} image captions)"
        )

    def query(
        self,
        question: str,
        return_context: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Query the multi-modal RAG system.

        The search can match against EITHER text chunks or image captions.
        Both are used as context for the answer LLM.

        Args:
            question:        User's question.
            return_context:  Whether to return retrieved contexts.

        Returns:
            Tuple of (answer_string, list_of_context_strings).
        """
        contexts, match_info = self.retriever.retrieve_context(question)

        if not contexts:
            return "No relevant information found in the document.", []

        answer = self.chat.chat_with_context(question, contexts)

        if return_context:
            return answer, contexts
        return answer, []

    def show_matches(self, question: str) -> None:
        """
        Debug helper: show what type of content matched (text vs image).

        This reveals whether the answer came from text or an image caption,
        which is the key insight of multi-modal RAG.

        Args:
            question:  Search query.
        """
        contexts, match_info = self.retriever.retrieve_context(question)

        print(f"\nQuery: {question}")
        print("=" * 70)

        for i, (ctx, info) in enumerate(zip(contexts, match_info)):
            content_type = info["type"]
            emoji = "🖼️" if content_type == "image_caption" else "📄"

            print(f"\n  {emoji} Match {i+1} ({content_type}):")
            print(f"    Page:  {info['page']}")
            print(f"    Score: {info['score']:.4f}")

            if info.get("image_path"):
                print(f"    Image: {info['image_path']}")

            print(f"    Content: {ctx[:250]}...")
            print(f"  {'-' * 50}")

        print("\n" + "=" * 70)



if __name__ == "__main__":

    pdf_path = r"data\TransformerClimateChange.pdf"

    rag = MultiModalRAG(
        file_path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        k=3,
        min_image_size=100,
    )

    # Show what matched — text or image?
    # rag.show_matches("What is the BLEU score of the Transformer base model?")
    # rag.show_matches("What does the Transformer architecture look like?")

    # Interactive loop
    print("\n[Multi-Modal RAG] Ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() == "exit":
            break

        answer, context = rag.query(question)
        print(f"\nAnswer: {answer}")
        print(f"Used contexts: {context}")
        print(f"Contexts used: {len(context)}")
        print()
