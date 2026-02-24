# Multi-Modal RAG: Answering Questions Hidden in Figures and Tables

---

## Introduction

Standard RAG is deaf to visuals. A research paper with BLEU score results only in a table image, an engineering manual with circuit diagrams, a financial report with charts showing quarterly trends — all of these have critical information locked in images that text-only RAG completely misses.

**Multi-Modal RAG** unlocks this by treating image captions as first-class retrieval units. Using a vision LLM (GPT-4o) to generate rich, retrieval-optimized captions for every image in a PDF, it places those captions into the same FAISS vector store alongside text chunks. Queries then search across both text and image descriptions — finding the right figure, table, or diagram just as easily as they find the right paragraph.

---

## The Information Types in a PDF

Consider "Attention is All You Need" (the Transformer paper):

| Location | Content | Accessible to text-only RAG? |
|----------|---------|------------------------------|
| Abstract, body text | Architecture description | ✅ Yes |
| Figure 1 | Transformer model architecture diagram | ❌ No |
| Table 2 | BLEU scores by model variant | ❌ No |
| Figure 3 | Attention visualization heatmaps | ❌ No |
| References | Citation list | ✅ Yes |

Text-only RAG can answer "What is the Transformer architecture?" using the text, but cannot answer "What does the Transformer architecture look like?" (figure) or "What is the BLEU score of the base model?" (table). These queries require image understanding.

---

## The Pipeline

```
PDF
 │
 ├─[Text Extraction]─────► Text pages → chunked → Documents (type: "text")
 │                                                           │
 │                                                           ▼
 └─[Image Extraction]─────► Images → [GPT-4o Vision] → Captions → Documents (type: "image_caption")
                                                                    │
                                                                    ▼
                                                     ┌──────────────────────────┐
                                                     │  FAISS Index            │
                                                     │  (text + captions mixed) │
                                                     └──────────────────────────┘
                                                                    │
                                                     User Query → FAISS Search
                                                                    │
                                                     Matched chunks (text or caption)
                                                                    │
                                                          GPT-4o generates answer
```

---

## Step 1: PDF Content Extraction

```python
class PDFContentExtractor:
    def extract(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract ALL content from a PDF: text and images.
        
        Returns:
            text_pages: [{"text": str, "page": int}, ...]
            images:     [{"image_bytes": bytes, "ext": str, "page": int,
                          "width": int, "height": int, ...}, ...]
        """
        text_pages = []
        images = []
        
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # Extract text
                text = page.get_text().strip()
                if text:
                    text_pages.append({"text": text, "page": page_num + 1})
                
                # Extract images
                for img_xref, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    base_image = pdf.extract_image(xref)
                    
                    # Filter small images (icons, decorations)
                    pil_img = Image.open(io.BytesIO(base_image["image"]))
                    w, h = pil_img.size
                    if w < self.min_image_size or h < self.min_image_size:
                        continue  # Skip tiny decorative images
                    
                    images.append({
                        "image_bytes": base_image["image"],
                        "ext": base_image["ext"],
                        "page": page_num + 1,
                        "width": w, "height": h
                    })
        
        return text_pages, images
```

**Why filter small images?** PDFs are full of decorative elements: logos, bullet point icons, horizontal rules. At `min_image_size=100` pixels, these are excluded — only substantive figures, tables, and diagrams are captioned. This prevents noise in the index from irrelevant image "captions."

---

## Step 2: Vision LLM Captioning

```python
class ImageCaptioner:
    def caption_image(self, image_bytes: bytes, image_ext: str) -> str:
        """
        Generate a retrieval-optimized caption using GPT-4o vision.
        
        Prompt design is critical here. Unlike a descriptive caption for a human
        ("A bar chart with blue bars"), a retrieval-optimized caption should
        contain all the specific data, numbers, and labels that might appear in
        a user's query.
        
        Compare:
        
        Descriptive caption (BAD for retrieval):
            "A table with several rows and columns containing performance metrics."
        
        Retrieval-optimized caption (GOOD):
            "Table 2: BLEU scores for EN-DE and EN-FR translation tasks.
             Transformer base model EN-DE: 27.3 BLEU, EN-FR: 38.1 BLEU.
             Transformer big model EN-DE: 28.4 BLEU, EN-FR: 41.0 BLEU.
             Comparison includes 6 prior methods including ByteNet and ConvS2S."
        """
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = {"png": "image/png", "jpg": "image/jpeg"}.get(image_ext.lower(), "image/png")
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
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
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image/table/figure for retrieval purposes:"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}",
                            "detail": "high"
                        }}
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
```

**"detail": "high"** tells GPT-4o to use its higher-resolution vision mode — critical for tables with small text, dense charts, or diagrams with labels.

### Caption Quality Examples

| Image Type | Bad Caption | Good (Retrieval-Optimized) Caption |
|-----------|------------|-----------------------------------|
| Bar chart | "A bar chart showing data" | "Figure 3: Average response time (ms) by training set size. GPT-4: 245ms at 1K, 312ms at 10K. Claude: 198ms at 1K, 267ms at 10K. Bars are color-coded blue (GPT-4) and orange (Claude)." |
| Architecture diagram | "A flowchart with boxes and arrows" | "Figure 1: Transformer encoder-decoder architecture. Left: encoder with multi-head attention + feed-forward layers. Right: decoder with masked multi-head attention, cross-attention, and feed-forward. N=6 stacked layers. Input/output embeddings with positional encoding." |
| Table | "A table with numbers" | "Table 1: Model performance metrics. BERT-base: F1=88.3, accuracy=84.6. BERT-large: F1=91.2, accuracy=87.4. RoBERTa: F1=93.1, accuracy=90.2." |

---

## Step 3: Unified Indexing

Both text chunks and image captions are wrapped in `Document` objects with a `type` metadata field:

```python
# Text chunk document
Document(
    content="The Transformer architecture relies on attention mechanisms...",
    metadata={"page": 3, "type": "text", "chunk_index": 12}
)

# Image caption document  
Document(
    content="Table 2: BLEU scores for EN-DE translation: base=27.3, big=28.4...",
    metadata={"page": 8, "type": "image_caption", "image_index": 2,
               "image_path": "extracted_images/image_p8_2.png"}
)
```

Both are embedded with `text-embedding-3-small` and stored in the same FAISS index. At query time, FAISS has no concept of "text vs. caption" — it simply returns the most semantically similar content, regardless of modality origin.

---

## Step 4: Query and Match Type Display

```python
def show_matches(self, question: str) -> None:
    """
    Display what type of content matched — text or image caption.
    Useful for understanding how multi-modal retrieval works.
    """
    contexts, match_info = self.retriever.retrieve_context(question)
    
    for i, (ctx, info) in enumerate(zip(contexts, match_info)):
        emoji = "🖼️" if info["type"] == "image_caption" else "📄"
        print(f"{emoji} Match {i+1}: {info['type']} — Page {info['page']} — Score {info['score']:.4f}")
        print(f"   {ctx[:250]}...")
```

**Example output:**

```
Query: "What is the BLEU score of the base model on EN-DE translation?"

🖼️ Match 1: image_caption — Page 8 — Score 0.894
   Table 2: BLEU scores comparison. Transformer base (EN-DE): 27.3 BLEU...

📄 Match 2: text — Page 7 — Score 0.731
   "We trained the base model for 100,000 steps with a batch size..."

🖼️ Match 3: image_caption — Page 9 — Score 0.689
   Figure 4: Training curves showing BLEU score progression over training steps...
```

Match 1 is an image caption — the answer is in a table that text-only RAG would have missed entirely.

---

## Cost and Indexing Time Analysis

For a 100-page PDF with 50 images:

| Phase | Operations | Time estimate |
|-------|-----------|--------------|
| Text extraction | PyMuPDF (fast) | ~2 seconds |
| Image extraction | PyMuPDF (fast) | ~3 seconds |
| Image captioning | 50 × GPT-4o calls | ~2-3 minutes (main bottleneck) |
| Text chunking | ~200 chunks | ~1 second |
| Embedding all (200 text + 50 captions) | OpenAI batch | ~10 seconds |
| FAISS indexing | 250 vectors | ~0.1 seconds |
| **Total** | | **~3-4 minutes** |

Image captioning is the bottleneck. For large documents, parallelize caption calls:

```python
import concurrent.futures

def caption_images_parallel(self, images: List[Dict], max_workers: int = 5) -> List[Dict]:
    """Caption multiple images in parallel using a thread pool."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.caption_image, img["image_bytes"], img["ext"]): i
            for i, img in enumerate(images)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            images[idx]["caption"] = future.result()
    return images
```

**Cost**: With `gpt-4o-mini` at `detail: high`, each image caption costs ~500-1000 input tokens + ~200 output tokens. For 50 images ≈ $0.05-0.10 in API costs.

---

## When to Use Multi-Modal RAG

Multi-Modal RAG is essential when critical information is locked in images rather than text. Scientific papers with results tables and methodology figures, technical documentation with architecture diagrams, financial reports where key metrics appear in charts, medical records with scan descriptions, engineering manuals with circuit drawings — documents like these contain content that text-only RAG simply cannot retrieve, regardless of how well the embedding model and retrieval pipeline are tuned. If users routinely ask questions like "What does the training loss curve look like?" or "What were the top benchmark results?" and those answers live in figures, text-only retrieval will fail by design.

For corpora where documents are entirely text-based — legal contracts, policy documents, general articles — the vision captioning pipeline adds cost without benefit. Likewise, if images are purely decorative (logos, stock photography, layout elements) or if queried information never maps to visual content, the overhead of extraction and captioning is not justified. The technique's value is proportional to how often answers to real user queries require reading a visual element.

---

## Summary

Multi-Modal RAG dismantles the assumption that all information in a document is in its text. By extracting images, captioning them with a vision LLM using retrieval-optimized prompts, and embedding those captions alongside text chunks in a unified FAISS index, it makes the full document searchable — including every figure, table, and diagram.

The implementation is elegant: the retrieval system doesn't need to "know" about modalities. Text and captions co-exist in the same embedding space, and queries find whichever form of content is most semantically relevant. For any application working with documents that contain significant visual content — research papers, technical manuals, financial reports — multi-modal RAG is not an optimization; it's a correctness requirement.
