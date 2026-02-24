# RAG Techniques with OpenAI

A comprehensive collection of **Retrieval-Augmented Generation (RAG)** techniques implemented with the OpenAI API. Each technique is accompanied by a detailed blog post explaining the motivation, architecture, and implementation.

---

## 📖 Blog Index

| # | Technique | Complexity | Key Concept | Blog |
|---|-----------|------------|-------------|------|
| 01 | Simple RAG | 🟢 Beginner | The fundamental index → retrieve → generate pipeline | [Read →](blogs/01_simple_rag.md) |
| 02 | Proposition Chunking | 🟡 Intermediate | Index atomic facts instead of text windows | [Read →](blogs/02_proposition_chunking.md) |
| 03 | RSE (Relevant Segment Extraction) | 🔴 Advanced | Max-sum-subarray retrieves contiguous relevant segments | [Read →](blogs/03_rse.md) |
| 04 | Context Chunk Headers | 🟢 Beginner | Prepend document title + summary to every chunk | [Read →](blogs/04_context_chunk_header.md) |
| 05 | Context Enrichment Window | 🟢 Beginner | Expand retrieved chunks with their neighbors | [Read →](blogs/05_context_enrichment_window.md) |
| 06 | Semantic Chunking | 🟡 Intermediate | Split text at topic boundaries using embedding similarity | [Read →](blogs/06_semantic_chunking.md) |
| 07 | Contextual Compression | 🟡 Intermediate | Extract only query-relevant excerpts from each chunk | [Read →](blogs/07_contextual_compression.md) |
| 08 | Document Augmentation | 🟡 Intermediate | Index chunks alongside LLM-generated questions they answer | [Read →](blogs/08_document_augmentation.md) |
| 09 | Fusion Retrieval | 🟡 Intermediate | Combine BM25 keyword search with dense vector search | [Read →](blogs/09_fusion_retrieval.md) |
| 10 | Reranking | 🟡 Intermediate | Two-stage retrieval: fast vector search + precise cross-encoder rerank | [Read →](blogs/10_reranking.md) |
| 11 | Hierarchical Indices | 🟡 Intermediate | Search page summaries first, then drill into detail chunks | [Read →](blogs/11_hierarchy_indices.md) |
| 12 | HyDE | 🟡 Intermediate | Generate a hypothetical answer and retrieve by its embedding | [Read →](blogs/12_hyde_rag.md) |
| 13 | HyPE | 🟡 Intermediate | Index question embeddings per chunk at build time | [Read →](blogs/13_hype_rag.md) |
| 14 | CRAG | 🔴 Advanced | Evaluate retrieved docs and fall back to web search if needed | [Read →](blogs/14_crag.md) |
| 15 | Self-RAG | 🔴 Advanced | Decide retrieval need, evaluate relevance, and self-critique responses | [Read →](blogs/15_self_rag.md) |
| 16 | Adaptive Retrieval | 🔴 Advanced | Route each query to a strategy tailored to its type | [Read →](blogs/16_adaptive_retrieval.md) |
| 17 | Feedback Loop RAG | 🔴 Advanced | Self-improving system that learns from user ratings over time | [Read →](blogs/17_feedback_loop_rag.md) |
| 18 | Reliable RAG | 🟡 Intermediate | Detect hallucinations by verifying each answer claim against context | [Read →](blogs/18_reliable_rag.md) |
| 19 | Explainable Retrieval | 🟡 Intermediate | Generate natural-language explanations for every retrieved chunk | [Read →](blogs/19_explainable_retrieval.md) |
| 20 | Query Transformations | 🟡 Intermediate | Rewrite, step-back, or decompose queries before retrieval | [Read →](blogs/20_query_transformations.md) |
| 21 | Dartboard RAG | 🔴 Advanced | Balance relevance and diversity via log-normal greedy selection | [Read →](blogs/21_dartboard_rag.md) |
| 22 | Multi-Modal RAG | 🔴 Advanced | Caption images with GPT-4o and index captions alongside text | [Read →](blogs/22_multimodal_rag.md) |
| 23 | RAPTOR | 🔴 Advanced | Recursive clustering + summarization for multi-resolution retrieval | [Read →](blogs/23_raptor_rag.md) |
| 24 | GraphRAG (Microsoft) | 🔴 Advanced | Knowledge graphs + community summaries for relational queries | [Read →](blogs/24_graph_rag.md) |
| 25 | Intelligent RAG | 🔴 Advanced | Budget-controlled orchestrator: temporal, multi-hop, decomposition | [Read →](blogs/25_intelligent_rag.md) |

---

## 🗂️ Repository Structure

```
RAG/
├── blogs/                          # 📝 Detailed blog posts (25 techniques)
│   ├── 01_simple_rag.md
│   ├── 02_proposition_chunking.md
│   ├── 03_rse.md
│   ├── 04_context_chunk_header.md
│   ├── 05_context_enrichment_window.md
│   ├── 06_semantic_chunking.md
│   ├── 07_contextual_compression.md
│   ├── 08_document_augmentation.md
│   ├── 09_fusion_retrieval.md
│   ├── 10_reranking.md
│   ├── 11_hierarchy_indices.md
│   ├── 12_hyde_rag.md
│   ├── 13_hype_rag.md
│   ├── 14_crag.md
│   ├── 15_self_rag.md
│   ├── 16_adaptive_retrieval.md
│   ├── 17_feedback_loop_rag.md
│   ├── 18_reliable_rag.md
│   ├── 19_explainable_retrieval.md
│   ├── 20_query_transformations.md
│   ├── 21_dartboard_rag.md
│   ├── 22_multimodal_rag.md
│   ├── 23_raptor_rag.md
│   ├── 24_graph_rag.md
│   └── 25_intelligent_rag.md
│
├── simple_rag/                     # 01 – Simple RAG baseline
├── context_enrichment/             # 02–06 – Chunking & enrichment (incl. RSE)
├── advanced_retrieval/             # 07–12, 21–22 – Fusion, HyDE, Dartboard, Multi-Modal
├── Querys/                         # 20 – Query transformation techniques
├── crag/                           # 14 – Corrective RAG
├── self_rag/                       # 15 – Self-Reflective RAG
├── Iterative_Techniques/           # 16–17 – Adaptive & Feedback Loop RAG
├── reliable_rag/                   # 18 – Reliable RAG
├── Explainbility/                  # 19 – Explainable Retrieval
├── raptor/                         # 23 – RAPTOR
├── microsoft_graph_rag/            # 24 – Microsoft GraphRAG
├── our_own_rag_system/             # 25 – Intelligent RAG orchestrator
├── RAG_CSV/                        # CSV-based RAG
├── data/                           # Sample PDFs and data files
├── evaluation/                     # Evaluation scripts and metrics
├── helper_function_openai.py       # Shared utility classes (embedder, FAISS, chat)
└── evaluate_across_all_rag_techniques.py
```

---

## 🧠 Learning Path

### Foundation (Start Here)
1. **[Simple RAG](blogs/01_simple_rag.md)** — Understand the core pipeline
2. **[Context Chunk Headers](blogs/04_context_chunk_header.md)** — A trivial win, start here
3. **[Context Enrichment Window](blogs/05_context_enrichment_window.md)** — Fix boundary artifacts

### Chunking Improvements
4. **[Semantic Chunking](blogs/06_semantic_chunking.md)** — Let content define its own boundaries
5. **[Proposition Chunking](blogs/02_proposition_chunking.md)** — Index atomic facts
6. **[RSE](blogs/03_rse.md)** — Retrieve coherent segments, not isolated fragments

### Query-Side Enhancements
7. **[Query Transformations](blogs/20_query_transformations.md)** — Fix the query before it hits the index
8. **[HyDE](blogs/12_hyde_rag.md)** — Search with a hypothetical answer
9. **[HyPE](blogs/13_hype_rag.md)** — Pre-generate questions at index time
10. **[Document Augmentation](blogs/08_document_augmentation.md)** — Bridge the vocabulary gap

### Retrieval Improvements
11. **[Fusion Retrieval](blogs/09_fusion_retrieval.md)** — Add BM25 to vector search
12. **[Reranking](blogs/10_reranking.md)** — Two-stage precision pipeline
13. **[Dartboard RAG](blogs/21_dartboard_rag.md)** — Relevance + diversity via log-normal selection
14. **[Hierarchical Indices](blogs/11_hierarchy_indices.md)** — Top-down document navigation
15. **[RAPTOR](blogs/23_raptor_rag.md)** — Multi-resolution tree for complex queries

### Context Refinement
16. **[Contextual Compression](blogs/07_contextual_compression.md)** — Extract only what's relevant

### Advanced & Agentic Techniques
17. **[CRAG](blogs/14_crag.md)** — Validate retrieval, fall back to web
18. **[Self-RAG](blogs/15_self_rag.md)** — Multi-step self-reflection
19. **[Adaptive Retrieval](blogs/16_adaptive_retrieval.md)** — Query-type-driven strategies
20. **[Feedback Loop RAG](blogs/17_feedback_loop_rag.md)** — Learn from users over time
21. **[Intelligent RAG](blogs/25_intelligent_rag.md)** — Budget-controlled full-pipeline orchestrator

### Specialized Capabilities
22. **[Multi-Modal RAG](blogs/22_multimodal_rag.md)** — Search images, tables, and figures
23. **[GraphRAG](blogs/24_graph_rag.md)** — Knowledge graph for relational queries

### Reliability & Transparency
24. **[Reliable RAG](blogs/18_reliable_rag.md)** — Catch hallucinations before they ship
25. **[Explainable Retrieval](blogs/19_explainable_retrieval.md)** — Open the black box

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/panchaldhruv27223/RAG_Techniques_WITH_OPENAI.git
cd RAG_Techniques_WITH_OPENAI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt  # or: uv sync

# Configure API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

---

## 🔑 Key Shared Utilities

All techniques build on the shared abstractions in `helper_function_openai.py`:

| Class | Purpose |
|-------|---------|
| `OpenAIEmbedder` | Wraps OpenAI embedding API with batch support |
| `FAISSVectorStore` | FAISS-powered similarity search with metadata |
| `OpenAIChat` | LLM generation with JSON mode support |
| `RAGRetriever` | Base class for all retrieval implementations |
| `Document` | Unified data model (content + metadata + embedding) |
| `RetrievalResult` | Scored retrieval result |
| `chunk_text` | Sliding-window text chunker |
| `read_pdf` | PDF text extraction via PyMuPDF |

---

## 📊 Evaluation

The repository includes a full evaluation framework:

```bash
# Evaluate all RAG techniques
python evaluate_across_all_rag_techniques.py

# Generate PDF report
python generate_rag_report.py
```

Evaluation metrics:
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: How precisely do retrieved chunks match the query?
- **Context Recall**: Are all relevant chunks retrieved?

---

## 🤝 Contributing

Contributions welcome! If you implement a new RAG technique:
1. Add the implementation in an appropriately named folder
2. Write a blog post in `blogs/` following the existing format
3. Add an entry to the table in this README
4. Update `evaluate_across_all_rag_techniques.py` to include your technique

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## ✍️ Author

**Dhruv Panchal** — [Medium](https://dhruv-panchal.medium.com/)
