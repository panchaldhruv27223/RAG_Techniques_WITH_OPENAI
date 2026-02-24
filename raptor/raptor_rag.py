"""
RAPTOR: Recursive Abstractive Processing and Thematic Organization for Retrieval

RAPTOR builds a hierarchical tree of document summaries BEFORE any queries.
At query time, it searches across ALL levels — specific questions match leaf
chunks while broad questions match higher-level summaries.


Offline (Tree Building):
    1. Chunk the document → Level 0 (original chunks)
    2. Embed all chunks
    3. Cluster embeddings using Gaussian Mixture Models (GMM)
    4. Summarize each cluster via LLM → Level 1 (cluster summaries)
    5. Repeat: embed Level 1 → cluster → summarize → Level 2
    6. Continue until single root summary or max_levels reached
    7. Index ALL nodes (all levels) into one FAISS vector store

Online (Query):
    1. Embed query
    2. Search across ALL levels in the vector store
    3. Optionally compress retrieved docs (extract relevant parts)
    4. Generate answer from retrieved context


"""

import os 
import sys 
import logging 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional, Tuple, Any 
from dataclasses import dataclass, field

from helper_function_openai import (
    OpenAIEmbedder,
    OpenAIChat,
    FAISSVectorStore,
    Document,
    RetrievalResult,
    read_pdf,
    chunk_text,
)


