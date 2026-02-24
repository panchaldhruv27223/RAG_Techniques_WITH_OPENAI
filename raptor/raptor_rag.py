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


from sklearn.mixture import GaussianMixture


@dataclass
class TreeNode:
    """A Single node in the RAPTOR tree.""" 
    text:str
    level:int
    embedding:Optional[List[float]] = None 
    clusted_id: Optional[int] = None
    origin: str = "Original"
    children: List[int] = field(default_factory=list)

@dataclass
class RAPTORResponse:
    """Complete RAPTOR response with metadata. """
    answer: str
    context_used: List[str]
    retrieved_levels: List[int]
    num_docs_retrieved: int
    tree_depth: int


class RAPTOROpenAI:
    """
    RAPTOR: Hierarchical Summary Tree + Multi-Level Retrieval.
    
    Builds a tree of summaries offline, then searches across all levels at query time.
    
    Args:
        file_path:      Path to PDF or text file to index
        chunk_size:     Characters per chunk for Level 0
        chunk_overlap:  Overlap between chunks
        max_levels:     Maximum tree depth (default 3)
        n_clusters:     Max clusters per level (actual = min(n_clusters, len(texts)//2))
        top_k:          Number of docs to retrieve per query
        chat_model:     OpenAI model for summarization + generation
        temperature:    LLM temperature
    
    """

    def __init__(
        self,
        file_path:str,
        chunk_size:int = 1000,
        chunk_overlap:int = 200,
        max_levels:int = 3,
        n_clusters:int = 10,
        top_k:int =3,
        chat_model:str = "gpt-4o-mini",
        temperature:float = 0.0
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_levels = max_levels
        self.n_clusters = n_clusters
        self.top_k = top_k

        self.embedder = OpenAIEmbedder(model="text-embedding-ada-002")

        self.llm = OpenAIChat(model_name=chat_model, temperature=temperature, max_tokens=5000)



        ## Tree Storage 
        self.tree_nodes: List[TreeNode] = []
        self.tree_depth: int = 0
        self.vector_store: Optional[FAISSVectorStore] = None


        self._build_tree()

    def _build_tree(self):
        """Build the full RAPTOR tree from document → summaries → root."""
        
        print(f"\nBuilding RAPTOR Tree")
        raw_text = read_pdf(self.file_path)
        chunks = chunk_text(raw_text, self.chunk_size, self.chunk_overlap)
        current_texts = chunks
        current_node_indices = []

        for i, text in enumerate(current_texts):

            node = TreeNode(
                text=text,
                level=0,
                origin="original",
            )

            self.tree_nodes.append(node)
            current_node_indices.append(len(self.tree_nodes) - 1)

        
        for level in range(1, self.max_levels + 1):

            embeddings = self.embedder.embed_texts(current_texts)

            for idx, embed in enumerate(embeddings):
                self.tree_nodes[idx].embedding = embed

            actual_clusters = min(self.n_clusters, max(2, len(current_texts) // 2))

            if len(current_texts) <= 2:
                ### Too Few texts to cluster - Just summarize everything
                print(f"   Only {len(current_texts)} texts - Summarizing everything")
                
                summary = self._summarize_texts(current_texts)
                summary_embed = self.embedder.embed_text(summary)

                node = TreeNode(
                    text=summary,
                    level=level,
                    origin=f"root_summary_level_{level}",
                    children=current_node_indices,
                )

                self.tree_nodes.append(node)
                self.tree_depth = level

                print(f"   Tree Completed at depth {level}")
                
                break


            print("Now lets use GMM to cluster the embeddings")

            embed_array = np.array(embeddings)

            gmm = GaussianMixture(
                n_components=actual_clusters,
                random_state=42,
                covariance_type="full"
            )

            cluster_labels = gmm.fit_predict(embed_array)

            for idx, label in zip(current_node_indices, cluster_labels):
                self.tree_nodes[idx].clusted_id = int(label)
            

            ### For next level now we need to create summaries of each cluster
            
            ## so lets summarize each cluster
            
            new_texts = []
            new_node_indices = []

            for cluster_id in range(actual_clusters):
                cluster_mask = cluster_labels == cluster_id

                cluster_texts = [current_texts[i] for i in range(len(current_texts)) if cluster_mask[i]]

                cluster_indices = [current_node_indices[i] for i in range(len(current_node_indices)) if cluster_mask[i]]

                if not cluster_texts:
                    continue

                summary = self._summarize_texts(cluster_texts)
                
                node = TreeNode(
                    text=summary,
                    level=level,
                    origin=f"summary_cluster_{cluster_id}_level_{level}",
                    children=cluster_indices,
                )


                self.tree_nodes.append(node)

                node_idx = len(self.tree_nodes) - 1

                new_texts.append(summary)
                new_node_indices.append(node_idx)

            print(f"   Created {len(new_texts)} summaries at level {level}")

            current_texts = new_texts

            current_node_indices = new_node_indices
            
            self.tree_depth = level

            if len(current_texts) <= 1:

                if current_texts:

                    final_embed = self.embedder.embed_text(current_texts)

                    for idx, emb in zip(current_node_indices, final_embed):
                        self.tree_nodes[idx].embedding = emb

                print(f"  Tree completed at depth {level}")
                break 

        self._build_vector_store()
        
        level_counts = {}

        for node in self.tree_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        print(f"  Total Nodes: {len(self.tree_nodes)}")
        print(f"  Tree Depth: {self.tree_depth}")

        for level in sorted(level_counts.keys()):
            print(f"   level {level}: {level_counts[level]} nodes")



    def _summarize_texts(self, texts: List[str]) -> str:
        """Summarize a list of texts into a single concise summary."""

        combined = "\n---\n".join(texts)

        # Truncate if too long (avoid token limit issues)
        if len(combined) > 12000:
            combined = combined[:12000] + "\n\n[truncated]"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarization expert. Create a concise but comprehensive "
                    "summary that captures the key information, main themes, and important "
                    "details from the provided texts. The summary should be self-contained "
                    "and informative enough to answer questions about the content."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize the following texts:\n\n{combined}",
            },
        ]
        return self.llm.chat(messages)

    def _build_vector_store(self):
        """Build FAISS vector store from all tree nodes."""

        print(f" [Vector Store] indexing {len(self.tree_nodes)} nodes..../|")


        nodes_without_emb = [n for n in self.tree_nodes if n.embedding is None]


        if nodes_without_emb:

            texts = [n.text for n in nodes_without_emb]

            embeddings = self.embedder.embed_texts(texts)
            
            for node, emb in zip(nodes_without_emb, embeddings):
                node.embedding = emb

        ## Create Document object for the vector store

        documents = []

        for i, node in enumerate(self.tree_nodes):

            doc = Document(
                content=node.text,
                metadata={
                    "level": node.level,
                    "origin": node.origin,
                    "node_index": i,
                    "has_children": len(node.children) > 0,
                },
                embedding= node.embedding,
            )

            documents.append(doc)



        self.vector_store = FAISSVectorStore(dimension= self.embedder.dimension)

        self.vector_store.add_documents(documents=documents)

        print(f"  [Vector Store] Vector store is ready to use ({len(documents)} documents)")


    # Query 

    def _retrieve(self, query:str, k:Optional[int]=None) -> List[RetrievalResult]:

        """
        Collapsed tree retrieval - search across all levels at once.
        Broad questions naturally match higher-level summaries.
        Specific questions match Level 0 chunks.
        """

        k = k or self.top_k
        query_embedding = self.embedder.embed_text(query)
        return self.vector_store.search(query_embeddings=query_embedding, k=k)


    def _compress_context(self, query: str, contexts: List[str]) -> str:
        """
        Contextual compression — extract only relevant parts from retrieved docs.
        This is equivalent to LangChain's LLMChainExtractor.
        """
        combined = "\n\n---\n\n".join(contexts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an information extractor. Given a question and multiple "
                    "context passages, extract ONLY the parts that are directly relevant "
                    "to answering the question. Remove irrelevant information. "
                    "Preserve important details and facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context passages:\n{combined}\n\n"
                    f"Relevant information:"
                ),
            },
        ]
        return self.llm.chat(messages)


    


    def _generate_answer(self, query:str, context:str) -> str:
        """ Generate the final answer from compressed context."""

        messages = [

            {
                "role":"system",
                "content": (
                    "You are a helpful assistant. Answer the question based on the provided context."
                    "Be consice and accurate. If the context doesn't contain enough information, say so."
                )
            },
            {
                "role":"user",
                "content":f"Context: \n{context} \n\n Question: {query}\n\nAnswer:"
            }
        ]

        return self.llm.chat(messages=messages)


    def query(self, question:str)->str:
        """
        Standard interface: returns (answer, context_list).
        Compatible with evaluation framework.
        """
        
        result = self.raptor_query(question)
        
        return result.answer, result.context_used
    
    def raptor_query(self, question:str, compress:bool=True) -> RAPTORResponse:
        """
        Full RAPTOR query pipeline.
        
        Args:
            query:    User's question
            compress: Whether to apply contextual compression (default True)
            
        Returns:
            RAPTORResponse with answer, context, and tree metadata
        """

        retrieved_docs = self._retrieve(question)

        contexts = []
        levels_retrieved = []

        for r in retrieved_docs:

            contexts.append(r.document.content)
            level = r.document.metadata.get("level", 0)
            levels_retrieved.append(level)

        # print(f"find contexts: {contexts}")

        
        if compress and contexts:
            compressed_context = self._compress_context(question, contexts)

        else:
            compressed_context = "\n\n---\n\n".join(contexts)

        answer = self._generate_answer(question, compressed_context)

        unique_levels = sorted(set(levels_retrieved))

        print(f"   LEVELS Used: {unique_levels}")
        print(f"   CONTEXT COUNT: {len(contexts)}")
        

        return RAPTORResponse(
            answer=answer,
            context_used=contexts,
            retrieved_levels=unique_levels,
            num_docs_retrieved=len(retrieved_docs),
            tree_depth=self.tree_depth,
        )

    def get_tree_stats(self):
        """Get statistics about the built tree."""
        level_counts = {}

        for node in self.tree_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        return {
            "total_nodes": len(self.tree_nodes),
            "tree_depth": self.tree_depth,
            "level_counts": level_counts,
            "vector_store_size": len(self.vector_store.documents) if self.vector_store else 0,
        }



if __name__ == "__main__":
    pdf_file_path = r"data\Understanding_Climate_Change.pdf"

    raptor_rag = RAPTOROpenAI(
        file_path=pdf_file_path,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        max_levels=10,
        n_clusters=25,
        chat_model="gpt-4o-mini"
    )

    stats = raptor_rag.get_tree_stats()

    answer, context = raptor_rag.query("What are the main causes of climate change?")

    print(f"\nAnswer: {answer}")
    print(f"\nContext: {context}")
    print(f"\nStats: {stats}")