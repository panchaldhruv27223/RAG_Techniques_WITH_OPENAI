# Microsoft GraphRAG: Knowledge Graphs Meet Retrieval-Augmented Generation

---

## Introduction

Standard vector-based RAG stores knowledge as independent chunks floating in embedding space. There's no explicit representation of *how concepts relate to each other* — no understanding that "Company A" acquired "Company B" in "Year C" and this acquisition led to "Strategic outcome D." That causal chain exists in the documents but is never captured in the index. Each chunk is an island.

**Microsoft GraphRAG** takes a fundamentally different approach. Instead of embedding text chunks, it:

1. **Extracts entities** (people, places, organizations, events, concepts) from the entire corpus
2. **Extracts relationships** between entities  
3. **Builds a knowledge graph** where entities are nodes and relationships are edges
4. **Generates community summaries** using the Leiden community-detection algorithm
5. **Answers queries** by querying the graph structure, not just chunk embeddings

The result is a system that can answer complex, multi-hop relational queries that are impossible for chunk-based RAG — and that excels at providing structured, sourced answers about how entities relate across an entire corpus.

---

## How GraphRAG Differs from Vector RAG

| Dimension | Vector RAG | GraphRAG |
|-----------|-----------|----------|
| Knowledge representation | Floating text chunks | Knowledge graph (entities + relationships) |
| Retrieval unit | Text chunk | Entity, relationship, or community summary |
| Multi-hop reasoning | Implicit (LLM must infer) | Explicit (graph traversal) |
| Cross-document synthesis | Weak | Strong (graph connects entities across docs) |
| Query types excelled at | Specific facts, semantic search | Relational queries, "who knows who", impact analysis |
| Index construction cost | Fast (embed chunks) | High (LLM-based entity extraction per chunk) |

---

## The Indexing Pipeline

GraphRAG's indexing pipeline is significantly more complex than standard RAG. Microsoft ships it as a Python package with a CLI:

```bash
# 1. Initialize a GraphRAG project
python -m graphrag.index --init --root ./my_rag_project

# 2. Run the full indexing pipeline
python -m graphrag.index --root ./my_rag_project
```

Internally, the pipeline executes these stages:

### Stage 1: Text Extraction and Chunking

Standard chunking of input documents — identical to other RAG techniques.

### Stage 2: Entity Extraction (LLM-intensive)

For each chunk, GPT-4 extracts a structured list of entities and relationships:

```python
# Conceptual representation of what GraphRAG extracts per chunk
{
    "entities": [
        {"name": "OpenAI", "type": "ORGANIZATION", "description": "AI research company"},
        {"name": "GPT-4", "type": "PRODUCT", "description": "Large language model by OpenAI"},
        {"name": "Sam Altman", "type": "PERSON", "description": "CEO of OpenAI"}
    ],
    "relationships": [
        {"source": "OpenAI", "target": "GPT-4", "type": "DEVELOPED", "strength": 1.0},
        {"source": "Sam Altman", "target": "OpenAI", "type": "LEADS", "strength": 0.9}
    ]
}
```

This runs for **every chunk** — meaning indexing a 1000-chunk corpus makes 1000+ LLM calls just for entity extraction. This is the primary cost driver of GraphRAG.

### Stage 3: Graph Construction

Extracted entities and relationships are merged across chunks into a unified knowledge graph:

```python
import networkx as nx

G = nx.Graph()

# Add entities as nodes
for entity in all_extracted_entities:
    G.add_node(entity["name"], 
               type=entity["type"], 
               description=entity["description"],
               sources=[...])  # which chunks this entity appears in

# Add relationships as edges
for rel in all_extracted_relationships:
    G.add_edge(rel["source"], rel["target"],
               type=rel["type"],
               weight=rel["strength"],
               sources=[...])
```

Entities mentioned in multiple chunks are merged — "OpenAI" in chunk 47 and chunk 483 become the same node, with both chunks as sources.

### Stage 4: Community Detection (Leiden Algorithm)

The Leiden algorithm partitions the graph into communities — densely connected subgraphs:

```
Knowledge Graph
├── Community A: AI Research Organizations
│   ├── OpenAI, DeepMind, Anthropic, Meta AI
│   └── Relationships: collaborations, competition, paper citations
├── Community B: Large Language Models
│   ├── GPT-4, Claude, Gemini, Llama
│   └── Relationships: benchmark comparisons, architecture similarities
├── Community C: Key Researchers
│   ├── Sam Altman, Ilya Sutskever, Demis Hassabis
│   └── Relationships: employment, publications, affiliations
...
```

Leiden outperforms Louvain (the previous standard) for community detection in large graphs — producing more balanced, hierarchically consistent communities.

### Stage 5: Community Summarization

For each community, GPT-4 generates a structured summary of all entities and relationships within it:

```python
community_summary_prompt = """
Summarize the following entities and relationships into a coherent description:

Entities: OpenAI (AI company), GPT-4 (LLM), Sam Altman (CEO)
Relationships: OpenAI DEVELOPED GPT-4, Sam Altman LEADS OpenAI

Summary: OpenAI, led by CEO Sam Altman, is an AI research company that developed 
GPT-4, one of the most capable large language models as of 2024. The organization 
focuses on safe artificial general intelligence development...
"""
```

These community summaries serve as high-level retrieval targets for broad queries.

---

## The Two Query Modes

GraphRAG supports two fundamentally different query modes:

### Mode 1: Global Search

For high-level questions requiring synthesis across the entire corpus:

```python
from graphrag.query.api import global_search

result = await global_search(
    config=settings,
    nodes=entities,
    entities=entities,
    community_reports=community_summaries,
    query="What are the major AI organizations and their main contributions?"
)
```

**How it works**:
1. Load all community summaries (these are pre-generated, not from vector search)
2. For each community summary, generate a partial response
3. Aggregate all partial responses into one comprehensive answer

**Best for**: Questions about general themes, trends, overviews, multi-entity comparisons — anything that requires synthesizing information from the full corpus.

### Mode 2: Local Search

For specific questions about particular entities or relationships:

```python
from graphrag.query.api import local_search

result = await local_search(
    config=settings,
    nodes=entities,
    text_units=chunks,
    relationships=relationships,
    query="What is GPT-4's performance on the MMLU benchmark?"
)
```

**How it works**:
1. Embed the query and find relevant entities (vector search over entity descriptions)
2. Expand to related entities and relationships (graph traversal)
3. Retrieve the source text chunks connected to these entities
4. Generate answer from entity context + source chunks

**Best for**: Specific fact queries, entity-level questions, relationship queries ("what did X do with Y?").

---

## GraphRAG vs. Standard RAG: When Each Wins

### Query: "What companies are competing in the AI assistant market?"

**Standard RAG**: Returns chunks mentioning "AI assistant companies" — isolated mentions, no cross-document synthesis. May miss companies not explicitly listed together in any chunk.

**GraphRAG**: Traverses the knowledge graph — finds "AI Assistant" entity cluster, retrieves all connected company entities (OpenAI, Google, Anthropic, Apple, Microsoft...) and their relationships. Comprehensive, structured answer covering the full competitive landscape.

### Query: "What was the specific benchmark score of Model X on Task Y?"

**Standard RAG**: Directly finds the chunk containing the specific score. Fast, precise.

**GraphRAG**: Also finds it, but with more overhead — entity extraction must have captured this as a relationship. If the fact was in a chart (not text), it may be missed.

**Winner for specific facts**: Standard RAG.  
**Winner for relational/systemic questions**: GraphRAG.

---

## Cost Reality Check

GraphRAG is significantly more expensive than standard RAG to deploy:

| Cost Dimension | Standard RAG | GraphRAG |
|---------------|-------------|----------|
| Indexing LLM calls | 0 | ~N chunk calls for entity extraction |
| Index construction time | Minutes | Hours (for large corpora) |
| Index storage | Small (vectors) | Medium (vectors + graph + community data) |
| Query LLM calls (Global) | 1 | ~C community summaries queried |
| Query LLM calls (Local) | 1 | 2-4 |
| Query latency | 1-3 seconds | 5-30 seconds |

For a corpus of 10,000 chunks, entity extraction alone may cost $10-50 in API calls and take several hours. This is a one-time indexing cost that amortizes over many queries — but it's a real deployment consideration.

---

## The `settings.yaml` Configuration

GraphRAG is configured entirely through `settings.yaml`:

```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini          # Use mini for extraction to reduce cost
  max_tokens: 2048

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  max_gleanings: 1            # How many times to retry extraction per chunk

community_reports:
  max_length: 2000            # Summary length per community

storage:
  type: file
  base_dir: "./output"
```

Key tuning decisions:
- **`model: gpt-4o-mini`** for entity extraction: Dramatically reduces cost vs. gpt-4o, with acceptable quality reduction for extraction tasks
- **`max_gleanings: 1`**: Set to 0 to skip extraction retries and halve LLM call count
- **`community_reports.max_length`**: Longer summaries = richer global search, but more tokens per query

---

## When to Use GraphRAG

GraphRAG is the right choice when the user's questions inherently require traversing relationships — "How are X and Y connected?", "Which organizations operate in both domain A and domain B?", "What chain of decisions led to outcome Z?" These multi-hop, relational, and comparative queries are where vector similarity fundamentally breaks down and graph traversal provides what embedding search cannot. For large corpora where users ask strategic or synthesis-level questions spanning many documents — competitive intelligence, organizational knowledge bases, research over thousands of pages — GraphRAG provides capabilities that no chunk-based system can match.

The tradeoffs are significant and should be weighed carefully. Indexing is substantially more expensive than standard RAG — entity extraction, relationship inference, and community summarization each require LLM calls per document batch. For corpora of text-heavy but entity-sparse content (literary texts, general essays), this expense yields little benefit. Documents that change frequently also create an expensive update burden: re-indexing the full graph for each revision is costly. And for queries that are simple factual lookups rather than relational traversals, a well-tuned standard RAG system will perform equivalently at a fraction of the cost.

---

## Summary

Microsoft GraphRAG represents a paradigm shift from "find relevant chunks" to "understand the knowledge structure of a corpus." By extracting entities and relationships, building a knowledge graph, running Leiden community detection, and generating community summaries, it creates a multi-level index that supports both specific lookups and corpus-level synthesis.

The tradeoffs are real: indexing is 10-100× more expensive than vector RAG, and queries are slower. But for applications where understanding *how information connects* matters as much as *what the information is* — competitive intelligence, organizational knowledge bases, research synthesis — GraphRAG provides capabilities that no chunk-based system can match.

It's the right tool when your questions sound like: "What is the relationship between X and Y?", "How did the acquisition of A affect B and C?", or "What are all the known connections between entity group 1 and entity group 2?" — questions that require reasoning across the entire document network, not just retrieving the most similar passage.
