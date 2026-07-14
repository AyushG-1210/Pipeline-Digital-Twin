# Hybrid Vector-Graph RAG Architecture

This document describes the upgraded RAG pipeline for the Pipeline Digital Twin. The system uses a hybrid approach, combining semantic vector similarity with structured graph traversal, and features a live web-search fallback for temporal/unknown queries.

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[Document Corpus] -->|Chunking| B(chunk_documents.py)
        B --> C[Embedding Pipeline (BGE-Large)]
        B --> D[LLM Graph Extractor]
        C -->|Vector Tensors| E[(ChromaDB)]
        D -->|Entities & Relations| F[(Neo4j AuraDB)]
    end

    subgraph Query Routing
        G[User Question] --> H{Temporal Keyword?}
        H -->|Yes| I[Web Retriever]
        H -->|No| J[Chroma Vector Query]
        
        J --> K{Confidence High?}
        K -->|No| I
        K -->|Yes| L[Extract Chunk IDs]
    end
    
    subgraph Context Assembly
        I --> M[DuckDuckGo Search]
        M --> N[HTML Scrape & Ephemeral Vector Filter]
        N --> O[Web Context]
        
        L --> P[Neo4j Cypher Query]
        P --> Q[1-Hop Graph Neighbors]
        
        L --> R[Vector Chunk Text]
        
        Q --> S[Merged Prompt Context]
        R --> S
        O --> S
    end
    
    S --> T[Gemini LLM]
    T --> U[Final Answer]
```

## Core Components

1. **Vector Store (ChromaDB)**
   - Holds text chunks embedded via `BAAI/bge-large-en`.
   - Used for initial semantic recall.

2. **Graph Database (Neo4j AuraDB)**
   - Shares credentials with `Environment Space/graph_ingestion.py`.
   - Stores `Operator`, `Pipeline`, `Incident`, and `Location` nodes.
   - Nodes extracted from documents are namespaced with `source: 'document_corpus'`.
   - `Chunk` nodes map Chroma chunk IDs to graph entities via `[:MENTIONS]` relationships.

3. **Web Retriever (DuckDuckGo)**
   - Triggers on keywords (e.g., "latest", "2026") or low vector confidence (threshold > 1.2).
   - Scrapes page HTML, strips boilerplate, and passes through an ephemeral ChromaDB instance to filter out irrelevant noise before sending to the LLM.
   - Caches results locally via SQLite (`web_cache.db`).
