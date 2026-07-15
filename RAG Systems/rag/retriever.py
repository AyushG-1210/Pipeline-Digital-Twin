import os
from database.vector_store import collection
from rag.web_retriever import retrieve_from_web
from dotenv import load_dotenv

load_dotenv()
WEB_RETRIEVAL_THRESHOLD = float(os.getenv("WEB_RETRIEVAL_THRESHOLD", "1.2"))

def is_temporal_query(question):
    temporal_keywords = ["latest", "recent", "newest", "today", "now", "2024", "2025", "2026", "news"]
    q_lower = question.lower()
    return any(kw in q_lower for kw in temporal_keywords)

def retrieve_documents(question):
    
    # 1. Check for temporal/breaking news queries
    if is_temporal_query(question):
        print("[Router] Temporal query detected. Routing to Web Search...")
        web_docs = retrieve_from_web(question)
        if web_docs:
            return web_docs

    # 2. Local Vector Search
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    documents = results["documents"][0] if "documents" in results and results["documents"] else []
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0]*len(documents)
    metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}]*len(documents)
    ids = results["ids"][0] if "ids" in results and results["ids"] else []

    # 3. Confidence/Relevance Gating
    valid_docs = []
    valid_ids = []
    for doc, dist, meta, chunk_id in zip(documents, distances, metadatas, ids):
        if dist < WEB_RETRIEVAL_THRESHOLD:
            source = meta.get("source", "Unknown Local Document")
            valid_docs.append(f"[Source: {source}]\n{doc}")
            valid_ids.append(chunk_id)

    if valid_docs:
        print(f"[Router] Found {len(valid_docs)} relevant local documents.")
        
        # Graph Augmentation
        from rag.graph_retriever import get_graph_context
        graph_facts = get_graph_context(valid_ids)
        if graph_facts:
            print(f"[Router] Augmented with {len(graph_facts)} facts from Neo4j Graph.")
            valid_docs.extend(graph_facts)
            
        return valid_docs
        
    # 4. Fallback to Web Search
    print(f"[Router] Local confidence too low (min distance: {distances[0] if distances else 'N/A'}). Falling back to Web Search...")
    web_docs = retrieve_from_web(question)
    
    if web_docs:
        return web_docs
        
    return []