import os
from ingestion.extract_text import load_documents
from ingestion.chunk_documents import chunk_documents
from ingestion.create_embeddings import create_embeddings
from database.vector_store import store_chunks
from ingestion.graph_extractor import extract_entities_from_chunk, push_to_neo4j

print("Loading documents...")

# Resolve path to the data directory dynamically relative to this script
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(RAG_DIR, "data")

docs = load_documents(data_folder)

print("Chunking documents...")

chunks = chunk_documents(docs)

print("Creating embeddings...")

chunks = create_embeddings(chunks)

print("Storing in vector database...")

store_chunks(chunks)

print("Extracting graph entities and pushing to Neo4j... (This may take a while due to LLM rate limits)")
for i, chunk in enumerate(chunks):
    chunk_id = str(chunk["metadata"]["chunk_id"]) # Or just use string(i) since vector_store uses str(i)
    # Actually vector_store.py uses str(i) for ids, where i is the index of chunk in chunks list. Let's match it.
    actual_chunk_id = str(i)
    text = chunk["text"]
    source = chunk["metadata"]["source"]
    
    print(f"  Processing chunk {i+1}/{len(chunks)} from {source}...")
    graph = extract_entities_from_chunk(text)
    if graph and graph.entities:
        push_to_neo4j(chunk_id=actual_chunk_id, chunk_text=text, source_doc=source, graph=graph)

print("Indexing complete!")