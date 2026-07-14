import os
from ingestion.extract_text import load_documents
from ingestion.chunk_documents import chunk_documents
from ingestion.create_embeddings import create_embeddings
from database.vector_store import store_chunks

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

print("Indexing complete!")