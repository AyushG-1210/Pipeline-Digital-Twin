from ingestion.extract_text import load_documents
from ingestion.chunk_documents import chunk_documents
from ingestion.create_embeddings import create_embeddings
from database.vector_store import store_chunks

print("Loading documents...")

docs = load_documents("data")

print("Chunking documents...")

chunks = chunk_documents(docs)

print("Creating embeddings...")

chunks = create_embeddings(chunks)

print("Storing in vector database...")

store_chunks(chunks)

print("Indexing complete!")