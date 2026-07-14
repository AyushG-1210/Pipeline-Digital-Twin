import os
import chromadb
from chromadb.utils import embedding_functions

# Use the SAME embedding model everywhere
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en"
)

# Persistent database path resolved dynamically relative to this file
DB_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(DB_DIR)
CHROMA_PATH = os.path.join(RAG_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="pipeline_docs",
    embedding_function=embedding_function
)

def store_chunks(chunks):

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        ids.append(str(i))

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )