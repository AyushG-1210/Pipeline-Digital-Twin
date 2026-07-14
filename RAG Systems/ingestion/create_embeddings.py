def create_embeddings(chunks):
    # Chroma DB's collection automatically handles embedding generation using the same BAAI/bge-large-en model,
    # so we pass chunks through to avoid redundant model loading and computation.
    return chunks