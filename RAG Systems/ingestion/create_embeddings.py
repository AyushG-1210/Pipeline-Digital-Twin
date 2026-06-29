from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-large-en")


def create_embeddings(chunks):

    for chunk in chunks:

        chunk["embedding"] = model.encode(chunk["text"]).tolist()

    return chunks