from database.vector_store import collection


def retrieve_documents(question):

    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    documents = results["documents"][0]

    return documents