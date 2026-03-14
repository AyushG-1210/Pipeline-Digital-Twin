from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    all_chunks = []

    for doc in documents:

        chunks = splitter.split_text(doc["text"])

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["source"],
                    "chunk_id": i
                }
            })

    return all_chunks