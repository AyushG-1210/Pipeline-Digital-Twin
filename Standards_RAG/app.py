from rag.retriever import retrieve_documents
from rag.prompt_builder import build_prompt
from rag.generator import generate_answer

print("RAG system ready!")

while True:

    question = input("\nAsk a question: ")

    retrieved_docs = retrieve_documents(question)

    context = "\n".join(retrieved_docs)

    prompt = build_prompt(context, question)

    answer = generate_answer(prompt)

    print("\nAnswer:\n", answer)