import sys
from rag.retriever import retrieve_documents
from rag.prompt_builder import build_prompt

# Try to import generator, but handle missing API key gracefully
try:
    from rag.generator import generate_answer
    has_generator = True
except ValueError as e:
    print(f"\n[Warning] {e}")
    print("[Warning] Running in Retrieval-Only mode. LLM answering is disabled.\n")
    has_generator = False

print("RAG system ready!")

while True:
    try:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.strip().lower() in ["exit", "quit"]:
            break
        
        if not question.strip():
            continue

        retrieved_docs = retrieve_documents(question)
        if not retrieved_docs:
            print("\nNo relevant documents found.")
            continue

        context = "\n".join(retrieved_docs)

        if has_generator:
            prompt = build_prompt(context, question)
            try:
                answer = generate_answer(prompt)
                print("\nAnswer:\n", answer)
            except Exception as err:
                print(f"\n[Error generating answer]: {err}")
                print("\nRetrieved Context:\n", context)
        else:
            print("\nRetrieved Context:\n", context)
            
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break