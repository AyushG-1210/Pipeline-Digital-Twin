import sys
import os

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RAG_DIR)

from rag.retriever import retrieve_documents
from rag.graph_retriever import get_graph_context

print("\n--- Test 1: Web Fallback (Temporal Query) ---")
try:
    web_results = retrieve_documents("What are the latest pipeline incidents in 2026?")
    if web_results:
        print(f"Success! Retrieved {len(web_results)} web chunks.")
        print(f"Sample:\n{web_results[0][:200]}...")
    else:
        print("Failed: No web results returned.")
except Exception as e:
    print(f"Error during web fallback: {e}")

print("\n--- Test 2: Graph Retrieval (AuraDB Connection) ---")
try:
    # Test with dummy chunk IDs
    graph_results = get_graph_context(["0", "1", "999"])
    print(f"Success! AuraDB Connection works. Retrieved {len(graph_results)} graph facts.")
    if graph_results:
        for res in graph_results:
            print(res)
except Exception as e:
    print(f"Error during graph retrieval: {e}")

print("\nTests complete!")
