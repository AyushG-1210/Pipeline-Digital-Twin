import sys
import os

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RAG_DIR)

from rag.graph_retriever import get_graph_context

print("\n--- Test: Graph Retrieval (AuraDB Connection) ---")
try:
    # Test with dummy chunk IDs
    graph_results = get_graph_context(["0", "1", "999"])
    print(f"Success! AuraDB Connection works. Retrieved {len(graph_results)} graph facts.")
    if graph_results:
        for res in graph_results:
            print(res)
except Exception as e:
    print(f"Error during graph retrieval: {e}")
