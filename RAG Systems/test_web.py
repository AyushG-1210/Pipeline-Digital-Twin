import sys
import os

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RAG_DIR)

# Mock chromadb and vector_store to prevent scipy DLL crash
import sys
from unittest.mock import MagicMock

sys.modules['chromadb'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['database.vector_store'] = MagicMock()

from rag.web_retriever import web_search, fetch_url

print("\n--- Test: Web Search & Fetching ---")
try:
    urls = web_search("What are the latest pipeline incidents in 2026?", max_results=2)
    if urls:
        print(f"Success! Found {len(urls)} URLs: {urls}")
        text = fetch_url(urls[0])
        if text:
            print(f"Success! Fetched content length: {len(text)}")
            print(f"Snippet: {text[:200]}...")
        else:
            print("Failed to fetch HTML content.")
    else:
        print("Failed: No URLs returned from DuckDuckGo.")
except Exception as e:
    print(f"Error during web retrieval test: {e}")
