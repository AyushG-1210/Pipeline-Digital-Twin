import os
import sqlite3
import requests
import json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from database.vector_store import embedding_function

DB_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DB_PATH = os.path.join(DB_DIR, "web_cache.db")

def setup_cache():
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS page_cache (
            url TEXT PRIMARY KEY,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_cache()

def fetch_url(url, timeout=10):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM page_cache WHERE url=?", (url,))
    row = cursor.fetchone()
    if row:
        conn.close()
        return row[0]
    
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        cursor.execute("INSERT OR REPLACE INTO page_cache (url, content) VALUES (?, ?)", (url, text))
        conn.commit()
        conn.close()
        return text
    except Exception as e:
        print(f"[Web Fetch Error] Could not fetch {url}: {e}")
        conn.close()
        return None

def web_search(query, max_results=3):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT results FROM search_cache WHERE query=?", (query,))
    row = cursor.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])
        
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        urls = [r.get("href") for r in results if r.get("href")]
        
        cursor.execute("INSERT OR REPLACE INTO search_cache (query, results) VALUES (?, ?)", (query, json.dumps(urls)))
        conn.commit()
        conn.close()
        return urls
    except Exception as e:
        print(f"[Web Search Error]: {e}")
        conn.close()
        return []

def retrieve_from_web(question, top_k=3):
    urls = web_search(question, max_results=3)
    if not urls:
        return []
        
    docs = []
    for url in urls:
        text = fetch_url(url)
        if text:
            docs.append({"source": f"Live Web Search - {url}", "text": text})
            
    if not docs:
        return []
        
    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {"source": doc["source"], "chunk_id": i}
            })
            
    if not all_chunks:
        return []
        
    # Ephemeral Chroma collection for vector search
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="temp_web", embedding_function=embedding_function)
    
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [str(i) for i in range(len(all_chunks))]
    
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, len(all_chunks))
    )
    
    if not results["documents"] or not results["documents"][0]:
        return []
        
    # Format the results to include provenance
    formatted_docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted_docs.append(f"[Source: {meta['source']}]\n{doc}")
        
    return formatted_docs
