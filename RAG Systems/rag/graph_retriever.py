import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

DB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(DB_DIR, ".env"))
load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+ssc://5b58c7e0.databases.neo4j.io")
NEO4J_USER = os.environ.get("NEO4J_USER", "5b58c7e0")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "1MicthVv2tKboUcSp5o4OUlIByAlEMMKh6Y1PVSS8WE")

def get_graph_context(chunk_ids):
    if not chunk_ids:
        return []
        
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
    OPTIONAL MATCH (c)-[:MENTIONS]->(e)
    OPTIONAL MATCH (e)-[r]-(neighbor)
    WHERE type(r) <> 'MENTIONS'
    WITH e, r, neighbor
    WHERE e IS NOT NULL
    RETURN labels(e)[0] AS entity_type, e.name AS entity_name, e.source AS entity_source, 
           type(r) AS relation, 
           labels(neighbor)[0] AS neighbor_type, neighbor.name AS neighbor_name, neighbor.source AS neighbor_source
    LIMIT 30
    """
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    context_sentences = []
    
    try:
        with driver.session() as session:
            result = session.run(query, chunk_ids=chunk_ids)
            for record in result:
                e_type = record["entity_type"]
                e_name = record["entity_name"]
                e_source = record.get("entity_source") or "unknown"
                
                rel = record["relation"]
                n_type = record["neighbor_type"]
                n_name = record["neighbor_name"]
                
                if rel and n_type and n_name:
                    n_source = record.get("neighbor_source") or "unknown"
                    context_sentences.append(f"[Graph Fact: {e_source}/{n_source}] {e_type} '{e_name}' is related via '{rel}' to {n_type} '{n_name}'.")
                else:
                    context_sentences.append(f"[Graph Fact: {e_source}] Mentioned {e_type}: '{e_name}'.")
                    
    except Exception as e:
        print(f"[Graph Retrieval Error] {e}")
    finally:
        driver.close()
        
    return list(set(context_sentences))
