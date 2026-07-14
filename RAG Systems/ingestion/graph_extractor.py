import os
import json
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from neo4j import GraphDatabase

DB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(DB_DIR, ".env"))
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+ssc://5b58c7e0.databases.neo4j.io")
NEO4J_USER = os.environ.get("NEO4J_USER", "5b58c7e0")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "1MicthVv2tKboUcSp5o4OUlIByAlEMMKh6Y1PVSS8WE")

try:
    client = genai.Client(api_key=api_key)
except Exception:
    client = None

class Entity(BaseModel):
    label: str = Field(description="One of: Operator, Pipeline, Incident, Location")
    name: str = Field(description="The primary identifier for this entity")
    attributes: dict = Field(default_factory=dict, description="Key-value pairs for properties")

class ExtractedGraph(BaseModel):
    entities: List[Entity]

def extract_entities_from_chunk(text: str) -> ExtractedGraph:
    if not client or not api_key or api_key == "PUT_YOUR_GEMINI_API_KEY_HERE":
        return ExtractedGraph(entities=[])
        
    prompt = f"""
    Extract any Pipeline Operators, Pipeline Systems, Incidents, or Locations mentioned in the following text.
    Only extract entities if you are highly confident.
    Text: {text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ExtractedGraph,
            },
        )
        return ExtractedGraph.model_validate_json(response.text)
    except Exception as e:
        print(f"[Graph Extraction Error] {e}")
        return ExtractedGraph(entities=[])

def push_to_neo4j(chunk_id: str, chunk_text: str, source_doc: str, graph: ExtractedGraph):
    if not graph.entities:
        return
        
    query_no_apoc = """
    MERGE (c:Chunk {chunk_id: $chunk_id})
    SET c.text = $chunk_text, c.source = $source_doc
    
    WITH c
    UNWIND $entities AS ent
    
    CALL {
        WITH c, ent
        WITH c, ent WHERE ent.label = 'Operator'
        MERGE (e:Operator {name: ent.name})
        SET e += ent.attributes, e.source = 'document_corpus'
        MERGE (c)-[:MENTIONS]->(e)
    }
    
    CALL {
        WITH c, ent
        WITH c, ent WHERE ent.label = 'Pipeline'
        MERGE (e:Pipeline {name: ent.name})
        SET e += ent.attributes, e.source = 'document_corpus'
        MERGE (c)-[:MENTIONS]->(e)
    }
    
    CALL {
        WITH c, ent
        WITH c, ent WHERE ent.label = 'Incident'
        MERGE (e:Incident {name: ent.name})
        SET e += ent.attributes, e.source = 'document_corpus'
        MERGE (c)-[:MENTIONS]->(e)
    }
    
    CALL {
        WITH c, ent
        WITH c, ent WHERE ent.label = 'Location'
        MERGE (e:Location {name: ent.name})
        SET e += ent.attributes, e.source = 'document_corpus'
        MERGE (c)-[:MENTIONS]->(e)
    }
    """
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        entities_data = [e.model_dump() for e in graph.entities]
        with driver.session() as session:
            session.run(query_no_apoc, chunk_id=chunk_id, chunk_text=chunk_text, source_doc=source_doc, entities=entities_data)
    except Exception as e:
        print(f"[Neo4j Push Error] {e}")
    finally:
        driver.close()
