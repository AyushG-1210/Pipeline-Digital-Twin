import pandas as pd
import os
from neo4j import GraphDatabase

# 1. Neo4j Connection Credentials
URI = "neo4j+ssc://5b58c7e0.databases.neo4j.io"
AUTH = ("5b58c7e0", "1MicthVv2tKboUcSp5o4OUlIByAlEMMKh6Y1PVSS8WE")

def build_graph_schema(tx, batch):
    """
    Creates Operator, Pipeline, and Incident nodes based on actual PHMSA headers.
    """
    query = """
    UNWIND $batch AS row
    
    // 1. Create the Operator Node
    MERGE (o:Operator {operator_id: toString(row.OPERATOR_ID)})
    SET o.name = toString(row.NAME)

    // 2. Create the Pipeline System Node
    // Using PIPE_FAC_NAME as the system identifier
    MERGE (p:Pipeline {system_id: toString(row.PIPE_FAC_NAME)})
    SET p.material = toString(row.MATERIAL_INVOLVED),
        p.installation_year = toString(row.INSTALLATION_YEAR)

    // 3. Create the Incident Node
    MERGE (i:Incident {report_number: toString(row.REPORT_NUMBER)})
    SET i.date = toString(row.LOCAL_DATETIME),
        i.cause = toString(row.CAUSE),
        i.latitude = toFloat(row.LOCATION_LATITUDE),
        i.longitude = toFloat(row.LOCATION_LONGITUDE)

    // 4. Map the Relationships
    MERGE (o)-[:OWNS]->(p)
    MERGE (i)-[:OCCURRED_ON]->(p)
    """
    tx.run(query, batch=batch)

def ingest_data():
    # Point directly to the Hazardous Liquid dataset
    file_path = os.path.join(
        'Raw_data', 
        'Raw_PHMSA_Pipeline_Safety_Flagged_Incidents', 
        'hl2010toPresent.xlsx'
    )
    print(f"Loading {file_path} into pandas... (This will take a minute)")
    
    # Read the data, skipping the "Data Dictionary" row if it exists
    df = pd.read_excel(file_path, sheet_name=1 if len(pd.ExcelFile(file_path).sheet_names) > 1 else 0)
    
    # Clean up column names (strip whitespace) just in case
    df.columns = df.columns.str.strip()
    
    # Fill missing data to avoid crashing later
    df = df.fillna("UNKNOWN")

    # Convert the entire dataframe to strings to prevent integer overflows
    df = df.astype(str)

    # Convert to a list of dictionaries for Neo4j
    data_dicts = df.to_dict('records')

    print("Connecting to Neo4j AuraDB and building the graph...")
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            # Push data in batches of 500 to keep memory stable
            batch_size = 500
            for i in range(0, len(data_dicts), batch_size):
                batch = data_dicts[i:i + batch_size]
                session.execute_write(build_graph_schema, batch)
                print(f"Successfully mapped records {i} to {i + len(batch)}")

    print("Graph ingestion complete! Nodes and relationships are live.")

if __name__ == "__main__":
    ingest_data()