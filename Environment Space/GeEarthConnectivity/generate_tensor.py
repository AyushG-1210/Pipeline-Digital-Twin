import numpy as np
import pandas as pd
from neo4j import GraphDatabase

# Neo4j Credentials
URI = "neo4j+ssc://5b58c7e0.databases.neo4j.io"
AUTH = ("5b58c7e0", "1MicthVv2tKboUcSp5o4OUlIByAlEMMKh6Y1PVSS8WE")

def generate_deeponet_tensor():
    print("Extracting spatial and climate features from Neo4j...")
    
    # Query to pull the coordinates and the climate data you just scraped
    query = """
    MATCH (i:Incident)
    WHERE i.latitude IS NOT NULL AND i.longitude IS NOT NULL
    RETURN 
        i.latitude AS lat, 
        i.longitude AS lon,
        i.soil_moisture AS moisture,
        i.temperature AS temp
    """
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            
    print(f"Extracted {len(records)} incidents.")
    
    # Load into Pandas for verification and manipulation
    df = pd.DataFrame(records)
    
    # --- VERIFICATION: Null Check & Imputation ---
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} missing values. Filling with 0.0 to satisfy branch net constraints...")
        df = df.fillna(0.0)
    else:
        print("✅ Null Check Passed: No missing values found in the dataset.")
    
    # Convert all columns to float32 (the standard precision for PyTorch/TensorFlow)
    df = df.astype(np.float32)
    
    # Flatten into a NumPy array
    tensor = df.to_numpy()
    
    # --- OUTPUT ---
    output_filename = "input_tensor.npy"
    np.save(output_filename, tensor)
    
    # Print the final Verification Report for your professor/team lead
    print("\n" + "="*45)
    print("🚀 ALIGNER TASK VERIFICATION REPORT")
    print("="*45)
    print(f"Output File      : {output_filename}")
    print(f"Tensor Shape     : {tensor.shape} -> [Samples={tensor.shape[0]}, Features={tensor.shape[1]}]")
    print(f"Data Type        : {tensor.dtype}")
    print(f"Final Null Count : {np.isnan(tensor).sum()}")
    print("="*45)

if __name__ == "__main__":
    generate_deeponet_tensor()