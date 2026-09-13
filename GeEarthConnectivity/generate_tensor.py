import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load variables from the .env file
load_dotenv()

# =============================================================================
# SOIL BRANCH (Branch 1) -- COLUMN SCHEMA
# =============================================================================
# Tensor shape (REAL data path): [N_incidents, 4]
#   This is the resolved answer to the Phase 4 shape conflict [5831, 4] vs [Batch, 50].
#   Decision: keep the 4 real-world columns and declare soil_dim=4 in FactoredMIONet
#   for the real-data inference path. Do NOT fabricate 46 synthetic dimensions.
#
# Col | Field         | Source                             | Units    | Texas range
# ----|---------------|------------------------------------|----------|-------------------
#  0  | lat           | PHMSA incident GPS                 | deg (N)  | 25.8  to  36.5
#  1  | lon           | PHMSA incident GPS                 | deg (E)  | -106.6 to -93.5
#  2  | soil_moisture | ERA5 volumetric_soil_water_layer_1 | m3/m3    | 0.05  to  0.55
#  3  | temperature   | ERA5 temperature_2m  (K -> degC)   | degC     | -5.0  to  45.0
SOIL_COL_LAT         = 0
SOIL_COL_LON         = 1
SOIL_COL_MOISTURE    = 2
SOIL_COL_TEMPERATURE = 3

# Normalization bounds for [0, 1] rescaling before model input.
# Source: Texas geographic extent + ERA5 climatological range.
SOIL_NORM_BOUNDS = {
    "lat":           (25.8,   36.5),    # Texas N-S extent
    "lon":           (-106.6, -93.5),   # Texas W-E extent
    "soil_moisture": (0.05,   0.55),    # m3/m3: dry caliche to saturated clay
    "temperature":   (-5.0,   45.0),    # degC: winter low to summer peak
}
SOIL_NORM_MINS = np.array([v[0] for v in SOIL_NORM_BOUNDS.values()], dtype=np.float32)
SOIL_NORM_MAXS = np.array([v[1] for v in SOIL_NORM_BOUNDS.values()], dtype=np.float32)

# =============================================================================
# FLUID BRANCH (Branch 2) -- SYNTHETIC DECLARATION
# =============================================================================
# STATUS: SYNTHETIC for Phase 4/5/6. No per-segment operating pressure or
# fluid flow rate exists in any currently ingested dataset.
#
# PHMSA HL Annual Report audit (2026-09-13):
#   Part A-E : Commodity type, operator metadata          -- NO pressure/flow
#   Part F-G : Mileage by HCA status                     -- NO pressure/flow
#   Part H   : Mileage binned by OD (4" to 58"+)         -- NO pressure/flow
#   Part J   : Mileage by seam x %SMYS bands             -- SYSTEM-LEVEL ONLY
#              (e.g. PARTJST20MOREON = miles at >=20% SMYS; not per-segment)
#   RRC GIS  : Static T-4 permit data                    -- NO pressure/flow
# FUTURE: Part J %SMYS bands proposed as Phase 7+ fluid branch constraint.
FLUID_BRANCH_IS_SYNTHETIC = True

# Neo4j Credentials
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

def generate_deeponet_tensor():
    print("Extracting spatial and climate features from Neo4j...")

    # Query pulls the 4 real-world soil columns that form Branch 1 input.
    # Column order matches SOIL_COL_* constants above: lat, lon, moisture, temp.
    query = """
    MATCH (i:Incident)
    WHERE i.latitude IS NOT NULL AND i.longitude IS NOT NULL
    RETURN 
        i.latitude     AS lat,
        i.longitude    AS lon,
        i.soil_moisture AS moisture,
        i.temperature  AS temp
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
    
    # Reorder columns to match SOIL_COL_* schema: lat, lon, moisture, temp
    df = df[["lat", "lon", "moisture", "temp"]]

    # Convert all columns to float32 (standard precision for PyTorch/TensorFlow)
    df = df.astype(np.float32)

    # Flatten into a NumPy array: shape [N_incidents, 4]
    tensor_raw = df.to_numpy()   # physical units: deg, deg, m3/m3, degC

    # Normalize each column to [0, 1] using SOIL_NORM_BOUNDS
    # Clamp first so out-of-range readings don't push beyond [0, 1]
    tensor_raw_clamped = np.clip(tensor_raw, SOIL_NORM_MINS, SOIL_NORM_MAXS)
    tensor_norm = ((tensor_raw_clamped - SOIL_NORM_MINS)
                   / (SOIL_NORM_MAXS - SOIL_NORM_MINS)).astype(np.float32)

    # --- OUTPUT ---
    # raw  : physical units  -- for logging / downstream diagnostics
    # norm : [0,1] scaled    -- this is what FactoredMIONet(soil_dim=4) consumes
    output_raw_filename  = "input_tensor_raw.npy"
    output_norm_filename = "input_tensor.npy"   # keep original name for compat
    np.save(output_raw_filename,  tensor_raw)
    np.save(output_norm_filename, tensor_norm)
    tensor = tensor_norm   # alias for the report below
    
    # Print the final Verification Report
    print("\n" + "="*60)
    print("  SOIL BRANCH (Branch 1) -- VERIFICATION REPORT")
    print("="*60)
    print(f"Raw  file        : {output_raw_filename}")
    print(f"Norm file        : {output_norm_filename}  (feeds FactoredMIONet soil_dim=4)")
    print(f"Tensor Shape     : {tensor.shape}  -- [N_incidents={tensor.shape[0]}, soil_dim={tensor.shape[1]}]")
    print(f"Data Type        : {tensor.dtype}")
    print(f"Null Count       : {np.isnan(tensor).sum()}")
    print()
    col_names = list(SOIL_NORM_BOUNDS.keys())
    for i, name in enumerate(col_names):
        lo, hi = SOIL_NORM_BOUNDS[name]
        raw_min = tensor_raw[:, i].min()
        raw_max = tensor_raw[:, i].max()
        norm_min = tensor[:, i].min()
        norm_max = tensor[:, i].max()
        print(f"  col {i} ({name:<14}): raw [{raw_min:8.3f}, {raw_max:8.3f}]  "
              f"norm [{norm_min:.4f}, {norm_max:.4f}]  expected [{lo}, {hi}]")
    print()
    print(f"  Fluid branch (Branch 2) : SYNTHETIC  (no per-segment pressure/flow in PHMSA data)")
    print("="*60)

if __name__ == "__main__":
    generate_deeponet_tensor()
