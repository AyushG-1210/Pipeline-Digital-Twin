import pandas as pd
from google import genai
import json
import time
import sys
from dotenv import load_dotenv 
import os

load_dotenv()

# --- CONFIG --- 
API_KEY = os.getenv("GEMINI_KEY")
if not API_KEY:
    print("ERROR: GEMINI_KEY environment variable is missing.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
model_id = 'gemini-2.5-flash' 

IN_FILE = r"C:\Users\anshu\Desktop\Major Project\Data_Cleaning\Cleaned_Physics_Index.csv"
OUT_FILE = "Master_Physics_Index_Gemini_Refined.csv"
CACHE_FILE = "gemini_progress_cache.json" # New: Saves progress as it goes

# The exact header string where it died
RESUME_HEADER = "Unnamed: 2" 

def refine_existing_index():
    try:
        df = pd.read_csv(IN_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find {IN_FILE}")
        return

    unique_headers = df['Original_Header'].dropna().unique().tolist()
    
    # --- RESUME LOGIC ---
    if RESUME_HEADER in unique_headers:
        start_index = unique_headers.index(RESUME_HEADER)
        print(f"Found resume point at index {start_index}. Skipping earlier headers...")
        headers_to_process = unique_headers[start_index:]
    else:
        print("Resume header not found! Starting from the beginning...")
        headers_to_process = unique_headers

    # Load previous cache if it exists so we don't overwrite earlier work
    decoded_map = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            decoded_map = json.load(f)

    batch_size = 40
    print(f"Refining {len(headers_to_process)} headers starting from '{RESUME_HEADER}'...")

    for i in range(0, len(headers_to_process), batch_size):
        batch = headers_to_process[i:i+batch_size]
        
        prompt = f"""
        Act as a Principal Pipeline Integrity Engineer. Refine the utility score and description for these PHMSA/RRC headers: 
        {batch}
        
        Return ONLY a raw JSON object where each key is the exact header and the value is a list of exactly 2 items:
        [Upgraded Physics Utility Score (1-10), Upgraded 1-Line Description for PINN modeling]
        
        Example: "PARTJSTR_UNK_ON": [8, "Flags if strain measurement method is unknown, increasing uncertainty in stress calculations."]
        Do not change the header key. Do not use markdown blocks, return ONLY raw JSON format.
        """
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            clean_json = response.text.replace('```json', '').replace('```', '').strip()
            
            # Update map and save to cache immediately
            new_data = json.loads(clean_json)
            decoded_map.update(new_data)
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(decoded_map, f)
                
            print(f"Batch {i} to {i+len(batch)} processed and cached.")
            time.sleep(3) 

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            print("Stopping to prevent further errors. Progress is saved in cache.")
            break

    # Map updates to DataFrame
    print("Applying Gemini updates from cache to the dataset...")
    
    def update_row(row):
        header = row['Original_Header']
        if header in decoded_map and len(decoded_map[header]) == 2:
            row['Physics_Score'] = decoded_map[header][0]
            row['Description'] = decoded_map[header][1]
        return row

    updated_df = df.apply(update_row, axis=1)
    updated_df.to_csv(OUT_FILE, index=False)
    print(f"Successfully saved refined data to {OUT_FILE}! Keys are untouched.")

if __name__ == "__main__":
    refine_existing_index()