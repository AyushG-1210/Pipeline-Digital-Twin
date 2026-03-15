import pandas as pd
from google import genai
from google.api_core import exceptions
import json
import time

# --- CONFIG ---
API_KEY = "API_KEY_HERE"  # Replace with your actual API key
client = genai.Client(api_key=API_KEY)
model_id = 'gemini-2.5-pro'

def gemini_sanity_check(csv_file="Master_Physics_Index.csv", start_batch=1000):
    df = pd.read_csv(csv_file)
    unique_features = df[['Original_Header', 'Description', 'Physics_Score']].drop_duplicates()
    data_to_check = unique_features.to_dict(orient='records')
    
    batch_size = 30 # Reduced size to stay under TPM limits
    corrections = {}

    # Jump to your failed batch
    print(f"Resuming from record index {start_batch}...")

    for i in range(start_batch, len(data_to_check), batch_size):
        batch = data_to_check[i:i+batch_size]
        
        prompt = f"""
        Audit these pipeline data headers and scores (1-10): {batch}
        If a score contradicts the description (e.g. 'ZIP_CODE' = 10), provide the fix.
        Return ONLY raw JSON: {{"Original_Header": New_Score}}
        """

        # --- RETRY LOGIC (Exponential Backoff) ---
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                
                if clean_json and clean_json != "{}":
                    corrections.update(json.loads(clean_json))
                
                print(f"Batch {i} successful.")
                time.sleep(2) # Baseline politeness delay
                break # Success! Move to next batch
                
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (2 ** attempt) + 5 # 6s, 9s, 13s...
                    print(f"Rate limit hit at batch {i}. Waiting {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Non-rate-limit error at batch {i}: {e}")
                    break 

    # --- SAVE PROGRESS ---
    if corrections:
        df['Physics_Score'] = df.apply(lambda r: corrections.get(r['Original_Header'], r['Physics_Score']), axis=1)
        df.to_csv("Audited_Physics_Index_v2.csv", index=False)
        print("Corrections applied and saved to 'Audited_Physics_Index_v2.csv'")

if __name__ == "__main__":
    # Change 1000 to whatever index you actually crashed on
    gemini_sanity_check(start_batch=1000)