import pandas as pd
from groq import Groq
from pathlib import Path
import json
import time

# --- CONFIG --- 
API_KEY = "API_KEY_HERE"
client = Groq(api_key=API_KEY)

# Use a valid Groq model (gpt-4o-mini is an OpenAI model, not Groq)
model_id = 'llama-3.1-8b-instant' 

IN_DIR = r"C:\Users\anshu\Desktop\Major Project\Cleaned_data"
OUT_FILE = "Master_Physics_Index.csv"

def get_real_headers(file_path):
    """Skips the trash rows at the top of PHMSA/RRC files."""
    try:
        temp_df = pd.read_excel(file_path, nrows=10, header=None) if file_path.suffix != '.csv' else pd.read_csv(file_path, nrows=10, header=None)
        header_idx = temp_df.apply(lambda x: x.astype(str).str.contains('ID|YEAR|OPERATOR|PART', case=False).sum(), axis=1).idxmax()
        df = pd.read_excel(file_path, skiprows=header_idx) if file_path.suffix != '.csv' else pd.read_csv(file_path, skiprows=header_idx)
        return df.columns.tolist()
    except:
        return []

def harvest_and_decode():
    all_data = []
    unique_headers = set()
    
    # 1. Collect everything
    for path in Path(IN_DIR).rglob('*.*'):
        if path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            headers = get_real_headers(path)
            if headers:
                all_data.append({"filename": path.name, "headers": headers})
                unique_headers.update(headers)

    # 2. Decode Unique Headers in Batches
    header_list = list(unique_headers)
    batch_size = 30
    decoded_map = {}

    print(f"Decoding {len(header_list)} unique headers...")

    for i in range(0, len(header_list), batch_size):
        batch = header_list[i:i+batch_size]
        
        prompt = f"""
        Act as a Pipeline Integrity Engineer. For these cryptic PHMSA/RRC headers: {batch}
        Return a JSON object where each key is the header and the value is a list containing exactly 4 items:
        [Plain English Name, Physics Utility Score (1-10), Engineering Category, 1-Line Description of Utility]
        Example: "PARTDCPBTOTAL": ["Miles Protected Bare Steel", 8, "Corrosion", "Defines the boundary condition for the rate of external wall thinning."]
        Do not use markdown blocks, return ONLY raw JSON format.
        """
        try:
            # Correct Groq syntax for completions
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Correct extraction path for Groq/OpenAI structure
            text = response.choices[0].message.content
            clean_json = text.replace('```json', '').replace('```', '').strip()
            decoded_map.update(json.loads(clean_json))
            time.sleep(1)  

        except Exception as e:
            print(f"Error in batch {i}: {e}")

    # 3. Flatten and Save
    final_rows = []
    desc_cache = {}

    def fetch_description_only(header: str) -> str:
        if header in desc_cache:
            return desc_cache[header]
        prompt = (
            f"Act as a Pipeline Integrity Engineer. Provide a single plain-English, one-line "
            f"description for the PHMSA/RRC header '{header}'. Return only the description text."
        )
        try:
            # Secondary API call fix
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            desc = resp.choices[0].message.content.strip()
        except Exception:
            desc = "No description available."
        desc_cache[header] = desc
        return desc

    for entry in all_data:
        fname = entry['filename']
        for h in entry['headers']:
            info = decoded_map.get(h, ["Unknown", 0, "N/A", "No description available."])
            while len(info) < 4:
                info.append("N/A")

            if info[3].strip().lower().startswith("no description"):
                info[3] = fetch_description_only(h)

            final_rows.append({
                "File": fname,
                "Original_Header": h,
                "Clean_Name": info[0],
                "Physics_Score": info[1],
                "Category": info[2],
                "Description": info[3]
            })

    # Sort by Score (High utility at the top)
    master_df = pd.DataFrame(final_rows).sort_values(by="Physics_Score", ascending=False)
    master_df.to_csv(OUT_FILE, index=False)
    print(f"Successfully created {OUT_FILE}!")

if __name__ == "__main__":
    harvest_and_decode()