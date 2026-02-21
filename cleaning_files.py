import pandas as pd
import numpy as np
from pathlib import Path

IN_DIR = r"C:\Users\anshu\Desktop\Major Project\Raw_data"
OUT_DIR = r"C:\Users\anshu\Desktop\Major Project\Cleaned_data"

def clean_data():
    in_path = Path(IN_DIR)
    out_path = Path(OUT_DIR)
    
    for file_path in in_path.rglob('*'):
        suffix = file_path.suffix.lower()
        if file_path.is_file() and suffix in ['.xlsx', '.xls', '.csv']:
            print(f"Processing: {file_path.name}...")
            
            try:
                # 1. Dynamic Loading
                if suffix == '.csv':
                    df = pd.read_csv(file_path, low_memory=False) 
                else:
                    df = pd.read_excel(file_path)
                
                num_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(num_cols) > 0:
                    mask_all_zeros = (df[num_cols] == 0).all(axis=1)
                    df_cleaned = df[~mask_all_zeros]
                else:
                    df_cleaned = df 
                
                rel_path = file_path.relative_to(in_path)
                target_path = out_path / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 2. Dynamic Saving
                if suffix == '.csv':
                    df_cleaned.to_csv(target_path, index=False)
                else:
                    df_cleaned.to_excel(target_path, index=False)
                    
                print(f" -> Saved cleaned file: {target_path}\n")
                
            except Exception as e:
                print(f" -> ERROR processing {file_path.name}: {e}\n")

if __name__ == "__main__":
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    clean_data()