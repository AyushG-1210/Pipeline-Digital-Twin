import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from pathlib import Path

# --- API CONFIGURATION ---
# Put your Gemini API key here. 
API_KEY = "KEY_HERE"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Initialize the Gemini model (using Gemini 1.5 Flash for fast, cheap text tasks)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- UI SETUP ---
st.set_page_config(page_title="PHMSA Data Decoder", layout="wide")
st.title("Pipeline Data Discovery & Decoder")

# --- FILE LOADING ---
DATA_DIR = r"C:\Users\anshu\Desktop\Major Project\Cleaned_data"
st.sidebar.header("Data Selection")

@st.cache_data
def load_data(file_path):
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path, low_memory=False)
    else:
        return pd.read_excel(file_path)

try:
    # Get all cleaned files
    all_files = list(Path(DATA_DIR).rglob('*.*'))
    valid_files = [f for f in all_files if f.suffix.lower() in ['.csv', '.xlsx', '.xls']]
    
    if not valid_files:
        st.warning(f"No valid files found in {DATA_DIR}")
    else:
        # File selector
        file_names = [f.name for f in valid_files]
        selected_file_name = st.sidebar.selectbox("Choose a file to analyze:", file_names)
        selected_file_path = valid_files[file_names.index(selected_file_name)]
        
        # Load and display data
        df = load_data(selected_file_path)
        st.write(f"### Previewing: {selected_file_name}")
        st.dataframe(df.head(10))
        
        # --- THE GEMINI DECODER ---
        st.divider()
        st.subheader("Header Decoder Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_header = st.selectbox("Select a cryptic header to decode:", df.columns)
            
            # Grab a sample of actual data from this column to give Gemini context
            sample_data = df[target_header].dropna().head(3).tolist()
            
            if st.button("Decode Header"):
                with st.spinner("Consulting digital engineer..."):
                    # The System Prompt
                    prompt = f"""
                    You are a Senior Pipeline Integrity Engineer analyzing Texas PHMSA/RRC regulatory data.
                    Decode this column header: '{target_header}'.
                    Here are 3 sample data points from this column: {sample_data}.
                    
                    Provide a response in this exact format:
                    1. Plain English Definition: (What this means)
                    2. Physics Utility: (How this can be used in a Physics-Informed Neural Network or burst pressure calculation)
                    """
                    
                    response = model.generate_content(prompt)
                    
                    with col2:
                        st.info("### Gemini Decode Result")
                        st.write(response.text)
                        
except Exception as e:
    st.error(f"Error loading dashboard: {e}")