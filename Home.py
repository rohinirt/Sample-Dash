# your_project/Home.py

import streamlit as st
import pandas as pd
from io import StringIO
# Import your utility functions to perform the initial checks
from utils import REQUIRED_COLS 

st.set_page_config(
    page_title="Cricket Dashboard",
    layout="wide"
)

# --- Check for uploaded data and display uploader ---
if 'data_df' not in st.session_state:
    st.title("🏏 Cricket Dashboard Data Upload")

    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            data = uploaded_file.getvalue().decode("utf-8")
            df_raw = pd.read_csv(StringIO(data))
            
            # Validation
            if not all(col in df_raw.columns for col in REQUIRED_COLS):
                missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
                st.error(f"The CSV file is missing required columns: {', '.join(missing_cols)}")
            else:
                # Store the DataFrame in session state for all pages to access
                st.session_state['data_df'] = df_raw
                st.success("Data uploaded successfully! Please navigate to a dashboard page.")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
else:
    st.title("📊 Global Dashboard")
    st.info("Data is loaded. Use the navigation on the left to switch between dashboards.")
    
    # You could add some simple summary stats here if needed
    st.write(f"Total Deliveries Loaded: {len(st.session_state['data_df'])}")
