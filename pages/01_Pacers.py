import streamlit as st
import pandas as pd
# Import the necessary functions from utils

# Set page configuration
st.set_page_config(
    page_title="Pacers Dashboard",
    layout="wide"
)

# =========================================================
# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
# =========================================================
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']

# =========================================================
# 2. BASE FILTER: ONLY SEAM DELIVERIES
# =========================================================
# Filter the raw data to include only Seam deliveries
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]

st.title("⚡ Pacers Dashboard (Seam Bowling Analysis)")

# =========================================================
# 3. FILTERS (Bowling Team and Bowler)
# =========================================================
filter_col1, filter_col2 = st.columns(2) 

# --- Filter Logic ---

# Check if 'BowlingTeam' column exists in the data
# If it doesn't, we will fall back to using 'BattingTeam' but warn the user.
if "BowlingTeam" in df_seam_base.columns:
    team_column = "BowlingTeam"
else:
    # Using 'BattingTeam' as a placeholder if 'BowlingTeam' is missing
    team_column = "BattingTeam" 
    st.warning("The 'BowlingTeam' column was not found. Displaying all Batting Teams as a fallback.")

# Get all teams based on the determined column
all_teams = ["All"] + sorted(df_seam_base[team_column].dropna().unique().tolist())
# Assuming 'BowlerName' column exists
all_bowlers = ["All"] + sorted(df_seam_base["BowlerName"].dropna().unique().tolist()) 

# 1. Bowling Team Filter (CORRECTED)
with filter_col1:
    bowl_team = st.selectbox("Bowling Team", all_teams, index=0)

# 2. Bowler Name Filter 
with filter_col2:
    bowler = st.selectbox("Bowler Name", all_bowlers, index=0)

# =========================================================
# 4. Apply Filters to the Base Seam Data
# =========================================================
df_filtered = df_seam_base.copy()

# Apply Bowling Team filter using the determined column name
if bowl_team != "All":
    df_filtered = df_filtered[df_filtered[team_column] == bowl_team]
    
# Apply Bowler filter (assuming 'BowlerName' column exists)
if bowler != "All":
    if "BowlerName" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["BowlerName"] == bowler]
    else:
        st.warning("BowlerName column not found for filtering.")
