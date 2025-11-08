import streamlit as st
import pandas as pd
# Import the new function and necessary existing functions from utils

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

# Get all bowling teams (use the original raw data for comprehensive team lists if needed, 
# but filtering by Batting Team on the Batters page seems correct based on the data structure.)
# Since we are filtering by *Bowler*, we need the list of BattingTeams they faced. 
# We'll stick to filtering the data itself for consistency, assuming BattingTeam is still a useful metric.
all_teams = ["All"] + sorted(df_seam_base["BattingTeam"].dropna().unique().tolist())
all_bowlers = ["All"] + sorted(df_seam_base["BowlerName"].dropna().unique().tolist()) # Assuming 'BowlerName' column exists

# 1. Bowling Team Filter (using BattingTeam for simplicity, or assume 'BowlingTeam' exists)
# NOTE: The provided column list only contains 'BattingTeam'. If a 'BowlingTeam' column exists 
# in the actual uploaded data, you should use that instead. Sticking to 'BattingTeam' for now.
with filter_col1:
    # A dedicated 'BowlingTeam' column would be better here if available.
    bat_team = st.selectbox("Opponent Batting Team", all_teams, index=0)

# 2. Bowler Name Filter 
with filter_col2:
    bowler = st.selectbox("Bowler Name", all_bowlers, index=0)

# =========================================================
# 4. Apply Filters to the Base Seam Data
# =========================================================
df_filtered = df_seam_base.copy()

if bat_team != "All":
    df_filtered = df_filtered[df_filtered["BattingTeam"] == bat_team]
    
# Apply Bowler filter (assuming 'BowlerName' column exists)
if bowler != "All":
    if "BowlerName" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["BowlerName"] == bowler]
    else:
        st.warning("BowlerName column not found for filtering.")
