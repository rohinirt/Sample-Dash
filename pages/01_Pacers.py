import streamlit as st
import pandas as pd
# Import the new function and necessary existing functions from utils
from utils import (
    create_seam_length_distribution, 
    create_pitch_map, 
    create_interception_side_on, 
    create_directional_split, 
    create_crease_beehive,
    create_zonal_analysis
    # Add any other required chart functions from utils here
)

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

# =========================================================
# 5. INITIAL CHART LAYOUT
# =========================================================

heading_text = bowler.upper() if bowler != "All" else "SEAM GLOBAL ANALYSIS"
st.header(f"**{heading_text}**")
st.markdown(f"Total Seam Deliveries in Selection: **{len(df_filtered)}**")

st.divider()

# Row 1: Pitch Length Distribution (New Chart)
st.markdown("### Pitch Length Analysis")
st.pyplot(create_seam_length_distribution(df_filtered), use_container_width=True)


# Row 2: Beehive and Pitch Map (Reusing Batter Charts)
st.markdown("### Line & Length Performance")
col_beehive, col_pitch_map = st.columns(2)

with col_beehive:
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_crease_beehive(df_filtered, "Seam"), use_container_width=True)

with col_pitch_map:
    st.markdown("###### PITCHMAP (BOUNCE LOCATION)")
    st.plotly_chart(create_pitch_map(df_filtered, "Seam"), use_container_width=True)


# Row 3: Interception Side-On and Directional Split
st.markdown("### Interception and Movement")
col_interception, col_directional = st.columns(2)

with col_interception:
    st.markdown("###### INTERCEPTION SIDE-ON (HEIGHT)")
    st.pyplot(create_interception_side_on(df_filtered, "Seam"), use_container_width=True)

with col_directional:
    st.markdown("###### SWING DIRECTION")
    st.pyplot(create_directional_split(df_filtered, "Swing", "Swing", "Seam"), use_container_width=True)

# You can continue adding more sections here, reusing functions like 
# create_zonal_analysis, create_lateral_performance_boxes, etc. 
# to view the bowler's data from the batsman's perspective.
