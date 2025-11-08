import streamlit as st
import pandas as pd
# 💥 Import ALL chart functions from the centralized utility file
from utils import (
    create_zonal_analysis, create_crease_beehive, create_lateral_performance_boxes,
    create_pitch_map, create_pitch_length_run_pct, create_interception_side_on,
    create_crease_width_split, create_interception_front_on, create_wagon_wheel, 
    create_left_right_split, create_directional_split
)

# Set page title (optional, but good practice)
st.set_page_config(
    page_title="BATTERS",
    layout="wide"
)

# =========================================================
# 💥 1. CRITICAL: GET DATA FROM SESSION STATE
# This check ensures the page cannot run without data uploaded via Home.py
# =========================================================
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    # Stop execution of the rest of the script if data is missing
    st.stop()
    
# Retrieve the full raw DataFrame
df_raw = st.session_state['data_df']

# =========================================================
# 🌟 FILTERS 🌟
# =========================================================
# Use columns to align the four filters horizontally
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4) 

# --- Filter Logic ---
all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())

# 1. Batting Team Filter (in column 1)
with filter_col1:
    bat_team = st.selectbox("Batting Team", all_teams, index=0)

# 2. Batsman Name Filter (Logic depends on Batting Team - in column 2)
if bat_team != "All":
    # Filter batsmen based on selected team
    batsmen_options = ["All"] + sorted(df_raw[df_raw["BattingTeam"] == bat_team]["BatsmanName"].dropna().unique().tolist())
else:
    # Show all batsmen if 'All' teams is selected
    batsmen_options = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())
    
with filter_col2:
    batsman = st.selectbox("Batsman Name", batsmen_options, index=0)

# 3. Innings Filter (in column 3)
# Check if 'Innings' column exists before creating options (Robustness)
if "Innings" in df_raw.columns:
    innings_options = ["All"] + sorted(df_raw["Innings"].dropna().unique().tolist())
    with filter_col3:
        selected_innings = st.selectbox("Innings", innings_options, index=0)
else:
    selected_innings = "All" # Default if column is missing
    with filter_col3:
        st.info("Innings filter unavailable.")

# 4. Bowler Hand Filter (in column 4)
# Check if 'IsBowlerRightHanded' column exists (CRITICAL FIX)
if "IsBowlerRightHanded" in df_raw.columns:
    bowler_hand_options = ["All", "Right Hand", "Left Hand"]
    with filter_col4:
        selected_bowler_hand = st.selectbox("Bowler Hand", bowler_hand_options, index=0)
else:
    selected_bowler_hand = "All" # Default if column is missing
    with filter_col4:
        st.info("Bowler Hand filter unavailable.")
    
# =========================================================

# --- Apply Filters to the Raw dataframes ---

def apply_filters(df):
    df_filtered = df.copy() # Work on a copy of the sub-dataframes

    if bat_team != "All":
        df_filtered = df_filtered[df_filtered["BattingTeam"] == bat_team]
        
    if batsman != "All":
        df_filtered = df_filtered[df_filtered["BatsmanName"] == batsman]
        
    # Apply Innings Filter (Only if column exists)
    if selected_innings != "All" and "Innings" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Innings"] == selected_innings]
        
    # Apply Bowler Hand Filter (Only if column exists)
    if selected_bowler_hand != "All" and "IsBowlerRightHanded" in df_filtered.columns:
        # True for Right Hand, False for Left Hand
        is_right = (selected_bowler_hand == "Right Hand") 
        df_filtered = df_filtered[df_filtered["IsBowlerRightHanded"] == is_right]
        
    return df_filtered

# Separate by delivery type BEFORE filtering to save a little processing, then apply filters
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]
df_spin_base = df_raw[df_raw["DeliveryType"] == "Spin"]

# Apply filters
df_seam = apply_filters(df_seam_base)
df_spin = apply_filters(df_spin_base)
    
heading_text = batsman.upper() if batsman != "All" else "GLOBAL ANALYSIS"
st.header(f"**{heading_text}**")

# --- 4. DISPLAY CHARTS IN TWO COLUMNS (SEAM vs. SPIN) ---
st.divider()

col1, col2 = st.columns(2)
    
# --- LEFT COLUMN: SEAM ANALYSIS ---
with col1:
    st.markdown("### SEAM")

    # Row 1: Zonal Analysis (Beehive Zones)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_zonal_analysis(df_seam, batsman, "Seam"), use_container_width=True)
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE")
    st.plotly_chart(create_crease_beehive(df_seam, "Seam"), use_container_width=True)

    # Row 3: Lateral Performance Boxes
    st.pyplot(create_lateral_performance_boxes(df_seam, "Seam", batsman), use_container_width=True)
    
    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_map_col, run_pct_col = st.columns([3, 1]) # 3:1 ratio

    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.plotly_chart(create_pitch_map(df_seam, "Seam"), use_container_width=True)    
    with run_pct_col:
        st.markdown("###### ")
        st.pyplot(create_pitch_length_run_pct(df_seam, "Seam"), use_container_width=True)
        
    st.divider()

    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-ON")
    st.pyplot(create_interception_side_on(df_seam, "Seam"), use_container_width=True)
    # Row 6: Interception Side-On Bins (Length Bins)
    st.pyplot(create_crease_width_split(df_seam, "Seam"), use_container_width=True)

    # Row 7: Interception Front-On and Scoring Areas (Side-by-Side)
    bottom_col_left, bottom_col_right = st.columns(2)
    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-ON")
        st.pyplot(create_interception_front_on(df_seam, "Seam"), use_container_width=True)
        
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")    
        # Two charts stacked vertically in the right column
        st.pyplot(create_wagon_wheel(df_seam, "Seam"), use_container_width=True)
        st.pyplot(create_left_right_split(df_seam, "Seam"), use_container_width=True)
        
    st.divider()
    
    # Row 8: Swing/Deviation Direction Analysis (Side-by-Side)
    final_col_swing, final_col_deviation = st.columns(2)

    with final_col_swing:
        st.markdown("###### SWING")
        st.pyplot(create_directional_split(df_seam, "Swing", "Swing", "Seam"), use_container_width=True)

    with final_col_deviation:
        st.markdown("###### DEVIATION")
        st.pyplot(create_directional_split(df_seam, "Deviation", "Deviation", "Seam"), use_container_width=True)    


# --- RIGHT COLUMN: SPIN ANALYSIS ---
with col2:
    st.markdown("### SPINE")
    
    # Row 1: Zonal Analysis (Beehive Zones)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_zonal_analysis(df_spin, batsman, "Spin"), use_container_width=True)
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE")
    st.plotly_chart(create_crease_beehive(df_spin, "Spin"), use_container_width=True)
    # Row 3: Lateral Performance Boxes
    st.pyplot(create_lateral_performance_boxes(df_spin, "Spin", batsman), use_container_width=True)

    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_map_col, run_pct_col = st.columns([3, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.plotly_chart(create_pitch_map(df_spin, "Spin"), use_container_width=True)    
        
    with run_pct_col:
        st.markdown("###### ")
        st.pyplot(create_pitch_length_run_pct(df_spin, "Spin"), use_container_width=True)
        
    st.divider()
    
    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-ON")
    st.pyplot(create_interception_side_on(df_spin, "Spin"), use_container_width=True)

    # Row 6: Interception Side-On Bins (Length Bins)
    st.pyplot(create_crease_width_split(df_spin, "Spin"), use_container_width=True)

    # Row 7: Interception Front-On and Scoring Areas (Side-by-Side)
    bottom_col_left, bottom_col_right = st.columns(2)

    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-ON")
        st.pyplot(create_interception_front_on(df_spin, "Spin"), use_container_width=True)
        
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")
        st.pyplot(create_wagon_wheel(df_spin, "Spin"), use_container_width=True)
        st.pyplot(create_left_right_split(df_spin, "Spin"), use_container_width=True)
            
    st.divider()

    # Row 8: Swing/Deviation Direction Analysis (Side-by-Side)
    final_col_swing, final_col_deviation = st.columns(2)

    with final_col_swing:
        st.markdown("###### DRIFT")
        # For spin, we often look at 'Drift' instead of 'Swing'
        st.pyplot(create_directional_split(df_spin, "Swing", "Drift", "Spin"), use_container_width=True)

    with final_col_deviation:
        st.markdown("###### TURN")
        # For spin, we often look at 'Turn' instead of 'Deviation'
        st.pyplot(create_directional_split(df_spin, "Deviation", "Turn", "Spin"), use_container_width=True)
