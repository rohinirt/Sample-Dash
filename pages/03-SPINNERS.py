import streamlit as st
import pandas as pd


# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']

# 2. BASE FILTER: ONLY SPIN DELIVERIES
df_spin_base = df_raw[df_raw["DeliveryType"] == "Spin"]

# Overall Dashboard Title
st.title("🧶 Spinners Dashboard (Spin Bowling Analysis)")

# Check if there is any spin data available
if df_spin_base.empty:
    st.warning("No deliveries categorized as 'Spin' found in the uploaded data.")
    st.stop()

# 3. FILTERS (Bowling Team and Bowler)
filter_col1, filter_col2 = st.columns(2) 

# --- Filter Logic ---
# Determine which column to use for Team filtering
if "BowlingTeam" in df_spin_base.columns:
    team_column = "BowlingTeam"
else:
    team_column = "BattingTeam" 
    st.warning("The 'BowlingTeam' column was not found. Displaying all Batting Teams as a fallback.")

all_teams = ["All"] + sorted(df_spin_base[team_column].dropna().unique().tolist())

with filter_col1:
    bowl_team = st.selectbox("Bowling Team", all_teams, index=0)

# Filter bowler list based on selected team
if bowl_team != "All":
    filtered_bowlers_df = df_spin_base[df_spin_base[team_column] == bowl_team]
else:
    filtered_bowlers_df = df_spin_base
    
filtered_bowlers = ["All"] + sorted(filtered_bowlers_df["BowlerName"].dropna().unique().tolist())

with filter_col2:
    bowler = st.selectbox("Bowler Name", filtered_bowlers, index=0)

# 4. Display Selected Bowler
# If a specific bowler is selected, show their name prominently
if bowler != "All":
    st.header(f"Selected Bowler: {bowler}")
else:
    st.header(f"Analysis for: {bowl_team} Bowlers")

# 5. Final Filtered DataFrame
# This is the DataFrame you will use for all your spinner chart functions below
df_spin_filtered = df_spin_base.copy()

if bowl_team != "All":
    df_spin_filtered = df_spin_filtered[df_spin_filtered[team_column] == bowl_team]
    
if bowler != "All":
    df_spin_filtered = df_spin_filtered[df_spin_filtered["BowlerName"] == bowler]


# --- START CHART CALLS HERE, using df_spin_filtered ---

# Example:
# st.markdown("### Spin Analysis Charts")
# st.pyplot(create_spinner_pitch_map(df_spin_filtered))
