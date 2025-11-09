# PAGE SETUP AND FILTERING
# =========================================================

st.set_page_config(
    layout="wide"
)

# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']

# 2. BASE FILTER: ONLY SPIN DELIVERIES
df_spin_base = df_raw[df_raw["DeliveryType"] == "Spin"]

st.title("PACERS")

# 3. FILTERS (Bowling Team and Bowler)
filter_col1, filter_col2 = st.columns(2) 

# --- Filter Logic ---
if "BowlingTeam" in df_spin_base.columns:
    team_column = "BowlingTeam"
else:
    team_column = "BattingTeam" 
    st.warning("The 'BowlingTeam' column was not found. Displaying all Batting Teams as a fallback.")

all_teams = ["All"] + sorted(df_spin_base[team_column].dropna().unique().tolist())
all_bowlers = ["All"] + sorted(df_spin_base["BowlerName"].dropna().unique().tolist()) 

with filter_col1:
    bowl_team = st.selectbox("Bowling Team", all_teams, index=0)

with filter_col2:
    bowler = st.selectbox("Bowler Name", all_bowlers, index=0)
st.header(f"{bowler}")
# 4. Apply Filters to the Base spin Data
df_filtered = df_spin_base.copy()

if bowl_team != "All":
    df_filtered = df_filtered[df_filtered[team_column] == bowl_team]
    
if bowler != "All":
    if "BowlerName" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["BowlerName"] == bowler]
    else:
        st.warning("BowlerName column not found for filtering.")

# =========================================================
# 5. SPLIT AND DISPLAY CHARTS (RHB vs LHB) 🏏
# =========================================================

# Check for the required column to split the data
if "IsBatsmanRightHanded" not in df_filtered.columns:
    st.error("Cannot split data by handedness: 'IsBatsmanRightHanded' column is missing.")
    st.stop()

# --- Data Split ---
# True is Right-Handed (RHB), False is Left-Handed (LHB)
df_rhb = df_filtered[df_filtered["IsBatsmanRightHanded"] == True]
df_lhb = df_filtered[df_filtered["IsBatsmanRightHanded"] == False]

st.divider()

# --- Display Layout ---
col_rhb, col_lhb = st.columns(2)
