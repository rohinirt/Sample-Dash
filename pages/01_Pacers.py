import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# Chart 1: CREASE BEEHIVE
# =========================================================

def create_pacer_crease_beehive(df_in, handedness_label):
    # This function is the equivalent of the original create_crease_beehive, 
    # but renamed and used here to specifically avoid the 'utils' file.
    
    if df_in.empty:
        fig = go.Figure().update_layout(
            title=f"No data for Beehive ({handedness_label})", height=400,
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return fig

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]

    # Boundaries (Runs = 4 or 6) from Non-Wickets
    boundaries = non_wickets_all[
        (non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)
    ]
    
    # Regular Balls (Runs != 4 and Runs != 6)
    regular_balls = non_wickets_all[
        (non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)
    ]
    
    fig_cbh = go.Figure()

    # 1. TRACE: Regular Balls (Non-Wicket, Non-Boundary) - Light Grey
    fig_cbh.add_trace(go.Scatter(
        x=regular_balls["CreaseY"], y=regular_balls["CreaseZ"], mode='markers', name="Regular Ball",
        marker=dict(color='lightgrey', size=10, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # 2. NEW TRACE: Boundary Balls (Runs 4 or 6) - Royal Blue
    fig_cbh.add_trace(go.Scatter(
        x=boundaries["CreaseY"], y=boundaries["CreaseZ"], mode='markers', name="Boundary",
        marker=dict(color='royalblue', size=12, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # 3. TRACE: Wickets - Red
    fig_cbh.add_trace(go.Scatter(
        x=wickets["CreaseY"], y=wickets["CreaseZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=12, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # Stump lines & Crease lines
    fig_cbh.add_vline(x=-0.18, line=dict(color="grey", dash="dot", width=0.5)) 
    fig_cbh.add_vline(x=0.18, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=0, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.5))
    fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_annotation(
        x=-1.5, y=0.78, text="Stump line", showarrow=False,
        font=dict(size=8, color="grey"), xanchor='left', yanchor='bottom'
    )
    
    # Layout update
    fig_cbh.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 2], showgrid=False, zeroline=True, visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        title=f"Crease Beehive vs. {handedness_label}" # Add title
    )
    
    return fig_cbh


# =========================================================
# PAGE SETUP AND FILTERING
# =========================================================

st.set_page_config(
    page_title="Pacers Dashboard",
    layout="wide"
)

# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']

# 2. BASE FILTER: ONLY SEAM DELIVERIES
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]

st.title("⚡ Pacers Dashboard (Seam Bowling Analysis)")

# 3. FILTERS (Bowling Team and Bowler)
filter_col1, filter_col2 = st.columns(2) 

# --- Filter Logic ---
if "BowlingTeam" in df_seam_base.columns:
    team_column = "BowlingTeam"
else:
    team_column = "BattingTeam" 
    st.warning("The 'BowlingTeam' column was not found. Displaying all Batting Teams as a fallback.")

all_teams = ["All"] + sorted(df_seam_base[team_column].dropna().unique().tolist())
all_bowlers = ["All"] + sorted(df_seam_base["BowlerName"].dropna().unique().tolist()) 

with filter_col1:
    bowl_team = st.selectbox("Bowling Team", all_teams, index=0)

with filter_col2:
    bowler = st.selectbox("Bowler Name", all_bowlers, index=0)

# 4. Apply Filters to the Base Seam Data
df_filtered = df_seam_base.copy()

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

st.header(f"**Analysis Split: Right-Handed vs. Left-Handed Batsmen**")
st.markdown(f"**RHB Deliveries:** {len(df_rhb)} | **LHB Deliveries:** {len(df_lhb)}")
st.divider()

# --- Display Layout ---
col_rhb, col_lhb = st.columns(2)

# === LEFT COLUMN: AGAINST RIGHT-HANDED BATSMEN (RHB) ===
with col_rhb:
    st.markdown("### 🧍 Right-Handed Batsmen (RHB)")
    
    # Chart 1: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_pacer_crease_beehive(df_rhb, "RHB"), use_container_width=True)

    # Placeholder for other charts (e.g., Pitch Map, Interception, etc.)
    st.markdown("---")
    st.info("Additional charts for RHB go here.")


# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
with col_lhb:
    st.markdown("### 👤 Left-Handed Batsmen (LHB)")

    # Chart 1: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_pacer_crease_beehive(df_lhb, "LHB"), use_container_width=True)

    # Placeholder for other charts (e.g., Pitch Map, Interception, etc.)
    st.markdown("---")
    st.info("Additional charts for LHB go here.")
