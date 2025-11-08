import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO
import base64
import matplotlib.patheffects as pe
from matplotlib import cm, colors, patches
import matplotlib.colors as mcolors


# =========================================================
# Chart 1a: CREASE BEEHIVE
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
    regular_balls = df_in[df_in["Wicket"] == False]
    

    
    fig_cbh = go.Figure()

    # 1. TRACE: Regular Balls (Non-Wicket, Non-Boundary) - Light Grey
    fig_cbh.add_trace(go.Scatter(
        x=regular_balls["CreaseY"], y=regular_balls["CreaseZ"], mode='markers', name="Regular Ball",
        marker=dict(color='lightgrey', size=10, line=dict(width=1, color="white"), opacity=0.95)
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
    )
    
    return fig_cbh

# =========================================================
# Chart 1b: CREASE BEEHIVE (Lateral Performace)
# =========================================================
def create_pacer_lateral_performance_boxes(df_in, handedness_label):
    from matplotlib import cm, colors, patches
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors # Explicitly import mcolors

    df_lateral = df_in.copy()
    if df_lateral.empty:
        fig, ax = plt.subplots(figsize=(7, 1)); ax.text(0.5, 0.5, f"No Data ({handedness_label})", ha='center', va='center'); ax.axis('off'); return fig    

    # 1. Define Zoning Logic (Same as before)
    def assign_lateral_zone(row):
        y = row["CreaseY"]
        if row["IsBatsmanRightHanded"] == True:
            if y > 0.18: return "LEG"
            elif y >= -0.18: return "STUMPS"
            elif y > -0.65: return "OUTSIDE OFF"
            else: return "WAY OUTSIDE OFF"
        else: # Left-Handed
            if y > 0.65: return "WAY OUTSIDE OFF"
            elif y > 0.18: return "OUTSIDE OFF"
            elif y >= -0.18: return "STUMPS"
            else: return "LEG"
    
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    # 2. Calculate Summary Metrics (This is Bowling Average logic)
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), 
            Wickets=("Wicket", lambda x: (x == True).sum()), 
            Balls=("Wicket", "count")
        )
    )
    
    # Order the zones from Way Outside Off to Leg (Left to Right)
    ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    summary = summary.reindex(ordered_zones).fillna(0)

    # Calculate Bowling Average (Runs / Wickets)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    
    # 3. Chart Setup
    fig_boxes, ax_boxes = plt.subplots(figsize=(7, 1)) 
    
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions # Fixed width for each box (total width = 1)
    left = 0
    
    # Color Normalization (based on Average)
    avg_values = summary["Avg Runs/Wicket"]
    # Normalize color range: Max Avg is capped at 50 for consistent color scaling
    avg_max_cap = 50 
    norm = mcolors.Normalize(vmin=0, vmax=avg_max_cap)
    cmap = cm.get_cmap('Reds') # Lower average (better bowling) is usually darker/redder
    
    # 4. Plotting Equal Boxes (Horizontal Heatmap)
    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
        
        # Determine color
        color = cmap(norm(avg)) if row["Balls"] > 0 else 'whitesmoke' 
        
        # Draw the Rectangle
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, 1, 
                              edgecolor="black", facecolor=color, linewidth=1)
        )
        
        # Add labels (Zone Name, Wickets, Average)
        label_wkts_avg = f"{wkts}W - Ave {avg:.1f}"
        
        # Calculate text color for contrast
        if row["Balls"] > 0:
            r, g, b, a = color
            # Calculate luminosity for text contrast
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        else:
            text_color = 'black' 

        # Label 1: Zone Name (Top of the box)
        ax_boxes.text(left + box_width / 2, 0.75, 
                      index,
                      ha='center', va='center', fontsize=10, color=text_color)
                      
        # Label 2: Wickets and Average (Middle of the box)
        ax_boxes.text(left + box_width / 2, 0.4, 
                      label_wkts_avg,
                      ha='center', va='center', fontsize= 10, fontweight = 'bold', color=text_color)
        
        left += box_width
        
    # 5. Styling
    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(0, 1)
    ax_boxes.axis('off') 

    plt.tight_layout(pad=0.5)
    return fig_boxes

# --- CHART 2b: LATERAL PERFORMANCE BOXES (BOWLING AVERAGE) ---

def create_pacer_lateral_performance_boxes(df_in, handedness_label):
    # This function now correctly reverses the lateral zones for LHB for visual consistency.
    df_lateral = df_in.copy()
    
    # Check if we are dealing with LHB data (important for zone ordering)
    is_lhb = handedness_label == "LHB"

    if df_lateral.empty:
        fig, ax = plt.subplots(figsize=(7, 1)); ax.text(0.5, 0.5, f"No Data ({handedness_label})", ha='center', va='center'); ax.axis('off'); return fig    

    # 1. Define Zoning Logic (Same as before)
    def assign_lateral_zone(row):
        y = row["CreaseY"]
        if row["IsBatsmanRightHanded"] == True:
            # RHB: Left side of pitch is Off (negative Y), Right side is Leg (positive Y)
            if y > 0.18: return "LEG"
            elif y >= -0.18: return "STUMPS"
            elif y > -0.65: return "OUTSIDE OFF"
            else: return "WAY OUTSIDE OFF"
        else: # Left-Handed
            # LHB: Left side of pitch is Leg (negative Y), Right side is Off (positive Y)
            if y > 0.65: return "WAY OUTSIDE OFF" # Off side
            elif y > 0.18: return "OUTSIDE OFF"
            elif y >= -0.18: return "STUMPS"
            else: return "LEG" # Leg side
    
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    # 2. Calculate Summary Metrics
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), 
            Wickets=("Wicket", lambda x: (x == True).sum()), 
            Balls=("Wicket", "count")
        )
    )
    
    # 3. Determine Zone Order based on handedness
    # Base order (RHB): Off side to Leg side
    base_ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    
    if is_lhb:
        # Reverse order for LHB: Leg side to Off side
        ordered_zones = base_ordered_zones[::-1]
    else:
        ordered_zones = base_ordered_zones
        
    summary = summary.reindex(ordered_zones).fillna(0)

    # Calculate Bowling Average (Runs / Wickets)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    
    # 4. Chart Setup
    fig_boxes, ax_boxes = plt.subplots(figsize=(7, 1)) 
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions # Fixed width for each box (total width = 1)
    left = 0
    
    # Color Normalization (based on Average)
    avg_values = summary["Avg Runs/Wicket"]
    avg_max_cap = 50 
    norm = mcolors.Normalize(vmin=0, vmax=avg_max_cap)
    cmap = cm.get_cmap('Reds') 
    
    # 5. Plotting Equal Boxes (Horizontal Heatmap)
    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
        
        # Determine color
        color = cmap(norm(avg)) if row["Balls"] > 0 else 'whitesmoke' 
        
        # Draw the Rectangle
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, 1, 
                              edgecolor="black", facecolor=color, linewidth=1)
        )
        
        # Add labels (Zone Name, Wickets, Average)
        label_wkts_avg = f"{wkts}W - Ave {avg:.1f}"
        
        # Calculate text color for contrast
        if row["Balls"] > 0:
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        else:
            text_color = 'black' 

        # Label 1: Zone Name (Top of the box)
        ax_boxes.text(left + box_width / 2, 0.75, 
                      index,
                      ha='center', va='center', fontsize=10, color=text_color)
                      
        # Label 2: Wickets and Average (Middle of the box)
        ax_boxes.text(left + box_width / 2, 0.4, 
                      label_wkts_avg,
                      ha='center', va='center', fontsize= 10, fontweight = 'bold', color=text_color)
        
        left += box_width
        
    # 6. Styling
    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(0, 1)
    ax_boxes.axis('off') 

    plt.tight_layout(pad=0.5)
    return fig_boxes
    
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import matplotlib.colors as mcolors
import pandas as pd

# --- CHART 2: ZONAL ANALYSIS (CBH Boxes) ---
def create_pacer_zonal_analysis(df_in, handedness_label):
    if df_in.empty:
        # Assuming plt is imported
        fig, ax = plt.subplots(figsize=(3, 2)); ax.text(0.5, 0.5, f"No Data ({handedness_label})", ha='center', va='center'); ax.axis('off'); return fig

    # 1. Handedness Determination and Zone Layout
    is_right_handed_batsman = (handedness_label == "RHB")
    
    # RHB Zones: Off side (negative Y) to Leg side (positive Y)
    # Y-axis (lateral): -0.72 (Wide Off) to 0.72 (Wide Leg)
    # Z-axis (height): 0 (Stump Base) to 1.91 (Above Head)
    right_hand_zones = { "Z1": (-0.72, 0, -0.45, 1.91), "Z2": (-0.45, 0, -0.18, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (-0.45, 0.71, -0.18, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.18, 1.91)}
    
    # LHB Zones: Leg side (negative Y) to Off side (positive Y)
    left_hand_zones = { "Z1": (0.45, 0, 0.72, 1.91), "Z2": (0.18, 0, 0.45, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (0.18, 0.71, 0.45, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.18, 1.91)}
    
    zones_layout = right_hand_zones if is_right_handed_batsman else left_hand_zones
        
    def assign_zone(row):
        x, y = row["CreaseY"], row["CreaseZ"]
        for zone, (x1, y1, x2, y2) in zones_layout.items():
            if x1 <= x <= x2 and y1 <= y <= y2: return zone
        return "Other"

    df_chart2 = df_in.copy(); 
    df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
    df_chart2 = df_chart2[df_chart2["Zone"] != "Other"]
        
    # 2. Calculate Summary Metrics (Bowler Focus)
    summary = (
        df_chart2.groupby("Zone").agg(
            Runs=("Runs", "sum"), 
            Wickets=("Wicket", lambda x: (x == True).sum()), 
            Balls=("Wicket", "count")
        )
        .reindex([f"Z{i}" for i in range(1, 7)]).fillna(0)
    )
    
    # Bowling Average (Runs / Wickets)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    
    # Bowling Strike Rate (Balls / Wickets)
    summary["BowlingSR"] = summary.apply(lambda row: row["Balls"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)

    # 3. Color Scaling (Based on Bowling Average)
    avg_values = summary["Avg Runs/Wicket"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 100 
    avg_max_cap = 50 # Cap max at 50 for visualization consistency
    
    # Assuming mcolors is imported
    norm = mcolors.Normalize(vmin=0, vmax=avg_max if avg_max > avg_max_cap else avg_max_cap)
    # Assuming cm is imported
    cmap = cm.get_cmap('Reds_r') # Higher Average (worse bowling) is darker red

    # Assuming plt and patches are imported
    fig_boxes, ax = plt.subplots(figsize=(3,2), subplot_kw={'xticks': [], 'yticks': []}) 
        
    # 4. Plotting Zones and Labels
    for zone, (x1, y1, x2, y2) in zones_layout.items():
        w, h = x2 - x1, y2 - y1
        z_key = zone.replace("Zone ", "Z")
            
        runs, wkts, bowling_avg, bowling_sr = (0, 0, 0, 0)
        
        if z_key in summary.index:
            runs = int(summary.loc[z_key, "Runs"])
            wkts = int(summary.loc[z_key, "Wickets"])
            bowling_avg = summary.loc[z_key, "Avg Runs/Wicket"]
            bowling_sr = summary.loc[z_key, "BowlingSR"]
            
        color = cmap(norm(bowling_avg)) if summary.loc[z_key, "Balls"] > 0 else 'white'

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=0.8))
        
        # Calculate text color for contrast
        text_color = "black"
        if summary.loc[z_key, "Balls"] > 0:
            r, g, b, a = color
            # Simple luminance check for text contrast
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'

        ax.text(x1 + w / 2, y1 + h / 2, 
        f"W: {wkts}\nSR: {bowling_sr:.1f}\nA: {bowling_avg:.1f}", 
        ha="center", 
        va="center", 
        fontsize=5,
        color=text_color, 
        linespacing=1.2)
        
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); 
    plt.tight_layout(pad=0.5) 
    return fig_boxes
    
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
    
    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_pacer_crease_beehive(df_rhb, "RHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    st.pyplot(create_pacer_lateral_performance_boxes(df_rhb, "RHB"), use_container_width=True)
    
    # Chart 2: ZONAL ANALYSIS (CBH Boxes)
    st.pyplot(create_pacer_zonal_analysis(df_rhb, "RHB"), use_container_width=True)


# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
with col_lhb:
    st.markdown("### 👤 Left-Handed Batsmen (LHB)")

    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_pacer_crease_beehive(df_lhb, "LHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    st.pyplot(create_pacer_lateral_performance_boxes(df_lhb, "LHB"), use_container_width=True)

    # Chart 2: ZONAL ANALYSIS (CBH Boxes)
    st.pyplot(create_pacer_zonal_analysis(df_lhb, "LHB"), use_container_width=True)
