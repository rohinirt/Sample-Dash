import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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
    ax_boxes.set_title(f"Lateral Bowling Performance vs. {handedness_label}", fontsize=12, fontweight='bold')
    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(0, 1)
    ax_boxes.axis('off') 

    plt.tight_layout(pad=0.5)
    return fig_boxes

# Place this function inside pages/Pacers.py, along with create_pacer_crease_beehive
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
    ax_boxes.set_title(f"Lateral Bowling Performance vs. {handedness_label}", fontsize=12, fontweight='bold')
    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(0, 1)
    ax_boxes.axis('off') 

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
    
    # Placeholder for other charts (e.g., Pitch Map, Interception, etc.)
    st.markdown("---")
    st.info("Additional charts for RHB go here.")


# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
with col_lhb:
    st.markdown("### 👤 Left-Handed Batsmen (LHB)")

    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE (IMPACT LOCATION)")
    st.plotly_chart(create_pacer_crease_beehive(df_lhb, "LHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    st.pyplot(create_pacer_lateral_performance_boxes(df_lhb, "LHB"), use_container_width=True)

    # Placeholder for other charts (e.g., Pitch Map, Interception, etc.)
    st.markdown("---")
    st.info("Additional charts for LHB go here.")
