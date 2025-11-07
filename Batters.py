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

# --- 1. GLOBAL UTILITY FUNCTIONS ---

# Required columns check
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over"
]

# Function to encode Matplotlib figure to image for Streamlit
def fig_to_image(fig):
    return fig

# --- CHART 1: ZONAL ANALYSIS (CBH Boxes) ---
def create_zonal_analysis(df_in, batsman_name, delivery_type):
    # ... (Zonal Analysis logic remains the same)
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig

    is_right_handed = True
    handed_data = df_in["IsBatsmanRightHanded"].dropna().unique()
    if len(handed_data) > 0 and batsman_name != "All": is_right_handed = handed_data[0]
        
    right_hand_zones = { "Z1": (-0.72, 0, -0.45, 1.91), "Z2": (-0.45, 0, -0.18, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (-0.45, 0.71, -0.18, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.45, 1.31, 0.18, 1.91)}
    left_hand_zones = { "Z1": (0.45, 0, 0.72, 1.91), "Z2": (0.18, 0, 0.45, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (0.18, 0.71, 0.45, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.45, 1.91)}
    zones_layout = right_hand_zones if is_right_handed else left_hand_zones
    
    def assign_zone(row):
        x, y = row["CreaseY"], row["CreaseZ"]
        for zone, (x1, y1, x2, y2) in zones_layout.items():
            if x1 <= x <= x2 and y1 <= y <= y2: return zone
        return "Other"

    df_chart2 = df_in.copy(); df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
    df_chart2 = df_chart2[df_chart2["Zone"] != "Other"]
    
    summary = (
        df_chart2.groupby("Zone").agg(Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count"))
        .reindex([f"Z{i}" for i in range(1, 7)]).fillna(0)
    )
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    summary["StrikeRate"] = summary.apply(lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1)

    avg_values = summary["Avg Runs/Wicket"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 1
    avg_min = avg_values[avg_values > 0].min() if avg_values[avg_values > 0].min() < avg_max else 0
    norm = mcolors.Normalize(vmin=avg_min, vmax=avg_max)
    cmap = cm.get_cmap('Reds')

    fig_boxes, ax = plt.subplots(figsize=(3,2), subplot_kw={'xticks': [], 'yticks': []}) 
    
    for zone, (x1, y1, x2, y2) in zones_layout.items():
        w, h = x2 - x1, y2 - y1
        z_key = zone.replace("Zone ", "Z")
        
        runs, wkts, avg, sr = (0, 0, 0, 0)
        if z_key in summary.index:
            runs = int(summary.loc[z_key, "Runs"])
            wkts = int(summary.loc[z_key, "Wickets"])
            avg = summary.loc[z_key, "Avg Runs/Wicket"]
            sr = summary.loc[z_key, "StrikeRate"]
        
        color = cmap(norm(avg)) if avg > 0 else 'white'

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=0.8))

        ax.text(x1 + w / 2, y1 + h / 2, 
        f"R: {runs}\nW: {wkts}\nSR: {sr:.0f}\nA: {avg:.0f}", 
        # ===============================================
        ha="center", 
        va="center", 
        fontsize=5,
        color="black" if norm(avg) < 0.6 else "white", 
        linespacing=1.2)
    ax.set_title(f"STRIKE RATE", 
                 fontsize=8, 
                 weight='bold', 
                 pad=10)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); 
    plt.tight_layout(pad=0.5) 
    return fig_boxes

# --- CHART 2a: CREASE BEEHIVE ---
def create_crease_beehive(df_in, delivery_type):
    # ... (Crease Beehive logic remains the same)
    if df_in.empty:
        return go.Figure().update_layout(title="No data for Beehive", height=400)

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    
    # 1. Filter Non-Wickets
    non_wickets_all = df_in[df_in["Wicket"] == False]

    # 2. Filter Boundaries (Runs = 4 or 6) from Non-Wickets
    boundaries = non_wickets_all[
        (non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)
    ]
    
    # 3. Filter Regular Balls (Runs != 4 and Runs != 6)
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

    # 3. TRACE: Wickets - Red (Kept as the largest marker size for emphasis)
    fig_cbh.add_trace(go.Scatter(
        x=wickets["CreaseY"], y=wickets["CreaseZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=12, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # Stump lines & Crease lines (No change)
    fig_cbh.add_vline(x=-0.18, line=dict(color="grey", dash="dot", width=0.5)) 
    fig_cbh.add_vline(x=0.18, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=0, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.5))
    fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_annotation(
        x=-1.5,                 # X-position on the far left
        y=0.78,                 # Y-position (on the line)
        text="Stump line",      # The label text
        showarrow=False,
        font=dict(size=8, color="grey"),
        xanchor='left',         # Anchor text to the left
        yanchor='bottom'        # Place text slightly above the line
    )
    fig_cbh.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 2], showgrid=False, zeroline=True, visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    
    return fig_cbh
    

# --- CHART 2b: LATERAL PERFORMANCE STACKED BAR ---
def create_lateral_performance_boxes(df_in, delivery_type, batsman_name):
    from matplotlib import cm, colors, patches
    
    df_lateral = df_in.copy()
    if df_lateral.empty:
        fig, ax = plt.subplots(figsize=(7, 1)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig     
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
    
    # 2. Calculate Summary Metrics
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

    # Calculate Average for coloring and labeling
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    
    # 3. Chart Setup
    fig_boxes, ax_boxes = plt.subplots(figsize=(7, 1)) 
    
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions # Fixed width for each box (total width = 1)
    left = 0
    
    # Color Normalization (based on Average)
    avg_values = summary["Avg Runs/Wicket"]
    # Normalize color range: Use 0 to 50 for a reasonable cap on average
    avg_max = avg_values.max() if avg_values.max() > 0 else 50
    norm = mcolors.Normalize(vmin=0, vmax=avg_max if avg_max > 50 else 50)
    cmap = cm.get_cmap('Reds') # Use inverted Reds_r: lower avg (good) is darker/redder
    
    # 4. Plotting Equal Boxes (Horizontal Heatmap)
    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
        
        # Color hue based on Average
        # Use white/light color if no data (zero balls)
        color = cmap(norm(avg)) if row["Balls"] > 0 else 'whitesmoke' 
        
        # Draw the Rectangle (Fixed width, full height)
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, 1, 
                              edgecolor="black", facecolor=color, linewidth=1)
        )
        
        # Add labels (Zone Name, Wickets, Average)
        label_wkts_avg = f"{wkts}W - Ave {avg:.1f}"
        
        # Calculate text color for contrast
        if row["Balls"] > 0:
            r, g, b, a = color # This is safe now, as 'color' is an RGBA tuple
            # Calculate luminosity for RGBA tuples (from cmap)
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        else:
            # If the color is 'whitesmoke' (from the initial check), use black text
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
    ax_boxes.axis('off') # Hide all axes/ticks/labels


    # Remove the border (spines)
    ax_boxes.spines['right'].set_visible(False)
    ax_boxes.spines['top'].set_visible(False)
    ax_boxes.spines['left'].set_visible(False)
    ax_boxes.spines['bottom'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    return fig_boxes

# --- CHART 3: PITCH MAP ---

# --- Helper function for Pitch Bins (Centralized) ---
def get_pitch_bins(delivery_type):
    if delivery_type == "Seam":
        # Seam Bins: 1.2-6: Full, 6-8 Length, 8-10 Short, 10-15 Bouncer
        return {
            "Full": [1.2, 6.0],
            "Length": [6.0, 8.0],
            "Short": [8.0, 10.0],
            "Bouncer": [10.0, 15.0],
        }
    elif delivery_type == "Spin":
        # Spin Bins: 1.22-2.22: OP, 2.22-4: full, 4-6: Good, 6-15: short
        return {
            "Over Pitched": [1.22, 2.22],
            "Full": [2.22, 4.0],
            "Good": [4.0, 6.0],
            "Short": [6.0, 15.0],
        }
    return {} # Default

# --- CHART 3: PITCH MAP (BOUNCE LOCATION) ---
def create_pitch_map(df_in, delivery_type):
    if df_in.empty:
        return go.Figure().update_layout(title=f"No data for Pitch Map ({delivery_type})", height=300)

    PITCH_BINS = get_pitch_bins(delivery_type)
    
    # Add a catch-all bin for plotting range/data outside the defined bins if needed
    if delivery_type == "Seam":
        PITCH_BINS["Full Toss"] = [-4.0, 1.2] 
    elif delivery_type == "Spin":
        PITCH_BINS["Full Toss"] = [-4.0, 1.22] 
        
    fig_pitch = go.Figure()
    
    # 1. Add Zone Lines & Labels (using y0, which is the start of the zone)
    # The dictionary keys must be sorted to ensure labels appear correctly if not iterating directly
    
    # Determine boundary keys for lines (excluding the start of the lowest bin)
    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0])

    for y_val in boundary_y_values:
        fig_pitch.add_hline(y=y_val, line=dict(color="lightgrey", width=1.0, dash="dot"))

    # Add zone labels
    for length, bounds in PITCH_BINS.items():
        if length != "Full Toss": # Skip full toss for placement of label
            mid_y = (bounds[0] + bounds[1]) / 2
            fig_pitch.add_annotation(x=-1.45, y=mid_y, text=length.upper(), showarrow=False,
                font=dict(size=8, color="grey", weight='bold'), xanchor='left')

    # 2. Add Stump lines
    fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0, line=dict(color="#777777", dash="dot", width=0.8))

    # 3. Plot Data
    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]

    fig_pitch.add_trace(go.Scatter(
        x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"], mode='markers', name="No Wicket",
        marker=dict(color='#D3D3D3', size=10, line=dict(width=1, color="white"), opacity=0.9)
    ))

    fig_pitch.add_trace(go.Scatter(
        x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket",
        marker=dict(color='red', size=12, line=dict(width=1, color="white")), opacity=0.95)
    )

    # 4. Layout
    fig_pitch.update_layout(
        height = 400,
        margin=dict(l=0, r=100, t=0, b=10),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
        # Ensure Y-axis range covers the custom bins
        yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False, visible=False), 
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    
    return fig_pitch

# --- CHART 3b: PITCH LENGTH RUN % (EQUAL SIZED BOXES) ---
def create_pitch_length_run_pct(df_in, delivery_type):
    # Adjust figsize height to accommodate the four boxes and title comfortably
    FIG_HEIGHT = 5.7
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(2, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', rotation=90); 
        ax.axis('off'); 
        return fig

    # Get the pitch bins from the helper function (must be available)
    try:
        PITCH_BINS_DICT = get_pitch_bins(delivery_type)
    except NameError:
        # Fallback if get_pitch_bins is not defined
        fig, ax = plt.subplots(figsize=(2, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "Setup Error", ha='center', va='center', rotation=90); 
        ax.axis('off'); 
        return fig
    
    
    # Define ordered keys for reindexing/plotting order (far to near)
    if delivery_type == "Seam":
        ordered_keys = ["Bouncer", "Short", "Length", "Full"]
        # Use a contrasting colormap (e.g., Reds for high run percentage)
        COLORMAP = 'Reds' 
    elif delivery_type == "Spin":
        ordered_keys = ["Short", "Good", "Full", "Over Pitched"]
        COLORMAP = 'Reds'
    else:
        # Fallback if delivery type is not recognized
        fig, ax = plt.subplots(figsize=(2, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "Invalid Type", ha='center', va='center', rotation=90); 
        ax.axis('off'); 
        return fig

    # 1. Data Preparation
    def assign_pitch_length(x):
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]: return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_length)
    
    # === CRITICAL DATA CHECK ===
    if df_pitch["PitchLength"].isnull().all() or df_pitch.empty:
        fig, ax = plt.subplots(figsize=(2, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "No Pitches Assigned", ha='center', va='center', rotation=90); 
        ax.axis('off'); 
        return fig

    # Aggregate data - ADD RUNS, BALLS, WICKETS
    df_summary = df_pitch.groupby("PitchLength").agg(
        Runs=("Runs", "sum"), 
        Wickets=("Wicket", lambda x: (x == True).sum()), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("PitchLength").reindex(ordered_keys).fillna(0)
    
    # --- CALCULATE AVG and SR ---
    df_summary["Average"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (row["Runs"] if row["Balls"] > 0 else 0), axis=1
    )
    df_summary["StrikeRate"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1
    )
    # -----------------------------
    
    total_runs = df_summary["Runs"].sum()
    df_summary["RunPercentage"] = (df_summary["Runs"] / total_runs) * 100 if total_runs > 0 else 0

    # 2. Chart Setup
    fig_stack, ax_stack = plt.subplots(figsize=(2, FIG_HEIGHT)) 

    # Plotting setup variables
    num_boxes = len(ordered_keys)
    box_height = 1.0 / num_boxes
    bottom = 0.0
    
    # Colormap and Normalization
    max_pct = df_summary["RunPercentage"].max() if df_summary["RunPercentage"].max() > 0 else 100
    norm = mcolors.Normalize(vmin=0, vmax=max_pct)
    cmap = cm.get_cmap(COLORMAP)
    
    # 3. Plotting Equal Boxes (Stacked Heat Map)
    for index, row in df_summary.iterrows():
        pct = row["RunPercentage"]
        avg = row["Average"] 
        sr = row["StrikeRate"] 
        
        # Determine box color
        color = cmap(norm(pct))
        
        # Draw the box (barh with width=1)
        ax_stack.bar( # <-- CORRECTED FUNCTION CALL
            x=0.5,           # X-position (center of the chart)
            height=box_height,
            width=1,         # Full width (from 0 to 1 on the X-axis)
            bottom=bottom,   # Y-start position
            color=color,
            edgecolor='black', 
            linewidth=1
        )
        
        # Add labels - UPDATING LABEL TEXT
        label_text = (
            f"{index}\n"
            f"{pct:.0f}%\n"
            f"Avg: {avg:.1f}\n"
            f"SR: {sr:.1f}"
        )
        
        # Calculate text color for contrast
        # Fixed: Only unpack R, G, B
        r, g, b = color[:3]
        luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = 'white' if luminosity < 0.5 else 'black'

        # Text color and plotting logic remain the same
        ax_stack.text(0.5, bottom + box_height / 2, 
                      label_text,
                      ha='center', va='center', fontsize=12, color=text_color, weight='bold', linespacing=1.2)
        bottom += box_height
        
    # 4. Styling
    ax_stack.set_xlim(0, 1); ax_stack.set_ylim(0, 1)
    ax_stack.axis('off') # Hide all axes/ticks/labels

    # Remove the border (spines)
    ax_stack.spines['right'].set_visible(False)
    ax_stack.spines['top'].set_visible(False)
    ax_stack.spines['left'].set_visible(False)
    ax_stack.spines['bottom'].set_visible(False)
    
    # Title remains hidden for space
    
    plt.tight_layout(pad=0.9)
    return fig_stack
    
# --- CHART 4a: INTERCEPTION SIDE-ON --- (Wide View)
def create_interception_side_on(df_in, delivery_type):
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        fig, ax = plt.subplots(figsize=(1.7, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    fig_7, ax_7 = plt.subplots(figsize=(3, 1.7), subplot_kw={'xticks': [], 'yticks': []}) 
    
    # 1. Plot Data (Layered for correct border visibility)
    
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
    ax_7.scatter(
        df_other["InterceptionX"] + 10, df_other["InterceptionZ"], 
        color='#D3D3D3', edgecolors='white', linewidths=0.3, s=20, label="Other"
    )
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
        ax_7.scatter(
            df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], 
            color=color_map[ctype],edgecolors='white', linewidths=0.3, s=30, label=ctype
        )

    # 2. Draw Vertical Dashed Lines with Labels (FIXED LINES: 0.0, 1.25, 2.0, 3.0)
    line_specs = {
        0.0: "Stumps",
        1.250: "Crease",
        2.000: "2m",     
        3.000: "3m" 
    }
    
    for x_val, label in line_specs.items():
        ax_7.axvline(x=x_val, color='lightgrey', linestyle='--', linewidth=0.6, alpha=0.7)  
        ax_7.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=0.6, alpha=0.7)   
        ax_7.text(x_val, 1.45, label.split(':')[-1].strip(), ha='center', va='center', fontsize=5, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Set Y limit as fixed
    y_limit = 1.5
    
    # Set X limit based on delivery type
    if delivery_type == "Seam":
        x_limit_max = 3.4
    elif delivery_type == "Spin":
        x_limit_max = 4.4
    else:
        # Fallback to the original seam limit if type is unknown
        x_limit_max = 3.4 
        
    x_limit_min = -0.2
    
    ax_7.set_xlim(x_limit_min, x_limit_max) 
    ax_7.set_ylim(0, y_limit) 
    # ... (Rest of the styling remains the same)
    ax_7.tick_params(axis='y', which='both', labelleft=False, left=False); ax_7.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_7.spines['right'].set_visible(False)
    ax_7.spines['top'].set_visible(False)
    ax_7.spines['left'].set_visible(False)
    ax_7.spines['bottom'].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig_7

# Chart 4b: Interception Side on Bins ---
def create_crease_width_split(df_in, delivery_type):
    # Adjust figsize width for horizontal display, height for four boxes
    FIG_WIDTH = 5
    FIG_HEIGHT = 1
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "No Data", ha='center', va='center'); 
        ax.axis('off'); 
        return fig
    
    # 1. Define Interception Bins and Order
    # NOTE: Assuming InterceptionX is centered around 0. We use X+10 for binning.
    
    # Lateral Bins (0m-1m, 1m-2m, 2m-3m, 3m+) based on InterceptionX + 10
    INTERCEPTION_BINS = {
        "0m-1m": [0, 1],
        "1m-2m": [1, 2],
        "2m-3m": [2, 3],
        "3m+": [3, 100] # Assuming max possible value is < 100
    }
    
    # Order: Wide to Close (e.g., 3m+ to 0m-1m)
    ordered_keys = ["0m-1m", "1m-2m", "2m-3m", "3m+"] 
    COLORMAP = 'Reds' # Color hue based on SR

    # 2. Data Preparation
    def assign_crease_width(x):
        # x is assumed to be InterceptionX + 10
        for width, bounds in INTERCEPTION_BINS.items():
            if bounds[0] <= x < bounds[1]: return width
        return None

    df_crease = df_in.copy()
    # Apply the required transformation: InterceptionX + 10
    df_crease["CreaseWidth"] = (df_crease["InterceptionX"] + 10).apply(assign_crease_width)
    
    if df_crease["CreaseWidth"].isnull().all() or df_crease.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "No Crease Width Assigned", ha='center', va='center'); 
        ax.axis('off'); 
        return fig

    # Aggregate data 
    df_summary = df_crease.groupby("CreaseWidth").agg(
        Runs=("Runs", "sum"), 
        Wickets=("Wicket", lambda x: (x == True).sum()), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("CreaseWidth").reindex(ordered_keys).fillna(0)
    
    # --- CALCULATE AVG and SR (Metric for color hue) ---
    df_summary["Average"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (row["Runs"] if row["Balls"] > 0 else 0), axis=1
    )
    df_summary["StrikeRate"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1
    )
    # -----------------------------
    
    # 3. Chart Setup
    fig_stack, ax_stack = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT)) 

    # Plotting setup variables (Now horizontal)
    num_boxes = len(ordered_keys)
    box_width = 1.0 / num_boxes # X-dimension split
    left = 0.0 # X-start position
    
    # Colormap and Normalization based on Strike Rate (SR)
    max_sr = df_summary["StrikeRate"].max() if df_summary["StrikeRate"].max() > 0 else 100
    norm = mcolors.Normalize(vmin=0, vmax=max_sr)
    cmap = cm.get_cmap(COLORMAP)
    
    # 4. Plotting Equal Boxes (Stacked Heat Map - Horizontal)
    for index, row in df_summary.iterrows():
        sr = row["StrikeRate"]
        avg = row["Average"] 
        
        # Determine box color based on SR
        color = cmap(norm(sr)) # This returns an RGBA tuple
        
        # Draw the box (barh with height=1)
        ax_stack.barh(
            y=0.5,           # Y-position (center of the chart)
            width=box_width,
            height=1,        # Full height (from 0 to 1 on the Y-axis)
            left=left,       # X-start position
            color=color,
            edgecolor='black', 
            linewidth=0.7
        )
        
        # --- Apply Dynamic Text Color Logic ---
        label_text = f"SR: {sr:.0f}\nAvg: {avg:.1f}"
        
        # Calculate text color for contrast
        r, g, b, a = color # 'color' is guaranteed to be an RGBA tuple from cmap
        # Calculate luminosity
        luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Set text color
        text_color = 'white' if luminosity < 0.5 else 'black'
        # -------------------------------------
        
        # Text position: Center of the box
        center_x = left + box_width / 2
        center_y = 0.5
        
        # Label 1: SR and Avg (Middle of the box)
        ax_stack.text(
            center_x, center_y, 
            label_text,
            ha='center', va='center', 
            fontsize=10, 
            color=text_color, weight='bold' # Using dynamic text_color
        )
        
        # Label 2: Crease Width Label (Below the box)
        ax_stack.text(
            center_x, -0.05, # Position slightly below the box
            index,           # The CreaseWidth label (e.g., '3m+')
            ha='center', va='top', 
            fontsize=10, 
            color='black' # Using dynamic text_color
        )

        left += box_width # Advance the starting position for the next box

    # 5. Styling

    # Hide all axis lines, ticks, and labels
    ax_stack.set_xlim(0, 1)
    ax_stack.set_ylim(0, 1)
    ax_stack.axis('off')

    plt.tight_layout(pad=0.5)
    
    return fig_stack

# --- CHART 5: INTERCEPTION FRONT-ON --- (Distance vs Width)
def create_interception_front_on(df_in, delivery_type):
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        fig, ax = plt.subplots(figsize=(3, 5)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    fig_8, ax_8 = plt.subplots(figsize=(3, 5), subplot_kw={'xticks': [], 'yticks': []}) 

    # 1. Plot Data
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on Y-axis (Distance) ===
    ax_8.scatter(
        df_other["InterceptionY"], df_other["InterceptionX"] + 10, 
        color='#D3D3D3', edgecolors='white', linewidths=0.5, s=60, label="Other"
    ) 
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on Y-axis (Distance) ===
        ax_8.scatter(
            df_slice["InterceptionY"], df_slice["InterceptionX"] + 10, 
            color=color_map[ctype],edgecolors='white', s=80, label=ctype
        ) 

    # 2. Draw Horizontal Dashed Lines with Labels (FIXED LINES: 0.0, 1.25)
    line_specs = {
        0.00: "Stumps",
        1.25: "Crease"        
    }
    for y_val, label in line_specs.items():
        ax_8.axhline(y=y_val, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), ha='left', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Boundary lines (FIXED LINES: -0.18, 0.18)
    ax_8.axvline(x=-0.18, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_8.axvline(x= 0.18, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_8.axvline(x= 0, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # 3. Set Axes Limits and Labels (FIXED LIMITS: Y-axis -0.2 to 3.5)
    ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
    ax_8.tick_params(axis='y', which='both', labelleft=False, left=False); ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_8.spines['right'].set_visible(False)
    ax_8.spines['top'].set_visible(False)
    ax_8.spines['left'].set_visible(False)
    ax_8.spines['bottom'].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig_8


# --- CHART 6: SCORING WAGON WHEEL ---
def calculate_scoring_wagon(row):
    """Calculates the scoring area based on LandingX/Y coordinates and handedness."""
    LX = row.get("LandingX"); LY = row.get("LandingY"); RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: return None
    
    # Safe arctan calculation to avoid division by zero
    def atan_safe(numerator, denominator): return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    # Right Handed Batsman Logic
    if RH == True: 
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
            # Note: The original atan_safe(LX, LY) usage here is non-standard for angle comparison, 
            # but preserved for code consistency. Assuming it defines the LONG OFF boundary.
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF" 
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON"
    # Left Handed Batsman Logic
    elif RH == False: 
        if LX <= 0 and LY > 0: return "THIRD MAN"
        elif LX <= 0 and LY <= 0: return "FINE LEG"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "SQUARE LEG"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG ON"
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "COVER"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG OFF"
    return None

def calculate_scoring_angle(area):
    """Defines the fixed angle size for each wedge."""
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0

# --- Main Wagon Wheel Function ---
def create_wagon_wheel(df_in, delivery_type):
    wagon_summary = pd.DataFrame() 
    try:
        df_wagon = df_in.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(TotalRuns=("Runs", "sum"), FixedAngle=("FixedAngle", 'first')).reset_index().dropna(subset=["ScoringWagon"])
        
        handedness_mode = df_in["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
        
        if is_right_handed:
            all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] 
        else:
            all_areas = ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
            
        template_df = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]})

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0) 
        wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
        wagon_summary = wagon_summary.sort_values("ScoringWagon")
        
        total_runs = wagon_summary["TotalRuns"].sum()
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100 if total_runs > 0 else 0 
        
        # Robust Angle Conversion
        wagon_summary["FixedAngle"] = pd.to_numeric(wagon_summary["FixedAngle"], errors='coerce').fillna(0).astype(int)
    
    except Exception:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "Calculation Error", ha='center', va='center'); ax.axis('off'); return fig
    
    
    # --- Data Extraction and CRITICAL Validation ---
    angles = wagon_summary["FixedAngle"].tolist()
    run_percentages = wagon_summary["RunPercentage"].tolist() 
    
    # Check for insufficient data
    if not angles or all(a == 0 for a in angles):
        fig, ax = plt.subplots(figsize=(4, 4)); 
        ax.text(0.5, 0.5, "Insufficient Data for Plot", ha='center', va='center'); 
        ax.axis('off'); 
        return fig

    # --- Color Logic (Top 1 Rank Only) ---
    wagon_summary['SortKey'] = wagon_summary['RunPercentage']
    wagon_summary['Rank'] = wagon_summary['RunPercentage'].rank(method='dense', ascending=False)
    
    COLOR_HIGH = '#d52221'
    COLOR_DEFAULT = 'white'

    colors = []
    for index, row in wagon_summary.iterrows():
        current_rank = row['Rank']
        
        if row['RunPercentage'] == 0:
            colors.append(COLOR_DEFAULT)
            continue
            
        if current_rank == 1:
            colors.append(COLOR_HIGH)
        else:
            colors.append(COLOR_DEFAULT)

    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    # --- Plotting Call ---
    pie_output = ax.pie(
        angles, 
        colors=colors, 
        wedgeprops={"width": 1, "edgecolor": "black"}, 
        startangle=90, 
        counterclock=False, 
        labels=None,
        labeldistance=1.1,
        autopct='%.0f',  # <--- CRITICAL ADDITION
        pctdistance=0.5
    )
    
    # FIX: Handle unpack error (expected 3, got 2)
    if len(pie_output) == 3:
        wedges, texts, autotexts = pie_output
    elif len(pie_output) == 2:
        wedges, texts = pie_output
        autotexts = [] # Assign an empty list if autotexts are missing
    else:
        fig, ax = plt.subplots(figsize=(4, 4)); 
        ax.text(0.5, 0.5, "Plotting Error (Return Value)", ha='center', va='center'); 
        ax.axis('off'); 
        return fig
    
    # === CRITICAL FIX: MANUALLY SET LABELS & ADD OUTLINE ===
    
    # Styling and label assignment
    for i, autotext in enumerate(autotexts):
        if i >= len(run_percentages): 
            break 
            
        percent = run_percentages[i]
        
        # 1. Set the actual percentage text
        if percent > 0:
            autotext.set_text(f'{percent:.0f}%')
            
            # 💥 FIX: Add a white stroke (outline) for text visibility
            autotext.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
        else:
            autotext.set_text('')
            
        # 2. Set text color based on background color for contrast
        color_rgb = mcolors.to_rgb(colors[i])
        luminosity = 0.2126 * color_rgb[0] + 0.7152 * color_rgb[1] + 0.0722 * color_rgb[2]
        
        # Use white text for the dark red wedges, black for white wedges
        autotext.set_color('white' if luminosity < 0.5 and colors[i] == COLOR_HIGH else 'black') 
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    # Ensure external texts are handled (UNCHANGED)
    for text in texts:
        text.set_color('black'); text.set_fontsize(8); text.set_fontweight('bold')
    ax.axis('equal'); 
    plt.tight_layout(pad=0.5)
    
    return fig
    
# --- CHART 7: LEFT/RIGHT SCORING SPLIT (100% Bar) ---
def create_left_right_split(df_in, delivery_type):
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    df_split = df_in.copy()
    
    # 1. Define Side
    # Assuming 'LandingY' < 0 is the batsman's right side (Off-side for RHB)
    # Since you label it LEFT/RIGHT, we'll keep that.
    df_split["Side"] = np.where(df_split["LandingY"] < 0, "LEFT", "RIGHT")
    
    # 2. Calculate Runs and Percentage
    summary = df_split.groupby("Side")["Runs"].sum().reset_index()
    total_runs = summary["Runs"].sum()
    
    # --- Check for No Runs (Remains the same) ---
    if total_runs == 0:
        # Decreased height applied here too
        fig, ax = plt.subplots(figsize=(4, 2.0)); ax.text(0.5, 0.5, "No Runs Scored", ha='center', va='center'); ax.axis('off'); return fig
        
    summary["Percentage"] = (summary["Runs"] / total_runs) * 100
    
    # Order the summary for consistent plotting
    summary = summary.set_index("Side").reindex(["LEFT", "RIGHT"]).fillna(0)
    
    left_pct = summary.loc["LEFT", "Percentage"]
    right_pct = summary.loc["RIGHT", "Percentage"]

    # 3. Apply Blue Hue Based on Percentage
    
    # Normalize percentage values (0 to 100)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    # Use a sequential blue colormap. Use Blues_r if you want a lower percentage to be darker blue.
    cmap = cm.get_cmap('Reds') 
    
    # Map the percentages to the colormap
    left_color = cmap(norm(left_pct))
    right_color = cmap(norm(right_pct))
    
    # 4. Create the 100% Stacked Bar Chart
    # DECREASED HEIGHT: Changed figsize to (4, 1.0)
    fig_split, ax_split = plt.subplots(figsize=(4, 2.0)) 

    # Plotting the left side
    ax_split.barh("Total", left_pct, color=left_color, edgecolor='black', linewidth=0.5)
    
    # Plotting the right side stacked on the left side
    ax_split.barh("Total", right_pct, left=left_pct, color=right_color, edgecolor='black', linewidth=0.5)
    
    # Add labels
    # Use a luminosity check to ensure white text on dark blue and black text on light blue
    
    def get_text_color(rgb_color):
        # Only unpack R, G, B (mcolors.to_rgb returns a 3-tuple)
        r, g, b = rgb_color 
        # Calculate luminosity (standard formula)
        luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Return contrast color
        return 'white' if luminosity < 0.5 else 'black'

    if left_pct > 0:
        text_color_left = get_text_color(mcolors.to_rgb(left_color))
        ax_split.text(left_pct / 2, 0, f"LEFT\n{left_pct:.0f}%", 
                      ha='center', va='center', color=text_color_left, weight='bold', fontsize=12)
                      
    if right_pct > 0:
        text_color_right = get_text_color(mcolors.to_rgb(right_color))
        ax_split.text(left_pct + right_pct / 2, 0, f"RIGHT\n{right_pct:.0f}%", 
                      ha='center', va='center', color=text_color_right, weight='bold', fontsize=12)

    # 5. Styling (Remains the same)
    ax_split.set_xlim(0, 100)
    
    # Remove all spines/borders
    ax_split.spines['right'].set_visible(False)
    ax_split.spines['top'].set_visible(False)
    ax_split.spines['left'].set_visible(False)
    ax_split.spines['bottom'].set_visible(False)
    
    # Hide ticks and labels
    ax_split.tick_params(axis='both', which='both', length=0)
    ax_split.set_yticklabels([]); ax_split.set_xticklabels([])
    
    plt.tight_layout(pad=0.5)
    return fig_split

# --- CHART 9/10: DIRECTIONAL SPLIT (Side-by-Side Bars) ---
def create_directional_split(df_in, direction_col, chart_title, delivery_type):
    df_dir = df_in.copy()
    if df_dir.empty:
        fig, ax = plt.subplots(figsize=(6, 2.5)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    
    # 1. Prepare Data
    df_dir["Direction"] = np.where(df_dir[direction_col] < 0, "LEFT", "RIGHT")
    
    summary = df_dir.groupby("Direction").agg(
        Runs=("Runs", "sum"), 
        Wickets=("Wicket", lambda x: (x == True).sum()), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("Direction").reindex(["LEFT", "RIGHT"]).fillna(0)
    
    summary["Average"] = summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (row["Runs"] if row["Balls"] > 0 else 0), axis=1
    )
    
    # --- Prepare for Butterfly Effect & Order (LEFT on top, RIGHT on bottom) ---
    # Reindex to plot RIGHT (index 0) then LEFT (index 1) for the desired vertical visual order
    summary = summary.reset_index().set_index("Direction").reindex(["RIGHT", "LEFT"]) 
    
    # Set LEFT side to negative values for mirroring
    summary.loc["LEFT", "Average_Mirrored"] = summary.loc["LEFT", "Average"] * -1
    summary.loc["RIGHT", "Average_Mirrored"] = summary.loc["RIGHT", "Average"]
    
    # Extract lists (Order: RIGHT, LEFT)
    directions = summary.index.tolist()
    averages_mirrored = summary["Average_Mirrored"].tolist()
    averages_abs = summary["Average"].tolist()
    wickets = summary["Wickets"].tolist()
    
    # 2. Create Plot
    fig_dir, ax_dir = plt.subplots(figsize=(6, 2.5)) 
    
    # --- Plotting the Bars ---
    y_positions = [0, 1] 
    colors = ['#d52221', '#d52221'] 
    
    # Plot horizontal bars
    bars = ax_dir.barh(y_positions, averages_mirrored, color=colors, edgecolor='black', linewidth=0.5, height=0.6)

    # 3. Add Labels and Styling
    
    # Set the y-axis labels. The list ['RIGHT', 'LEFT'] matches y_positions [0, 1]
    ax_dir.set_yticks(y_positions)
    ax_dir.set_yticklabels(directions, fontsize=12, color='black') 

    # Calculate max absolute value for x-axis limit
    max_abs_avg = summary["Average"].max()
    x_limit = max_abs_avg * 1.15 if max_abs_avg > 0 else 10 
    ax_dir.set_xlim(-x_limit, x_limit)
    
    # Custom X-Axis: HIDE AXIS AND LABELS
    ax_dir.set_xticks([]) 
    ax_dir.set_xticklabels([]) 

    # --- Add Data Labels (Wickets and Average) ---
    for i, bar in enumerate(bars):
        avg = averages_abs[i]
        wkts = wickets[i]
        label = f"{int(wkts)}W\n{avg:.1f} Ave"
        
        # Determine bar properties
        bar_end_x = bar.get_x() + bar.get_width() # The tip of the bar
        padding = 0.05 * x_limit 

        if directions[i] == 'LEFT': # LEFT bar (index 1, negative values)
            # Positioned inside the bar: move right (positive direction) from the tip (negative)
            text_x = bar_end_x + padding 
            ha_align = 'left' 
        else: # RIGHT bar (index 0, positive values)
            # Positioned inside the bar: move left (negative direction) from the tip (positive)
            text_x = bar_end_x - padding 
            ha_align = 'right' 

        # Set text color to white for contrast, and apply black outline for guaranteed visibility
        text_color = 'black' 

        text_object = ax_dir.text(text_x, 
                    bar.get_y() + bar.get_height() / 2, 
                    label,
                    ha=ha_align, va='center', 
                    fontsize=14, 
                    color=text_color, weight='bold') 

    # --- Final Styling and Spines ---
    ax_dir.set_title(chart_title, fontsize=14, weight='bold', color='black', pad=10)
    
    # Hide all spines
    ax_dir.spines['top'].set_visible(False)
    ax_dir.spines['bottom'].set_visible(False) 
    ax_dir.spines['left'].set_visible(False)
    ax_dir.spines['right'].set_visible(False)
    
    # Add a subtle vertical line at x=0 for the axis center
    ax_dir.axvline(0, color='gray', linewidth=0.8)
    
    # Remove y-ticks
    ax_dir.tick_params(axis='y', which='both', length=0)
    
    plt.tight_layout(pad=1.0)
    return fig_dir

# --- 3. MAIN STREAMLIT APP STRUCTURE ---

st.set_page_config(layout="wide")

# --- 3. MAIN STREAMLIT APP STRUCTURE ---
# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
if uploaded_file is not None:
    # Read the data from the uploaded file
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df_raw = pd.read_csv(StringIO(data))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Initial validation and required column check
    if not all(col in df_raw.columns for col in REQUIRED_COLS):
        missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
        st.error(f"The CSV file is missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Data separation
    df_seam = df_raw[df_raw["DeliveryType"] == "Seam"].copy()
    df_spin = df_raw[df_raw["DeliveryType"] == "Spin"].copy()

    # =========================================================
    # 🌟 FILTERS MOVED TO TOP OF MAIN BODY 🌟
    # =========================================================
    # Use columns to align the three filters horizontally
    # Use columns to align the four filters horizontally
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4) 

    # --- Filter Logic ---
    all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())

    # 1. Batting Team Filter (in column 1)
    with filter_col1:
        bat_team = st.selectbox("Batting Team", all_teams, index=0)

    # 2. Batsman Name Filter (Logic depends on Batting Team - in column 2)
    if bat_team != "All":
        batsmen_options = ["All"] + sorted(df_raw[df_raw["BattingTeam"] == bat_team]["BatsmanName"].dropna().unique().tolist())
    else:
        batsmen_options = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())
    
    with filter_col2:
        batsman = st.selectbox("Batsman Name", batsmen_options, index=0)

    # 3. Innings Filter (in column 3) - NEW
    innings_options = ["All"] + sorted(df_raw["Innings"].dropna().unique().tolist())
    with filter_col3:
        selected_innings = st.selectbox("Innings", innings_options, index=0)
    
    # 4. Bowler Hand Filter (in column 4) - NEW
    bowler_hand_options = ["All", "Right Hand", "Left Hand"]
    with filter_col4:
        selected_bowler_hand = st.selectbox("Bowler Hand", bowler_hand_options, index=0)
    
# =========================================================

    # --- Apply Filters to Seam and Spin dataframes ---
    def apply_filters(df):
        if bat_team != "All":
            df = df[df["BattingTeam"] == bat_team]
        
        if batsman != "All":
            df = df[df["BatsmanName"] == batsman]
        
        # Apply Innings Filter
        if selected_innings != "All":
            df = df[df["Innings"] == selected_innings]
        
        # Apply Bowler Hand Filter
        if selected_bowler_hand != "All":
            is_right = (selected_bowler_hand == "Right Hand")
              # Ensure column name 'IsBowlerRightHanded' is correct
            df = df[df["IsBowlerRightHanded"] == is_right]
        
        return df

    # Apply filters
    df_seam = apply_filters(df_seam)
    df_spin = apply_filters(df_spin)
    
    heading_text = batsman.upper() if batsman != "All" else "GLOBAL ANALYSIS"
    st.header(f"**{heading_text}**")

    # --- 4. DISPLAY CHARTS IN TWO COLUMNS ---
    
    col1, col2 = st.columns(2)
    
    # --- LEFT COLUMN: SEAM ANALYSIS ---
    with col1:
        # Use a smaller Markdown header (e.g., h4)
        st.markdown("#### SEAM")

        st.markdown("###### CREASE BEEHIVE ZONES")
        st.pyplot(create_zonal_analysis(df_seam, batsman, "Seam"), use_container_width=True)
        
        st.markdown("###### CREASE BEEHIVE")
        st.plotly_chart(create_crease_beehive(df_seam, "Seam"), use_container_width=True)

        st.pyplot(create_lateral_performance_boxes(df_seam, "Seam", batsman), use_container_width=True)
        
        # Charts 3: Pitch Map and Vertical Run % Bar (Side-by-Side)
        pitch_map_col, run_pct_col = st.columns([3, 1]) # 3:1 ratio for Pitch Map and Bar

        with pitch_map_col:
            st.markdown("###### PITCHMAP")
            st.plotly_chart(create_pitch_map(df_seam, "Seam"), use_container_width=True)
            
        with run_pct_col:
            st.markdown("###### ")
            st.pyplot(create_pitch_length_run_pct(df_seam, "Seam"), use_container_width=True)
        
        # --- NEW LAYOUT START ---
        
        # Chart 4a: Interception Side-On (Wide View) - Takes full width4
        st.markdown("###### INTERCEPTION SIDE-ON")
        st.pyplot(create_interception_side_on(df_seam, "Seam"), use_container_width=True)

        # Chart 4b: Interception Side-On Bins
        st.pyplot(create_crease_width_split(df_seam, "Seam"), use_container_width=True)

        # Charts 5 & 6: Interception Front-On and Scoring Areas (Side-by-Side)
        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown("###### INTERCEPTION TOP-ON")
            st.pyplot(create_interception_front_on(df_seam, "Seam"), use_container_width=True)
        
        with bottom_col_right:
            st.markdown("###### SCORING AREAS")    
            st.pyplot(create_wagon_wheel(df_seam, "Seam"), use_container_width=True)
            st.pyplot(create_left_right_split(df_seam, "Seam"), use_container_width=True)
    
         # Charts 9 & 10: Swing/Deviation Direction Analysis (Side-by-Side)
        final_col_swing, final_col_deviation = st.columns(2)

        with final_col_swing:
            st.pyplot(create_directional_split(df_seam, "Swing", "Swing", "Seam"), use_container_width=True)

        with final_col_deviation:
            st.pyplot(create_directional_split(df_seam, "Deviation", "Deviation", "Seam"), use_container_width=True)   
        # --- NEW LAYOUT END ---




    # --- RIGHT COLUMN: SPIN ANALYSIS ---
    with col2:
        # Use a smaller Markdown header (e.g., h4)
        st.markdown("#### SPIN")
        
        st.markdown("###### CREASE BEEHIVE ZONES")
        st.pyplot(create_zonal_analysis(df_spin, batsman, "Spin"), use_container_width=True)
        st.markdown("###### CREASE BEEHIVE")
        st.plotly_chart(create_crease_beehive(df_spin, "Spin"), use_container_width=True)

        st.pyplot(create_lateral_performance_boxes(df_spin, "Spin", batsman), use_container_width=True)

        # Charts 3 & 8: Pitch Map and Vertical Run % Bar (Side-by-Side)
        pitch_map_col, run_pct_col = st.columns([3, 1]) # 3:1 ratio for Pitch Map and Bar

        with pitch_map_col:
            # CORRECTED: Use df_spin and "Spin"
            st.markdown("###### PITCHMAP")
            st.plotly_chart(create_pitch_map(df_spin, "Spin"), use_container_width=True) 
            
        with run_pct_col:
            # CORRECTED: Use df_spin and "Spin"
            st.markdown("###### ")
            st.pyplot(create_pitch_length_run_pct(df_spin, "Spin"), use_container_width=True)
        
        # --- NEW LAYOUT START (Mirroring Left Column) ---
        
        # Chart 4a: Interception Side-On (Wide View) - Takes full width
        st.markdown("###### INTERCEPTION SIDE-ON")
        st.pyplot(create_interception_side_on(df_spin, "Spin"), use_container_width=True)

        # Chart 4b: Interception Side-On Bins
        st.pyplot(create_crease_width_split(df_spin, "Spin"), use_container_width=True)

        # Charts 5 & 6: Interception Front-On and Scoring Areas (Side-by-Side)
        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown("###### INTERCEPTION TOP-ON")
            st.pyplot(create_interception_front_on(df_spin, "Spin"), use_container_width=True)
        
        with bottom_col_right:
            st.markdown("###### SCORING AREAS")
            st.pyplot(create_wagon_wheel(df_spin, "Spin"), use_container_width=True)
            st.pyplot(create_left_right_split(df_spin, "Spin"), use_container_width=True)
            
        # Charts 9 & 10: Swing/Deviation Direction Analysis (Side-by-Side)
        final_col_swing, final_col_deviation = st.columns(2)

        with final_col_swing:
            # CORRECTED: Use df_spin and "Spin"
            st.pyplot(create_directional_split(df_spin, "Swing", "Drift", "Spin"), use_container_width=True)

        with final_col_deviation:
            # CORRECTED: Use df_spin and "Spin"
            st.pyplot(create_directional_split(df_spin, "Deviation", "Turn", "Spin"), use_container_width=True)    
        # --- NEW LAYOUT END ---

else:
    st.info("⬆️ Please upload a CSV file to begin the analysis.")
