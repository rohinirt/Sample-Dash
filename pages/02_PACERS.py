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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# =========================================================
# Chart 2: CREASE BEEHIVE 
# ========================================================
def create_pacer_crease_beehive(df_in, handedness_label): # Renamed function and parameter
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 5)); 
        ax.text(0.5, 0.5, f"No data for Analysis ({handedness_label})", ha='center', va='center', fontsize=12); 
        ax.axis('off'); 
        return fig

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]
    boundaries = non_wickets_all[(non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)]
    regular_balls = non_wickets_all[(non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)]
    
    # --- Lateral Zone Data Prep (Chart 2b) ---
    df_lateral = df_in.copy()
    
    # DETERMINE HANDEDNESS FOR ZONE REVERSAL
    # If the function is called with a single handedness filter (RHB or LHB), this will be consistent.
    is_rhb = handedness_label == "RHB" 

    def assign_lateral_zone(row):
        y = row["CreaseY"]
        if row["IsBatsmanRightHanded"] == True:
            # RHB: Off side is negative Y, Leg side is positive Y
            if y > 0.18: return "LEG"
            elif y >= -0.18: return "STUMPS"
            elif y > -0.65: return "OUTSIDE OFF"
            else: return "WAY OUTSIDE OFF"
        else: # Left-Handed
            # LHB: Leg side is negative Y, Off side is positive Y
            if y > 0.65: return "WAY OUTSIDE OFF"
            elif y > 0.18: return "OUTSIDE OFF"
            elif y >= -0.18: return "STUMPS"
            else: return "LEG"
            
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count")
        )
    )
    
    # 1. Define standard zone order (WOO to LEG)
    ordered_zones_base = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    
    # 2. HANDEDNESS AWARE REVERSAL: Reverse order for LHB for visual consistency
    ordered_zones = ordered_zones_base if is_rhb else ordered_zones_base[::-1]
    
    summary = summary.reindex(ordered_zones).fillna(0)
    
    # BOWLING AVERAGE CALCULATION (Same formula as before, just interpreted differently)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)

    # -----------------------------------------------------------
    # --- 1. SETUP SUBPLOTS ---
    fig = plt.figure(figsize=(7, 5)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.005) 
    ax_bh = fig.add_subplot(gs[0, 0])      
    ax_boxes = fig.add_subplot(gs[1, 0])   
    fig.patch.set_facecolor('white')
    

    # -----------------------------------------------------------
    ## --- 2. CHART 2a: CREASE BEEHIVE (ax_bh) ---
    
    # --- Traces ---
    ax_bh.scatter(regular_balls["CreaseY"], regular_balls["CreaseZ"], s=40, c='lightgrey', edgecolor='white', linewidths=1.0, alpha=0.95, label="Regular Ball")
    ax_bh.scatter(boundaries["CreaseY"], boundaries["CreaseZ"], s=80, c='royalblue', edgecolor='white', linewidths=1.0, alpha=0.95, label="Boundary")
    ax_bh.scatter(wickets["CreaseY"], wickets["CreaseZ"], s=80, c='red', edgecolor='white', linewidths=1.0, alpha=0.95, label="Wicket")

    # --- Reference Lines ---
    ax_bh.axvline(x=-0.18, color="grey", linestyle="--", linewidth=0.5) 
    ax_bh.axvline(x=0.18, color="grey", linestyle="--", linewidth=0.5)
    ax_bh.axvline(x=0, color="grey", linestyle="--", linewidth=0.5) 
    ax_bh.axvline(x=-0.92, color="grey", linestyle="-", linewidth=0.5) 
    ax_bh.axvline(x=0.92, color="grey", linestyle="-", linewidth=0.5)
    ax_bh.axhline(y=0.78, color="grey", linestyle="-", linewidth=0.5)

    # --- Annotation ---
    ax_bh.text(-1.5, 0.78, "Stump line", ha='left', va='bottom', fontsize=8, color="grey", transform=ax_bh.transData)
    
    # --- Formatting ---
    ax_bh.set_xlim([-2, 2])
    ax_bh.set_ylim([0, 2])
    ax_bh.set_aspect('equal', adjustable='box')
    ax_bh.set_xticks([]); ax_bh.set_yticks([]); ax_bh.grid(False)
    for spine in ax_bh.spines.values():
        spine.set_visible(False)
    ax_bh.set_facecolor('white')
    
    # -----------------------------------------------------------
    ## --- 3. CHART 2b: LATERAL PERFORMANCE BOXES (ax_boxes) ---
    
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions
    box_height = 0.4 
    left = 0
    
    # Color Normalization
    avg_values = summary["Avg Runs/Wicket"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 50
    # Capping max at 50 for consistent coloring/normalization
    norm = mcolors.Normalize(vmin=0, vmax=avg_max if avg_max > 50 else 50) 
    cmap = cm.get_cmap('Reds') 

    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
        
        color = cmap(norm(avg)) if row["Balls"] > 0 else 'whitesmoke' 
        
        # Draw the Rectangle
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, box_height, 
                              edgecolor="white", facecolor=color, linewidth=1)
        )
        
        # Label 1: Zone Name (Above the box)
        ax_boxes.text(left + box_width / 2, box_height + 0.1, 
                      index, 
                      ha='center', va='bottom', fontsize=7, color='black')
        
        # Calculate text color for contrast
        text_color = 'black'
        if row["Balls"] > 0:
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        
        # Label 2: Wickets and Average (Middle of the box)
        label_wkts_avg = f"{wkts}W - Ave {avg:.1f}"
        ax_boxes.text(left + box_width / 2, box_height * 0.5, 
                      label_wkts_avg,
                      ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)
        
        left += box_width

    # Formatting
    ax_boxes.set_xlim(0, 1)
    ax_boxes.set_ylim(0, box_height + 0.3) 
    ax_boxes.axis('off')
    for spine in ax_boxes.spines.values():
        spine.set_visible(False)
    ax_boxes.set_facecolor('white')

    # -----------------------------------------------------------
    ## --- 4. DRAW SINGLE COMPACT BORDER AROUND THE ENTIRE FIGURE ---
    
    plt.tight_layout(pad=0.2)
    
    PADDING = 0.008

    bh_bbox = ax_bh.get_position()
    box_bbox = ax_boxes.get_position()
    
    x0_orig = min(bh_bbox.x0, box_bbox.x0)
    y0_orig = box_bbox.y0
    x1_orig = max(bh_bbox.x1, box_bbox.x1)
    y1_orig = bh_bbox.y1
    
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    border_rect = patches.Rectangle(
        (x0_pad, y0_pad), 
        width_pad, 
        height_pad, 
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, 
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig

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
    avg_values = summary["Wickets"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 100 
    avg_max_cap = 50 # Cap max at 50 for visualization consistency
    
    # Assuming mcolors is imported
    norm = mcolors.Normalize(vmin=0, vmax=avg_max if avg_max > avg_max_cap else avg_max_cap)
    # Assuming cm is imported
    cmap = cm.get_cmap('Reds') # Higher Average (worse bowling) is darker red

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
    plt.tight_layout(pad=0.1)
    bbox = ax.get_position()
    LINE_THICKNESS = 0.3
    
    # 2. DEFINE CUSTOM PADDING FOR EACH SIDE (in figure coordinates, e.g., 0.01 = 1% of figure dimension)
    # Adjust these values to shift the border relative to the plot content:
    custom_padding = {
        'left': 0.0002,   # Increase for wider gap on the left
        'bottom': 0.03, # Decrease for tighter gap on the bottom
        'right': 0.0002,  # Increase for wider gap on the right
        'top': 0.00001     # Decrease for tighter gap on the top
    }
    
    # 3. CALCULATE NEW RECTANGLE POSITION AND SIZE
    
    # New X start position (original X start minus left padding)
    x_start = bbox.x0 - custom_padding['left']
    
    # New Y start position (original Y start minus bottom padding)
    y_start = bbox.y0 - custom_padding['bottom']
    
    # New Width (original width + left padding + right padding)
    new_width = (bbox.x1 - bbox.x0) + custom_padding['left'] + custom_padding['right']
    
    # New Height (original height + bottom padding + top padding)
    new_height = (bbox.y1 - bbox.y0) + custom_padding['bottom'] + custom_padding['top']
    
    # Create the border rectangle
    border_rect = patches.Rectangle(
        (x_start, y_start), 
        new_width, 
        new_height, 
        facecolor='none', 
        edgecolor='black', 
        linewidth=LINE_THICKNESS, 
        transform=fig_boxes.transFigure, 
        clip_on=False
    )
    
    # Add the border to the figure
    fig_boxes.patches.append(border_rect)
    
    return fig_boxes
    return fig_boxes

# --- Helper function for Pitch Bins (Centralized) ---
def get_pitch_bins():
    """Defines the pitch length ranges for Seam bowling."""
    # Seam Bins: 1.2-6: Full, 6-8 Length, 8-10 Short, 10-15 Bouncer (Distance from batsman's stumps in meters)
    return {
        "Over Pitched": [1.22, 2.22],
        "Full": [2.22, 4.0],
        "Good": [4.0, 6.0],
        "Short": [6.0, 15.0],
    }
    
# --- CHART 3a: PITCH MAP (BOUNCE LOCATION) ---
# --- CHART 3: PITCH MAP (BOUNCE LOCATION) ---
def create_pacer_pitch_map(df_in): 
 
    # Define Pacer Bins (Delivery Type is fixed as Seam)
    # Bins: 1.2-6: Full, 6-8 Length, 8-10 Short, 10-15 Bouncer
    PITCH_BINS = {
        "Full Toss": [-4.0, 1.2], # Added Full Toss based on Seam logic
        "Full": [1.2, 6.0],
        "Length": [6.0, 8.0],
        "Short": [8.0, 10.0],
        "Bouncer": [10.0, 15.0],
    }

    if df_in.empty:
        # Create an empty figure with a text note if data is missing
        fig, ax = plt.subplots(figsize=(4,6))
        ax.text(0.5, 0.5, f"No data for Pacer Pitch Map", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- Data Filtering ---
    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]
    
    # --- Chart Setup ---
    fig, ax = plt.subplots(figsize=(4,6)) # Maintained figsize=(4,6)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # --- 1. Add Zone Lines & Labels (Horizontal Lines) ---
    
    # Determine boundary Y values to draw lines (excluding the start of the lowest bin)
    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0], reverse=True)

    for y_val in boundary_y_values:
        ax.axhline(y=y_val, color="lightgrey", linewidth=1.0, linestyle="--")

    # Add zone labels (Annotation)
    for length, bounds in PITCH_BINS.items():
        if length != "Full Toss": 
            mid_y = (bounds[0] + bounds[1]) / 2
            # Use ax.text for annotation, positioned on the far left (x=-1.45)
            ax.text(
                x=-1.45, 
                y=mid_y, 
                s=length.upper(), 
                ha='left', 
                va='center', 
                fontsize=8, 
                color="grey", 
                fontweight='bold'
            )

    
    # --- 3. Plot Data (Scatter Traces) ---
    
    # Non-Wickets (light grey)
    ax.scatter(
        pitch_non_wickets["BounceY"], pitch_non_wickets["BounceX"], 
        s=60, 
        c='#D3D3D3', 
        edgecolor='white', 
        linewidths=1.0, 
        alpha=0.9,
        label="No Wicket"
    )

    # Wickets (red)
    ax.scatter(
        pitch_wickets["BounceY"], pitch_wickets["BounceX"], 
        s=90, 
        c='red', 
        edgecolor='white', 
        linewidths=1.0, 
        alpha=0.95,
        label="Wicket"
    )
    
    # --- 2. Add Stump lines (Vertical Lines) ---
    ax.axvline(x=-0.18, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(x=0.18, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(x=0, color="#777777", linestyle="--", linewidth=0.8)
    
    # --- 4. Layout (Axis and Spines) ---
    
    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    # Reverse the axis to match the cricket visual (batter at bottom)
    ax.set_ylim([16.0, -4.0]) 

    # Hide all axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    
    # Hide axis spines (plot border)
    spine_color = 'black'
    spine_width = 0.5
    for spine_name in ['left', 'top', 'bottom','right']:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(spine_color)
        ax.spines[spine_name].set_linewidth(spine_width)
        
    plt.tight_layout()
    
    return fig

# --- CHART 3b: PITCH LENGTH METRICS (BOWLER FOCUS) ---
# --- Helper function for Pitch Bins (Hardcoded for Seam) ---
def get_pacer_pitch_bins():
    """Returns fixed pitch bins for Seam bowlers."""
    # Seam Bins: 1.2-6: Full, 6-8 Length, 8-10 Short, 10-15 Bouncer
    return {
        "Full": [1.2, 6.0],
        "Length": [6.0, 8.0],
        "Short": [8.0, 10.0],
        "Bouncer": [10.0, 15.0],
    }

# --- CHART 3b: PITCH LENGTH BOWLER METRICS (EQUAL SIZED BOXES) ---
def create_pacer_pitch_length_bars(df_in):
    # Increased height to accommodate three stacked charts comfortably
    FIG_SIZE = (4, 6) 
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Pacer Pitch Length Comparison", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Get the pitch bins and define order (Fixed for Pacer)
    PITCH_BINS_DICT = get_pacer_pitch_bins()
    ordered_keys = ["Full", "Length", "Short", "Bouncer"]
    
    # 1. Data Preparation
    def assign_pitch_length(x):
        # We don't need to consider 'Full Toss' in the assignment logic for this chart 
        # as it's typically excluded from length comparison analysis.
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]: return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_length)
    
    # Aggregate data
    df_summary = df_pitch.groupby("PitchLength").agg(
        Runs=("Runs", "sum"), 
        Wickets=("Wicket", lambda x: (x == True).sum()), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("PitchLength").reindex(ordered_keys).fillna(0)
    
    # Calculate Bowling Metrics
    # Bowling Average = Runs / Wickets
    df_summary["BowlingAverage"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (100 if row["Balls"] > 0 else 0), axis=1
    )
    # Bowling Strike Rate = Balls / Wickets * 100
    df_summary["BowlingStrikeRate"] = df_summary.apply(
        lambda row: row["Balls"] / row["Wickets"] if row["Wickets"] > 0 else (row["Balls"] if row["Balls"] > 0 else 0), axis=1
    )
    # Use 'Wickets' as the count for Dismissals
    df_summary["Dismissals"] = df_summary["Wickets"]
    
    # Categories for plotting (reversed for barh)
    categories = df_summary.index.tolist()[::-1]
    
    # 2. Chart Setup (3 Rows, 1 Column)
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharey=True) 
    plt.subplots_adjust(hspace=0.4) 

    # --- Metrics and Titles (Order: Dismissals, Bowling Average, Bowling Strike Rate) ---
    metrics = ["Dismissals", "BowlingAverage", "BowlingStrikeRate"]
    titles = ["Dismissals", "Bowling Average", "Bowling Strike Rate"]

    # Define limits for each chart to ensure proper scaling
    max_wkts = df_summary["Dismissals"].max() * 1.5 if df_summary["Dismissals"].max() > 0 else 5
    max_avg = df_summary["BowlingAverage"].max() * 1.2 if df_summary["BowlingAverage"].max() > 0 else 60
    max_sr = df_summary["BowlingStrikeRate"].max() * 1.2 if df_summary["BowlingStrikeRate"].max() > 0 else 100 # SR can be very high if few wickets

    xlim_limits = {
        "Dismissals": (0, max_wkts),
        "BowlingAverage": (0, max_avg),
        "BowlingStrikeRate": (0, max_sr)
    }

    # --- Plotting Loop ---
    for i, ax in enumerate(axes):
        metric = metrics[i]
        title = titles[i]
        
        # Data values (reversed to align with category order)
        values = df_summary[metric].values[::-1] 
        
        # Define x limits
        ax.set_xlim(xlim_limits[metric])
        
        # Horizontal Bar Chart
        ax.barh(categories, values, height=0.5, color='Red', zorder=3, alpha=0.9)
        
        # --- Annotations ---
        for j, (cat, val) in enumerate(zip(categories, values)):
            # Format value
            if metric == "Dismissals":
                label = f"{int(val)}"
            else:
                label = f"{val:.1f}" # Use 1 decimal place for averages/rates
            
            # Place label slightly to the right of the bar tip
            ax.text(val, j, label, 
                    ha='left', va='center', 
                    fontsize=10, fontweight = 'bold', color='black',
                    bbox=dict(facecolor='White', alpha=0.8, edgecolor='none', pad=2),
                    zorder=4)

        # --- Formatting ---
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5) # Reduced title size slightly to fit
        ax.set_facecolor('white')

        # Set Ticks and Spines
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', length=0) # Hide y ticks

        # Set Y-axis labels only on the bottom-most chart (ax[2])
        if i == 2:
            ax.set_yticks(np.arange(len(categories)), labels=[c.upper() for c in categories], fontsize=9)
        else:
            # Remove y-tick labels for the top two charts
            ax.set_yticks(np.arange(len(categories)), labels=[''] * len(categories))
            
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False)

        # Hide x labels/ticks and enforce xlim
        ax.set_xticks([]) 
        ax.set_xlim(0, xlim_limits[metric][1]) 
        
        # --- Custom Spines: Right, Top, Bottom ---
        spine_color = 'lightgray'
        spine_width = 1.0 
        for spine_name in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine_name].set_visible(False)
            ax.spines[spine_name].set_color(spine_color)
            ax.spines[spine_name].set_linewidth(spine_width)
            
    plt.tight_layout(pad=0.5)
    return fig
    
# --- CHART 4: RELEASE SPEED DISTRIBUTION ---
def create_pacer_release_speed_distribution(df_in, handedness_label):
    FIG_SIZE = (4, 4.4)

    if df_in.empty or "ReleaseSpeed" not in df_in.columns or df_in["ReleaseSpeed"].empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No Data or Missing 'ReleaseSpeed' for {handedness_label}", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Prepare Data and Determine Histogram Parameters
    speeds = df_in["ReleaseSpeed"].dropna().values
    total_balls = len(speeds)
    
    if total_balls == 0:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Deliveries Found", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Calculate the range to ensure fixed bin width of 5 km/h
    min_speed = np.floor(speeds.min() / 5) * 5
    max_speed = np.ceil(speeds.max() / 5) * 5
    
    # Generate bins with a fixed width of 5 km/h
    bin_width = 5
    bins = np.arange(min_speed, max_speed + bin_width, bin_width)
    
    # Calculate histogram counts and edges
    counts, bin_edges = np.histogram(speeds, bins=bins)
    
    # 2. Process Data for Plotting & Filtering
    
    # Filter out bins with less than 5 balls
    MIN_BALLS = 5
    valid_counts = []
    valid_bin_labels = []
    
    for i in range(len(counts)):
        if counts[i] >= MIN_BALLS:
            lower = int(bin_edges[i])
            upper = int(bin_edges[i+1])
            label = f"{lower}-{upper}"
            
            valid_counts.append(counts[i])
            valid_bin_labels.append(label)

    if not valid_counts:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No Bins Meet the {MIN_BALLS}-Ball Minimum Filter", ha='center', va='center', fontsize=10)
        ax.axis('off')
        return fig
        
    # Calculate percentages for valid bins only
    valid_percentages = (np.array(valid_counts) / total_balls) * 100
    
    # Reverse order for horizontal bar chart (fastest speeds typically at the top)
    plot_percentages = valid_percentages[::-1]
    plot_labels = valid_bin_labels[::-1]
    plot_counts = valid_counts[::-1]
    
    # 3. Chart Generation (Horizontal Bar / Histogram)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    
    y_pos = np.arange(len(plot_labels))
    
    ax.barh(
        y_pos,
        plot_percentages,
        color='Red', # Single, uniform color
        height=0.6
    )

    
    # Add percentage labels
    for i, pct in enumerate(plot_percentages):
        count = plot_counts[i]
        # Display percentage (e.g., 25%)
        label_text = f'{pct:.0f}%'
        
        # Placement logic: Inside if bar is > 10%, otherwise outside
        x_pos = pct - 1 if pct > 10 else pct + 0.5
        ha = 'right' if pct > 10 else 'left'
        text_color = 'white' if pct > 10 else 'black'
        
        ax.text(
            x_pos, 
            i, 
            label_text, 
            ha=ha, va='center', fontsize=12, color=text_color, fontweight='bold'
        )
        


    # 4. Formatting
    
    # Set Y-axis labels
    ax.set_yticks(y_pos, labels=plot_labels, fontsize=10)
    
    # Set X-axis limit slightly higher than the max percentage for clean labels
    max_pct = np.max(plot_percentages) if len(plot_percentages) > 0 else 0
    ax.set_xlim(0, max(max_pct * 1.1, 10)) 
    
    # Hide axis ticks/labels
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    # Remove all spines 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # --- ADDING SHARP BORDER ---
    # We create a custom Rectangle patch with 'miter' joinstyle and add it to the figure.
    # Get the bounding box of the axes in figure coordinates
    ax_bbox = ax.get_position()
    
    # Calculate padding based on figure dimensions to ensure a consistent border
    # Use 0.01 for x and y to give a small padding
    padding_x = 0.2* FIG_SIZE[0] / fig.get_size_inches()[0] # Scale padding based on total figure width
    padding_y = 0.01 * FIG_SIZE[1] / fig.get_size_inches()[1] # Scale padding based on total figure height

    border_rect = patches.Rectangle(
        (ax_bbox.x0 - padding_x, ax_bbox.y0 - padding_y), # Start (x,y)
        ax_bbox.width + 2 * padding_x,                    # Width
        ax_bbox.height + 2 * padding_y,                   # Height
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        transform=fig.transFigure, # Use figure coordinates
        clip_on=False,             # Ensure it's not clipped
        joinstyle='miter'          # THIS ENSURES SHARP CORNERS
    )
    fig.add_artist(border_rect) # Add the custom rectangle to the figure

    return fig

# Chart 5 Bowler Release Map
def create_pacer_release_analysis(df_in, handedness_label): 
    FIG_SIZE = (4, 4) # Increased height for both charts

    if df_in.empty or "ReleaseY" not in df_in.columns or "ReleaseZ" not in df_in.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No data for Release Analysis vs. {handedness_label}", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- 1. Calculate Lateral Release Performance (LEFT vs RIGHT) ---
    df_temp = df_in.copy()
    
    # Categorize based on ReleaseY sign
    df_temp["ReleaseCategory"] = np.where(
        df_temp["ReleaseY"] < 0, "LEFT (<0)", 
        np.where(df_temp["ReleaseY"] > 0, "RIGHT (>0)", "CENTER (=0)")
    )
    
    df_temp = df_temp[df_temp["ReleaseCategory"] != "CENTER (=0)"]
    
    # Calculation functions
    def calculate_ba(row):
        # Use np.nan as a flag for "N/A"
        return row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else np.nan

    def calculate_sr(row):
        # Strike Rate = Balls per Wicket (normalized by 6 for Cricket SR)
        return (row["Balls"] / row["Wickets"]) * 6 if row["Wickets"] > 0 else np.nan
        
    summary = df_temp.groupby("ReleaseCategory").agg(
        Wickets=("Wicket", lambda x: (x == True).sum()),
        Runs=("Runs", "sum"),
        Balls=("Wicket", "count")
    )

    # Ensure both categories are present for consistent plotting
    summary = summary.reindex(["LEFT (<0)", "RIGHT (>0)"]).fillna(0)
    
    summary["BA"] = summary.apply(calculate_ba, axis=1)
    summary["SR"] = summary.apply(calculate_sr, axis=1)

    # Formatting helper
    def format_metric(value, is_wickets=False):
        if is_wickets:
            return f"{int(value)}"
        if np.isnan(value) or value == np.inf:
            return "N/A"
        return f"{value:.1f}"

    left = summary.loc["LEFT (<0)"]
    right = summary.loc["RIGHT (>0)"]

    # --- 2. Setup Figure and GridSpec ---
    fig = plt.figure(figsize=FIG_SIZE, facecolor='white')
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1.2], hspace=0.1)
    
    ax_map = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[1, 0])

    # --- 3. Plot Release Zone Map (ax_map) ---
    
    release_wickets = df_in[df_in["Wicket"] == True]
    release_non_wickets = df_in[df_in["Wicket"] == False]
    
    # Non-Wickets (light grey)
    ax_map.scatter(
        release_non_wickets["ReleaseY"], release_non_wickets["ReleaseZ"], 
        s=40, color='#D3D3D3', alpha=0.8, edgecolors='white', linewidths=0.5, label="No Wicket"
    )

    # Wickets (red)
    ax_map.scatter(
        release_wickets["ReleaseY"], release_wickets["ReleaseZ"], 
        s=80, color='red', alpha=1.0, edgecolors='white', linewidths=1.0, label="Wicket", zorder=5
    )
    
    # Add Stump Lines
    stump_lines = [-0.18, 0, 0.18]
    for y_val in stump_lines:
        ax_map.axvline(x=y_val, color="#777777", linestyle="--", linewidth=1.0)
    
    # Formatting Map
    ax_map.set_xlim(-1.5, 1.5)
    ax_map.set_ylim(0.5, 2.5)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_facecolor('white')
    ax_map.grid(True)

    
    # Hide all map spines
    for spine in ax_map.spines.values():
        spine.set_visible(False)
        
    # --- 4. Draw Lateral Metrics Table (ax_metrics) ---
    
    # Hide all metrics spines/ticks/labels
    ax_metrics.axis('off')
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(-0.5, 1)

    # Titles
    # Metric Labels (Left Alignment for labels)
    ax_metrics.text(0.05, 1, "W:", ha='right', va='center', fontsize=10, fontweight='bold')
    ax_metrics.text(0.05, 0.5, "Avg:", ha='right', va='center', fontsize=10, fontweight='bold')
    ax_metrics.text(0.05, 0, "SR:", ha='right', va='center', fontsize=10, fontweight='bold')

    # LEFT Values
    ax_metrics.text(0.2, 0.7, format_metric(left["Wickets"], is_wickets=True), ha='center', va='center', fontsize=12, color='red', fontweight='bold')
    ax_metrics.text(0.2, 0.45, format_metric(left["BA"]), ha='center', va='center', fontsize=12, color='darkred', fontweight='bold')
    ax_metrics.text(0.2, 0.2, format_metric(left["SR"]), ha='center', va='center', fontsize=12, color='darkred', fontweight='bold')

    # RIGHT Values
    ax_metrics.text(0.9, 0.7, format_metric(right["Wickets"], is_wickets=True), ha='center', va='center', fontsize=12, color='red', fontweight='bold')
    ax_metrics.text(0.9, 0.45, format_metric(right["BA"]), ha='center', va='center', fontsize=12, color='darkblue', fontweight='bold')
    ax_metrics.text(0.9, 0.2, format_metric(right["SR"]), ha='center', va='center', fontsize=12, color='darkblue', fontweight='bold')
    
    # --- 5. Add Sharp Border to Figure ---
    plt.tight_layout(pad=0.1)
    
    # Create and add a custom Rectangle patch for sharp border
    ax_bbox = ax_map.get_position()
    # Calculate padding relative to figure size
    padding_x = 0.001 * FIG_SIZE[0] / fig.get_size_inches()[0] 
    padding_y = 0.001 * FIG_SIZE[1] / fig.get_size_inches()[1] 
    
    border_rect = patches.Rectangle(
        (0, 0.0), 
        0.99, 
        0.99, 
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        transform=fig.transFigure,
        clip_on=False,
        joinstyle='miter' # Ensures sharp corners
    )
    fig.add_artist(border_rect)

    return fig

# --- CHARTS 6 & 7: SWING/DEVIATION DIRECTIONAL SPLIT (100% Stacked Bar) ---
def create_directional_split(df_in, column_name, handedness_label):
    from matplotlib import pyplot as plt
    import pandas as pd
    import matplotlib.patheffects as pe 
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    if df_in.empty or column_name not in df_in.columns:
        fig, ax = plt.subplots(figsize=(8, 1)); ax.text(0.5, 0.5, f"No Data or Missing '{column_name}'", ha='center', va='center'); ax.axis('off'); return fig

    # 1. Categorization (IF < 0 THEN "LEFT" ELSE "RIGHT")
    df = df_in.copy()
    # Note: Assuming negative value means movement towards the left
    df['Direction'] = df[column_name].apply(lambda x: 'LEFT' if x < 0 else 'RIGHT')
    
    # 2. Calculation
    total_balls = len(df)
    if total_balls == 0:
        fig, ax = plt.subplots(figsize=(8, 1)); ax.text(0.5, 0.5, "No Deliveries Found", ha='center', va='center'); ax.axis('off'); return fig
        
    df_counts = df['Direction'].value_counts().reset_index()
    df_counts.columns = ['Direction', 'Count']
    df_counts['Percentage'] = (df_counts['Count'] / total_balls) * 100
    
    # 3. Preparation for Stacked Bar
    df_plot = pd.DataFrame({
        'Direction': ['LEFT', 'RIGHT'],
        'Percentage': [0.0, 0.0]
    })
    
    df_plot.set_index('Direction', inplace=True)
    df_counts.set_index('Direction', inplace=True)
    df_plot.update(df_counts['Percentage'])
    df_plot = df_plot.T
    
    # 4. Chart Generation
    
    fig, ax = plt.subplots(figsize=(8, 1.5))
    
    # Define Colormap: 'Reds_r' is reversed Reds, applying a darker shade to the larger percentage
    cmap = cm.get_cmap('Reds') 
    norm = mcolors.Normalize(vmin=0, vmax=100)
    
    # Plotting order: LEFT first (starts at 0), then RIGHT
    categories = ['LEFT', 'RIGHT']
    left = 0
    
    for category in categories:
        pct = df_plot.loc['Percentage', category]
        
        # Get dynamic color based on percentage
        bar_color = cmap(norm(pct))
        
        # Plot the bar segment
        ax.barh(
            y=[0], 
            width=pct, 
            left=left, 
            color=bar_color, 
            label=category,
            height=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add percentage label (using outline for visibility)
        if pct > 0.5:
            # Use black text if segment is light (low percentage), white otherwise
            text_color = 'black' if pct < 30 else 'white'
            
            ax.text(
                left + pct / 2, 
                0, 
                f'{category.upper()}\n{pct:.0f}%', 
                ha='center', va='center', 
                color=text_color, fontsize=18, fontweight='bold',
                # Path effects give text a sharp edge against the background
                path_effects=[pe.withStroke(linewidth=2, foreground='none')] 
            )
        
        left += pct

    # 5. Formatting (Minimalist Look)
    # Hide all axis ticks, labels, and borders
    ax.set_xlim(0, 100)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_yticklabels([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

# Chart 8: Swing Distribution
def create_swing_distribution_histogram(df_in, handedness_label):
    # 0. Initial Check
    if df_in.empty or "Swing" not in df_in.columns:
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.text(0.5, 0.5, f"No Swing data for ({handedness_label})", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Ensure 'Swing' is not NaN and is numeric
    swing_data = df_in["Swing"].dropna().astype(float)
    if swing_data.empty:
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.text(0.5, 0.5, f"No valid Swing data for ({handedness_label})", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Define Bins of Size 1
    min_swing = np.floor(swing_data.min())
    max_swing = np.ceil(swing_data.max())
    bins = np.arange(min_swing, max_swing + 1.1, 1)

    # 2. Calculate Counts (N) and Bin Edges
    counts, bin_edges = np.histogram(swing_data, bins=bins)
    total_balls = len(swing_data)
    percentages = (counts / total_balls) * 100

    # 3. Prepare for plotting: Bar centers and labels
    # Use the lower edge of the bin for positioning and labeling
    lower_bin_edges = bin_edges[:-1] # Exclude the final upper boundary
    bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.9

    # Create tick labels: use only the lower limit of the bin
    # Use floor to ensure clean integer/single decimal labels
    tick_labels = [f"{b:.0f}" for b in lower_bin_edges] 

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot the bars, centered correctly
    rects = ax.bar(bar_centers, percentages, width=bar_width, 
                   color='red', linewidth=1.0)

    ax.set_xticks(lower_bin_edges)
    ax.set_xticklabels(tick_labels, ha='right', fontsize=16)
    
    # 5. Annotation (Percentages on top of bars)
    for rect, pct in zip(rects, percentages):
        if pct > 0:
            height = rect.get_height()
            # Ensure text is readable: only show % if > 0.5%
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5,
                    f'{pct:.0f}%',
                    ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax.set_ylim(0, percentages.max() * 1.25 if percentages.max() > 0 else 10)
    # Hide X and Y ticks and tick labels
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # Hide axis spines (the border lines)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    
    return fig
    
#Chart 9 Deviation Dstribution
def create_deviation_distribution_histogram(df_in, handedness_label):
    # 0. Initial Check
    if df_in.empty or "Deviation" not in df_in.columns:
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.text(0.5, 0.5, f"No Deviation data for ({handedness_label})", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Ensure 'Deviation' is not NaN and is numeric
    Deviation_data = df_in["Deviation"].dropna().astype(float)
    if Deviation_data.empty:
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.text(0.5, 0.5, f"No valid Deviation data for ({handedness_label})", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Define Bins of Size 1
    min_Deviation = np.floor(Deviation_data.min())
    max_Deviation = np.ceil(Deviation_data.max())
    bins = np.arange(min_Deviation, max_Deviation + 1.1, 1)

    # 2. Calculate Counts (N) and Bin Edges
    counts, bin_edges = np.histogram(Deviation_data, bins=bins)
    total_balls = len(Deviation_data)
    percentages = (counts / total_balls) * 100

    # 3. Prepare for plotting: Bar centers and labels
    # Use the lower edge of the bin for positioning and labeling
    lower_bin_edges = bin_edges[:-1] # Exclude the final upper boundary
    bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.9 

    # Create tick labels: use only the lower limit of the bin
    # Use floor to ensure clean integer/single decimal labels
    tick_labels = [f"{b:.0f}" for b in lower_bin_edges] 

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot the bars, centered correctly
    rects = ax.bar(bar_centers, percentages, width=bar_width, 
                   color='red', linewidth=1.0)
    
    ax.set_xticks(lower_bin_edges)
    ax.set_xticklabels(tick_labels, ha='right', fontsize=16)
    
    # 5. Annotation (Percentages on top of bars)
    for rect, pct in zip(rects, percentages):
        if pct > 0:
            height = rect.get_height()
            # Ensure text is readable: only show % if > 0.5%
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5,
                    f'{pct:.0f}%',
                    ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax.set_ylim(0, percentages.max() * 1.25 if percentages.max() > 0 else 10)
    # Hide X and Y ticks and tick labels
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # Hide axis spines (the border lines)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()

# PAGE SETUP LAYOUT

st.set_page_config(
    layout="wide"
)

# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']

# 2. BASE FILTER: ONLY SEAM DELIVERIES
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]

st.title("PACERS")

# --- Prepare Initial Filter Options ---
if "BowlingTeam" in df_seam_base.columns:
    team_column = "BowlingTeam"
else:
    team_column = "BattingTeam" 
    st.warning("The 'BowlingTeam' column was not found. Displaying all Batting Teams as a fallback.")

# 3. FILTERS (Bowling Team, Bowler, and Innings)
filter_col1, filter_col2, filter_col3 = st.columns(3) 

# --- Render Bowling Team Filter (Col 1) ---
all_teams = ["All"] + sorted(df_seam_base[team_column].dropna().unique().tolist())
with filter_col1:
    bowl_team = st.selectbox("Bowling Team", all_teams, index=0)

# --- Determine Bowlers based on selected Team ---
df_for_bowlers = df_seam_base.copy()

if bowl_team != "All":
    # Filter the DataFrame used for populating the bowler list
    df_for_bowlers = df_for_bowlers[df_for_bowlers[team_column] == bowl_team]

if "BowlerName" in df_for_bowlers.columns:
    # Generate the list of bowlers from the team-filtered DataFrame
    relative_bowlers = ["All"] + sorted(df_for_bowlers["BowlerName"].dropna().unique().tolist())
else:
    relative_bowlers = ["All"]
    
# --- Render Bowler Name Filter (Col 2) ---
with filter_col2:
    bowler = st.selectbox("Bowler Name", relative_bowlers, index=0)

# --- Render Inningss Filter (Col 3) ---
Innings_options = ["All"]
if "Innings" in df_seam_base.columns:
    valid_Inningss = df_seam_base["Innings"].dropna().astype(int).unique()
    Innings_options.extend(sorted([str(i) for i in valid_Inningss]))
with filter_col3:
    selected_Innings = st.selectbox("Innings", Innings_options, index=0)

st.header(f"{bowler}")

# 4. Apply Filters to the Base Seam Data
df_filtered = df_seam_base.copy()

# Apply Team Filter
if bowl_team != "All":
    df_filtered = df_filtered[df_filtered[team_column] == bowl_team]
    
# Apply Bowler Filter (This uses the value selected in the relative dropdown)
if bowler != "All":
    if "BowlerName" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["BowlerName"] == bowler]
    else:
        st.warning("BowlerName column not found for filtering.")

# Apply Innings Filter
if selected_Innings != "All" and "Innings" in df_filtered.columns:
    Innings_int = int(selected_Innings)
    df_filtered = df_filtered[df_filtered["Innings"] == Innings_int]
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

# === LEFT COLUMN: AGAINST RIGHT-HANDED BATSMEN (RHB) ===
with col_rhb:
    st.markdown("###  V RIGHT-HAND BATSMAN")    
    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE ")
    st.pyplot(create_pacer_crease_beehive(df_rhb, "RHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    # st.pyplot(create_pacer_lateral_performance_boxes(df_rhb, "RHB"), use_container_width=True)
    
    # Chart 2: ZONAL ANALYSIS (CBH Boxes)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_pacer_zonal_analysis(df_rhb, "RHB"), use_container_width=True)

    # Chart 3: PITCHMAP
    pitch_map_col, run_pct_col = st.columns([1, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.pyplot(create_pacer_pitch_map(df_rhb), use_container_width=True)    
    with run_pct_col:
        st.markdown("##### ")
        st.pyplot(create_pacer_pitch_length_bars(df_rhb), use_container_width=True)


     # Chart 4/5: RELEASE
    pace_col, release_col = st.columns([2, 2])
    with pace_col:
        st.markdown("###### RELEASE SPEED")
        st.pyplot(create_pacer_release_speed_distribution(df_rhb, "RHB"), use_container_width=True)
    with release_col:
        st.markdown("###### RELEASE")
        st.pyplot(create_pacer_release_analysis(df_rhb, "RHB"), use_container_width=True)
        
    #Chart 8/9: Swing Deviation Distribution
    swing_dist, deviation_dist = st.columns([2,2])
    with swing_dist:
        st.pyplot(create_swing_distribution_histogram(df_rhb, "RHB"))
    with deviation_dist:
        st.pyplot(create_deviation_distribution_histogram(df_rhb, "RHB"))  
    
     # Chart 6/7: Lateral Movement
    swing_col, deviation_col = st.columns([2, 2]) 
    with swing_col:
        st.markdown("###### SWING")
        st.pyplot(create_directional_split(df_rhb, "Swing", "RHB"), use_container_width=True)
    with deviation_col:
        st.markdown("###### DEVIATION")
        st.pyplot(create_directional_split(df_rhb, "Deviation", "RHB"), use_container_width=True)


# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
with col_lhb:
    st.markdown("###  V LEFT-HAND BATSMAN)")

    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE")
    st.pyplot(create_pacer_crease_beehive(df_lhb, "LHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    # st.pyplot(create_pacer_lateral_performance_boxes(df_lhb, "LHB"), use_container_width=True)

    # Chart 2: ZONAL ANALYSIS (CBH Boxes)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_pacer_zonal_analysis(df_lhb, "LHB"), use_container_width=True)

    # Chart 3: PITCHMAP
    pitch_map_col, run_pct_col = st.columns([1, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.pyplot(create_pacer_pitch_map(df_lhb), use_container_width=True)    
    with run_pct_col:
        st.markdown("##### ")
        st.pyplot(create_pacer_pitch_length_bars(df_lhb), use_container_width=True)

    # Chart 4/5: RELEASE
    pace_col, release_col = st.columns([2, 2]) 
    with pace_col:
        st.markdown("###### RELEASE SPEED")
        st.pyplot(create_pacer_release_speed_distribution(df_lhb, "LHB"), use_container_width=True)
    with release_col:
        st.markdown("###### RELEASE")
        st.pyplot(create_pacer_release_analysis(df_lhb, "LHB"), use_container_width=True)
        
    #Chart 8/9: Swing Deviation Distribution
    swing_dist, deviation_dist = st.columns([2,2])
    with swing_dist:
        st.pyplot(create_swing_distribution_histogram(df_lhb, "LHB"))
    with deviation_dist:
        st.pyplot(create_deviation_distribution_histogram(df_lhb, "LHB"))
        
    # Chart 6/7: Lateral Movement
    swing_col, deviation_col = st.columns([2, 2]) 
    with swing_col:
        st.markdown("###### SWING")
        st.pyplot(create_directional_split(df_lhb, "Swing", "RHB"), use_container_width=True)
    with deviation_col:
        st.markdown("###### DEVIATION")
        st.pyplot(create_directional_split(df_lhb, "Deviation", "RHB"), use_container_width=True)
