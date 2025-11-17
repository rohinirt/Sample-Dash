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
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); return fig

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
         # 1. Set line style for all spines you want visible
        spine_color = 'black'
        spine_width = 0.5
        for spine_name in ['left', 'top', 'bottom','right']:  
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(spine_color)
            ax.spines[spine_name].set_linewidth(spine_width)
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

# Chart 2: CREASE BEEHIVE
def create_crease_beehive(df_in, delivery_type):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 5)); 
        ax.text(0.5, 0.5, "No data for Analysis", ha='center', va='center', fontsize=12); 
        ax.axis('off'); 
        return fig

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]
    boundaries = non_wickets_all[(non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)]
    regular_balls = non_wickets_all[(non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)]
    
    # --- Lateral Zone Data Prep (Chart 2b) ---
    df_lateral = df_in.copy()
    is_rhb = df_in["IsBatsmanRightHanded"].iloc[0] if not df_in.empty and "IsBatsmanRightHanded" in df_in.columns else True

    

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
    
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count")
        )
    )
    
    # 2. Define standard zone order (RHB: Left to Right == WOO to LEG)
    ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    summary = summary.reindex(ordered_zones).fillna(0)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else np.nan, axis=1)

    # 3. HANDEDNESS AWARE REVERSAL: Reverse order for LHB
    if not is_rhb:
        # Reverses the DataFrame for LHB (LEG, STUMPS, OUTSIDE OFF, WAY OUTSIDE OFF)
        summary = summary.iloc[::-1]

    

    # -----------------------------------------------------------
    # --- 1. SETUP SUBPLOTS (Increased Figure Width) ---
    # Increased width from 7 to 8 for a wider Beehive chart relative to height
    fig = plt.figure(figsize=(7, 5)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.005) 
    ax_bh = fig.add_subplot(gs[0, 0])      # Top subplot (Beehive)
    ax_boxes = fig.add_subplot(gs[1, 0])   # Bottom subplot (Lateral Boxes)
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
    # Ensure NaN/Inf values are handled before max calculation for robust normalization
    avg_values = summary["Avg Runs/Wicket"].replace([np.inf, -np.inf], np.nan)
    avg_max_val = avg_values.max() if avg_values.max() > 0 else 50
    avg_max = avg_max_val if avg_max_val > 50 else 50
    norm = mcolors.Normalize(vmin=0, vmax=avg_max)
    cmap = cm.get_cmap('Reds') 


    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
    
    # --- Conditional Logic for N/A Average (Wickets = 0) ---
        if np.isnan(avg) or avg == np.inf:
            color = 'white'
            text_color = 'black'
            avg_display = 'N/A'
        else:
            color = cmap(norm(avg))
            avg_display = f"{avg:.1f}"

            # Calculate text color for contrast (white on dark, black on light)
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        
        # Draw the Rectangle
        ax_boxes.add_patch(
        patches.Rectangle((left, 0), box_width, box_height, 
                          edgecolor="black", facecolor=color, linewidth=1)
    )
    
        # Label 1: Zone Name (Above the box)
        ax_boxes.text(left + box_width / 2, box_height + 0.1, 
                  index, 
                  ha='center', va='bottom', fontsize=7, color='black')
    
        # Label 2: Wickets and Average (Middle of the box)
        label_wkts_avg = f"{wkts}W - Ave {avg_display}"
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
    
    # 1. Ensure plots are drawn tight (removes outer whitespace)
    ## --- 4. DRAW SINGLE COMPACT BORDER WITH PADDING ---
    
    # 1. Ensure plots are drawn tight
    plt.tight_layout(pad=0.2)
    
    # Define Padding Value (in figure coordinates)
    PADDING = 0.008

    # 2. Get the bounding box of the two subplots in Figure coordinates
    bh_bbox = ax_bh.get_position()
    box_bbox = ax_boxes.get_position()
    
    # Determine the total bounds (original compact bounds)
    x0_orig = min(bh_bbox.x0, box_bbox.x0)
    y0_orig = box_bbox.y0
    x1_orig = max(bh_bbox.x1, box_bbox.x1)
    y1_orig = bh_bbox.y1
    
    # 3. Apply Padding
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    # Width and Height must be increased by 2*PADDING (one for each side)
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    # 4. Draw the custom Rectangle using the padded bounds
    border_rect = patches.Rectangle(
        (x0_pad, y0_pad), 
        width_pad, 
        height_pad,  
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, # Use the figure's coordinate system
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig


# --- CHART 3: PITCHMAP ---
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
        # Create an empty figure with a text note if data is missing
        fig, ax = plt.subplots(figsize=(4,6))
        ax.text(0.5, 0.5, f"No data for Pitch Map ({delivery_type})", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- Data Filtering ---
    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]
    
    # --- Chart Setup ---
    fig, ax = plt.subplots(figsize=(4,6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # --- Pitch Bins & Full Toss Adjustment ---
    PITCH_BINS = get_pitch_bins(delivery_type)
    
    # Add Full Toss bin based on delivery type
    if delivery_type == "Seam":
        PITCH_BINS["Full Toss"] = [-4.0, 1.2] 
    elif delivery_type == "Spin":
        PITCH_BINS["Full Toss"] = [-4.0, 1.22] 
    
    # --- 1. Add Zone Lines & Labels (Horizontal Lines) ---
    
    # Determine boundary Y values to draw lines (excluding the start of the lowest bin)
    # The 'Full Toss' bin is assumed to start at -4.0, which is the bottom plot limit.
    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0], reverse=True)

    for y_val in boundary_y_values:
        # ax.axhline is the Matplotlib equivalent of fig_pitch.add_hline
        ax.axhline(y=y_val, color="lightgrey", linewidth=1.0, linestyle="--")

    # Add zone labels (equivalent to fig_pitch.add_annotation)
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
        s=60, # Matplotlib size equivalent to Plotly size=10
        c='#D3D3D3', 
        edgecolor='white', 
        linewidths=1.0, 
        alpha=0.9,
        label="No Wicket"
    )

    # Wickets (red)
    ax.scatter(
        pitch_wickets["BounceY"], pitch_wickets["BounceX"], 
        s=90, # Matplotlib size equivalent to Plotly size=12
        c='red', 
        edgecolor='white', 
        linewidths=1.0, 
        alpha=0.95,
        label="Wicket"
    )
    # --- 2. Add Stump lines (Vertical Lines) ---
    # ax.axvline is the Matplotlib equivalent of fig_pitch.add_vline
    ax.axvline(x=-0.18, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(x=0.18, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(x=0, color="#777777", linestyle="--", linewidth=0.8)
    # --- 4. Layout (Axis and Spines) ---
    
    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    # Note: Matplotlib typically plots y-axis increasing upwards, but here we set 
    # the range from [16.0, -4.0] to reverse the axis and match the Plotly visual 
    # where lower values (closer to batter) are at the bottom.
    ax.set_ylim([16.0, -4.0])

    # Hide all axis elements (equivalent to visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    
    # Hide axis spines (plot border)
    # 1. Set line style for all spines you want visible
    spine_color = 'black'
    spine_width = 0.5
    for spine_name in ['left', 'top', 'bottom','right']:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(spine_color)
        ax.spines[spine_name].set_linewidth(spine_width)
        
    plt.tight_layout()
    
    return fig

# --- CHART 3b: PITCH LENGTH RUN % (EQUAL SIZED BOXES) ---
def create_pitch_length_bars(df_in, delivery_type):
    """
    Generates a figure with three vertically stacked horizontal bar charts 
    for Batting Average, Strike Rate, and Dismissals by Pitch Length.
    """
    # Increased height to accommodate three stacked charts comfortably
    FIG_SIZE = (4, 6) 
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Pitch Length Comparison", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Get the pitch bins and define order
    PITCH_BINS_DICT = get_pitch_bins(delivery_type)
    
    if delivery_type == "Seam":
        ordered_keys = ["Full","Length", "Short", "Bouncer" ]
    elif delivery_type == "Spin":
        ordered_keys = ["Over Pitched", "Full" , "Good", "Short"]
    else:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "Invalid Delivery Type", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Data Preparation
    def assign_pitch_length(x):
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
    
    # Calculate Metrics
    df_summary["Average"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (row["Runs"] if row["Balls"] > 0 else 0), axis=1
    )
    df_summary["StrikeRate"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1
    )
    # Categories for plotting (reversed for barh)
    categories = df_summary.index.tolist()[::-1]
    
    # 2. Chart Setup (3 Rows, 1 Column)
    # sharex=False is default, sharey=True forces Y-axis to be the same, 
    # which is what we want for aligning the bar labels.
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharey=True) 
    # Adjust space between charts to minimize it vertically
    plt.subplots_adjust(hspace=0.4) 

    metrics = ["Average", "StrikeRate", "Wickets"]
    titles = ["Batting Average", "Batting Strike Rate", "Dismissals"]
    colors = ['Reds', 'Reds', 'Reds']

    # Define limits for each chart to ensure proper scaling
    max_avg = df_summary["Average"].max() * 1.1 if df_summary["Average"].max() > 0 else 60
    max_sr = df_summary["StrikeRate"].max() * 1.1 if df_summary["StrikeRate"].max() > 0 else 100
    max_wkts = df_summary["Wickets"].max() * 1.5 if df_summary["Wickets"].max() > 0 else 5

    xlim_limits = {
        "Average": (0, max_avg),
        "StrikeRate": (0, max_sr),
        "Wickets": (0, max_wkts)
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
            if metric == "Wickets":
                label = f"{int(val)}"
            else:
                label = f"{val:.2f}"
            
            # Place label slightly to the right of the bar tip
            ax.text(val, j, label, 
                    ha='left', va='center', 
                    fontsize=10,fontweight = 'bold', color='black',
                    bbox=dict(facecolor='White', alpha=0.8, edgecolor='none', pad=2),
                    zorder=4)

        # --- Formatting ---
        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
        ax.set_facecolor('white')

        # Set Ticks and Spines
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', length=0) # Hide y ticks

        # Set Y-axis labels only on the bottom-most chart (ax[2])
        # This keeps the labels at the bottom, mimicking the style in your image
        if i == 2:
            ax.set_yticks(np.arange(len(categories)), labels=[c.upper() for c in categories], fontsize=9)
        else:
             # Remove y-tick labels for the top two charts
            ax.set_yticks(np.arange(len(categories)), labels=[''] * len(categories))
            
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False)

        # Hide x labels/ticks
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
    
  
# --- CHART 4a: INTERCEPTION SIDE-ON --- (Wide View)
# --- Helper function for Interception Bins ---

def get_interception_bins():
    """Defines the bins for the Crease Width Split chart."""
    return {
        "0m-1m": [0, 1],
        "1m-2m": [1, 2],
        "2m-3m": [2, 3],
        "3m+": [3, 100]  # Assuming max possible value is < 100
    }

def create_interception_side_on(df_in, delivery_type):
    # Define Figure Size (slightly narrower and taller for the vertical stack)
    FIG_WIDTH = 7
    FIG_HEIGHT = 5
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

    if df_in.empty or df_in["InterceptionX"].isnull().all():
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Combined Interception Analysis", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- SETUP GRID FOR TWO ROWS ---
    # Top: Scatter Plot (Larger) | Bottom: Bar Chart (Smaller)
    fig = plt.figure(figsize=FIG_SIZE)
    # Ratio: 80% for scatter plot, 20% for bar chart
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1) 
    
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    
    fig.patch.set_facecolor('white')

    # ----------------------------------------------------------------------
    ## --- PART 1: CHART 4a - INTERCEPTION SIDE-ON SCATTER (ax_scatter) ---
    # ----------------------------------------------------------------------
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()    
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    # 1. Plot Data (Layered for correct border visibility)
    
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
    ax_scatter.scatter(
        df_other["InterceptionX"] + 10, df_other["InterceptionZ"], 
        color='#D3D3D3', edgecolors='white', linewidths=0.3, s=40, label="Other"
    )
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
        ax_scatter.scatter(
            df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], 
            color=color_map[ctype],edgecolors='white', linewidths=0.3, s=60, label=ctype
        )

    # 2. Draw Vertical Dashed Lines with Labels (FIXED LINES: 0.0, 1.25, 2.0, 3.0)
    line_specs = {
        0.0: "Stumps",
        1.250: "Crease",
        2.000: "2m",     
        3.000: "3m" 
    }
    
    for x_val, label in line_specs.items():
        ax_scatter.axvline(x=x_val, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7)     
        ax_scatter.text(x_val, 1.45, label.split(':')[-1].strip(), ha='center', va='center', fontsize=8, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    ax_scatter.axhline(y=0.78, color="grey", linestyle="-", linewidth=0.5)
    # --- Annotation ---
    ax_scatter.text(0.1, 0.78, "Stumps Height", ha='left', va='bottom', fontsize=7, color="grey", transform=ax_scatter.transData)
    
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
    
    ax_scatter.set_xlim(x_limit_min, x_limit_max) 
    ax_scatter.set_ylim(0, y_limit) 
    # ... (Rest of the styling remains the same)
    ax_scatter.tick_params(axis='y', which='both', labelleft=False, left=False); ax_scatter.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['left'].set_visible(False)
    ax_scatter.spines['bottom'].set_visible(False)


    # ----------------------------------------------------------------------
    ## --- PART 2: CHART 4b - CREASE WIDTH SPLIT BARS (ax_bar) ---
    # ----------------------------------------------------------------------
    
    # 1. Data Preparation (Same as previous function)
    INTERCEPTION_BINS = get_interception_bins()
    ordered_keys = ["0m-1m", "1m-2m", "2m-3m", "3m+"]  # Order: Close to Wide
    COLORMAP = 'Reds'
    
    def assign_crease_width(x):
        for width, bounds in INTERCEPTION_BINS.items():
            if bounds[0] <= x < bounds[1]: return width
        return None

    df_crease = df_in.copy()
    df_crease["CreaseWidth"] = (df_crease["InterceptionX"] + 10).apply(assign_crease_width)
    
    df_summary = df_crease.groupby("CreaseWidth").agg(
        Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count")
    ).reset_index().set_index("CreaseWidth").reindex(ordered_keys).fillna(0)
    
    df_summary["Average"] = df_summary.apply(
    lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else np.nan, axis=1
)
    
    # 2. Plotting Equal Boxes
    num_boxes = len(ordered_keys)
    box_width = 1.0 / num_boxes 
    left = 0.0

    # Normalization for the heatmap colors
    # Use only non-NaN values for max for accurate normalization
    max_avg_val = df_summary["Average"].replace([np.inf, -np.inf], np.nan).max()
    max_avg = max_avg_val if max_avg_val > 0 else 100
    
    norm = mcolors.Normalize(vmin=0, vmax=max_avg)
    cmap = cm.get_cmap(COLORMAP)
    
    for index, row in df_summary.iterrows():
        wickets = row["Wickets"]
        avg = row["Average"] 
        
        # --- CONDITIONAL STYLING LOGIC (The Fix) ---
        if np.isnan(avg) or avg == np.inf:
            # If average is N/A (Wickets=0)
            avg_display = 'N/A'
            color = 'white'  # Rule: Bar should be white
            text_color = 'black' # Rule: Label should be black
        else:
            # If average is valid
            avg_display = f"{avg:.1f}"
            color = cmap(norm(avg)) 
            
            # Luminosity Check for Text Color
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
            
        # Draw the box  
        ax_bar.barh(
            y=0.5,             
            width=box_width,
            height=0.6,          
            left=left,         
            color=color,
            edgecolor='black',
            linewidth=0.4
        )
        
        # --- Text Label (Wickets and Average) ---
        label_text = f"{int(wickets)}W - Ave {avg_display}"
        
        center_x = left + box_width / 2
        center_y = 0.5
        
        ax_bar.text(
            center_x, center_y, 
            label_text,
            ha='center', va='center', 
            fontsize=9, # Adjusted fontsize for fitting
            fontweight = 'bold',
            color=text_color
        )
        
        # --- Crease Width Label (Top of the box) ---
        ax_bar.text(
            center_x, 0.8, 
            index,           
            ha='center', va='bottom', 
            fontsize=9, 
            color='black',
        )

        left += box_width

    # 3. Styling for Bar Chart
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1) 
    ax_bar.axis('off')


    # ----------------------------------------------------------------------
    ## --- PART 3: DRAW SINGLE COMPACT BORDER ---
    # ----------------------------------------------------------------------
    
    plt.tight_layout(pad=0.2) 
    
    PADDING = 0.005 

    # Get the bounding box of the top (scatter) and bottom (bar) charts
    scatter_bbox = ax_scatter.get_position()
    bar_bbox = ax_bar.get_position() 
    # Determine the total bounds (figure coordinates)
    x0_orig = scatter_bbox.x0         
    y0_orig = bar_bbox.y0  
    x1_orig = scatter_bbox.x1     
    y1_orig = scatter_bbox.y1         
    
    # Apply Padding
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    # Draw the custom Rectangle 
    border_rect = patches.Rectangle(
        (x0_pad-0.008, y0_pad+0.02), 
        width_pad+0.017, 
        height_pad,  
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, 
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig


    
# --- CHART 5: INTERCEPTION FRONT-ON --- (Distance vs Width)
def create_interception_front_on(df_in, delivery_type):
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        fig, ax = plt.subplots(figsize=(4, 6)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    fig_8, ax_8 = plt.subplots(figsize=(4, 6), subplot_kw={'xticks': [], 'yticks': []}) 

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
        ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), ha='left', va='center', fontsize=12, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Boundary lines (FIXED LINES: -0.18, 0.18)
    ax_8.axvline(x=-0.18, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax_8.axvline(x= 0.18, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax_8.axvline(x= 0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    
    # 3. Set Axes Limits and Labels (FIXED LIMITS: Y-axis -0.2 to 3.5)
    ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
    ax_8.tick_params(axis='y', which='both', labelleft=False, left=False); ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
     # Hide axis spines (plot border)
    # 1. Set line style for all spines you want visible
    spine_color = 'black'
    spine_width = 0.5
    
    for spine_name in ['left', 'top', 'bottom','right']:
        ax_8.spines[spine_name].set_visible(True)
        ax_8.spines[spine_name].set_color(spine_color)
        ax_8.spines[spine_name].set_linewidth(spine_width)
    plt.tight_layout(pad=0.5)
    return fig_8
    

# Chart 6 Scoring wagon wheel
def calculate_scoring_wagon(row):
    """Calculates the scoring area based on LandingX/Y coordinates and handedness."""
    LX = row.get("LandingX"); LY = row.get("LandingY"); RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: return None
    
    def atan_safe(numerator, denominator): return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    # Right Handed Batsman Logic
    if RH == True: 
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
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

# --- Main Combined Function ---
def create_wagon_wheel(df_in, delivery_type):
    FIG_WIDTH = 10.0
    FIG_HEIGHT = 15.1 # Adjusted height for the vertical stack
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Combined Scoring Analysis", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- SETUP GRID FOR TWO ROWS ---
    # Top: Wagon Wheel (Larger) | Bottom: Left/Right Split (Smaller)
    fig = plt.figure(figsize=FIG_SIZE)
    # Ratio: 75% for Wagon Wheel, 25% for Left/Right Split
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1) 
    
    ax_wagon = fig.add_subplot(gs[0, 0])
    ax_split = fig.add_subplot(gs[1, 0])
    
    fig.patch.set_facecolor('white')

    # ----------------------------------------------------------------------
    ## --- PART 1: CHART 6 - SCORING WAGON WHEEL (ax_wagon) ---
    # ----------------------------------------------------------------------
    wagon_summary = pd.DataFrame() 
    try:
        df_wagon = df_in.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(TotalRuns=("Runs", "sum"), FixedAngle=("FixedAngle", 'first')).reset_index().dropna(subset=["ScoringWagon"])
        
        handedness_mode = df_in["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
        
        if is_right_handed:
            # RHB areas start from Fine Leg (top left) and go clockwise
            all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] 
        else:
            # LHB areas start from Third Man (top left) and go clockwise
            all_areas = ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
            
        template_df = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]})

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0) 
        wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
        wagon_summary = wagon_summary.sort_values("ScoringWagon")
        
        total_runs = wagon_summary["TotalRuns"].sum()
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100 if total_runs > 0 else 0 
        
        wagon_summary["FixedAngle"] = pd.to_numeric(wagon_summary["FixedAngle"], errors='coerce').fillna(0).astype(int)
    
    except Exception as e:
        ax_wagon.text(0.5, 0.5, f"Wagon Wheel Calculation Error: {e}", ha='center', va='center', fontsize=8)
        ax_wagon.axis('off')
        return fig # Return early if data processing fails

    
    # --- Data Extraction and CRITICAL Validation ---
    angles = wagon_summary["FixedAngle"].tolist()
    run_percentages = wagon_summary["RunPercentage"].tolist() 
    labels = wagon_summary["ScoringWagon"].tolist()
    
    if not angles or all(a == 0 for a in angles):
        ax_wagon.text(0.5, 0.5, "Insufficient Wagon Wheel Data", ha='center', va='center', fontsize=8) 
        ax_wagon.axis('off')
        # Skip plotting the pie chart, but allow the rest of the combined chart to proceed
    else:
        # --- Color Logic (Top 1 Rank Only) ---
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

        # --- Plotting Call ---
        pie_output = ax_wagon.pie(
            angles, 
            colors=colors, 
            wedgeprops={"width": 1, "edgecolor": "black"}, 
            startangle=90, 
            counterclock=False, 
            autopct='%.0f', 
            pctdistance=0.6 # Keeps percentage label centered in radius
        )
        
        if len(pie_output) == 3:
            wedges, texts, autotexts = pie_output
        elif len(pie_output) == 2:
            wedges, texts = pie_output
            autotexts = [] # Assign an empty list if autotexts are missing
        else:
            # Handle unexpected plot output
            ax_wagon.text(0.5, 0.5, "Wagon Wheel Plotting Error", ha='center', va='center', fontsize=8)
            ax_wagon.axis('off')
            return fig
        
        # === CRITICAL FIX: CENTERING PERCENTAGE LABELS AND STYLING ===
        for i, autotext in enumerate(autotexts):
            if i >= len(run_percentages): break
                
            percent = run_percentages[i]
            
            # 1. Set the actual percentage text
            if percent > 0:
                autotext.set_text(f'{percent:.0f}%')
                
                # 💥 FIX: Ensure percentage text is centered in the slice (horizontally and vertically)
                autotext.set_horizontalalignment('center')
                autotext.set_verticalalignment('center')
                
                # Add a white stroke (outline) for text visibility
            else:
                autotext.set_text('')
                
            # 2. Set text color based on background color for contrast
            color_rgb = mcolors.to_rgb(colors[i])
            luminosity = 0.2126 * color_rgb[0] + 0.7152 * color_rgb[1] + 0.0722 * color_rgb[2]
            
            autotext.set_color('white' if luminosity < 0.5 and colors[i] == COLOR_HIGH else 'black') 
            autotext.set_fontsize(20)
            autotext.set_fontweight('bold')
        
        ax_wagon.axis('equal'); 

    # ----------------------------------------------------------------------
    ## --- PART 2: CHART 7 - LEFT/RIGHT SCORING SPLIT (ax_split) ---
    # ----------------------------------------------------------------------
    
    df_split = df_in.copy()
    
    # 1. Define Side based on LandingY
    df_split["Side"] = np.where(df_split["LandingY"] < 0, "LEFT", "RIGHT")
    
    # 2. Calculate Runs and Percentage
    summary = df_split.groupby("Side")["Runs"].sum().reset_index()
    total_runs = summary["Runs"].sum()
    
    if total_runs == 0:
        ax_split.text(0.5, 0.5, "No Runs Scored in Split", ha='center', va='center', fontsize=8) 
        ax_split.axis('off')
    else:
        summary["Percentage"] = (summary["Runs"] / total_runs) * 100
        summary = summary.set_index("Side").reindex(["LEFT", "RIGHT"]).fillna(0)
        
        left_pct = summary.loc["LEFT", "Percentage"]
        right_pct = summary.loc["RIGHT", "Percentage"]

        # 3. Apply Color Map (Reds hue based on percentage)
        norm = mcolors.Normalize(vmin=0, vmax=100)
        cmap = cm.get_cmap('Reds') 
        left_color = cmap(norm(left_pct))
        right_color = cmap(norm(right_pct))
        
        # 4. Create the 100% Stacked Bar Chart
        ax_split.barh("Total", left_pct, color=left_color, edgecolor='black', linewidth=0.5)
        ax_split.barh("Total", right_pct, left=left_pct, color=right_color, edgecolor='black', linewidth=0.5)
        
        # Add labels
        def get_text_color(rgb_color):
            r, g, b = mcolors.to_rgb(rgb_color)
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return 'white' if luminosity < 0.5 else 'black'

        if left_pct > 0:
            text_color_left = get_text_color(left_color)
            ax_split.text(left_pct / 2, 0, f"LEFT\n{left_pct:.0f}%", 
                          ha='center', va='center', color=text_color_left, weight='bold', fontsize=20)
            
        if right_pct > 0:
            text_color_right = get_text_color(right_color)
            ax_split.text(left_pct + right_pct / 2, 0, f"RIGHT\n{right_pct:.0f}%", 
                          ha='center', va='center', color=text_color_right, weight='bold', fontsize=20)

        # 5. Styling
        ax_split.set_xlim(0, 100)
        ax_split.axis('off') # Hide all axes/ticks


    # ----------------------------------------------------------------------
    ## --- PART 3: DRAW SINGLE COMPACT BORDER ---
    # ----------------------------------------------------------------------
    
    plt.tight_layout() 
    
    PADDING = 0.005 

    # Get the bounding box of the top (wagon) and bottom (split) charts
    wagon_bbox = ax_wagon.get_position()
    split_bbox = ax_split.get_position()
    
    # Determine the total bounds (figure coordinates)
    x0_orig = min(wagon_bbox.x0, split_bbox.x0)         
    y0_orig = split_bbox.y0         
    x1_orig = max(wagon_bbox.x1, split_bbox.x1)     
    y1_orig = wagon_bbox.y1         
    
    # Apply Padding
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    # Draw the custom Rectangle 
    border_rect = patches.Rectangle(
        (x0_pad-0.03, y0_pad-0.053), 
        width_pad+0.05, 
        height_pad+0.038,  
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, 
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig

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
    ax_dir.set_yticks([]) 
    ax_dir.set_yticklabels([]) 

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
    
    
    # Add a subtle vertical line at x=0 for the axis center
    ax_dir.axvline(0, color='gray', linewidth=0.8)
    spine_color = 'black'
    spine_width = 0.5
    for spine_name in ['left', 'top', 'bottom','right']:
        ax_dir.spines[spine_name].set_visible(True)
        ax_dir.spines[spine_name].set_color(spine_color)
        ax_dir.spines[spine_name].set_linewidth(spine_width)
    
    # Remove y-ticks
    ax_dir.tick_params(axis='y', which='both', length=0)
    ax_dir.tick_params(axis='x', which='both', length=0)
    
    plt.tight_layout(pad=1.0)
    return fig_dir


# Set page title (optional, but good practice)
st.set_page_config(
    page_title="BATTERS",
    layout="wide"
)
st.title("BATTERS")
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

# Ensure columns exist before attempting to convert them
if "BatsmanName" in df_raw.columns:
    df_raw["BatsmanName"] = df_raw["BatsmanName"].astype(str).str.upper()
if "BowlerName" in df_raw.columns:
    # Assuming 'BowlerName' is used elsewhere, convert it here too for consistency
    df_raw["BowlerName"] = df_raw["BowlerName"].astype(str).str.upper()
# NOTE: BattingTeam is often case-sensitive, but converting Batsman/Bowler is key here.
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
    st.markdown("### v SEAM")

    # Row 1: Zonal Analysis (Beehive Zones)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_zonal_analysis(df_seam, batsman, "Seam"), use_container_width=True)
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE")
    st.pyplot(create_crease_beehive(df_seam, "Seam"), use_container_width=True)
    
    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP")
        st.pyplot(create_pitch_map(df_seam, "Seam"), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_length_bars(df_seam, "Seam"), use_container_width=True)   

    # Row 5: Interception Side-On (Wide View)
    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-VIEW")
    st.pyplot(create_interception_side_on(df_seam, "Seam"), use_container_width=True)

    # Row 7: Interception and Scoring Areas (Side-by-Side)
    bottom_col_left, bottom_col_right = st.columns(2)
    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-VIEW")
        st.pyplot(create_interception_front_on(df_seam, "Seam"), use_container_width=True)
        
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")    
        # Two charts stacked vertically in the right column
        st.pyplot(create_wagon_wheel(df_seam, "Seam"), use_container_width=True)
        
    
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
    st.markdown("### v SPIN")
    
    # Row 1: Zonal Analysis (Beehive Zones)
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(create_zonal_analysis(df_spin, batsman, "Spin"), use_container_width=True)
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE")
    st.pyplot(create_crease_beehive(df_spin, "Spin"), use_container_width=True)
 

    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP")
        st.pyplot(create_pitch_map(df_spin, "Spin"), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_length_bars(df_spin, "Spin"), use_container_width=True)    
    
    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-VIEW")
    st.pyplot(create_interception_side_on(df_spin, "Spin"), use_container_width=True)

    # Row 7: Interception Front-On and Scoring Areas (Side-by-Side)
    bottom_col_left, bottom_col_right = st.columns(2)

    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-VIEW")
        st.pyplot(create_interception_front_on(df_spin, "Spin"), use_container_width=True)
        
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")
        st.pyplot(create_wagon_wheel(df_spin,'djf'), use_container_width=True)
            

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
