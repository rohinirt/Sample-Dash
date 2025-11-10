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
from plotly.subplots import make_subplots



def create_spinner_crease_beehive(df_in):
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
def create_spinner_lateral_performance_boxes(df_in):

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

# --- CHART 2a: PITCH MAP (BOUNCE LOCATION) ---
def create_spinner_pitch_map(df_in):
    # Imports needed if not at the top of the file
    import plotly.graph_objects as go
    if df_in.empty:
        return go.Figure().update_layout(title=f"No data for Pitch Map (Seam)", height=300)

    PITCH_BINS = {
            "Over Pitched": [1.22, 2.22],
            "Full": [2.22, 4.0],
            "Good": [4.0, 6.0],
            "Short": [6.0, 15.0],
        }
    # Add a catch-all bin for Full Tosses (always Seam logic)
    PITCH_BINS["Full Toss"] = [-4.0, 1.2]  
        
    fig_pitch = go.Figure()
    
    # 1. Add Zone Lines & Labels
    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0])

    for y_val in boundary_y_values:
        fig_pitch.add_hline(y=y_val, line=dict(color="lightgrey", width=1.0, dash="dot"))

    # Add zone labels
    for length, bounds in PITCH_BINS.items():
        if length != "Full Toss": 
            mid_y = (bounds[0] + bounds[1]) / 2
            fig_pitch.add_annotation(x=-1.45, y=mid_y, text=length.upper(), showarrow=False,
                font=dict(size=8, color="grey", weight='bold'), xanchor='left')

    # 2. Add Stump lines
    fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0, line=dict(color="#777777", dash="dot", width=0.8))

   # 3. Plot Data (Wickets, Boundaries, and Others)
    
    # 1. Wickets (Highest priority, Red)
    pitch_wickets = df_in[df_in["Wicket"] == True]

    # 2. Boundaries (Non-Wicket, Runs = 4 or 6, Royal Blue)
    pitch_boundaries = df_in[(df_in["Wicket"] == False) & (df_in["Runs"].isin([4, 6]))]

    # 3. Other Balls (Non-Wicket, Non-Boundary, Light Grey)
    # This filters for balls that are NOT wickets AND NOT boundaries (i.e., 0, 1, 2, 3 runs)
    pitch_other = df_in[(df_in["Wicket"] == False) & (~df_in["Runs"].isin([4, 6]))]

    # Plot Other Balls (Bottom Layer - Light Grey)
    fig_pitch.add_trace(go.Scatter(
        x=pitch_other["BounceY"], y=pitch_other["BounceX"], mode='markers', name="Other",
        marker=dict(color='#D3D3D3', size=10, line=dict(width=1, color="white"), opacity=0.9)
    ))
    # Plot Boundaries (Middle Layer - Royal Blue)
    fig_pitch.add_trace(go.Scatter(
        x=pitch_boundaries["BounceY"], y=pitch_boundaries["BounceX"], mode='markers', name="Boundary",
        marker=dict(color='royalblue', size=11, line=dict(width=1, color="white")), opacity=0.95)
    )
    
    # Plot Wickets (Top Layer - Red)
    fig_pitch.add_trace(go.Scatter(
        x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket",
        marker=dict(color='red', size=12, line=dict(width=1, color="white")), opacity=0.95)
    )
    

    # 4. Layout
    fig_pitch.update_layout(
        height = 400,
        margin=dict(l=0, r=100, t=30, b=10),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False, visible=False), 
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    
    return fig_pitch

# --- CHART 2b: PITCH LENGTH METRICS (BOWLER FOCUS) ---
def create_spinner_pitch_length_metrics(df_in):
    FIG_HEIGHT = 5.7
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(2, FIG_HEIGHT)); 
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', rotation=90); 
        ax.axis('off'); 
        return fig

    PITCH_BINS_DICT = {
            "Over Pitched": [1.22, 2.22],
            "Full": [2.22, 4.0],
            "Good": [4.0, 6.0],
            "Short": [6.0, 15.0],
        } # Simplified call
        
    # Define ordered keys for plotting order (far to near) - Only Seam
    ordered_keys = ["Bouncer", "Short", "Length", "Full"]
    COLORMAP = 'Reds' # Red indicates higher runs/run percentage (worse for bowler)
    
    # 1. Data Preparation
    def assign_pitch_length(x):
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]: return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_length)
    
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
    
    # --- CALCULATE BOWLER METRICS ---
    df_summary["Average"] = df_summary.apply(
        # Bowling Average (Runs / Wickets)
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else (0), axis=1
    )
    df_summary["StrikeRate"] = df_summary.apply(
        # Bowling Strike Rate (Balls / Wickets)
        lambda row: row["Balls"] / row["Wickets"] if row["Wickets"] > 0 else (0), axis=1
    )
    # --- Calculate Run Percentage (for color mapping) ---
    total_runs = df_summary["Runs"].sum()
    df_summary["RunPercentage"] = (df_summary["Runs"] / total_runs) * 100 if total_runs > 0 else 0

    # 2. Chart Setup
    fig_stack, ax_stack = plt.subplots(figsize=(2, FIG_HEIGHT)) 

    # Plotting setup variables
    num_boxes = len(ordered_keys)
    box_height = 1.0 / num_boxes
    bottom = 0.0
    
    # Colormap and Normalization based on Run Percentage
    max_pct = df_summary["RunPercentage"].max() if df_summary["RunPercentage"].max() > 0 else 100
    norm = mcolors.Normalize(vmin=0, vmax=max_pct)
    cmap = cm.get_cmap(COLORMAP)
    
    # 3. Plotting Equal Boxes (Vertical Heat Map)
    for index, row in df_summary.iterrows():
        pct = row["RunPercentage"]
        wkts = int(row["Wickets"])
        avg = row["Average"] 
        sr = row["StrikeRate"] 
        
        # Determine box color
        color = cmap(norm(pct))
        
        # Draw the box 
        ax_stack.bar( 
            x=0.5,            
            height=box_height,
            width=1,          
            bottom=bottom,    
            color=color,
            edgecolor='black', 
            linewidth=1
        )
        
        # Add labels - UPDATING LABEL TEXT
        label_text = (
            f"{index.upper()}\n"
            f"{pct:.0f}% Runs\n"
            f"W: {wkts}\n"
            f"Avg: {avg:.1f}\n"
            f"SR: {sr:.1f}"
        )
        
        # Calculate text color for contrast
        r, g, b = color[:3]
        luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = 'white' if luminosity < 0.5 else 'black'

        # Text plotting
        ax_stack.text(0.5, bottom + box_height / 2, 
                      label_text,
                      ha='center', va='center', fontsize=9, color=text_color, weight='bold', linespacing=1.2)
        
        bottom += box_height
        
    # 4. Styling
    ax_stack.set_xlim(0, 1); ax_stack.set_ylim(0, 1)
    ax_stack.axis('off') 

    plt.tight_layout(pad=0.9)
    return fig_stack
    
# --- CHART 3a: RELEASE SPEED DISTRIBUTION ---
def create_spinner_release_speed_distribution(df_in, handedness_label):
    from matplotlib import pyplot as plt
    import pandas as pd
    
    # 1. Define Speed Bins (in km/h) with simplified labels
    SPEED_BINS = {
        ">90": [90, 150],
        "85-90": [85, 90],
        "80-85": [80, 85],
        "75-80": [75, 80],
        "<75": [0, 75]      
    }
    # Define plotting order (Slowest to Fastest)
    ordered_bins = list(SPEED_BINS.keys())
    
    if df_in.empty or "ReleaseSpeed" not in df_in.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Data or Missing 'ReleaseSpeed'", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 2. Assign Balls to Speed Bins
    def assign_speed_bin(speed):
        for label, bounds in SPEED_BINS.items():
            if bounds[0] <= speed < bounds[1]:
                return label
        return None

    df_speed = df_in.copy()
    df_speed["SpeedBin"] = df_speed["ReleaseSpeed"].apply(assign_speed_bin)
    
    # 3. Aggregate Data (Total balls and Percentage)
    total_balls = len(df_speed) 
    
    if total_balls == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Deliveries Found", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Count balls in each bin
    df_summary = df_speed.groupby("SpeedBin").size().reset_index(name="Count")
    
    # Calculate percentage
    df_summary["Percentage"] = (df_summary["Count"] / total_balls) * 100

    # Prepare for plotting, ensuring correct order
    df_summary = df_summary.set_index("SpeedBin").reindex(ordered_bins, fill_value=0).reset_index()
    plot_data = df_summary.set_index("SpeedBin")
    
    # 4. Chart Generation (Simple Horizontal Bar)
    
    fig, ax = plt.subplots(figsize=(4,4.4))
    
    # Plot a single horizontal bar series
    ax.barh(
        plot_data.index,
        plot_data["Percentage"],
        color='Red', # Single, uniform color
        height=0.6,
        edgecolor='black',
        linewidth=0.5
    )

    
    # Add percentage and count labels
    for i, (bin_label, row) in enumerate(plot_data.iterrows()):
        pct = row["Percentage"]
        count = row["Count"]
        
        if pct > 0:
            # Display percentage and raw count (e.g., 25.4% (32 balls))
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

    # Set X-axis limit slightly higher than the max percentage for clean labels
    max_pct = plot_data["Percentage"].max()
    ax.set_xlim(0, max(max_pct * 1.1, 10)) 
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove legend as there is only one series
    plt.tight_layout()
    return fig

# --- CHART 5: RELEASE ZONE MAP ---
def create_spinner_release_zone_map(df_in, handedness_label):
    import plotly.graph_objects as go   
    if df_in.empty:
        return go.Figure().update_layout(title=f"No data for Release Zone Map vs. {handedness_label}", height=450)

    # 1. Calculate KPIs
    runs = df_in["Runs"].sum()
    wickets = (df_in["Wicket"] == True).sum()
    balls = len(df_in)
    
    average = runs / wickets if wickets > 0 else 0
    strike_rate = balls / wickets if wickets > 0 else 0
    
    # Format KPIs
    kpi_wickets = str(wickets)
    kpi_average = f"{average:.1f}" if average > 0 else "-"
    kpi_sr = f"{strike_rate:.1f}" if strike_rate > 0 else "-"

    # 2. Setup Figure
    fig = go.Figure()
    
    # Filter data for plotting
    release_wickets = df_in[df_in["Wicket"] == True]
    release_non_wickets = df_in[df_in["Wicket"] == False]
    
    # 3. Plot Data
    
    # Non-Wickets (light grey)
    fig.add_trace(go.Scatter(
        x=release_non_wickets["ReleaseY"], y=release_non_wickets["ReleaseZ"], mode='markers', name="No Wicket",
        marker=dict(color='#D3D3D3', size=7, opacity=0.8,line=dict(width=1, color="white")), hoverinfo='none'
    ))

    # Wickets (red)
    fig.add_trace(go.Scatter(
        x=release_wickets["ReleaseY"], y=release_wickets["ReleaseZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=9, line=dict(width=1, color="white")), opacity=1.0, hoverinfo='text',
        text=[f"Wicket<br>Speed: {s:.1f} km/h" for s in release_wickets["ReleaseSpeed"]]
    ))
    
    # 4. Add Stump Lines (Vertical)
    # Lines for Off Stump (-0.18), Middle (0), Leg Stump (0.18)
    stump_lines = [-0.18, 0, 0.18]
    for y_val in stump_lines:
        fig.add_vline(x=y_val, line=dict(color="#777777", dash="dot", width=1.0))
        
    # 5. Add KPI Annotations (FIXED TO XREF="PAPER")
    
    # Map data coordinates (-1.0, 0.0, 1.0) to paper coordinates (0.1, 0.5, 0.9)
    paper_x_map = {
        -1.0: 0.1, 
        0.0: 0.5, 
        1.0: 0.9
    }
    
    kpi_data = [
        ("Wickets", kpi_wickets, -1.0),
        ("Avg", kpi_average, 0.0),
        ("SR", kpi_sr, 1.0),
    ]
    
    # Add KPI Headers
    for label, _, x_data_pos in kpi_data:
        fig.add_annotation(
            x=paper_x_map[x_data_pos], y=-0.15, xref="paper", yref="paper", 
            text=f"<b>{label.upper()}</b>", showarrow=False, xanchor='center',
            font=dict(size=11, color="grey")
        )

    # Add KPI Values
    for _, value, x_data_pos in kpi_data:
        fig.add_annotation(
            x=paper_x_map[x_data_pos], y=-0.25, xref="paper", yref="paper", 
            text=f"<b>{value}</b>", showarrow=False, xanchor='center',
            font=dict(size=16, color="black")
        )
    # 6. Layout and Styling
    fig.update_layout(
        height = 250,
        margin=dict(l=0, r=0, t=0, b=0), # Increased bottom margin for KPIs
        xaxis=dict(
            range=[-1.5, 1.5], 
            showgrid=True,showticklabels=False, zeroline=False
        ),
        yaxis=dict(
            range=[0, 2.5], 
            showgrid=True,showticklabels=False, zeroline=False
        ), 
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    return fig

#Chart 5b: Release performance 
def create_spinner_releasey_performance(df_in, handedness_label):
    # 1. Define ReleaseY Category
    if df_in.empty or "ReleaseY" not in df_in.columns:
        fig, ax = plt.subplots(figsize=(7, 2)); 
        ax.text(0.5, 0.5, f"No ReleaseY data for ({handedness_label})", ha='center', va='center'); 
        ax.axis('off'); 
        return fig
        
    df_temp = df_in.copy()
    
    # Categorize based on ReleaseY sign
    df_temp["ReleaseCategory"] = np.where(
        df_temp["ReleaseY"] < 0, "LEFT (<0)", 
        np.where(df_temp["ReleaseY"] > 0, "RIGHT (>0)", "CENTER (=0)")
    )
    
    # Filter out "CENTER" if you only want the two main lateral zones
    df_temp = df_temp[df_temp["ReleaseCategory"] != "CENTER (=0)"]
    
    if df_temp.empty:
        fig, ax = plt.subplots(figsize=(7, 2)); 
        ax.text(0.5, 0.5, f"No lateral release data for ({handedness_label})", ha='center', va='center'); 
        ax.axis('off'); 
        return fig

    # 2. Calculate Metrics
    summary = df_temp.groupby("ReleaseCategory").agg(
        Wickets=("Wicket", lambda x: (x == True).sum()),
        Runs=("Runs", "sum"),
        Balls=("Wicket", "count")
    )
    
    # Ensure both categories are present
    if "LEFT (<0)" not in summary.index: summary.loc["LEFT (<0)"] = [0, 0, 0]
    if "RIGHT (>0)" not in summary.index: summary.loc["RIGHT (>0)"] = [0, 0, 0]
    
    # Calculate BA and SR (using 999.0 for "N/A" equivalent)
    summary["BA"] = summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 999.0, axis=1)
    summary["SR"] = summary.apply(
        lambda row: row["Balls"] / row["Wickets"] * 6 if row["Wickets"] > 0 else 999.0, axis=1)
    
    # Reorder the index for consistent plotting order (e.g., LEFT then RIGHT)
    summary = summary.reindex(["LEFT (<0)", "RIGHT (>0)"])

    # 3. Chart Setup
    metrics = ["Wickets", "BA", "SR"]
    titles = ["Wickets", "Bowling Average", "Bowling Strike Rate"]
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 2)) # Adjust figsize as needed for your Streamlit layout
    plt.subplots_adjust(wspace=0.3) 

    colors = ['#1f77b4', '#d62728'] # Using blue/red for contrast
    y_labels = summary.index.tolist()
    
    # Determine maximum value for setting consistent x-limits
    max_wickets = summary["Wickets"].max() * 1.2
    max_ba = summary[summary["BA"] < 999.0]["BA"].max() * 1.2
    max_sr = summary[summary["SR"] < 999.0]["SR"].max() * 1.2

    max_values = {
        "Wickets": max_wickets if max_wickets > 0 else 5,
        "BA": max_ba if max_ba > 0 else 100,
        "SR": max_sr if max_sr > 0 else 100,
    }

    # 4. Plotting Loop
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = summary[metric].tolist()
        
        # Bar plotting
        # Note: Matplotlib automatically plots bars in the order of the y_labels list
        bars = ax.barh(y_labels, data, 
                       color=[colors[0], colors[1]], 
                       height=0.5, zorder=3) 
        
        # Titles
        ax.set_title(titles[i], fontsize=10, pad=5)
        
        # X-axis limits (Hiding ticks and labels)
        ax.set_xlim(0, max_values[metric])
        ax.xaxis.set_visible(False) 
        
        # Y-axis Labels (Only show on the first chart)
        if i == 0:
            ax.tick_params(axis='y', length=0) 
            ax.set_yticks([0, 1])
            ax.set_yticklabels(y_labels, fontsize=10, weight='bold', color='black')
        else:
            ax.yaxis.set_visible(False) 
            
        # Add labels on the bars
        for bar, value in zip(bars, data):
            if metric == "Wickets":
                text = f"{int(value)}"
            else:
                text = f"{value:.2f}" if value < 999.0 else "N/A"
            
            # Place the text inside the bar
            ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2, 
                    text, 
                    ha='right', va='center', fontsize=9, color='white', 
                    weight='bold', zorder=10)

        # Hide axis spines (borders) and gridlines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False) 
    
    plt.tight_layout(pad=0.5)
    
    return fig
# --- CHARTS 6 & 7: SWING/DEVIATION DIRECTIONAL SPLIT (100% Stacked Bar) ---
def create_directional_split(df_in, column_name, handedness_label):
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

# --- CHART 8: HITTING VS MISSING STUMPS MAP ---
def create_spinner_hitting_missing_map(df_in, handedness_label):

    # 0️⃣ Early exit if data is empty
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, f"No data for Hitting/Missing ({handedness_label})",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    df_map = df_in.copy()

    # 1️⃣ Define Hitting/Missing Category (Target box: Y=[-0.18, 0.18], Z=[0, 0.78])
    is_hitting_target = (
        (df_map["StumpsY"] >= -0.18) &
        (df_map["StumpsY"] <= 0.18) &
        (df_map["StumpsZ"] >= 0) &
        (df_map["StumpsZ"] <= 0.78)
    )
    df_map["HittingCategory"] = np.where(is_hitting_target, "HITTING", "MISSING")

    # 2️⃣ Calculate Percentages
    if not df_map.empty:
        counts = df_map["HittingCategory"].value_counts(normalize=True).mul(100).round(1)
        hitting_pct = counts.get("HITTING", 0.0)
        missing_pct = counts.get("MISSING", 0.0)
    else:
        hitting_pct = 0.0
        missing_pct = 0.0

    # 3️⃣ Setup Figure
    fig, ax = plt.subplots(figsize=(7, 4))

    # 4️⃣ Split Data
    df_missing = df_map[df_map["HittingCategory"] == "MISSING"]
    df_hitting = df_map[df_map["HittingCategory"] == "HITTING"]
    ax.axvline(x=-0.18, color='grey', linestyle='--', linewidth=1, zorder=20)
    ax.axvline(x=0, color='grey', linestyle=':', linewidth=1, zorder=20)
    ax.axvline(x=0.18, color='grey', linestyle='--', linewidth=1, zorder=20)
    ax.axhline(y=0.78, color='grey', linestyle='--', linewidth=1, zorder=20)

    # 5️⃣ Plot MISSING (Grey)
    ax.scatter(
        df_missing["StumpsY"], df_missing["StumpsZ"],
        color='#D3D3D3', s=45, edgecolor='white',
        linewidth=0.4, alpha=0.8, label='_nolegend_'
    )

    # 6️⃣ Plot HITTING (Red)
    ax.scatter(
        df_hitting["StumpsY"], df_hitting["StumpsZ"],
        color='red', s=55, edgecolor='white',
        linewidth=0.4, alpha=0.9, label='_nolegend_'
    )

    # 8️⃣ Format Plot
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.4)
    ax.axis('off')  # clean look, no ticks
    plt.tight_layout(pad=0.5)

    # 🔟 Add Hitting and Missing Text Labels
    ax.text(
        1.05, 1.35, f"Hitting: {hitting_pct:.0f}%",
        transform=ax.transData, ha='right', va='top',
        fontsize=14, color='red', weight='bold'
    )

    ax.text(
        1.05, 1.25, f"Missing: {missing_pct:.0f}%",
        transform=ax.transData, ha='right', va='top',
        fontsize=14, color='#D3D3D3', weight='bold'
    )

    return fig

# Chart 9 Hitting Missing Performance
def create_spinner_h_m_performance_bars(df_in, handedness_label):
    # 1. Define Hitting/Missing Category
    is_hitting_target = (
        (df_in["StumpsY"] >= -0.18) & 
        (df_in["StumpsY"] <= 0.18) &
        (df_in["StumpsZ"] >= 0) & 
        (df_in["StumpsZ"] <= 0.78)
    )
    df_in["HittingCategory"] = np.where(is_hitting_target, "HITTING", "MISSING")
    
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 2)); 
        ax.text(0.5, 0.5, f"No Data ({handedness_label})", ha='center', va='center'); 
        ax.axis('off'); 
        return fig

    # 2. Calculate Metrics
    summary = df_in.groupby("HittingCategory").agg(
        Wickets=("Wicket", lambda x: (x == True).sum()),
        Runs=("Runs", "sum"),
        Balls=("Wicket", "count")
    )
    
    # Ensure both categories are present
    if "HITTING" not in summary.index: summary.loc["HITTING"] = [0, 0, 0]
    if "MISSING" not in summary.index: summary.loc["MISSING"] = [0, 0, 0]
    
    # Calculate BA and SR (using 999.0 for "N/A" equivalent)
    summary["BA"] = summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 999.0
    , axis=1)
    summary["SR"] = summary.apply(
        lambda row: row["Balls"] / row["Wickets"] * 6 if row["Wickets"] > 0 else 999.0
    , axis=1)
    

    # 3. Chart Setup
    metrics = ["Wickets", "BA", "SR"]
    titles = ["Wickets", "Average", "Strike Rate"]
    
    # Use figsize=(7, 2) for a compact, horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(7, 2))
    plt.subplots_adjust(wspace=0.3) # Adjust space between the subplots

    colors = ['red', '#A9A9A9'] # Red for HITTING, Grey for MISSING
    y_labels = summary.index.tolist()
    
    # Determine maximum value for setting consistent x-limits
    max_wickets = summary["Wickets"].max() * 1.2
    max_ba = summary[summary["BA"] < 999.0]["BA"].max() * 1.2
    max_sr = summary[summary["SR"] < 999.0]["SR"].max() * 1.2

    max_values = {
        "Wickets": max_wickets if max_wickets > 0 else 5,
        "BA": max_ba if max_ba > 0 else 100,
        "SR": max_sr if max_sr > 0 else 100,
    }

    # 4. Plotting Loop
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Data for the current metric (HITTING, then MISSING)
        data = summary[metric].tolist()
        
        # Bar plotting - using barh for horizontal bars
        bars = ax.barh(y_labels, data, color=colors, height=0.5) # height=0.5 matches the bar width of the Plotly version
        
        # Titles
        ax.set_title(titles[i], fontsize=10, pad=5)
        
        # X-axis limits (Hiding ticks and labels)
        ax.set_xlim(0, max_values[metric])
        ax.xaxis.set_visible(False) 
        
        # Y-axis Labels (Only show on the first chart)
        if i == 0:
            ax.tick_params(axis='y', length=0) # Remove tick marks
            ax.set_yticks([0, 1])
            ax.set_yticklabels(y_labels, fontsize=10, weight='bold', color='black')
        else:
            ax.yaxis.set_visible(False) 
            
        # Add labels on the bars
        for bar, value in zip(bars, data):
            if metric == "Wickets":
                text = f"{int(value)}"
            else:
                if value >= 999.0:
                    text = "N/A"
                else:
                    text = f"{value:.2f}" 
            
            # Place the text inside the bar
            ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2, 
                    text, 
                    ha='right', va='center', fontsize=9, color='white', 
                    weight='bold', zorder=10)

        # Hide axis spines (borders) and gridlines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False) # Explicitly turn off grid
    
    plt.tight_layout(pad=0.5)
    
    return fig

# Chart 9: Swing Distribution
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
    
#Chart 10 Deviation Dstribution
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
    
# =========================================================
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

st.title("SPINNERS")

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
col_rhb, col_lhb = st.columns(2)
# === LEFT COLUMN: AGAINST RIGHT-HANDED BATSMEN (RHB) ===
with col_rhb:
    st.markdown("###  Right-Handed Batsmen (RHB)")    
    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE ")
    st.plotly_chart(create_spinner_crease_beehive(df_rhb), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    st.pyplot(create_spinner_lateral_performance_boxes(df_rhb), use_container_width=True)


    # Chart 2: PITCHMAP
    pitch_map_col, run_pct_col = st.columns([3, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.plotly_chart(create_spinner_pitch_map(df_rhb), use_container_width=True)    
    with run_pct_col:
        st.markdown("##### ")
        st.pyplot(create_spinner_pitch_length_metrics(df_rhb), use_container_width=True) 

    # Chart 3: RELEASE
    pace_col, release_col = st.columns([2, 2])
    with pace_col:
        st.markdown("###### RELEASE SPEED")
        st.pyplot(create_spinner_release_speed_distribution(df_rhb, "RHB"), use_container_width=True)
    with release_col:
        st.markdown("###### RELEASE")
        st.plotly_chart(create_spinner_release_zone_map(df_rhb, "RHB"), use_container_width=True)
        st.pyplot(create_spinner_releasey_performance(df_rhb, "RHB"))

    #Chart 9/10: Swing Deviation Distribution
    swing_dist, deviation_dist = st.columns([2,2])
    with swing_dist:
        st.pyplot(create_swing_distribution_histogram(df_rhb, "RHB"))
    with deviation_dist:
        st.pyplot(create_deviation_distribution_histogram(df_rhb, "RHB")) 
    
     # Chart 4: Lateral Movement
    swing_col, deviation_col = st.columns([2, 2]) 
    with swing_col:
        st.markdown("###### DRIFT")
        st.pyplot(create_directional_split(df_rhb, "Swing", "RHB"), use_container_width=True)
    with deviation_col:
        st.markdown("###### TURN")
        st.pyplot(create_directional_split(df_rhb, "Deviation", "RHB"), use_container_width=True)
    
    # Chart 8: Missing Hitting    
    st.pyplot(create_spinner_hitting_missing_map(df_rhb, "RHB"), use_container_width=True)
    st.pyplot(create_spinner_h_m_performance_bars(df_rhb,"RHB"), use_container_width=True)
    
# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
    with col_lhb:
        st.markdown("###  Left-Handed Batsmen (LHB)")
         # Chart 1a: Crease Beehive (using the new local function)
        st.markdown("###### CREASE BEEHIVE")
        st.plotly_chart(create_spinner_crease_beehive(df_lhb), use_container_width=True)
        # Chart 1b: Lateral Performance Boxes (Bowling Avg)
        st.pyplot(create_spinner_lateral_performance_boxes(df_lhb), use_container_width=True)

        # Chart 2: PITCHMAP
        pitch_map_col, run_pct_col = st.columns([3, 1]) 
        with pitch_map_col:
            st.markdown("###### PITCHMAP")
            st.plotly_chart(create_spinner_pitch_map(df_lhb), use_container_width=True)    
        with run_pct_col:
            st.markdown("##### ")
            st.pyplot(create_spinner_pitch_length_metrics(df_lhb), use_container_width=True) 

         # Chart 4/5: RELEASE
        pace_col, release_col = st.columns([2, 2]) 
        with pace_col:
            st.markdown("###### RELEASE SPEED")
            st.pyplot(create_spinner_release_speed_distribution(df_lhb, "LHB"), use_container_width=True)
        with release_col:
            st.markdown("###### RELEASE")
            st.plotly_chart(create_spinner_release_zone_map(df_lhb, "LHB"), use_container_width=True)
            st.pyplot(create_spinner_releasey_performance(df_lhb, "RHB"))

        #Chart 9/10: Swing Deviation Distribution
        swing_dist, deviation_dist = st.columns([2,2])
        with swing_dist:
            st.pyplot(create_swing_distribution_histogram(df_lhb, "RHB"))
        with deviation_dist:
            st.pyplot(create_deviation_distribution_histogram(df_lhb, "RHB"))
        
        # Chart 6/7: Lateral Movement
        swing_col, deviation_col = st.columns([2, 2]) 
        with swing_col:
            st.markdown("###### DRIFT")
            st.pyplot(create_directional_split(df_lhb, "Swing", "LHB"), use_container_width=True)
        with deviation_col:
            st.markdown("###### TURN")
            st.pyplot(create_directional_split(df_lhb, "Deviation", "LHB"), use_container_width=True)
        
        
        # Chart 8: Missing Hitting    
        st.pyplot(create_spinner_hitting_missing_map(df_lhb, "LHB"), use_container_width=True)
        # Assuming you use the columns col_rhb and col_lhb:
        st.pyplot(create_spinner_h_m_performance_bars(df_lhb,"LHB"), use_container_width=True)
