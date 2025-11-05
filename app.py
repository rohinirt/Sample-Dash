import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from io import StringIO
import base64

# --- 1. DASH INITIALIZATION & EXTERNAL STYLES ---
# Use a minimal Bootstrap theme for clean layout
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP])
server = app.server # Expose the Flask server instance for Render

# Custom CSS for absolute minimum spacing and full width
CUSTOM_CSS = """
/* Target main app content container */
.container-fluid {
    padding-left: 5px !important;
    padding-right: 5px !important;
    padding-top: 5px !important;
    padding-bottom: 5px !important;
}
/* Target card bodies/chart wrappers to reduce padding */
.card-body {
    padding: 5px !important;
}
/* Remove margin/padding from default Plotly graphs */
.dcc-graph {
    margin: 0px !important;
    padding: 0px !important;
}
/* Ensure the main content uses full width */
.dash-app {
    width: 100% !important;
}
"""

# Inject custom CSS
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

# --- 2. GLOBAL UTILITY FUNCTIONS (Adapted from Streamlit code) ---

# Global DataFrame to hold the loaded data
df = None
INITIAL_BATSMAN = "All"
INITIAL_TEAM = "All"

def calculate_scoring_wagon(row):
    # Same logic as provided in the Streamlit code
    LX = row.get("LandingX"); LY = row.get("LandingY"); RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: return None
    def atan_safe(numerator, denominator): return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    if RH == True: # Right Handed Batsman
        if LX <= 0 and LY > 0: return "FINE LEG";
        elif LX <= 0 and LY <= 0: return "THIRD MAN";
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER";
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF";
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG";
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON";
    elif RH == False: # Left Handed Batsman
        if LX <= 0 and LY > 0: return "THIRD MAN";
        elif LX <= 0 and LY <= 0: return "FINE LEG";
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "SQUARE LEG";
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG ON";
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "COVER";
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG OFF";
    return None

def calculate_scoring_angle(area):
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0

# Function to encode Matplotlib figure to image for Dash
def fig_to_uri(in_fig, close_all=True):
    out_img = StringIO()
    in_fig.savefig(out_img, format='png', bbox_inches='tight', pad_inches=0.05)
    if close_all:
        plt.close('all')
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

# --- 3. CHART GENERATION FUNCTIONS (Rewritten for Dash/Plotly) ---

# --- CHART 1: CREASE BEEHIVE BOXES (Matplotlib Image) ---
def create_zonal_analysis(df_in, batsman_name):
    if df_in.empty:
        return html.Div("No data for Zonal Analysis.", style={'padding': '20px'})

    # Detect handedness (Default to Right Hand if data is ambiguous/missing)
    is_right_handed = True
    handed_data = df_in["IsBatsmanRightHanded"].dropna().unique()
    if len(handed_data) > 0 and batsman_name != "All":
        is_right_handed = handed_data[0]
        
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
    norm = mcolors.Normalize(vmin=avg_values[avg_values > 0].min(), vmax=avg_values.max()) if avg_values.max() > 0 else mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Blues')

    # Adjusted figsize for compactness: (4, 4) or similar ratio
    fig_boxes, ax = plt.subplots(figsize=(4, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    for zone, (x1, y1, x2, y2) in zones_layout.items():
        w, h = x2 - x1, y2 - y1
        z_key = zone.replace("Zone ", "Z") # Map back to Z1, Z2, etc.
        
        runs, wkts, avg, sr = (0, 0, 0, 0)
        if z_key in summary.index:
            runs = int(summary.loc[z_key, "Runs"])
            wkts = int(summary.loc[z_key, "Wickets"])
            avg = summary.loc[z_key, "Avg Runs/Wicket"]
            sr = summary.loc[z_key, "StrikeRate"]
        
        color = cmap(norm(avg)) if avg > 0 else 'white'

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=1.5))

        ax.text(x1 + w / 2, y1 + h / 2, 
                f"{z_key}\nR:{runs} W:{wkts}\nA:{avg:.1f} SR:{sr:.1f}", 
                ha="center", va="center", weight="bold", fontsize=7,
                color="black" if norm(avg) < 0.6 else "white",
                linespacing=1.2)

    ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); plt.tight_layout(pad=0) 
    return html.Img(src=fig_to_uri(fig_boxes), style={'width': '100%', 'height': '100%'})

# --- CHART 2: CREASE BEEHIVE (Plotly Figure) ---
def create_crease_beehive(df_in):
    if df_in.empty:
        return go.Figure().update_layout(title="No data for CBH")

    wickets = df_in[df_in["Wicket"] == True]
    non_wickets = df_in[df_in["Wicket"] == False]
    fig_cbh = go.Figure()

    # Non-wickets (light grey)
    fig_cbh.add_trace(go.Scatter(
        x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"], mode='markers', name="No Wicket",
        marker=dict(color='lightgrey', size=4, line=dict(width=0), opacity=0.95)
    ))

    # Wickets (red)
    fig_cbh.add_trace(go.Scatter(
        x=wickets["StumpsY"], y=wickets["StumpsZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=8, line=dict(width=0), opacity=0.95)
    ))

    # --- Stump lines & Background zones (Removed color boxes, keeping lines) ---
    fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1)) # Stumps
    fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1))
    fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.8)) # Crease edges
    fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.8))
    fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.8)) # Mid-wicket height
    
    fig_cbh.update_layout(
        # Adjusted size for single column, prioritizing height
        height=300, 
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0.5, 2], showgrid=True, zeroline=False, visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    return dcc.Graph(figure=fig_cbh, config={'displayModeBar': False}, style={'height': '300px'})


# --- CHART 3: PITCH MAP (Plotly Figure) ---
def create_pitch_map(df_in):
    if df_in.empty:
        return go.Figure().update_layout(title="No data for Pitch Map")

    PITCH_BINS = {
        "Short": {"y0": 8.60, "y1": 16.0, "color": "#5d3bb3"},
        "Length": {"y0": 5.0, "y1": 8.60, "color": "#ae4fa1"},
        "Slot": {"y0": 2.8, "y1": 5.0, "color": "#cc5d54"},
        "Yorker": {"y0": 0.9, "y1": 2.8, "color": "#c7b365"},
        "Full Toss": {"y0": -4.0, "y1": 0.9, "color": "#6e9d4f"},
    }
    
    fig_pitch = go.Figure()
    
    # 1. Add Background Zones (Removed color boxes, keeping lines only)
    for length, params in PITCH_BINS.items():
        # Add Horizontal lines to separate zones
        fig_pitch.add_hline(y=params["y0"], line=dict(color="lightgrey", width=1.0, dash="dot"))
        
        # Add zone label
        mid_y = (params["y0"] + params["y1"]) / 2
        fig_pitch.add_annotation(x=-1.45, y=mid_y, text=length.upper(), showarrow=False,
            font=dict(size=8, color="grey", weight='bold'), xanchor='left')

    # 2. Add Stump lines
    fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))

    # 3. Plot Data
    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]

    fig_pitch.add_trace(go.Scatter(
        x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"], mode='markers', name="No Wicket",
        marker=dict(color='white', size=4, line=dict(width=1, color="grey"), opacity=0.9)
    ))

    fig_pitch.add_trace(go.Scatter(
        x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket",
        marker=dict(color='red', size=8, line=dict(width=0), opacity=0.95)
    ))

    # 4. Layout
    fig_pitch.update_layout(
        height=350, # Adjusted height for single column
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False, visible=False), # Reversed Y-axis
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    return dcc.Graph(figure=fig_pitch, config={'displayModeBar': False}, style={'height': '350px'})


# --- CHART 4: INTERCEPTION SIDE-ON (Matplotlib Image) ---
def create_interception_side_on(df_in):
    # Filter out invalid interception points (where InterceptionX is < -999)
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        return html.Div("No valid interception data.", style={'padding': '20px'})
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    # Adjusted figsize for compactness: (3, 4) or similar ratio
    fig_7, ax_7 = plt.subplots(figsize=(3, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    # Plot Data (Layered)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    ax_7.scatter(df_other["InterceptionX"] + 10, df_other["InterceptionZ"], color='white', edgecolors='grey', linewidths=0.5, s=25, label="Other") 
    
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        ax_7.scatter(df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], color=color_map[ctype], s=40, label=ctype) 

    # Draw Vertical Dashed Lines with Labels
    line_specs = {0.0: "Stumps", 1.250: "Crease", 2.000: "2m", 3.000: "3m"}
    for x_val, label in line_specs.items():
        ax_7.axvline(x=x_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)    
        ax_7.text(x_val, 1.45, label.split(':')[-1].strip(), ha='center', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Set Axes Limits and Labels
    ax_7.set_xlim(-0.2, 3.4); ax_7.set_ylim(0, 1.5) 
    ax_7.tick_params(axis='y', which='both', labelleft=False, left=False); ax_7.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_7.set_xlabel("Distance (m)", fontsize=8); ax_7.set_ylabel("Height (m)", fontsize=8) 
    ax_7.legend(loc='upper right', fontsize=6); ax_7.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(pad=0)
    
    return html.Img(src=fig_to_uri(fig_7), style={'width': '100%', 'height': '100%'})

# --- CHART 5: INTERCEPTION FRONT-ON (Matplotlib Image) ---
def create_interception_front_on(df_in):
    # Filter out invalid interception points (where InterceptionX is < -999)
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        return html.Div("No valid interception data.", style={'padding': '20px'})
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    # Adjusted figsize for compactness: (3, 4) or similar ratio
    fig_8, ax_8 = plt.subplots(figsize=(3, 4), subplot_kw={'xticks': [], 'yticks': []}) 

    # Plot Data (Layered)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    ax_8.scatter(df_other["InterceptionY"], df_other["InterceptionX"] + 10, color='white', edgecolors='grey', linewidths=0.5, s=25, label="Other") 
    
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        ax_8.scatter(df_slice["InterceptionY"], df_slice["InterceptionX"] + 10, color=color_map[ctype], s=40, label=ctype) 

    # Draw Horizontal Dashed Lines with Labels
    line_specs = {0.00: "Stumps", 1.25: "Crease"}
    for y_val, label in line_specs.items():
        ax_8.axhline(y=y_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), ha='left', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Add stump lines (vertical)
    ax_8.axvline(x=-0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
    ax_8.axvline(x= 0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Set Axes Limits and Labels
    ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
    ax_8.tick_params(axis='y', which='both', labelleft=False, left=False); ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_8.set_xlabel("Width (m)", fontsize=8); ax_8.set_ylabel("Distance (m)", fontsize=8) 
    ax_8.legend(loc='lower right', fontsize=6); ax_8.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(pad=0)
    
    return html.Img(src=fig_to_uri(fig_8), style={'width': '100%', 'height': '100%'})


# --- CHART 6: SCORING WAGON WHEEL (Matplotlib Image) ---
def create_wagon_wheel(df_in):
    # Data Prep (Same as Streamlit logic)
    wagon_summary = pd.DataFrame() 
    try:
        df_wagon = df_in.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(TotalRuns=("Runs", "sum"), FixedAngle=("FixedAngle", 'first')).reset_index().dropna(subset=["ScoringWagon"])
        handedness_mode = df_in["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
        
        all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] if is_right_handed else ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
        template_df = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]})

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0) 
        wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
        wagon_summary = wagon_summary.sort_values("ScoringWagon").reset_index(drop=True)
        
        total_runs = wagon_summary["TotalRuns"].sum()
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100 if total_runs > 0 else 0 
        wagon_summary["FixedAngle"] = wagon_summary["FixedAngle"].astype(int) 
    except Exception:
        return html.Div("No scoring shots or missing columns.", style={'padding': '20px'})


    # Chart Creation
    angles = wagon_summary["FixedAngle"].tolist()
    runs = wagon_summary["TotalRuns"].tolist()
    
    labels = [f"{area}\n({pct:.0f}%)" for area, pct in zip(wagon_summary["ScoringWagon"], wagon_summary["RunPercentage"])]
    
    run_min = min(runs)
    run_max = max(runs)
    norm = mcolors.Normalize(vmin=run_min, vmax=run_max) if run_max > run_min else mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Greens')
    colors = cmap(norm(runs))
    for i, run_count in enumerate(runs):
        if run_count == 0: colors[i] = (1.0, 1.0, 1.0, 1.0) # White for 0 runs

    # Adjusted figsize for compactness: (4, 4) or similar ratio
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    wedges, texts = ax.pie(
        angles, 
        colors=colors, 
        wedgeprops={"width": 1, "edgecolor": "black"}, 
        startangle=90, 
        counterclock=False, 
        labels=labels, 
        labeldistance=1.1 # Push labels outside
    )
    
    for text in texts:
        text.set_color('black'); text.set_fontsize(8); text.set_fontweight('bold')

    ax.axis('equal'); plt.tight_layout(pad=0)
    
    return html.Img(src=fig_to_uri(fig), style={'width': '100%', 'height': '100%'})


# --- 4. APP LAYOUT AND COMPONENT STRUCTURE ---

# Placeholder for Data Upload (This should ideally be done before deployment for Render)
# For a deployed Render app, the data must be included in the repository. 
# We'll use a placeholder structure for the final production code.
# In a real deployed app, you'd load the CSV directly here:
df = pd.read_csv(r"odi sample data.csv")

# Temporary data loading to enable app structure if deployed
try:
    # Attempt to load a sample file or expect the user to adjust this line
    df = pd.read_csv("your_data_file.csv") # CHANGE THIS LINE IF YOUR FILE IS NAMED DIFFERENTLY
    
    # Check for required columns
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", "InterceptionZ", "InterceptionY", "Over"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in CSV: {set(required_cols) - set(df.columns)}")
    
    # Pre-filter for Seam only (as requested)
    df = df[df["DeliveryType"] == "Seam"].copy()

    BAT_TEAM_OPTIONS = ["All"] + sorted(df["BattingTeam"].dropna().unique().tolist())
    BATSMAN_OPTIONS = ["All"] + sorted(df["BatsmanName"].dropna().unique().tolist())
    OVER_OPTIONS = ["All"] + sorted(df["Over"].dropna().unique().tolist())
    
except Exception as e:
    print(f"Error loading or validating initial data: {e}. Using dummy data for layout.")
    # Create dummy DataFrame to prevent KeyErrors during layout generation
    df = pd.DataFrame(
        columns=["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", "InterceptionZ", "InterceptionY", "Over"],
        data=[["Dummy Player", "Seam", False, 0, 1, "Dummy Team", 0, 1, 1, True, 10, 1, 1, 1, 1, 1, 1, 1]]
    )
    BAT_TEAM_OPTIONS = ["All", "Dummy Team"]
    BATSMAN_OPTIONS = ["All", "Dummy Player"]
    OVER_OPTIONS = ["All", 1]

# --- Top Filters Row ---
filter_layout = dbc.Row([
    dbc.Col(dcc.Dropdown(id='bat_team_filter', options=BAT_TEAM_OPTIONS, value='All', placeholder="Select Batting Team"), width=4),
    dbc.Col(dcc.Dropdown(id='batsman_filter', options=BATSMAN_OPTIONS, value='All', placeholder="Select Batsman"), width=4),
    dbc.Col(dcc.Dropdown(id='over_filter', options=OVER_OPTIONS, value='All', placeholder="Select Over"), width=4),
], className="g-1 mb-1") # g-1 for minimal gutter/spacing

# --- Main App Layout ---
app.layout = dbc.Container([
    # 1. Filters Row
    filter_layout,
    
    # 2. Batsman Heading
    dbc.Row(dbc.Col(html.H2(id='batsman_heading', className='text-center mt-0 mb-1', style={'fontSize': '1.5rem', 'fontWeight': 'bold'}))),
    
    # 3. Main Chart Area
    dbc.Row([
        # Single Column for all Charts (width=12 ensures full container width)
        dbc.Col([
            # 3.1 Row 1: Zonal Analysis (CBH Boxes)
            dbc.Card([dbc.CardHeader("1. Zonal Analysis (Crease Impact)"), dbc.CardBody(html.Div(id='chart-zonal', style={'height': '300px'}))], className="mb-1"),
            
            # 3.2 Row 2: Crease Beehive
            dbc.Card([dbc.CardHeader("2. Crease Beehive"), dbc.CardBody(html.Div(id='chart-cbh', style={'height': '300px'}))], className="mb-1"),
            
            # 3.3 Row 3: Pitch Map
            dbc.Card([dbc.CardHeader("3. Pitch Map (Bounce Location)"), dbc.CardBody(html.Div(id='chart-pitch', style={'height': '350px'}))], className="mb-1"),
            
            # 3.4 Row 4: Interception and Wagon Wheel (Side-by-Side)
            dbc.Row([
                # Left Side (Interception Side-On)
                dbc.Col(
                    dbc.Card([dbc.CardHeader("4. Interception Side-On"), dbc.CardBody(html.Div(id='chart-int-side', style={'height': '300px'}))], className="mb-1"),
                    width=4 # Use 4/12 width for the smaller chart
                ),
                # Middle Side (Interception Front-On)
                dbc.Col(
                    dbc.Card([dbc.CardHeader("5. Interception Front-On"), dbc.CardBody(html.Div(id='chart-int-front', style={'height': '300px'}))], className="mb-1"),
                    width=4 # Use 4/12 width
                ),
                # Right Side (Scoring Areas - Wagon Wheel)
                dbc.Col(
                    dbc.Card([dbc.CardHeader("6. Scoring Areas"), dbc.CardBody(html.Div(id='chart-wagon', style={'height': '300px'}))], className="mb-1"),
                    width=4 # Use 4/12 width
                ),
            ], className="g-1"), # g-1 for minimal gutter/spacing
            
        ], width=12) # Full width column for the single stack
    ], className="g-0", style={'marginBottom': '10px'}),
    
], fluid=True)


# --- 5. CALLBACKS (INTERACTIVITY) ---

# Callback to update Batsman Options (Cascading filter)
@app.callback(
    Output('batsman_filter', 'options'),
    Output('batsman_filter', 'value'),
    [Input('bat_team_filter', 'value')]
)
def update_batsman_options(selected_team):
    dff = df if selected_team == "All" else df[df["BattingTeam"] == selected_team]
    batsman_options = ["All"] + sorted(dff["BatsmanName"].dropna().unique().tolist())
    # Reset batsman selection if the current one is no longer available
    new_value = 'All' 
    return batsman_options, new_value

# Callback to update Over Options (Cascading filter)
@app.callback(
    Output('over_filter', 'options'),
    Output('over_filter', 'value'),
    [Input('bat_team_filter', 'value'), Input('batsman_filter', 'value')]
)
def update_over_options(selected_team, selected_batsman):
    dff = df if selected_team == "All" else df[df["BattingTeam"] == selected_team]
    dff = dff if selected_batsman == "All" else dff[dff["BatsmanName"] == selected_batsman]
    over_options = ["All"] + sorted(dff["Over"].dropna().unique().tolist())
    new_value = 'All' 
    return over_options, new_value


# Master Callback to Update ALL Charts and Heading
@app.callback(
    [Output('batsman_heading', 'children'),
     Output('chart-zonal', 'children'),
     Output('chart-cbh', 'children'),
     Output('chart-pitch', 'children'),
     Output('chart-int-side', 'children'),
     Output('chart-int-front', 'children'),
     Output('chart-wagon', 'children')],
    [Input('bat_team_filter', 'value'),
     Input('batsman_filter', 'value'),
     Input('over_filter', 'value')]
)
def update_all_charts(bat_team, batsman, selected_over):
    # 1. Filter Data
    dff = df.copy()
    if bat_team != "All":
        dff = dff[dff["BattingTeam"] == bat_team]
    if batsman != "All":
        dff = dff[dff["BatsmanName"] == batsman]
    if selected_over != "All":
        dff = dff[dff["Over"] == selected_over]

    # 2. Update Heading
    heading_text = batsman.upper() if batsman != "All" else "SEAM ANALYSIS: ALL BATSMEN"
    
    # 3. Generate Charts (ensure the order matches the layout)
    chart_zonal = create_zonal_analysis(dff, batsman)
    chart_cbh = create_crease_beehive(dff)
    chart_pitch = create_pitch_map(dff)
    chart_int_side = create_interception_side_on(dff)
    chart_int_front = create_interception_front_on(dff)
    chart_wagon = create_wagon_wheel(dff)

    return (heading_text, chart_zonal, chart_cbh, chart_pitch, chart_int_side, chart_int_front, chart_wagon)


if __name__ == '__main__':
    # NOTE: When deploying to Render, you will use gunicorn and the 'server' object.
    # e.g., gunicorn app:server
    # Local run:
    app.run_server(debug=True)
