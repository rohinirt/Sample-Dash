import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import base64, io

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Needed for Render

# -------------------- LAYOUT --------------------
app.layout = dbc.Container([
    html.H2("🏏 Cricket Analysis Dashboard (Dash)"),
    html.Hr(),

    # File upload section
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📁 Drag & Drop or ', html.A('Select Hawkeye CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
        },
        multiple=False
    ),

    # Filters: Team → Batsman → Delivery
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='team', placeholder="Select Batting Team"), width=4),
        dbc.Col(dcc.Dropdown(id='batsman', placeholder="Select Batsman"), width=4),
        dbc.Col(dcc.Dropdown(id='delivery', placeholder="Select Delivery Type"), width=4),
    ], className="mb-3"),

    # Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id='crease-beehive'), width=6),
        dbc.Col(dcc.Graph(id='pitch-map'), width=6)
    ])
], fluid=True)


# -------------------- CALLBACKS --------------------

# Store uploaded data
@app.callback(
    Output('team', 'options'),
    Input('upload-data', 'contents')
)
def load_data(contents):
    if contents is None:
        return []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Validate columns
    required_cols = ["BattingTeam", "BatsmanName", "DeliveryType", "Wicket",
                     "StumpsY", "StumpsZ", "BounceY", "BounceX"]
    if not all(col in df.columns for col in required_cols):
        return []

    teams = [{'label': t, 'value': t} for t in df["BattingTeam"].dropna().unique()]
    # Store df in dcc.Store for later use
    app.server.df = df
    return teams


# Update batsman dropdown based on team
@app.callback(
    Output('batsman', 'options'),
    Input('team', 'value')
)
def update_batsmen(team):
    df = getattr(app.server, 'df', None)
    if df is None or team is None:
        return []
    batsmen = df[df["BattingTeam"] == team]["BatsmanName"].dropna().unique()
    return [{'label': b, 'value': b} for b in batsmen]


# Update delivery dropdown based on batsman
@app.callback(
    Output('delivery', 'options'),
    [Input('team', 'value'), Input('batsman', 'value')]
)
def update_deliveries(team, batsman):
    df = getattr(app.server, 'df', None)
    if df is None or team is None or batsman is None:
        return []
    deliveries = df[(df["BattingTeam"] == team) & (df["BatsmanName"] == batsman)]["DeliveryType"].dropna().unique()
    return [{'label': d, 'value': d} for d in deliveries]


# Update charts when filters change
@app.callback(
    [Output('crease-beehive', 'figure'),
     Output('pitch-map', 'figure')],
    [Input('team', 'value'),
     Input('batsman', 'value'),
     Input('delivery', 'value')]
)
def update_charts(team, batsman, delivery):
    df = getattr(app.server, 'df', None)
    if df is None:
        return go.Figure(), go.Figure()

    # Apply filters
    if team:
        df = df[df["BattingTeam"] == team]
    if batsman:
        df = df[df["BatsmanName"] == batsman]
    if delivery:
        df = df[df["DeliveryType"] == delivery]

    if df.empty:
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="No data for selected filters", x=0.5, y=0.5, showarrow=False)
        return fig_empty, fig_empty

    # --- Crease Beehive ---
    wickets = df[df["Wicket"] == True]
    non_wickets = df[df["Wicket"] == False]

    fig_cbh = go.Figure()
    fig_cbh.add_trace(go.Scatter(
        x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"],
        mode='markers', marker=dict(color='lightgrey', size=8, opacity=0.8)
    ))
    fig_cbh.add_trace(go.Scatter(
        x=wickets["StumpsY"], y=wickets["StumpsZ"],
        mode='markers', marker=dict(color='red', size=10, opacity=0.9)
    ))

    fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1))
    fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1))

    fig_cbh.update_layout(
        title=f"Crease Beehive - {batsman or 'All'}",
        xaxis=dict(range=[-1.6, 1.6], visible=False),
        yaxis=dict(range=[0, 2.5], visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )

    # --- Pitch Map ---
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(
        x=df["BounceY"], y=df["BounceX"],
        mode='markers', marker=dict(color='dodgerblue', size=8, opacity=0.85)
    ))
    fig_pitch.update_layout(
        title=f"Pitch Map - {batsman or 'All'}",
        xaxis=dict(range=[-2, 2], visible=False),
        yaxis=dict(range=[-1, 20], visible=False),
        plot_bgcolor="#f5f5f5", paper_bgcolor="white", showlegend=False
    )

    return fig_cbh, fig_pitch


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

