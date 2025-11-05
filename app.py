import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import base64, io

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Render deployment

# -------------------- LAYOUT --------------------
app.layout = dbc.Container([
    html.H2("🏏 Cricket Analysis Dashboard (Dash)"),
    html.Hr(),

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

    dbc.Row([
        dbc.Col(dcc.Dropdown(id='batsman', placeholder="Select Batsman"), width=6),
        dbc.Col(dcc.Dropdown(id='delivery', placeholder="Select Delivery Type"), width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='crease-beehive'), width=6),
        dbc.Col(dcc.Graph(id='pitch-map'), width=6)
    ])
], fluid=True)


# -------------------- CALLBACKS --------------------
@app.callback(
    [Output('batsman', 'options'),
     Output('delivery', 'options'),
     Output('crease-beehive', 'figure'),
     Output('pitch-map', 'figure')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, filename):
    if contents is None:
        # Empty charts before upload
        return [], [], go.Figure(), go.Figure()

    # Decode uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Check required columns
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BounceY", "BounceX"]
    if not all(col in df.columns for col in required_cols):
        fig_error = go.Figure()
        fig_error.add_annotation(text="Missing required columns in CSV!", x=0.5, y=0.5, showarrow=False)
        return [], [], fig_error, fig_error

    # Dropdown options
    batsman_opts = [{'label': b, 'value': b} for b in df['BatsmanName'].dropna().unique()]
    delivery_opts = [{'label': d, 'value': d} for d in df['DeliveryType'].dropna().unique()]

    # --- Chart 1: Crease Beehive ---
    wickets = df[df["Wicket"] == True]
    non_wickets = df[df["Wicket"] == False]

    fig_cbh = go.Figure()
    fig_cbh.add_trace(go.Scatter(
        x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"],
        mode='markers', marker=dict(color='lightgrey', size=8, opacity=0.8),
        name="No Wicket"
    ))
    fig_cbh.add_trace(go.Scatter(
        x=wickets["StumpsY"], y=wickets["StumpsZ"],
        mode='markers', marker=dict(color='red', size=10, opacity=0.9),
        name="Wicket"
    ))

    # Stump lines
    fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1.2))
    fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1.2))

    fig_cbh.update_layout(
        title="Crease Beehive View",
        xaxis=dict(range=[-1.6, 1.6], visible=False),
        yaxis=dict(range=[0, 2.5], visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )

    # --- Chart 2: Pitch Map ---
    pitch_wickets = df[df["Wicket"] == True]
    pitch_non_wickets = df[df["Wicket"] == False]

    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(
        x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"],
        mode='markers', marker=dict(color='lightblue', size=8, opacity=0.8),
        name="No Wicket"
    ))
    fig_pitch.add_trace(go.Scatter(
        x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"],
        mode='markers', marker=dict(color='crimson', size=10, opacity=0.9),
        name="Wicket"
    ))

    fig_pitch.update_layout(
        title="Pitch Map",
        xaxis=dict(range=[-2, 2], visible=False),
        yaxis=dict(range=[-1, 20], visible=False),
        plot_bgcolor="#f0f0f0", paper_bgcolor="white", showlegend=False
    )

    return batsman_opts, delivery_opts, fig_cbh, fig_pitch


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
