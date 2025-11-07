import streamlit as st
import pandas as pd
from Batters import create_crease_beehive, create_pitch_map, create_wagon_wheel  # import what you need

st.title("Pacers Dashboard")

uploaded_file = st.file_uploader("Upload CSV", key="pacers_upload")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # filter pacers: adjust column name as per your dataset (e.g., BowlerType or DeliveryType)
    pacers = df[df["DeliveryType"].str.lower().str.contains("seam|pacer", na=False)]
    # top filters if needed:
    team = st.selectbox("Batting Team", ["All"] + sorted(pacers["BattingTeam"].unique().tolist()))
    # apply filters...
    st.plotly_chart(create_crease_beehive(pacers, "Seam"), use_container_width=True)
    st.plotly_chart(create_pitch_map(pacers, "Seam"), use_container_width=True)
    st.pyplot(create_wagon_wheel(pacers, "Seam"), use_container_width=True)
else:
    st.info("Upload CSV to show Pacers dashboard")


