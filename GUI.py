import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Input feature names (must match training set)
features = [
    'lw', 'hw', 'tw', "f′c", 'fyt', 'fysh', 'fyl', 'fybl',
    'ρt', 'ρsh', 'ρl', 'ρbl', "P/(Agf′c)", 'b0', 'db', 's/db',
    'AR', 'M/Vlw', 'Ductility Ratio', 'DriftRatio'
]

# Title and layout
st.markdown("<h1 style='text-align: center;'>Wall Characteristic</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Enter your wall data:</h3>", unsafe_allow_html=True)

# Input fields layout
input_values = {}
cols = st.columns(4)
for idx, col in enumerate(features):
    input_values[col] = cols[idx % 4].number_input(f"{col}", format="%.5f")

# Prepare input for prediction
input_df = pd.DataFrame([input_values])

# Log setup
if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame()

# Predict and display
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    input_df["Damping Ratio"] = prediction
    st.write(f"### Predicted Damping Ratio: {prediction:.5f}")
    st.session_state.log_df = pd.concat([st.session_state.log_df, input_df], ignore_index=True)

# Clear logs
if st.button("Clear Logs"):
    st.session_state.log_df = pd.DataFrame()
    st.success("Logs cleared.")

# Show and download logs
if not st.session_state.log_df.empty:
    csv = st.session_state.log_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "damping_ratio_log.csv", "text/csv")
    st.write("### Log of Predictions:")
    st.dataframe(st.session_state.log_df)
