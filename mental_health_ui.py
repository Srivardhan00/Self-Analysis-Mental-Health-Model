import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load Models and Scaler
try:
    rf_depression = load('rf_depression.joblib')
    rf_anxiety = load('rf_anxiety.joblib')
    scaler = load('scaler.joblib')  # Load the scaler you saved during training
except FileNotFoundError:
    st.error("Model files not found. Please ensure they are in the correct directory.")
    st.stop()

# Define input features (WITHOUT 'internally')
input_features = ['anxiousness', 'phq_score', 'depressiveness', 'age', 'gad_score']

st.title("Mental Health Severity Prediction")

# Input Fields (WITHOUT 'internally')
user_inputs = {}
for feature in input_features:
    if feature in ['anxiousness', 'depressiveness']:  # Boolean inputs
        user_inputs[feature] = st.selectbox(f"{feature.capitalize()}:", [0, 1])
    elif feature == 'age':
        user_inputs[feature] = st.number_input(f"{feature.capitalize()}:", min_value=0, step=1)
    elif feature == 'phq_score':  # PHQ range 0-27
        user_inputs[feature] = st.number_input(f"{feature.capitalize()}:", min_value=0, max_value=27, step=1)
    elif feature == 'gad_score':  # GAD range 0-21
        user_inputs[feature] = st.number_input(f"{feature.capitalize()}:", min_value=0, max_value=21, step=1)
    # No 'internally' input


# Prediction Button
if st.button("Predict"):
    # Create DataFrame from user inputs (respecting the order, WITHOUT 'internally')
    input_df = pd.DataFrame([user_inputs])[input_features]  # Reorder columns

    # Scale the input features using the loaded scaler
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_features)

    try:
        # Make predictions
        depression_prediction = rf_depression.predict(input_scaled_df)[0]
        anxiety_prediction = rf_anxiety.predict(input_scaled_df)[0]

        # Display predictions (decode if needed)
        depression_severity_labels = {0: 'None', 1: 'None-minimal', 2: 'Mild', 3: 'Moderate', 4: 'Moderately severe', 5: 'Severe'}
        anxiety_severity_labels = {0: 'None-minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

        st.write(f"**Depression Severity:** {depression_severity_labels.get(depression_prediction, 'Unknown')}")
        st.write(f"**Anxiety Severity:** {anxiety_severity_labels.get(anxiety_prediction, 'Unknown')}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
