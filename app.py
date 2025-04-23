
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("bra_model_classified.h5")
scaler = joblib.load("scaler.pkl")
band_mapping = joblib.load("band_mapping.pkl")
cup_mapping = joblib.load("cup_mapping.pkl")

st.title("👗 Bra Size Predictor")
st.markdown("Enter the following body measurements to predict your bra size:")

# Input fields
height = st.number_input("Height (cm)", min_value=100.0, max_value=200.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
body_fat = st.number_input("Body Fat (%)", min_value=5.0, max_value=60.0, step=0.1)

if st.button("Predict"):
    X_input = scaler.transform([[height, weight, body_fat]])
    band_pred, cup_pred = model.predict(X_input)
    band_pred = int(np.round(band_pred[0][0]))
    cup_pred = int(np.round(cup_pred[0][0]))

    band_size = band_mapping.get(band_pred, f"Unknown ({band_pred})")
    cup_size = cup_mapping.get(cup_pred, f"Unknown ({cup_pred})")

    st.success(f"Predicted Bra Size: **{band_size}{cup_size}**")
