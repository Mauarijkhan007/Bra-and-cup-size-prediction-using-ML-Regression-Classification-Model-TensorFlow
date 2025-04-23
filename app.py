import streamlit as st
import numpy as np
import tensorflow as tf
import joblib


st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
        }
        .stTextInput, .stNumberInput, .stButton {
            background-color: #1a1a1a;
            color: white;
        }
        .stButton:hover {
            background-color: #FFD700;
            color: black;
        }
        .css-1d391kg {
            color: white;
        }
        .stTitle {
            color: #FFD700;
        }
        .stSidebar {
            background-color: #1a1a1a;
        }
        .stSidebar .stTextInput, .stSidebar .stNumberInput {
            background-color: #1a1a1a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
# Load model without compiling to avoid 'mse' deserialization issue
model = tf.keras.models.load_model("bra_model_classified.h5", compile=False)
# Load scaler and mappings
scaler = joblib.load("scaler.pkl")
band_mapping = joblib.load("band_mapping.pkl")  # idx_to_band
idx_to_band = band_mapping

# Cup size numeric to letter mapping
cup_letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'DD', 6: 'E', 7: 'F'}

# Streamlit interface
st.title("Bra Size Predictor")

# Input form
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
body_fat = st.number_input("Body Fat %", min_value=1.0, max_value=75.0, value=25.0)

if st.button("Predict"):
    # Prepare and scale input
    input_data = np.array([[height, weight, body_fat]])
    input_scaled = scaler.transform(input_data)

    # Predict
    band_probs, cup_pred = model.predict(input_scaled)
    band_class = np.argmax(band_probs[0])
    band_size = idx_to_band.get(band_class, "Unknown Band")
    cup_numeric = round(cup_pred[0][0])
    cup_letter = cup_letters.get(cup_numeric, f"Unknown({cup_numeric})")

    # Display result
    st.success(f"Predicted Bra Size: {band_size}{cup_letter}")
