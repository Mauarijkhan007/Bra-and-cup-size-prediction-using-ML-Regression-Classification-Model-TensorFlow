import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
st.markdown("""
    <style>
    /* Title styling with badge */
    .title-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .badge {
        background-color: #ffb900;
        color: black;
        padding: 0.25em 0.75em;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }

    /* Predict button styling */
    .stButton>button {
        background-color: #ffb900;
        color: black;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
    }

    .stButton>button:active {
        background-color: #ffb900 !important;
        color: black !important;
    }

    .stButton>button:hover {
        background-color: #e6a700;
        color: black;
    }

    /* Custom success styling */
    .stSuccess {
        background-color: #ffb900 !important;
        color: black !important;
        font-weight: bold;
        padding: 1em;
        border-radius: 10px;
    }

    /* Input fields dark style */
    input, .stNumberInput input {
        background-color: #1e1e1e !important;
        color: white !important;
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
st.markdown("""
    <div class="title-container">
        <h1 style="color: white; margin: 0;">Bra Size</h1>
        <div class="badge">Predictor</div>
    </div>
""", unsafe_allow_html=True)

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
