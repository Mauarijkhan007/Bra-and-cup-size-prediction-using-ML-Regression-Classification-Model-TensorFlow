import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
# Custom CSS
st.markdown("""
    <style>
    /* Custom font and layout for the Pornhub-style logo */
    .logo-container {
        display: flex;
        align-items: center;
        font-size: 2.5em;
        font-weight: 800;
        font-family: sans-serif;
        margin-bottom: 20px;
    }

    .logo-text {
        color: white;
        margin-right: 10px;
    }

    .logo-highlight {
        background-color: #ffb900;
        color: black;
        padding: 0.1em 0.5em;
        border-radius: 0.3em;
    }

    /* Predict button style */
        .stButton>button {
        background-color: #ffb900 !important;
        color: black !important;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }

    .stButton>button:active {
        background-color: #e6a700 !important;
        color: black !important;
    }

    .stButton>button:hover {
        background-color: #e6a700 !important;
        color: black !important;
    }

    /* Input dark style */
    input, .stNumberInput input {
        background-color: #1e1e1e !important;
        color: white !important;
    }

    /* Custom success box override */
    div.stAlert.success {
        background-color: #ffb900 !important;
        color: black !important;
        border-left: 0.25rem solid black !important;
        font-weight: bold;
        border-radius: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Pornhub-style logo
st.markdown("""
    <div class="logo-container">
        <div class="logo-text">Bra Size</div>
        <div class="logo-highlight">Predictor</div>
    </div>
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
        # Custom styled success message (yellow box with black text)
    st.markdown(f"""
        <div style='
            background-color: #ffb900;
            color: black;
            padding: 1em;
            border-radius: 0.5em;
            font-weight: bold;
            margin-top: 1em;
        '>
            Predicted Bra Size: {band_size}{cup_letter}
        </div>
    """, unsafe_allow_html=True)
