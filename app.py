import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model without compiling to avoid the 'mse' function deserialization issue
model = tf.keras.models.load_model("bra_model_classified.h5", compile=False)

# Load scaler and band mapping
scaler = joblib.load("scaler.pkl")
band_mapping = joblib.load("band_mapping.pkl")  # This is idx_to_band
idx_to_band = band_mapping

# Inject custom CSS for further styling
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

# Streamlit app interface
st.title("Bra Size Prediction")

# Input fields for user
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
body_fat = st.number_input("Body Fat Percentage", min_value=0.0, max_value=100.0, value=25.0)
cup_letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'DD', 6: 'E', 7: 'F'}
# Predict when user clicks the button
if st.button("Predict Bra Size"):
    # Prepare input features for prediction
    input_data = np.array([[height, weight, body_fat]])
    
    # Normalize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions using the model
    band_pred, cup_pred = model.predict(input_data_scaled)
    
    # Post-process the predictions
    band_pred = round(band_pred[0][0])
    cup_pred = round(cup_pred[0][0])
    
    # Display predictions
    band_size = idx_to_band.get(band_pred, "Unknown Band Size")
    cup_size = cup_pred  # You can map this to letter sizes if needed
    
    st.success(f"Predicted Bra Size: {band_size}{cup_letter}")
