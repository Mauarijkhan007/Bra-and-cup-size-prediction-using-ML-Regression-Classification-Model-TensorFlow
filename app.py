import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load the trained model, scaler, and mappings
model = tf.keras.models.load_model("bra_model_classified.h5")
scaler = joblib.load("scaler.pkl")
band_mapping = joblib.load("band_mapping.pkl")
cup_mapping = joblib.load("cup_mapping.pkl")

# Ensure the model is compiled
model.compile(optimizer='adam', loss={'band_output': 'sparse_categorical_crossentropy', 'cup_output': 'mse'},
              metrics={'band_output': 'accuracy', 'cup_output': 'mae'})

# Streamlit app interface
st.title("Bra Size Prediction")

# Input fields for user
height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=60)
body_fat = st.number_input("Body Fat Percentage", min_value=0.0, max_value=100.0, value=25.0)

# Predict when user clicks the button
if st.button("Predict Bra Size"):
    # Prepare input features for prediction
    input_data = np.array([[height, weight, body_fat]])
    
    # Normalize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions using the model
    band_pred_probs, cup_pred = model.predict(input_data_scaled)
    
    # Post-process the predictions
    band_pred = np.argmax(band_pred_probs, axis=1)[0]
    cup_pred = round(cup_pred[0][0])
    
    # Convert predictions to actual sizes
    band_size = band_mapping.get(band_pred, "Unknown Band Size")
    cup_size = cup_mapping.get(cup_pred, "Unknown Cup Size")
    
    # Display predictions
    st.write(f"Predicted Band Size: {band_size}")
    st.write(f"Predicted Cup Size: {cup_size}")
