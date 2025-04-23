
# 🧠 Bra Size Predictor

A machine learning-powered web app that predicts bra sizes based on a person’s height, weight, and body fat percentage. Built using TensorFlow, trained on synthetic data, and deployed with a simple Streamlit interface.

---

## 🔍 Overview

This project predicts **band size** (e.g., 32, 34, 36) as a **classification task** and **cup size** (e.g., A, B, C...) as a **regression task**, combining them to estimate a full bra size like `34C`.

Key Components:
- Multi-output Keras model
- Scaled inputs with `StandardScaler`
- Mappings for readable predictions
- Interactive prediction interface with Streamlit

---

## 📁 Project Structure

```
📦 bra-size-predictor/
├── app.py                     # Streamlit app
├── training_data.csv          # Synthetic training data
├── bra_model_classified.h5    # Trained ML model
├── scaler.pkl                 # Scaler used for input normalization
├── band_mapping.pkl           # Band class index → actual size mapping
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🧪 Model Training

The model is trained using a mix of synthetic features:

- Inputs: `height_cm`, `weight_kg`, `body_fat`
- Outputs:
  - `band_size`: Classification using `softmax`
  - `cup_size_numeric`: Regression with `MSE` loss

Model architecture:
- Dense(64) → Dense(32)
- Outputs:
  - `band_output`: Dense(N), softmax
  - `cup_output`: Dense(1), linear

Training is done with `sparse_categorical_crossentropy` and `mean_squared_error`.

Model performance is evaluated using:
- Accuracy, Precision, Recall for band size
- MAE and correlation plots for cup size

---

## 💻 Running the App

### 🔧 Setup

Install required libraries:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
streamlit
tensorflow
scikit-learn
numpy
joblib
matplotlib
pandas
```

---

### ▶️ Launch

Run the app using Streamlit:

```bash
streamlit run app.py
```

This opens a local web interface where users can input their stats and get a bra size prediction.

---

## 🧠 How It Works

1. User inputs height, weight, and body fat %.
2. The input is scaled using a pre-fit `StandardScaler`.
3. The model predicts:
   - **Band class probabilities**
   - **Cup size as a float**
4. Band is decoded using a class-to-size mapping.
5. Cup is rounded and translated from a number to a letter (e.g., `2 → B`).

---

## 🔡 Cup Size Legend

| Numeric | Cup |
|---------|-----|
| 1       | A   |
| 2       | B   |
| 3       | C   |
| 4       | D   |
| 5       | DD  |
| 6       | E   |
| 7       | F   |

---
## model specs
Band Size Evaluation:
Accuracy: 0.8201438848920863
Precision: 0.8081733726845872
Recall: 0.8201438848920863

Cup Size Evaluation:
Accuracy: 0.841726618705036
Precision: 0.8532695374800638
Recall: 0.841726618705036

Cup Size Evaluation:
Mean Absolute Error (MAE): 0.158273383975029
Mean Squared Error (MSE): 0.158273383975029
---
## Launched at :https://ml-model-no1-by-mojo.streamlit.app/
---
## 🧠 Author

**Mauarij Khan** (a.k.a Mojo)  
Junior Software Engineer, ML Enthusiast, and App Developer  
📍 Based in Pakistan  

---

## 📜 License

This project is open-source and freely usable for educational and experimental purposes.
