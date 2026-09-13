import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
model = tf.keras.models.load_model("diabetes_ann_model.keras")
scaler = joblib.load("diabetes_scaler.pkl")

# Title
st.title("🩺 Diabetes Prediction Dashboard")
st.write("Enter patient information to predict diabetes.")

st.divider()

# Input fields
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.divider()

# Prediction button
if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    probability = prediction * 100

    st.subheader("Prediction Result")

    if prediction >= 0.5:
        st.error("⚠️ Diabetes Predicted")
    else:
        st.success("✅ No Diabetes Predicted")

    st.metric(
        "Diabetes Probability",
        f"{probability:.2f}%"
    )