import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🚀 Page configuration
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("💰 Medical Insurance Cost Predictor")

# 🧠 Load trained pipeline model (includes preprocessor)
try:
    model = joblib.load("insurance_model_pipeline.joblib")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'insurance_model_pipeline.joblib' is in the same directory.")
    st.stop()

# 📝 Input form
st.header("📋 Enter Your Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", options=["male", "female"])
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# 🧮 Prediction
if st.button("Predict Insurance Cost"):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        # Predict
        prediction = model.predict(input_data)[0]

        # 💬 Output
        st.success(f"Estimated Insurance Cost: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction:\n\n{str(e)}")

# 📘 About section
st.markdown("""
---
### ℹ️ About This App
This application estimates **medical insurance costs** based on a machine learning model trained on features like:
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Residential region

The model was built using a fully integrated **scikit-learn pipeline**, including preprocessing and hyperparameter tuning.

Make sure to enter realistic values to get a reliable prediction.
""")
