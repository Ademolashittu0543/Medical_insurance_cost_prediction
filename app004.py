import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page title
st.title("Medical Insurance Cost Predictor")

# Load the trained model and preprocessor
try:
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please ensure 'best_model.pkl' and 'preprocessor.pkl' are in the same directory.")
    st.stop()

# Create input fields for user
st.header("Enter Your Details")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
bmi = st.number_input("BMI", min_value=10, max_value=60, value=25, step=1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

# Categorical inputs
sex = st.selectbox("Sex", options=["male", "female"])
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Button to make prediction
if st.button("Predict Insurance Cost"):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    try:
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Display the result
        st.success(f"Predicted Insurance Cost: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some information about the app
st.markdown("""
---
### About This App
This app predicts medical insurance costs based on user inputs using a trained machine learning model. The model was trained on a dataset with features including age, sex, BMI, number of children, smoking status, and region. Ensure all inputs are valid to get an accurate prediction.
""")
