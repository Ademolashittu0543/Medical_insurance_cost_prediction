import streamlit as st
import pandas as pd
import pickle

# Load the saved model
try:
    with open('insurance_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"🚨 Error loading model: {str(e)}")
    st.stop()

# Streamlit app UI
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("💊 Medical Insurance Cost Predictor 💉")
st.write("📋 Enter the details below to predict the medical insurance cost! 🩺")

# Input form
with st.form("insurance_form"):
    st.subheader("🧑‍⚕️ Patient Details")
    
    # Numerical inputs
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0, step=1)
    
    # Categorical inputs
    sex = st.selectbox("🚻 Sex", options=["male", "female"])
    smoker = st.selectbox("🚬 Smoker", options=["yes", "no"])
    region = st.selectbox("🌍 Region", options=["northeast", "northwest", "southeast", "southwest"])
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict Cost")

# Prediction logic
if submitted:
    if age < 18 or bmi < 10.0 or children < 0:
        st.warning("⚠️ Please enter valid values for age (≥18), BMI (≥10), and children (≥0).")
    else:
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'smoker': [smoker],
                'region': [region],
                'children': [children]
            })
            
            # Make prediction
            predicted_cost = model.predict(input_data)[0]
            
            # Display prediction
            st.success(f"🎉 **Predicted Insurance Cost:** ${predicted_cost:,.2f} 💸")
            
        
        except Exception as e:
            st.error(f"🚨 Error during prediction: {str(e)}")

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
