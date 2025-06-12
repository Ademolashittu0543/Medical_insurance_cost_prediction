# Medical Insurance Cost Predictor 💊💉

This project implements a medical insurance cost prediction system using a Random Forest Regressor to estimate insurance charges based on patient details like age, sex, BMI, smoking status, region, and number of children. 📋

## Purpose ✨

The goal of this project is to demonstrate how machine learning can predict medical insurance costs. It includes data preprocessing, feature encoding, model training, evaluation, and deployment as a Streamlit web app. Try the deployed version here: [Medical Insurance Cost Predictor](https://medicalinsurancecostprediction-ev89nczdpen4ioxctefohn.streamlit.app/) 🚀

## Technologies Used 💻

- Python
- pandas
- numpy
- scikit-learn
- pickle
- streamlit

## Dataset 🧾

The dataset used is `Train_Data.csv`, containing features like age, sex, BMI, smoker status, region, children, and insurance charges. It’s loaded and preprocessed in `Medical_Insurance_cost_.ipynb`, where numerical columns are rounded to integers.

## Project Steps 📖

- **Data Loading**: Load `Train_Data.csv` using pandas.
- **Data Preprocessing**: Round numerical columns (age, BMI, children) to integers.
- **Feature Extraction**: Use one-hot encoding for categorical variables (sex, smoker, region) via `ColumnTransformer`.
- **Model Training**: Train a `RandomForestRegressor` with 100 estimators in a scikit-learn pipeline.
- **Model Evaluation**: Assess performance using R² score and Mean Absolute Error (MAE).
- **Prediction**: Deploy the model in a Streamlit app to predict costs for new inputs.
- **Deployment**: Host the app on Streamlit Cloud for interactive use.

## Results ✅

The Random Forest model achieved an **R² score of 0.89** and a ** RMSE: $3502.88** on the test set, indicating strong predictive accuracy. The Streamlit app provides predictions with a confidence metric based on this R² score. 🎉

## Using the Trained Model 🧰

The trained Random Forest model and preprocessing pipeline are saved as `insurance_model.pkl`. To use it for predictions:

1. Load the model using `pickle`.
2. Prepare input data in a DataFrame matching the training format.
3. Predict the insurance cost.

**Example code**:
```python
import pandas as pd
import pickle

# Load the model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create input data
new_data = pd.DataFrame({
    'age': [30],
    'sex': ['male'],
    'bmi': [25.0],
    'smoker': ['no'],
    'region': ['northeast'],
    'children': [1]
})

# Predict
predicted_cost = model.predict(new_data)[0]
print(f"Predicted Insurance Cost: ${predicted_cost:,.2f}")
