import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load trained model
model = joblib.load("House_Price_Prediction_Model.pk1")

# Prediction function
def predict_price(
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, grade, yr_built, yr_renovated,
    lat, long, sqft_living15, sqft_lot15, month_sold
):
    # Create single-row DataFrame for prediction
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'grade': grade,
        'yr_built'_
