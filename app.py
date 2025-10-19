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
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15,
        'month_sold': month_sold
    }])

    # Predict log(price)
    log_pred = model.predict(input_data)

    # Convert back to actual price
    price_pred = np.exp(log_pred)

    # Return nicely formatted string
    return f"${price_pred[0]:,.2f}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Sqft Living"),
        gr.Number(label="Sqft Lot"),
        gr.Number(label="Floors"),
        gr.Number(label="Waterfront (0/1)"),
        gr.Number(label="View (0-4)"),
        gr.Number(label="Grade (1-13)"),
        gr.Number(label="Year Built"),
        gr.Number(label="Year Renovated"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
        gr.Number(label="Sqft Living 15"),
        gr.Number(label="Sqft Lot 15"),
        gr.Number(label="Month Sold (1-12)")
    ],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="🏠 House Price Prediction App",
    description="Enter the features of a house to get the predicted price."
)

# Launch app
interface.launch()
