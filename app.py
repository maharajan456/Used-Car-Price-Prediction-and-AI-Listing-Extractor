import streamlit as st
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from custom_transformer import OrderedTargetEncoder

# Load the trained pipeline and category dictionary
pipe = joblib.load("final_pipeline.joblib")
cat_dict = joblib.load("cat_dict.joblib")

st.title("Used Car Price Predictor")

# User inputs for car features
brand = st.selectbox("Choose Brand Name", cat_dict['all_brands'])

# Check if the selected brand is in the category dictionary, else assign 'Other'
if brand in cat_dict['brand']:
    brand = brand

else:
    brand = 'Other'

# Model name input
model = st.text_input("Enter Model Name")

# Model year input
model_year = st.number_input("Enter Model Year")

# Engine type selection
engine_type = st.selectbox("Choose Engine Type", cat_dict["engine_type"])

# Transmission type selection
transmission = st.selectbox("Choose transmission Type", cat_dict["transmission"])

# Fuel type selection
fuel_type = st.selectbox("Choose Fuel Type", cat_dict["fuel_type"])

# Exterior color input
ext_col = st.text_input("Choose Extrior Color")

# Mileage input
mileage = st.number_input("Enter Mileage in Odometer (In Miles)")

# Damage reported selection
damage = st.selectbox("Choose Damage Reported or not", cat_dict["damage"])


# Predict button for prediction the price of the car based on user inputs
if st.button("Predict"):
    # Create a DataFrame with the user inputs
    df = pd.DataFrame({"brand": [brand], 
                       "model": [model], 
                       "model_year": [model_year], 
                       "engine_type": [engine_type], 
                       "transmission": [transmission],
                       "fuel_type": [fuel_type],
                       "ext_col": [ext_col],
                       "mileage in miles": [mileage], 
                       "damage": [damage]})
    
    # Predict the log price using the pipeline and then convert it back to the original price
    log_prediction = pipe.predict(df)
    predicted_price = np.exp(log_prediction)

    
    # Display the predicted price to the user
    st.title(f"Predicted Price of Your Configured Model is ${predicted_price[0]:,.2f}")
