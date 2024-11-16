import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load pre-trained model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Title of the web app
st.title("Crop Yield Prediction")

# Add input fields for user inputs
st.header("Enter Crop Data")

# Input fields for features
year = st.number_input("Year", min_value=1900, max_value=2024, value=1990)
average_rain_fall = st.number_input("Average Rainfall (mm per year)", value=1485.0)
pesticides_tonnes = st.number_input("Pesticides (Tonnes)", value=121.0)
avg_temp = st.number_input("Average Temperature (°C)", value=16.37)
area = st.selectbox("Select Area", ['Albania', 'Zimbabwe', 'India', 'USA', 'Brazil'])  # Add other areas if needed
item = st.selectbox("Select Crop", ['Maize', 'Sorghum', 'Wheat', 'Rice'])  # Add other crops if needed

# Function to make prediction
def predict_yield(year, rainfall, pesticides, temp, area, item):
    # Prepare input features as a numpy array
    features = np.array([[year, rainfall, pesticides, temp, area, item]], dtype=object)
    
    # Apply the preprocessor (scaling and encoding)
    transformed_features = preprocessor.transform(features)
    
    # Make prediction using the trained model
    predicted_yield = dtr.predict(transformed_features)
    
    return predicted_yield[0]

# Button to make prediction
if st.button("Predict Yield"):
    predicted_yield = predict_yield(year, average_rain_fall, pesticides_tonnes, avg_temp, area, item)
    st.write(f"The predicted crop yield is: {predicted_yield:.2f} hg/ha")

# Footer
st.write("### About")
st.write("This app predicts crop yield based on various inputs using a trained Decision Tree Regressor model.")
