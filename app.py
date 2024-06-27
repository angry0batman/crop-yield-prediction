import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and the scaler
with open('crop_yield_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the crop names and their corresponding codes
crop_mapping = {
    'Cocoa, beans': 0,
    'Oil palm fruit': 1,
    'Rice, paddy': 2,
    'Soybean': 3
}

# Streamlit app title
st.title('Crop Yield Prediction')

# HTML and CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
        .title {
            color: #4CAF50;
            text-align: center;
        }
        .input {
            margin-bottom: 20px;
        }
        .submit-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 10px;
        }
        .result {
            font-size: 24px;
            text-align: center;
            color: #4CAF50;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

# Input fields
st.markdown('<h2 class="title">Enter the features for prediction</h2>', unsafe_allow_html=True)

st.markdown('<div class="input">', unsafe_allow_html=True)
crop_name = st.selectbox('Crop', options=list(crop_mapping.keys()), key='1')
crop_code = crop_mapping[crop_name]
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input">', unsafe_allow_html=True)
precipitation = st.number_input('Precipitation (mm day-1)', min_value=0.0, step=0.1, key='2')
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input">', unsafe_allow_html=True)
specific_humidity = st.number_input('Specific Humidity at 2 Meters (g/kg)', min_value=0.0, step=0.1, key='3')
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input">', unsafe_allow_html=True)
relative_humidity = st.number_input('Relative Humidity at 2 Meters (%)', min_value=0.0, step=0.1, key='4')
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input">', unsafe_allow_html=True)
temperature = st.number_input('Temperature at 2 Meters (C)', min_value=-50.0, step=0.1, key='5')
st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
if st.button('Predict', key='6'):
    # Create a DataFrame for the input
    input_features = pd.DataFrame([[crop_code, precipitation, specific_humidity, relative_humidity, temperature]],
                                  columns=['Crop', 'Precipitation (mm day-1)', 'Specific Humidity at 2 Meters (g/kg)',
                                           'Relative Humidity at 2 Meters (%)', 'Temperature at 2 Meters (C)'])

    # Scale the input features using the loaded scaler
    input_scaled = scaler.transform(input_features)

    # Make the prediction
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.markdown(f'<div class="result">Predicted Yield: {prediction[0]:.2f}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
