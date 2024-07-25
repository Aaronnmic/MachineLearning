import streamlit as st
import pandas as pd
import joblib

# Load trained models
nb_model = joblib.load('nb_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# User inputs
st.title("Weather Prediction App")
MaxTemp = st.number_input("Max Temperature")
WindSpeed9am = st.number_input("Wind Speed at 9 AM")
Pressure9am = st.number_input("Pressure at 9 AM")
Rainfall = st.number_input("Rainfall")

# Make prediction
input_data = [[MaxTemp, WindSpeed9am, Pressure9am, Rainfall]]
nb_prediction = nb_model.predict(input_data)
dt_prediction = dt_model.predict(input_data)
rf_prediction = rf_model.predict(input_data)

# Display prediction
st.write(f"Naive Bayes Prediction: {'Yes' if nb_prediction[0] == 'Yes' else 'No'}")
st.write(f"Decision Tree Prediction: {'Yes' if dt_prediction[0] == 'Yes' else 'No'}")
st.write(f"Random Forest Prediction: {'Yes' if rf_prediction[0] == 'Yes' else 'No'}")
