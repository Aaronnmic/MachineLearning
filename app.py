import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Muat model
model = joblib.load('random_forest_model.pkl')

# Judul aplikasi
st.title('Prediksi Cuaca - Apakah akan hujan besok?')

# Input data pengguna
MinTemp = st.number_input('Min Temperature')
MaxTemp = st.number_input('Max Temperature')
Rainfall = st.number_input('Rainfall')
Humidity3pm = st.number_input('Humidity at 3pm')
Pressure9am = st.number_input('Pressure at 9am')
Temp9am = st.number_input('Temperature at 9am')
Temp3pm = st.number_input('Temperature at 3pm')

# Prediksi
if st.button('Predict'):
    input_data = pd.DataFrame({
        'MinTemp': [MinTemp],
        'MaxTemp': [MaxTemp],
        'Rainfall': [Rainfall],
        'Humidity3pm': [Humidity3pm],
        'Pressure9am': [Pressure9am],
        'Temp9am': [Temp9am],
        'Temp3pm': [Temp3pm]
    })
    prediction = model.predict(input_data)
    result = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Apakah akan hujan besok? {result}')
