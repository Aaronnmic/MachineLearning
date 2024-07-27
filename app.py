import streamlit as st
from weather_model import load_and_preprocess_data, train_and_evaluate_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.title('Weather Prediction')

X_train, X_test, y_train, y_test = load_and_preprocess_data('weatherAUS.csv')

model_choice = st.selectbox('Choose Model', ['Naive Bayes', 'Decision Tree', 'Random Forest'])

if model_choice == 'Naive Bayes':
    model = GaussianNB()
elif model_choice == 'Decision Tree':
    model = DecisionTreeClassifier()
else:
    model = RandomForestClassifier()

accuracy, cm = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)

st.write(f'Accuracy: {accuracy}')
st.write('Confusion Matrix:')
st.write(cm)
