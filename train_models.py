import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_preprocess_data(filepath):
    # Load Dataset
    data = pd.read_csv(filepath)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Drop the 'Date' column
    data.drop(columns=['Date'], inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Split the data into features and target variable
    X = data.drop(columns=['RainTomorrow'])
    y = data['RainTomorrow']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('weatherAUS.csv')

    # Initialize the models
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    for name, model in models.items():
        accuracy, cm = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)
        print(f'{name} Accuracy: {accuracy}')
        print(f'{name} Confusion Matrix:\n {cm}')
