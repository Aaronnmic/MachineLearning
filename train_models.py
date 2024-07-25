import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('weatherAUS.csv')
df = df.dropna()

# Adjusted features and target based on actual column names
X = df[['MaxTemp', 'WindSpeed9am', 'Pressure9am', 'Rainfall']]
y = df['RainTomorrow']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model, 'nb_model.pkl')

# Train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'dt_model.pkl')

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.pkl')

# Evaluate models
nb_preds = nb_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
