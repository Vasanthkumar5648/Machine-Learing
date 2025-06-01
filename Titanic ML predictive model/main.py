import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load Titanic dataset (for preprocessing reference only)
@st.cache_data
def load_data():
    df = pd.read_csv(https://raw.github.com/Vasanthkumar5648/Machine-Learing/main/Titanic%20ML%20predictive%20model/titanic-2.csv)  # Ensure this file exists
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
    X = df[features]
    y = df['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(model, "logistic_model.pkl")

    return model, scaler

# Load model and scaler
try:
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model, scaler = load_data()

# Streamlit UI
st.title("Titanic Survival Predictor 🚢")

st.sidebar.header("Passenger Details")

def get_user_input():
    Pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.sidebar.selectbox("Sex", ["male", "female"])
    Age = st.sidebar.slider("Age", 0, 100, 30)
    SibSp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
    Fare = st.sidebar.slider("Fare Paid", 0.0, 600.0, 50.0)
    Embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

    Sex = 1 if Sex == "male" else 0
    Embarked_Q = 1 if Embarked == "Q" else 0
    Embarked_S = 1 if Embarked == "S" else 0

    data = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked_Q": Embarked_Q,
        "Embarked_S": Embarked_S
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

# Display results
st.subheader("Prediction")
st.write("✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive")

st.subheader("Survival Probability")
st.write(f"{probability:.2%}")
