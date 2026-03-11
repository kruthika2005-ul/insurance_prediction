import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Insurance Cost Prediction")

# Load dataset
df = pd.read_csv("insurance.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Model Score
score = r2_score(y_test, predictions)
st.write("Model R2 Score:", score)

st.subheader("Enter Customer Details")

age = st.number_input("Enter Age", min_value=0, max_value=100)
bmi = st.number_input("Enter BMI")
children = st.number_input("Children", min_value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex=="male" else 0],
        "smoker_yes": [1 if smoker=="yes" else 0],
        "region_northwest": [1 if region=="northwest" else 0],
        "region_southeast": [1 if region=="southeast" else 0],
        "region_southwest": [1 if region=="southwest" else 0]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
