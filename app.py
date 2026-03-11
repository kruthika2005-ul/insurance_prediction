import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("insurance.csv")

# Convert smoker column
df["smoker"] = df["smoker"].map({"yes":1, "no":0})

# Feature Engineering
df["smoker_bmi_interaction"] = df["smoker"] * df["bmi"]

# Input and Output
X = df[["age","bmi","children","smoker","smoker_bmi_interaction"]]
y = df["charges"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

# Streamlit UI
st.title("Insurance Charges Prediction App")

st.write("Enter details to predict medical insurance cost")

age = st.slider("Age",18,65)
bmi = st.number_input("BMI",15.0,50.0)
children = st.number_input("Children",0,5)

smoker = st.selectbox("Smoker",["No","Yes"])

# Convert smoker input
smoker_val = 1 if smoker == "Yes" else 0

# Interaction Feature
interaction = smoker_val * bmi

# Prediction
if st.button("Predict Charges"):
    
    input_data = np.array([[age,bmi,children,smoker_val,interaction]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Insurance Charges: ${prediction[0]:.2f}")