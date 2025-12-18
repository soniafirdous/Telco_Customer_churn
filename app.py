import streamlit as st
import requests
import pandas as pd

# ---------------- Sidebar ----------------
st.sidebar.title("Telecom Customer Churn")

st.sidebar.image("churn_image.jpeg", caption="Customer Churn", width=250)

st.sidebar.info("""
Customer churn occurs when customers stop using a company's service.
Predicting churn helps businesses improve retention.
""")

# ---------------- Main App ----------------
st.title("Customer Churn Prediction")
st.write("Enter customer details:")

# ----- Inputs -----
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = int(st.selectbox("Senior Citizen", [0, 1]))
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, value=1)

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0)
TotalCharges = st.number_input("Total Charges", min_value=0)

class_names = ["No Churn", "Churn"]

# ----- Prediction -----
if st.button("Predict Churn"):
    payload = {
        "features": [
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        ]
    }

    url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            st.error(response.text)
        else:
            result = response.json()
            label = int(result["class_label"])
            st.success(f"Prediction: **{class_names[label]}**")

            probs = result["probabilities"][0]
            df = pd.DataFrame(probs, index=class_names, columns=["Probability"])
            st.bar_chart(df)

    except Exception as e:
        st.error(str(e))
