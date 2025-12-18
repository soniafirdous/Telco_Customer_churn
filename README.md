# 📉 Customer Churn Prediction System

A complete **Machine Learning–based Customer Churn Prediction project** that compares multiple classification models, exposes predictions via a **FastAPI backend (GET & POST methods)**, and provides an interactive **Streamlit web app** for user-friendly testing.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. This project demonstrates an **end-to-end ML workflow**, from model training and evaluation to API deployment and frontend integration.

The focus of this project is:

* Comparing multiple ML models
* Practicing **FAST APIs (GET & POST)** using FastAPI
* Connecting ML models with a **Streamlit UI**

---

## 🔍 Feature Importance & Key Insights

The model analysis shows that **billing behavior, customer tenure, and contract type** dominate churn predictions. The most influential drivers are:

* **Billing & Tenure:** Total charges, monthly charges, and tenure are the strongest predictors. Customers with **higher bills or shorter tenure** are more likely to churn.
* **Contract Type:** Customers on **month-to-month contracts** show significantly higher churn, while **longer-term contracts reduce churn risk**.
* **Payment Method:** Paying via **electronic check** is associated with a slightly higher likelihood of churn.
* **Service Features:** Lack of **tech support** or **online security** moderately increases churn probability.
* **Low-Impact Features:** Demographics (e.g., gender) and minor services (streaming, no-internet indicators) have minimal effect and can be deprioritized.

---

## 🧠 Models Used & Accuracy

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Logistic Regression          | **73%**  |
| Decision Tree                | **71%**  |
| Random Forest                | **77%**  |
| XGBoost                      | **77%**  |
| Support Vector Machine (SVM) | **75%**  |
| Naive Bayes                  | **69%**  |

📌 **Best Performing Models:** Random Forest & XGBoost

---

## 🛠 Tech Stack

* **Python**
* **Scikit-learn**
* **XGBoost**
* **FastAPI** (GET & POST requests)
* **Uvicorn** (ASGI server)
* **Streamlit** (Frontend UI)
* **NumPy & Pandas**

---

## 🔁 Workflow

1. Data preprocessing & feature encoding
2. Train multiple ML classification models
3. Evaluate and compare accuracy scores
4. Save the best-performing model
5. Build FastAPI endpoints for prediction
6. Consume API using Streamlit app

---

## 🌐 FastAPI Endpoints

### ▶️ GET Method (Health Check)

```http
GET /
```

**Response:**

```json
{"message": "Customer Churn API is running"}
```

---

### ▶️ POST Method (Prediction)

```http
POST /predict
```

**Input:**

```json
{
  "features": [
    "Male", 0, "Yes", "No", 12, "Yes", "No", "DSL",
    "Yes", "No", "Yes", "No", "No", "No",
    "Month-to-month", "Yes", "Electronic check", 29.85, 356.15
  ]
}
```

**Response:**

```json
{
  "prediction": "No",
  "class_label": 0
}
```

---

## 🖥 Streamlit Application

The Streamlit app allows users to:

* Enter customer details
* Send data to FastAPI using POST requests
* View churn prediction instantly

📌 The Streamlit app acts as a **client**, while FastAPI serves as the **ML backend**.

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start FastAPI Server

```bash
uvicorn main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Key Learning Outcomes

* Model comparison & evaluation
* Practical usage of **GET & POST APIs**
* Integrating ML models with FastAPI
* Connecting backend APIs to Streamlit frontend
* Debugging API & deployment errors

---

## 📂 Project Structure

```
CustomerChurn/
│── main.py            # FastAPI backend
│── app.py             # Streamlit frontend
│── model.pkl          # Trained ML model
│── requirements.txt
│── README.md

---

## 🔗 Author

**Sonia Firdous**
soniafirdous1985@gmail.com
Aspiring Machine Learning Engineer

---

⭐ If you found this project helpful, don’t forget to star the repository!
