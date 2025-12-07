# 📡 Telco Customer Churn Prediction

> **An end-to-end Data Science project predicting telecom customer churn using XGBoost and deployed with Streamlit.**

This project demonstrates a complete machine learning pipeline — from **data cleaning and feature engineering** to **model building, evaluation, and deployment** — applied to the **Telco Customer Churn Dataset**.  

The final app allows users to input customer details and instantly receive a **churn probability** with interpretive recommendations.

---

## 🚀 Live App

🎯 **Try it here:** [https://telcochurnprediction-dpnmsgt9en5fhbpfywtu2n.streamlit.app/](https://telcochurnprediction-dpnmsgt9en5fhbpfywtu2n.streamlit.app/)

---

## 🎯 Project Overview

Telecom companies lose substantial revenue every year when customers discontinue their services (**churn**).  
This project predicts **which customers are likely to churn**, based on demographic, contractual, and service-related features.  

By identifying these customers early, companies can **take proactive retention measures** like discounts, service improvements, or loyalty programs.

---

## 🧱 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Language** | Python |
| **Framework** | Streamlit |
| **Libraries** | Pandas, NumPy, Scikit-learn, XGBoost, Joblib |
| **Visualization** | Matplotlib, Seaborn, SHAP |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git + GitHub |

---

## 🧠 Workflow Breakdown

### 🩺 **1. Data Loading & Cleaning**
- Loaded Telco Customer Churn dataset (~7,000 rows).
- Converted `TotalCharges` to numeric.
- Imputed missing values and dropped `customerID`.

### 📊 **2. Exploratory Data Analysis (EDA)**
Explored churn patterns:
- **Shorter tenure** → higher churn.
- **Month-to-month contracts** → highest churn.
- **Electronic check payments** → strong churn correlation.
- **No online security/tech support** → increased churn.

### ⚙️ **3. Feature Engineering**
- Created `AvgChargePerMonth = TotalCharges / tenure`.
- One-hot encoded categorical variables.
- Standard-scaled numeric features.

### 🤖 **4. Model Building**
Two models trained and compared:
- **Logistic Regression** – Baseline, interpretable model.
- **XGBoost (tuned)** – Achieved best performance (87% accuracy, 0.90 ROC-AUC).

### 🧾 **5. Evaluation Metrics**
Used multiple metrics for reliability:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Precision–Recall Curve

### 🔍 **6. Model Explainability**
Used **SHAP (SHapley Additive Explanations)** to interpret feature influence.

**Top churn drivers:**
- Low tenure
- Month-to-month contract
- High monthly charges
- Electronic check payments

### 🌐 **7. Deployment**
Deployed as an interactive Streamlit app:
- Users input customer info.
- Model predicts churn probability.
- The app displays intuitive feedback and business recommendations.

---

## 📈 Results Summary

| Model | Accuracy | F1-Score | ROC-AUC |
|--------|-----------|----------|----------|
| Logistic Regression | 0.82 | 0.74 | 0.85 |
| **XGBoost (Tuned)** | **0.87** | **0.79** | **0.90** |

✅ Improved recall for churners by ~10%.  
✅ Achieved interpretable, high-performing, and business-actionable predictions.  

---

## 🧮 Streamlit App Features

- 🧾 **Customer Input Form** – Enter tenure, billing, and service details.  
- ⚡ **Instant Prediction** – Real-time churn probability with classification.  
- 💡 **Interpretive Feedback** – Explains whether a customer is high/medium/low risk.  
- ☁️ **Live Deployment** – Hosted publicly via Streamlit Cloud.  

---

## 🗂️ Folder Structure

```text
telco-churn-app/
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── models/                 # Saved model artifacts
│   ├── xgboost_tuned.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── telco_churn_pipeline.ipynb  # Full notebook with EDA & modeling
└── data/
    └── Telco-Customer-Churn.csv (optional, for local retraining)
