# ============================================================
# Telco Customer Churn Prediction - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier  # needed so joblib can load the model
from sklearn.preprocessing import StandardScaler  # for type hints only

# ------------------------------
# 0. Streamlit Page Config (MUST be first Streamlit command)
# ------------------------------

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ------------------------------
# 1. Load trained artifacts
# ------------------------------

@st.cache_resource
def load_artifacts():
    """
    Load the trained model, scaler, and feature columns.
    Assumes they are stored in the 'models/' directory.
    """
    xgb_model = joblib.load("models/xgboost_tuned.pkl")
    scaler: StandardScaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return xgb_model, scaler, feature_columns

xgb_model, scaler, feature_columns = load_artifacts()

# Numeric features we scaled during training
NUM_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']

st.title("Telco Customer Churn Prediction")
st.write(
    """
    This app predicts the **probability of churn** for a telecom customer  
    using a trained XGBoost model built on the Telco Customer Churn dataset.
    """
)

st.markdown("---")

# ------------------------------
# 3. Collect User Input
# ------------------------------

st.subheader("Customer Details")

# Layout: two columns for cleaner UI
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months with company)", min_value=0, max_value=100, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

st.markdown("### 🔌 Additional Services")

col3, col4 = st.columns(2)

with col3:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

with col4:
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

st.markdown("### Billing Details")

col5, col6 = st.columns(2)

with col5:
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with col6:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=8000.0, value=1000.0, step=10.0)

st.markdown("---")

# ------------------------------
# 4. Build a single-row DataFrame from inputs
# ------------------------------

def build_input_dataframe():
    """
    Turn the raw Streamlit form inputs into a one-row DataFrame
    with the same structure as the original raw Telco dataset
    (before encoding).
    """
    data = {
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    }
    df_input = pd.DataFrame(data)
    return df_input

# ------------------------------
# 5. Preprocessing function (mirror training steps)
# ------------------------------

def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing as in the notebook:
    - Engineer AvgChargePerMonth
    - One-hot encode categorical variables (drop_first=True)
    - Align columns to training feature_columns
    - Scale numeric features using the fitted scaler
    """
    # Feature engineering
    df_raw = df_raw.copy()
    df_raw["AvgChargePerMonth"] = df_raw["TotalCharges"] / df_raw["tenure"].replace(0, 1)

    # One-hot encode categoricals in the same way: drop_first=True
    df_encoded = pd.get_dummies(df_raw, drop_first=True)

    # Align columns to the training feature set: add missing, drop extra
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric features
    df_aligned[NUM_FEATURES] = scaler.transform(df_aligned[NUM_FEATURES])

    return df_aligned

# ------------------------------
# 6. Predict button
# ------------------------------

if st.button("Predict Churn"):
    # Build raw df from inputs
    df_input_raw = build_input_dataframe()
    st.markdown("#### ▶ Input Summary")
    st.dataframe(df_input_raw)

    # Preprocess for model
    X_input = preprocess_input(df_input_raw)

    # Predict probability
    churn_proba = xgb_model.predict_proba(X_input)[:, 1][0]
    churn_pred = xgb_model.predict(X_input)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    st.write(f"**Churn probability:** `{churn_proba:.3f}`")

    if churn_pred == 1:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **unlikely to churn**.")

    # Add a little interpretation
    if churn_proba >= 0.7:
        st.write(
            "The churn probability is quite high. Consider offering retention incentives, "
            "such as discounts, contract benefits, or better support."
        )
    elif churn_proba >= 0.4:
        st.write(
            "This customer is at moderate risk. It may be useful to monitor their usage patterns "
            "and proactively reach out with engagement campaigns."
        )
    else:
        st.write(
            "This churn risk is relatively low compared to typical churners. "
            "Focus on maintaining good service quality."
        )

else:
    st.info("Fill in the customer details and click **Predict Churn** to get a prediction.")
