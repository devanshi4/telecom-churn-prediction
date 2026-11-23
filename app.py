"""Streamlit dashboard: score a customer's churn risk and explain the prediction with SHAP."""
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Telecom Customer Retention Dashboard")
st.markdown("""
This tool uses **XGBoost Machine Learning** to predict if a customer is likely to cancel their service (Churn).
Adjust the customer profile in the sidebar to see how the risk score changes.
""")

# Load the trained model and encoders
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    encoders = joblib.load('encoders.pkl')
    features = joblib.load('features.pkl')
    return model, encoders, features

try:
    model, encoders, features = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model files not found. Did you run 'train_model.py'?")
    st.stop()

# Sidebar: customer profile
st.sidebar.header("📝 Customer Profile")

def user_input_features():
    # Same columns and category labels as the training data.
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

    st.sidebar.markdown("---")
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.sidebar.markdown("---")
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=monthly * tenure)

    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': device,
        'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
        'PaperlessBilling': billing, 'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Preprocess inputs
# Encode with the encoders saved at training time, so "Yes" maps to the same number.
proc_df = input_df.copy()

for col, le in encoders.items():
    if col in proc_df.columns:
        # An unseen category falls back to 0 instead of crashing the app.
        proc_df[col] = proc_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

# XGBoost expects the training column order.
proc_df = proc_df[features]

# Prediction & display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Prediction")
    prediction = model.predict(proc_df)
    probability = model.predict_proba(proc_df)

    churn_prob = probability[0][1]

    if churn_prob > 0.5:
        st.error(f"⚠️ HIGH CHURN RISK: {churn_prob:.1%}")
    else:
        st.success(f"✅ LOYAL CUSTOMER: {churn_prob:.1%}")

    st.write(f"This customer has a **{churn_prob:.1%}** chance of leaving the company.")

# Explainability (SHAP)
with col2:
    st.subheader("💡 Why this prediction?")
    st.write("The chart shows which factors had the most influence on this customer's risk score.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(proc_df)

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.summary_plot(shap_values, proc_df, plot_type="bar", show=False, max_display=10)
    st.pyplot(fig)

st.divider()
st.subheader("🔍 Raw Input Data")
st.dataframe(input_df)