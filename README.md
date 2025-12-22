# 📉 Telco Customer Churn Prediction & Risk Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)

### 🚀 Project Overview
This project focuses on **Customer Retention Analytics** for the telecommunications industry. Using a dataset of 7,000+ customers, I built a machine learning application that predicts the likelihood of a customer cancelling their service (Churn).

Unlike "Black Box" models, this tool integrates **Explainable AI (SHAP)** to tell business stakeholders *why* a customer is at risk (e.g., *"High Monthly Charges"* or *"Short Tenure"*), enabling targeted retention strategies.

### 💼 Business Value
* **Proactive Retention:** Identify at-risk customers *before* they leave.
* **Explainability:** Provides actionable reasons for churn, allowing support teams to offer specific discounts or contract upgrades.
* **Scenario Planning:** The interactive dashboard allows managers to simulate changes (e.g., "If we offer a 1-year contract, does the risk drop?") in real-time.

### 🛠️ Technical Workflow
1.  **Data Engineering:** Cleaned messy data (handling missing values in `TotalCharges`), encoded categorical variables (One-Hot/Label Encoding), and engineered features.
2.  **Modeling:** Trained an **XGBoost Classifier** (Extreme Gradient Boosting) optimized for tabular data.
3.  **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** values to visualize feature importance dynamically.
4.  **Deployment:** Built a user-friendly web interface using **Streamlit**.

### 💻 How to Run Locally

**1. Clone the Repository**
```bash
git clone https://github.com/devanshi4/churn-prediction.git
cd churn-prediction

2. Install Dependencies

Bash
pip install -r requirements.txt

3. Run the App

Bash
# (Optional) Retrain the model
python train_model.py

# Launch the Dashboard
streamlit run app.py

📊 Model Performance

Algorithm: XGBoost Classifier
Accuracy: ~79%
Key Drivers: Contract Type, Tenure, Electronic Check Payment

📂 Project Structure


├── train_model.py      # Script to clean data and train XGBoost
├── app.py              # Streamlit Dashboard application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation


## 🚀 Future Roadmap
- **Deployment:** Deploying the Streamlit app to the cloud (Streamlit Community Cloud or AWS).
- **Model Monitoring:** Implementing drift detection to alert when customer data patterns change over time.
- **Explainability:** Adding detailed "What-If" analysis allowing business users to simulate changes in customer plans.