import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==========================================
# 2. CLEANING & PREPROCESSING
# ==========================================
# 'TotalCharges' is read as text because of empty spaces. Force it to number.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop useless columns
df.drop(columns=['customerID'], inplace=True)

# Encode categorical variables (Turn "Yes"/"No" into 1/0)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ==========================================
# 3. SPLIT DATA
# ==========================================
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TRAIN MODEL (XGBoost)
# ==========================================
print("Training XGBoost Model...")
# Using XGBoost because it handles tabular data better than Deep Learning
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Trained! Accuracy: {accuracy:.2%}")

# ==========================================
# 5. SAVE THE BRAIN
# ==========================================
# We save the model AND the encoders so the app can understand user input
joblib.dump(model, 'churn_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
joblib.dump(X.columns, 'features.pkl') # Save column names to keep order

print("💾 Model saved to 'churn_model.pkl'")