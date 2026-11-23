"""Train the churn model on the IBM Telco Customer Churn dataset and save it for the app."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Load data
print("Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaning & preprocessing
# TotalCharges is blank for brand-new customers, so it loads as text. Convert, then fill.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# customerID is an identifier, not a signal.
df.drop(columns=['customerID'], inplace=True)

# Keep each column's encoder: the app must encode inputs exactly the same way.
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (xgboost)
print("Training XGBoost Model...")
# Gradient-boosted trees are a strong default for tabular data like this.
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained. Test accuracy: {accuracy:.2%}")

# Save artifacts
joblib.dump(model, 'churn_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
joblib.dump(X.columns, 'features.pkl')

print("Saved churn_model.pkl, encoders.pkl and features.pkl")