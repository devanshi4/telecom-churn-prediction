"""Train the churn model on the IBM Telco Customer Churn dataset and save it for the app."""
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_and_prepare(path=DATA_FILE):
    """Return features, target and the fitted encoders."""
    df = pd.read_csv(path)

    # TotalCharges is blank for brand-new customers, so it loads as text. Convert, then fill.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID is an identifier, not a signal.
    df = df.drop(columns=["customerID"])

    # Keep each column's encoder: the app must encode inputs exactly the same way.
    label_encoders = {}
    for column in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y, label_encoders


def split(X, y):
    # Stratified so train and test keep the same churn rate (about 27%).
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def build_model():
    # Gradient-boosted trees are a strong default for tabular data like this.
    return xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)


def main():
    print("Loading data...")
    X, y, label_encoders = load_and_prepare()
    X_train, X_test, y_train, y_test = split(X, y)

    print("Training XGBoost model...")
    model = build_model()
    model.fit(X_train, y_train)
    print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.2%}")

    joblib.dump(model, "churn_model.pkl")
    joblib.dump(label_encoders, "encoders.pkl")
    joblib.dump(X.columns, "features.pkl")
    print("Saved churn_model.pkl, encoders.pkl and features.pkl")


if __name__ == "__main__":
    main()
