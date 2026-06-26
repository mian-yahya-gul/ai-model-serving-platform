"""
train_export_model.py
Trains the churn classifier and exports model + encoders as artifacts
for the serving API to load. This script is intentionally separate
from the serving app — in production, training and serving are
different lifecycle stages.

Run:
    python models/train_export_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.pkl")

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def main():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    df = df.drop(columns=DROP_COLS)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(
        {
            "feature_order": list(X.columns),
            "accuracy": acc,
            "roc_auc": auc,
            "model_version": "v1.0.0",
        },
        METADATA_PATH,
    )

    print(f"Model exported to {MODEL_PATH}")
    print(f"Test accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
