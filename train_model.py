"""
train_model.py
---------------
Trains four classifiers (Decision Tree, Random Forest, KNN, XGBoost) on the
loan applicant dataset, evaluates them, and saves the best-performing model
(along with the label encoders and feature list) so app.py can load it for
real-time prediction.

Run from the project root:
    python model/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- run: pip install xgboost")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_data.csv")
MODEL_DIR = os.path.dirname(__file__)

FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
TARGET_COL = "Loan_Status"


def load_and_clean(path):
    df = pd.read_csv(path)

    # Fill missing values
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History",
                "ApplicantIncome", "CoapplicantIncome"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df, encoders=None, fit=True):
    """Label-encode categorical columns. Reuses fitted encoders at inference time."""
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )
    return df, encoders


def main():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].map({"Y": 1, "N": 0})

    X_encoded, encoders = encode_features(X, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42
        )

    results = {}
    fitted_models = {}

    print("\nTraining & evaluating models...\n" + "-" * 40)
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"train_acc": train_acc, "test_acc": test_acc}
        fitted_models[name] = model
        print(f"{name:15s} | Train Accuracy: {train_acc*100:5.1f}% | Test Accuracy: {test_acc*100:5.1f}%")

    # Pick the best model by test accuracy
    best_name = max(results, key=lambda k: results[k]["test_acc"])
    best_model = fitted_models[best_name]
    print("-" * 40)
    print(f"Best model: {best_name} "
          f"(Train: {results[best_name]['train_acc']*100:.1f}%, "
          f"Test: {results[best_name]['test_acc']*100:.1f}%)")

    # Save model, encoders, and feature order
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "model_name.pkl"))

    print(f"\nSaved: model/best_model.pkl, model/encoders.pkl, "
          f"model/feature_cols.pkl, model/model_name.pkl")


if __name__ == "__main__":
    main()
