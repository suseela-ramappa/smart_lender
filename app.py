"""
app.py
------
Flask web application for the Smart Lender project. Loads the trained
model (produced by model/train_model.py) and serves a form where a user
enters applicant details and instantly gets a loan approval prediction.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
model_name = joblib.load(os.path.join(MODEL_DIR, "model_name.pkl"))

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def encode_input(form_data):
    row = {
        "Gender": form_data["Gender"],
        "Married": form_data["Married"],
        "Dependents": form_data["Dependents"],
        "Education": form_data["Education"],
        "Self_Employed": form_data["Self_Employed"],
        "ApplicantIncome": float(form_data["ApplicantIncome"]),
        "CoapplicantIncome": float(form_data["CoapplicantIncome"]),
        "LoanAmount": float(form_data["LoanAmount"]),
        "Loan_Amount_Term": float(form_data["Loan_Amount_Term"]),
        "Credit_History": float(form_data["Credit_History"]),
        "Property_Area": form_data["Property_Area"],
    }
    df = pd.DataFrame([row])

    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = str(df.at[0, col])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    return df[feature_cols]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = encode_input(request.form)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        result = "Approved" if pred == 1 else "Not Approved"
        confidence = f"{proba*100:.1f}%" if proba is not None else "N/A"

        return render_template(
            "index.html",
            model_name=model_name,
            result=result,
            confidence=confidence,
            form_data=request.form,
        )
    except Exception as e:
        return render_template(
            "index.html",
            model_name=model_name,
            error=str(e),
            form_data=request.form,
        )


if __name__ == "__main__":
    app.run(debug=True)
