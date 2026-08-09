"""
generate_data.py
-----------------
Generates a synthetic loan-applicant dataset (data/loan_data.csv) so the
Smart Lender project can be trained and demoed end-to-end without needing
to manually source an external CSV.

If you already have a real loan-prediction dataset (e.g. from Kaggle),
just replace data/loan_data.csv with that file using the SAME column
names used below, and skip running this script.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 800  # number of synthetic applicants

genders = np.random.choice(["Male", "Female"], size=N, p=[0.7, 0.3])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.8, 0.2])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.15, 0.85])
applicant_income = np.random.gamma(shape=5, scale=1200, size=N).astype(int) + 1500
coapplicant_income = np.random.gamma(shape=2, scale=800, size=N).astype(int)
coapplicant_income = np.where(married == "No", 0, coapplicant_income)
loan_amount = (np.random.gamma(shape=4, scale=30, size=N)).astype(int) + 50
loan_term = np.random.choice([360, 180, 120, 60], size=N, p=[0.75, 0.15, 0.06, 0.04])
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.85, 0.15])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N)

# Build a "true" probability of approval using a simple weighted rule,
# then sample the label -- this gives the models real signal to learn.
income_total = applicant_income + coapplicant_income
score = (
    0.45 * credit_history
    + 0.20 * (income_total > np.median(income_total)).astype(float)
    + 0.15 * (loan_amount < np.median(loan_amount)).astype(float)
    + 0.10 * (education == "Graduate").astype(float)
    + 0.10 * (property_area != "Rural").astype(float)
)
prob_approved = np.clip(score + np.random.normal(0, 0.08, size=N), 0, 1)
loan_status = np.where(prob_approved > 0.5, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{1000+i}" for i in range(N)],
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# Sprinkle in a few missing values, like real-world data
for col in ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    idx = df.sample(frac=0.03, random_state=1).index
    df.loc[idx, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/loan_data.csv", index=False)
print(f"Synthetic dataset created: data/loan_data.csv ({len(df)} rows)")
