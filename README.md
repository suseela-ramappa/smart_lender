# Smart Lender

Machine learning–powered web app that predicts loan applicant creditworthiness
using Decision Tree, Random Forest, KNN, and XGBoost classifiers, served through
a Flask web interface.

## Project structure
```
smart-lender/
├── data/
│   └── loan_data.csv        (created by generate_data.py, or replace with a real dataset)
├── model/
│   ├── train_model.py       (trains all 4 models, saves the best one)
│   ├── best_model.pkl        <- created after training
│   ├── encoders.pkl           <- created after training
│   ├── feature_cols.pkl       <- created after training
│   └── model_name.pkl         <- created after training
├── templates/
│   └── index.html            (prediction form UI)
├── static/
├── generate_data.py          (creates a synthetic dataset to train on)
├── app.py                    (Flask web app)
├── requirements.txt
└── README.md
```

## Setup (Windows / VS Code)

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Generate the dataset (skip this step if you already have a real
   loan-prediction CSV — just place it at `data/loan_data.csv` with the
   same column names):
   ```
   python generate_data.py
   ```
5. Train the models:
   ```
   python model/train_model.py
   ```
   This prints the accuracy of all four models and saves the best one.
6. Run the web app:
   ```
   python app.py
   ```
7. Open your browser at **http://127.0.0.1:5000** and try a prediction.

## Using a real dataset instead of the synthetic one

Search Kaggle for **"Loan Prediction Problem Dataset"**. Download the CSV and
save it as `data/loan_data.csv`, keeping these exact column names:

```
Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Property_Area, Loan_Status
```

Then just re-run steps 5–6 above (no need to run `generate_data.py`).

## Uploading to GitHub

```
git init
git add .
git commit -m "Initial commit - Smart Lender"
git branch -M main
git remote add origin https://github.com/<your-username>/smart-lender.git
git push -u origin main
```

## Recording a demo video

1. Run `python app.py` and open the app in your browser.
2. Use Windows Game Bar (`Windows + G`) or any screen recorder.
3. Show: entering applicant details → clicking Predict → viewing the result,
   for 2–3 different scenarios (low risk, high risk).
4. Upload the recording to YouTube (Unlisted) or Google Drive (shared link)
   and paste that link into the Skill Wallet "Add Demo Link" field.
