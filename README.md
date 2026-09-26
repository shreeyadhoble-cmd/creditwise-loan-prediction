# CreditWise Loan Approval — Streamlit

Streamlit deployment of the supplied supervised-learning loan approval notebook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app first looks for `loan_approval_data.csv` in the project folder. If it is not present, it downloads the public dataset used by the matching project repository at runtime.

## Deploy

1. Put `app.py`, `requirements.txt`, and optionally `loan_approval_data.csv` in a GitHub repository.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository and `app.py` as the main file.
4. Deploy.

## Model workflow

The app follows the supplied notebook's final workflow:

- Mean imputation for numeric columns
- Most-frequent imputation for categorical columns
- Drop `Applicant_ID`
- Label encoding for `Education_Level` and `Loan_Approved`
- One-hot encoding for nominal categorical fields
- `DTI_Ratio_sq`
- `Credit_Score_sq`
- `Total_Income`
- `Loan_Income_Ratio`
- `Loan_Collateral_Ratio`
- StandardScaler
- RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
