import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="CreditWise Loan Approval", page_icon="🏦", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/nisargpatel1906/creditWise_loan_system/main/loan_approval_data.csv"

NUMERIC_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term"
]
CATEGORICAL_COLS = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category"
]
OHE_COLS = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Gender", "Employer_Category"
]

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("loan_approval_data.csv")
    except FileNotFoundError:
        return pd.read_csv(DATA_URL)

@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data().copy()

    # Same preprocessing approach used in the supplied notebook.
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    education_encoder = LabelEncoder()
    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])

    target_encoder = LabelEncoder()
    df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[OHE_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(OHE_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=OHE_COLS), encoded_df], axis=1)

    # Same final feature engineering used in the supplied notebook.
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    df["Loan_Income_Ratio"] = df["Loan_Amount"] / (df["Total_Income"] + 1)
    df["Loan_Collateral_Ratio"] = df["Loan_Amount"] / (df["Collateral_Value"] + 1)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "ohe": ohe,
        "education_encoder": education_encoder,
        "target_encoder": target_encoder,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "rows": len(df),
        "approval_rate": float(y.mean()),
    }
    return artifact


def build_input_row(values, artifact):
    row = pd.DataFrame([values])

    # Match notebook imputation behavior for user-entered values.
    for col in NUMERIC_COLS:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    row[NUMERIC_COLS] = row[NUMERIC_COLS].fillna(0)

    row["Education_Level"] = artifact["education_encoder"].transform(row["Education_Level"])

    encoded = artifact["ohe"].transform(row[OHE_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=artifact["ohe"].get_feature_names_out(OHE_COLS),
        index=row.index,
    )

    row = pd.concat([row.drop(columns=OHE_COLS), encoded_df], axis=1)

    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row["Total_Income"] = row["Applicant_Income"] + row["Coapplicant_Income"]
    row["Loan_Income_Ratio"] = row["Loan_Amount"] / (row["Total_Income"] + 1)
    row["Loan_Collateral_Ratio"] = row["Loan_Amount"] / (row["Collateral_Value"] + 1)

    row = row.reindex(columns=artifact["feature_columns"], fill_value=0)
    return row


# ---------- UI ----------
st.markdown("# 🏦 CreditWise")
st.markdown("### Intelligent Loan Approval Prediction System")
st.caption("Educational machine-learning demonstration based on the supplied supervised-learning notebook. It is not a real lending decision system.")

try:
    artifact = train_model()
except Exception as e:
    st.error("The model could not be loaded or trained.")
    st.code(str(e))
    st.stop()

with st.sidebar:
    st.header("📊 Model Information")
    st.write(f"Training records: **{artifact['rows']}**")
    st.write(f"Random Forest trees: **200**")
    st.write(f"Max depth: **8**")
    st.divider()
    st.caption("The app follows the preprocessing and final Random Forest workflow from the supplied notebook.")

st.subheader("Applicant Information")

with st.form("loan_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0.0, value=20000.0, step=1000.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])

    with c2:
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=700.0, step=1.0)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=1)
        dti = st.number_input("DTI Ratio", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        savings = st.number_input("Savings (₹)", min_value=0.0, value=100000.0, step=5000.0)
        collateral = st.number_input("Collateral Value (₹)", min_value=0.0, value=300000.0, step=5000.0)
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    with c3:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, value=200000.0, step=5000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=120, value=60)
        employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        employer = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])
        purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Car", "Business", "Education"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

if submitted:
    values = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": property_area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer,
    }

    try:
        input_df = build_input_row(values, artifact)
        scaled = artifact["scaler"].transform(input_df)
        prediction = int(artifact["model"].predict(scaled)[0])
        probability = float(artifact["model"].predict_proba(scaled)[0][1])

        st.divider()
        if prediction == 1:
            st.success(f"### ✅ LOAN APPROVED\n\nEstimated approval probability: **{probability * 100:.2f}%**")
        else:
            st.error(f"### ❌ LOAN REJECTED\n\nEstimated approval probability: **{probability * 100:.2f}%**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Approval Probability", f"{probability * 100:.1f}%")
        m2.metric("Loan Amount", f"₹{loan_amount:,.0f}")
        m3.metric("Credit Score", f"{credit_score:.0f}")

        with st.expander("View calculated features"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Please check the entered values and dataset categories.")
        st.code(str(e))

st.divider()
st.subheader("📈 Model Evaluation")
cols = st.columns(5)
for col, (name, value) in zip(cols, artifact["metrics"].items()):
    col.metric(name, f"{value * 100:.2f}%")

st.info("⚠️ This application is for academic/demo purposes. A real lending decision requires validated data, regulatory review, human oversight, and appropriate fairness/risk controls.")
