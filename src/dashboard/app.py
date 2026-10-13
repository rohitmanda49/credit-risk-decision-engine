"""Streamlit dashboard for the Credit Risk Decision Engine."""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# ---------- Configuration ----------
API_URL = "http://localhost:8000/predict"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

st.set_page_config(
    page_title="Credit Risk Decision Engine",
    page_icon="💳",
    layout="wide"
)

# Human-readable labels for SHAP features
FEATURE_LABELS = {
    'EXT_SOURCE_MEAN': 'Average external credit score',
    'EXT_SOURCE_MIN': 'Lowest external credit score',
    'EXT_SOURCE_MAX': 'Highest external credit score',
    'EXT_SOURCE_PROD': 'Combined external credit score',
    'EXT_SOURCE_STD': 'External credit score (spread)',
    'EXT_SOURCE_1': 'External credit score #1',
    'EXT_SOURCE_2': 'External credit score #2',
    'EXT_SOURCE_3': 'External credit score #3',
    'CREDIT_TERM_MONTHS': 'Loan term (months)',
    'CREDIT_INCOME_RATIO': 'Credit-to-income ratio',
    'ANNUITY_INCOME_RATIO': 'Payment-to-income ratio',
    'CREDIT_GOODS_RATIO': 'Credit-to-goods ratio',
    'EMPLOYED_AGE_RATIO': 'Employment-to-age ratio',
    'INCOME_PER_FAMILY': 'Income per family member',
    'INCOME_PER_CHILD': 'Income per child',
    'AGE_YEARS': 'Age',
    'DAYS_BIRTH': 'Age (in days)',
    'DAYS_EMPLOYED': 'Employment length',
    'DAYS_ID_PUBLISH': 'Days since ID published',
    'DAYS_REGISTRATION': 'Days since registration',
    'DAYS_LAST_PHONE_CHANGE': 'Days since last phone change',
    'AMT_CREDIT': 'Credit amount',
    'AMT_ANNUITY': 'Monthly payment',
    'AMT_INCOME_TOTAL': 'Annual income',
    'AMT_GOODS_PRICE': 'Goods price',
    'NAME_EDUCATION_TYPE': 'Education level',
    'NAME_INCOME_TYPE': 'Income type',
    'NAME_FAMILY_STATUS': 'Marital status',
    'NAME_HOUSING_TYPE': 'Housing type',
    'NAME_CONTRACT_TYPE': 'Contract type',
    'OCCUPATION_TYPE': 'Occupation',
    'ORGANIZATION_TYPE': 'Employer type',
    'OWN_CAR_AGE': 'Car age',
    'REGION_RATING_CLIENT': 'Region rating',
    'REGION_RATING_CLIENT_W_CITY': 'Region rating (with city)',
    'DOCUMENT_COUNT': 'Documents provided',
    'DAYS_EMPLOYED_ANOMALY': 'Not in standard employment',
}


def label(feature_name: str) -> str:
    """Return a human-readable label, or the raw name if unmapped."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' ').title())


# ---------- Header ----------
st.title("💳 Credit Risk Decision Engine")
st.caption("Score loan applications, get decisions with reason codes, and estimate expected loss.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model summary")
    st.markdown(
        """
        **Model:** LightGBM + isotonic calibration  
        **Test AUC:** 0.766  
        **Calibration:** Brier 0.178 → 0.067  
        **Decision policy:** approve / review / reject
        """
    )
    st.divider()
    st.markdown("**Cost assumptions**")
    st.markdown(
        """
        - Loss Given Default (LGD): **65%**
        - Profit margin: **10%**
        - Review cost: **0.5%**
        - Exposure at Default (EAD): **$100,000**
        - Review capacity: **5%** of applicants
        """
    )
    st.divider()
    st.markdown("**API docs:** [localhost:8000/docs](http://localhost:8000/docs)")


# ---------- Load sample data ----------
@st.cache_data
def load_test_sample():
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    return test.sample(n=500, random_state=42).reset_index(drop=True)


try:
    sample_df = load_test_sample()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load test data: {e}")
    data_loaded = False


# ---------- Session state ----------
if "applicant" not in st.session_state:
    st.session_state.applicant = None
if "result" not in st.session_state:
    st.session_state.result = None


# ---------- Helpers ----------
def prepare_applicant_for_ui(row_dict: dict) -> dict:
    """Convert a raw test-set row into a dict ready for UI display."""
    a = dict(row_dict)

    # Age in years
    db = a.get('DAYS_BIRTH')
    if db is not None and not pd.isna(db):
        a['_age_years'] = int(round(-db / 365.25))
    else:
        a['_age_years'] = 40

    # Employment
    de = a.get('DAYS_EMPLOYED')
    if de is not None and not pd.isna(de):
        if de == 365243:
            a['_not_standard_employment'] = True
            a['_years_employed'] = 0.0
        else:
            a['_not_standard_employment'] = False
            a['_years_employed'] = round(-de / 365.25, 1)
    else:
        a['_not_standard_employment'] = False
        a['_years_employed'] = 0.0

    return a


def to_json_safe(d: dict) -> dict:
    """Remove internal keys, convert NaN/numpy to JSON-safe values."""
    out = {}
    for k, v in d.items():
        if k.startswith('_'):
            continue
        if v is None:
            out[k] = None
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif hasattr(v, 'item'):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ---------- Input section ----------
st.subheader("Step 1 — Choose an applicant")

col1, col2 = st.columns([3, 1])
with col1:
    if st.button(
        "🎲 Load random applicant from test set",
        use_container_width=True,
        disabled=not data_loaded,
    ):
        idx = np.random.randint(0, len(sample_df))
        row = sample_df.iloc[idx].drop(['TARGET', 'SK_ID_CURR']).to_dict()
        st.session_state.applicant = prepare_applicant_for_ui(row)
        st.session_state.result = None
with col2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.applicant = None
        st.session_state.result = None


if st.session_state.applicant is not None:
    st.divider()
    st.subheader("Step 2 — Review and edit applicant details")
    st.caption("Values are pre-filled from the random sample. Edit any field, then score.")

    a = st.session_state.applicant

    # ------- Financials -------
    with st.expander("💰 Financial details", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            a['AMT_INCOME_TOTAL'] = st.number_input(
                "Annual income",
                value=float(a.get('AMT_INCOME_TOTAL') or 150000),
                step=5000.0,
                help="Total annual income of the applicant.",
            )
        with f2:
            a['AMT_CREDIT'] = st.number_input(
                "Credit amount",
                value=float(a.get('AMT_CREDIT') or 500000),
                step=10000.0,
                help="Total loan amount requested.",
            )
        with f3:
            a['AMT_ANNUITY'] = st.number_input(
                "Monthly payment",
                value=float(a.get('AMT_ANNUITY') or 25000),
                step=1000.0,
                help="Monthly loan payment (annuity).",
            )
        st.caption("**Rule of thumb:** credit ≤ 3× income and payment ≤ 30% of monthly income.")

    # ------- Credit scores -------
    with st.expander("📊 External credit scores", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            v2 = a.get('EXT_SOURCE_2')
            v2 = 0.5 if v2 is None or pd.isna(v2) else float(v2)
            a['EXT_SOURCE_2'] = st.slider(
                "External credit score #2", 0.0, 1.0, v2,
                help="Higher is better. Typical range: 0.3 – 0.8.",
            )
        with c2:
            v3 = a.get('EXT_SOURCE_3')
            v3 = 0.4 if v3 is None or pd.isna(v3) else float(v3)
            a['EXT_SOURCE_3'] = st.slider(
                "External credit score #3", 0.0, 1.0, v3,
                help="Higher is better. Typical range: 0.2 – 0.7.",
            )
        st.caption(
            "**External credit scores are the single strongest predictor.** "
            "A high score can offset weakness in other fields."
        )

    # ------- Personal -------
    with st.expander("👤 Personal details", expanded=True):
        p1, p2 = st.columns(2)
        with p1:
            a['_age_years_ui'] = st.number_input(
                "Age (years)",
                min_value=18, max_value=80,
                value=int(a.get('_age_years') or 40),
                step=1,
                help="Applicant's age in years.",
            )
            a['DAYS_BIRTH'] = -int(a['_age_years_ui'] * 365.25)

        with p2:
            a['_not_standard_employment'] = st.checkbox(
                "Not in standard employment (pensioner, student, other)",
                value=bool(a.get('_not_standard_employment', False)),
                help="Check if the applicant is a pensioner, student, or otherwise not in standard employment.",
            )
            if a['_not_standard_employment']:
                a['DAYS_EMPLOYED'] = 365243
                st.caption("Using placeholder value for non-standard employment.")
            else:
                a['_years_employed_ui'] = st.number_input(
                    "Years employed",
                    min_value=0.0, max_value=60.0,
                    value=float(a.get('_years_employed') or 5.0),
                    step=0.5,
                    help="How many years the applicant has been employed.",
                )
                a['DAYS_EMPLOYED'] = -int(a['_years_employed_ui'] * 365.25)

        p3, p4 = st.columns(2)
        with p3:
            edu_options = [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ]
            current_edu = a.get('NAME_EDUCATION_TYPE')
            edu_idx = edu_options.index(current_edu) if current_edu in edu_options else 0
            a['NAME_EDUCATION_TYPE'] = st.selectbox(
                "Education level", edu_options, index=edu_idx,
                help="Higher education is associated with lower default risk.",
            )
        with p4:
            current_ct = a.get('NAME_CONTRACT_TYPE')
            ct_idx = 1 if current_ct == "Revolving loans" else 0
            a['NAME_CONTRACT_TYPE'] = st.selectbox(
                "Contract type", ["Cash loans", "Revolving loans"], index=ct_idx,
                help="Cash loans are riskier than revolving loans.",
            )

        st.caption("**Note:** Gender is deliberately excluded from the model for fair-lending compliance.")

    # ------- Score -------
    st.divider()
    if st.button("🚀 Score applicant", type="primary", use_container_width=True):
        payload = {"features": to_json_safe(a)}
        with st.spinner("Scoring..."):
            try:
                r = requests.post(API_URL, json=payload, timeout=15)
                if r.status_code == 200:
                    st.session_state.result = r.json()
                else:
                    try:
                        detail = r.json().get('detail', r.text)
                    except Exception:
                        detail = r.text
                    st.error(f"API error {r.status_code}: {detail}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")


# ---------- Results ----------
if st.session_state.result is not None:
    st.divider()
    st.subheader("Step 3 — Decision")

    r = st.session_state.result
    pd_val = r['default_probability']
    decision = r['decision']
    exp_loss = r['expected_loss']

    if decision == 'approve':
        st.success(f"### ✅ APPROVE  \nDefault probability: **{pd_val*100:.2f}%**")
    elif decision == 'review':
        st.warning(f"### 🟡 MANUAL REVIEW  \nDefault probability: **{pd_val*100:.2f}%**")
    else:
        st.error(f"### 🔴 REJECT  \nDefault probability: **{pd_val*100:.2f}%**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{pd_val*100:.2f}%")
    m2.metric(
        "Expected Loss", f"${exp_loss:,.0f}",
        help="Expected loss = PD × LGD × EAD, assuming $100K exposure.",
    )
    m3.metric("Decision", decision.upper())

    st.markdown("**Risk level**")
    st.progress(min(pd_val, 1.0), text=f"PD = {pd_val*100:.2f}%")
    st.caption(
        "**Thresholds:** PD < 10% → approve  |  10% – 20% → manual review  |  > 20% → reject"
    )

    st.markdown("### Top reason codes")
    st.caption(
        "SHAP-based contributions to the model's prediction. "
        "Impact is in log-odds units — positive values increase risk."
    )
    reasons = pd.DataFrame(r['reason_codes'])
    reasons['feature'] = reasons['feature'].apply(label)
    reasons['direction'] = reasons['direction'].map({
        'increases_risk': '⬆ increases risk',
        'decreases_risk': '⬇ decreases risk',
    })
    reasons['impact'] = reasons['impact'].round(4)
    reasons = reasons.rename(columns={
        'feature': 'Feature',
        'value': 'Applicant value',
        'direction': 'Effect',
        'impact': 'Impact',
    })
    st.dataframe(reasons, hide_index=True, use_container_width=True)


# ---------- Footer ----------
st.divider()
st.caption(
    "Built with FastAPI + Streamlit. "
    "Model: LightGBM + isotonic calibration. "
    "Data: Home Credit Default Risk (Kaggle). "
    "Full documentation: reports/model_card.md."
)