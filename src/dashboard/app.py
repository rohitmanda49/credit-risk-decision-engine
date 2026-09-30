"""Streamlit dashboard for the Credit Risk Decision Engine."""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

# ---------- Configuration ----------
API_URL = "http://localhost:8000/predict"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

st.set_page_config(
    page_title="Credit Risk Decision Engine",
    page_icon="💳",
    layout="wide"
)

# ---------- Header ----------
st.title("💳 Credit Risk Decision Engine")
st.caption("Score loan applications, get decisions with reason codes, and estimate expected loss.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Model:** LightGBM + isotonic calibration
        **Test AUC:** 0.766
        **Calibration:** Brier 0.178 → 0.067
        **Decision policy:** approve / review / reject
        **Cost assumptions:** LGD=0.65, EAD=$100K, review capacity=5%
        """
    )
    st.markdown("---")
    st.markdown("**Endpoints:**")
    st.code("POST /predict\nGET /health")
    st.markdown("API docs: [localhost:8000/docs](http://localhost:8000/docs)")

# ---------- Load sample data ----------
@st.cache_data
def load_test_sample():
    """Load a small cached sample of the test set for random selection."""
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

# ---------- Input section ----------
st.subheader("1. Choose an applicant")

col1, col2 = st.columns(2)
with col1:
    if st.button("🎲 Load random applicant from test set", use_container_width=True) and data_loaded:
        idx = np.random.randint(0, len(sample_df))
        st.session_state.applicant = sample_df.iloc[idx].drop(['TARGET', 'SK_ID_CURR']).to_dict()
        st.session_state.result = None
with col2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.applicant = None
        st.session_state.result = None

# Editable fields for the key business-relevant features
if st.session_state.applicant is not None:
    st.markdown("### Applicant details")
    st.caption("You can edit the values below, then click **Score applicant**.")

    a = st.session_state.applicant

    c1, c2, c3 = st.columns(3)
    with c1:
        a['AMT_INCOME_TOTAL'] = st.number_input("Income (total)", value=float(a.get('AMT_INCOME_TOTAL') or 150000), step=5000.0)
        a['AMT_CREDIT'] = st.number_input("Credit amount", value=float(a.get('AMT_CREDIT') or 500000), step=10000.0)
        a['AMT_ANNUITY'] = st.number_input("Annuity", value=float(a.get('AMT_ANNUITY') or 25000), step=1000.0)
    with c2:
        a['DAYS_BIRTH'] = st.number_input("Days since birth (negative)", value=int(a.get('DAYS_BIRTH') or -14000), step=-100)
        a['DAYS_EMPLOYED'] = st.number_input("Days employed (negative)", value=int(a.get('DAYS_EMPLOYED') or -2000), step=-100)
        a['EXT_SOURCE_2'] = st.slider("EXT_SOURCE_2", 0.0, 1.0, float(a.get('EXT_SOURCE_2') or 0.5))
    with c3:
        a['EXT_SOURCE_3'] = st.slider("EXT_SOURCE_3", 0.0, 1.0, float(a.get('EXT_SOURCE_3') or 0.4))
        a['NAME_EDUCATION_TYPE'] = st.selectbox(
            "Education",
            ["Higher education", "Secondary / secondary special", "Incomplete higher",
             "Lower secondary", "Academic degree"],
            index=0
        )
        a['CODE_GENDER'] = st.selectbox("Gender (for fairness audit only)", ["F", "M"], index=0)

    # Convert NaN and pandas types to JSON-safe
    def to_json_safe(d):
        out = {}
        for k, v in d.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[k] = None
            elif hasattr(v, 'item'):
                out[k] = v.item()
            else:
                out[k] = v
        return out

    if st.button("🚀 Score applicant", type="primary", use_container_width=True):
        payload = {"features": to_json_safe(a)}
        with st.spinner("Scoring..."):
            try:
                r = requests.post(API_URL, json=payload, timeout=10)
                if r.status_code == 200:
                    st.session_state.result = r.json()
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")

# ---------- Results section ----------
if st.session_state.result is not None:
    st.markdown("---")
    st.subheader("2. Decision")

    r = st.session_state.result

    # Top row: decision + PD + expected loss
    d1, d2, d3 = st.columns(3)

    decision_color = {"approve": "🟢", "review": "🟡", "reject": "🔴"}
    d1.metric("Decision", f"{decision_color.get(r['decision'],'')} {r['decision'].upper()}")
    d2.metric("Default Probability", f"{r['default_probability']*100:.2f}%")
    d3.metric("Expected Loss", f"${r['expected_loss']:,.0f}")

    # Reason codes
    st.markdown("### Top reason codes")
    reasons = pd.DataFrame(r['reason_codes'])
    reasons['direction'] = reasons['direction'].map({
        'increases_risk': '⬆ increases risk',
        'decreases_risk': '⬇ decreases risk'
    })
    reasons['impact'] = reasons['impact'].round(4)
    st.dataframe(
        reasons[['feature', 'value', 'direction', 'impact']],
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Reason codes are SHAP-based contributions to the model's prediction. "
        "Impact is in log-odds units."
    )

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with FastAPI + Streamlit. Model: LightGBM + isotonic calibration. Data: Home Credit (Kaggle).")