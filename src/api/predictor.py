"""Prediction and decision logic for the credit risk API."""
import numpy as np
import pandas as pd
import shap
from typing import List

from src.features.build_features import build_features
from src.api.schemas import ReasonCode


class Predictor:
    """Wraps the model, calibrator, and decision policy for serving."""

    def __init__(self, model_bundle):
        self.bundle = model_bundle
        self.explainer = shap.TreeExplainer(model_bundle.model)

    def predict(self, raw_features: dict) -> dict:
        """Score a single applicant and return decision + explanations."""

        # 1. Wrap raw features in a single-row DataFrame
        df = pd.DataFrame([raw_features])

        # 2. Ensure all training features exist. Missing columns become NaN.
        df = df.reindex(
            columns=self.bundle.metadata['feature_cols'],
            fill_value=np.nan
        )

        # 3. Build engineered features (safe now — all raw columns exist)
        df = build_features(df)

        # 4. Enforce exact column order after build_features
        X = df.reindex(columns=self.bundle.metadata['feature_cols'])

        # 5a. Coerce non-categorical columns to numeric (fixes 'object' dtype from JSON nulls)
        num_cols = [c for c in X.columns if c not in self.bundle.metadata['categorical_cols']]
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors='coerce')

        # 5b. Convert categorical columns to 'category' dtype for LightGBM
        for col in self.bundle.metadata['categorical_cols']:
            if col in X.columns:
                X[col] = X[col].astype('category')

        # 6. Predict raw probability, then calibrate
        pd_raw = float(self.bundle.model.predict(X)[0])
        pd_cal = float(self.bundle.calibrator.predict([pd_raw])[0])

        # 7. Apply decision policy
        decision = self._decide(pd_cal)

        # 8. Compute expected loss (PD × LGD × EAD)
        costs = self.bundle.decision_policy['business_costs']
        expected_loss = pd_cal * costs['LGD'] * costs['EAD']

        # 9. Compute SHAP reason codes
        reason_codes = self._explain(X, top_k=3)

        return {
            'default_probability': pd_cal,
            'decision': decision,
            'expected_loss': float(expected_loss),
            'reason_codes': reason_codes,
            'model_version': 'v1.0'
        }

    def _decide(self, pd_cal: float) -> str:
        """Apply the three-way decision policy."""
        low = self.bundle.decision_policy['low_threshold']
        high = self.bundle.decision_policy['high_threshold']
        if pd_cal < low:
            return 'approve'
        elif pd_cal > high:
            return 'reject'
        else:
            return 'review'

    def _explain(self, X: pd.DataFrame, top_k: int = 3) -> List[ReasonCode]:
        """Generate top-K SHAP reason codes for a single applicant."""
        shap_values = self.explainer.shap_values(X)

        # For binary classifiers, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # positive class, single row
        else:
            sv = shap_values[0]

        # Top-K by absolute impact
        top_idx = np.argsort(np.abs(sv))[::-1][:top_k]

        codes = []
        for i in top_idx:
            feature = X.columns[i]
            value = X.iloc[0, i]

            # Format value safely
            if pd.isna(value):
                value_str = None
            else:
                value_str = str(value)[:40]

            codes.append(ReasonCode(
                feature=feature,
                value=value_str,
                impact=float(sv[i]),
                direction='increases_risk' if sv[i] > 0 else 'decreases_risk'
            ))

        return codes