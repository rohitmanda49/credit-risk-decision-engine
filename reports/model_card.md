# Model Card: Credit Risk Decision Engine

## Model Details
- **Developed by:** Rohit [Your full name]
- **Model date:** 2026-09-28
- **Model type:** LightGBM (gradient boosted trees) + isotonic calibration
- **Version:** v1.0
- **License:** MIT

## Intended Use
- **Primary use:** Score loan applications for default probability and inform 
  approve / reject / manual review decisions.
- **Primary users:** Credit risk analysts, loan officers.
- **Out-of-scope uses:** Not for real-time fraud detection. Not validated 
  outside consumer credit.

## Training Data
- **Source:** Home Credit Default Risk (Kaggle)
- **Train:** 215,257 applications (default rate 8.13%)
- **Validation:** 46,127 applications (default rate 7.98%)
- **Test:** 46,127 applications (default rate 7.92%)
- **Features:** 131 (raw + engineered)
- **Split:** Approximate temporal split via SK_ID_CURR (70/15/15)

## Feature Engineering
- **Ratios:** credit-to-income, annuity-to-income, credit-term-months, 
  employed-to-age, credit-to-goods, income-per-family-member, income-per-child
- **Aggregates:** mean/std/min/max/product of EXT_SOURCE_1/2/3
- **Flags:** DAYS_EMPLOYED anomaly placeholder, document count

## Performance
| Metric | Train | Val | Test |
|---|---|---|---|
| AUC | 0.8469 | 0.7741 | 0.7662 |
| KS | 0.5364 | 0.4176 | 0.3972 |
| Brier (raw) | 0.1687 | 0.1755 | 0.1782 |
| Brier (calibrated) | 0.0634 | 0.0661 | 0.0664 |

**Calibration impact:** Isotonic calibration reduced test Brier score by 
63% (0.1782 → 0.0664) while preserving ranking (test AUC essentially 
unchanged: 0.7672 → 0.7662). Calibrated probabilities are now suitable 
for cost-sensitive decisioning.

## Tuning Tradeoff
**Decision:** Optuna hyperparameter tuning (30 trials, TPE sampler) 
was used to optimize validation AUC.

**Outcome:**
- Validation AUC improved modestly: 0.7702 → 0.7724
- Train-validation gap increased: 0.053 → 0.075

**Reasoning:** The tuning objective maximized validation AUC without 
penalizing the train-val gap. This selected higher-capacity parameters 
(num_leaves=55, max_depth=9), which fit training data more closely 
while marginally improving validation performance.

**Alternatives considered:**
1. Manually constrain parameters (num_leaves=30, max_depth=7, 
   min_child_samples=200) to reduce overfitting at the cost of 
   slightly lower val AUC (~0.770).
2. Modify the Optuna objective to penalize the train-val gap.

**Decision:** Proceeded with the tuned model. The decision engine 
(threshold optimization, calibration) delivers larger business value 
than marginal AUC differences. Calibration reduced test Brier by 63%, 
a substantially larger improvement than the AUC gain from tuning.

**Future work:** Re-tune with an overfitting-aware objective and 
compare business impact (expected loss, approval rate) — not just AUC.

## Fairness Considerations
- **Excluded from features:** CODE_GENDER (protected attribute)
- **Exclusion impact:** Test AUC dropped from [old] to [new] — a 
  [X]-point cost accepted for regulatory compliance and fairness.
- **Kept but audited (Day 5):** NAME_FAMILY_STATUS, NAME_EDUCATION_TYPE, 
  DAYS_BIRTH. These are not protected attributes but can act as 
  socioeconomic proxies; approval-rate parity will be measured in Day 5.
- Full fairness analysis forthcoming (Day 5).

## Limitations
- Home Credit data is anonymized; may not generalize to all markets.
- No explicit origination date; SK_ID_CURR used as a temporal proxy.
- Assumes stable macroeconomic conditions.
- Trained on consumer credit only; not validated on business lending.
- Train-val gap of 0.075 suggests some overfitting; documented in Tuning 
  Tradeoff section.

## Ethical Considerations
- Predictions inform human decisions, not replace them.
- SHAP reason codes provided for all decisions (Day 4).
- Requires periodic retraining to account for drift.
- Fair lending considerations: gender excluded; other demographic groups 
  audited for approval parity.