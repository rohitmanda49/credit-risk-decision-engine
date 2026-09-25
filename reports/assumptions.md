# Project Assumptions & Business Framing

## Problem Statement
Given a loan application, predict the probability of default and recommend
an action: approve, reject, or manual review. The goal is to minimize
expected credit losses while maintaining fair access to credit and
respecting operational review capacity.

## Dataset
- Source: Home Credit Default Risk (Kaggle)
- Rows: 307,511 loan applications
- Features: 122 application-time variables
- Target: TARGET — 1 = default, 0 = repaid
- Class balance: 91.93% repaid, 8.07% default

## Target Definition
- TARGET = 1 → client had payment difficulties (default)
- TARGET = 0 → all payments made on time (repaid)

## Business Cost Assumptions
| Parameter | Value | Rationale |
|---|---|---|
| Loss Given Default (LGD) | 0.65 | Industry-typical for unsecured consumer credit |
| Profit margin per good loan | 10% of principal | Reasonable net margin for consumer lending |
| Manual review capacity | Top 5% of applications | Operational constraint on review team |
| Cost of manual review | 0.5% of principal | Human + ops cost per reviewed application |

## Decision Policy
Three-way decision based on predicted probability of default (PD):
- Approve: PD below low threshold
- Reject: PD above high threshold
- Manual Review: PD in middle band (up to 5% of applications)

Thresholds will be set to maximize expected profit, not accuracy.

## Success Metrics
### Model
- AUC (discrimination)
- KS statistic
- Calibration (Brier score, reliability curve)

### Business
- Expected loss reduction vs rule-based baseline
- Approval rate at target loss level
- Precision @ 5% review capacity

### Fairness
- Approval rate parity across groups (gender, income band, age band)
- Equal opportunity (true positive rate parity)

## Validation Strategy
- Temporal split where possible — train on earlier loans, test on later.
- No post-origination features (no target leakage).

## Out of Scope (MVP)
- Fraud detection
- Real-time streaming
- Deep learning / GNNs
- Full MLOps (CI/CD, Kubernetes)

## Resume Target
Built end-to-end credit risk decision engine on 300K+ loan records;
LightGBM with calibrated probabilities and cost-sensitive thresholds
reduced expected loss vs rule-based baseline; deployed with SHAP
reason codes, fairness audit, and drift monitoring.

## Validation Strategy
- Approximate temporal split using SK_ID_CURR as a proxy for origination
  order (70/15/15 train/val/test).
- Train: 215,257 rows, default rate 8.13%
- Val:    46,127 rows, default rate 7.98%
- Test:   46,127 rows, default rate 7.92%
- Default rate drift of 0.21pp across splits confirms SK_ID_CURR is a
  reasonable temporal proxy (mild, realistic drift, not data leakage).
- Limitation: Home Credit dataset lacks an explicit origination date.
  SK_ID_CURR correlates with time but is not a clean timestamp.
  Documented in model card.
- No post-origination features (no target leakage).s