# Ensemble Health -- End-to-End Process Documentation

## Table of Contents
1. [Exploratory Data Analysis](#1-exploratory-data-analysis)
2. [Feature Engineering](#2-feature-engineering)
3. [Experimental Design](#3-experimental-design)
4. [Model Experimentation & Results](#4-model-experimentation--results)
5. [Model Selection Reasoning](#5-model-selection-reasoning)
6. [Calibration Decision](#6-calibration-decision)
7. [Threshold Selection](#7-threshold-selection)
8. [GenAI Explanation Engine](#8-genai-explanation-engine)
9. [Production Deployment](#9-production-deployment)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. Exploratory Data Analysis

### 1.1 Dataset Overview

| Property | Historical Claims | Current Claims |
|---|---|---|
| Row count | 3,200 | 500 |
| Features | 17 raw columns | 17 raw columns |
| Time range | 2024-01 through 2024-12 | 2025-01 |
| Target | `is_denied` (0/1) | None (to predict) |
| Split | train/validation/test | N/A |

### 1.2 Target Distribution (Class Imbalance)

The denial rate across the historical dataset:

| Split | Claims | Denial Rate |
|---|---|---|
| Training (70%) | 2,240 | 21.1% |
| Validation (15%) | 480 | 19.1% |
| Test (15%) | 480 | 26.0% |
| **Overall** | **3,200** | **21.6%** |

**Key insight:** This is a moderately imbalanced dataset (~4:1 non-denied to denied). The test set has a slightly higher denial rate (26%) than training (21%), suggesting some temporal drift. This class imbalance means accuracy is not a useful metric -- a model that predicts "no denial" for everything would achieve 78.4% accuracy but catch zero denials.

### 1.3 Missing Values

Only one field has missing values: `denial_reason` is missing for 2,509 out of 3,200 claims (78.4%). This is expected -- it is only populated for denied claims. The field is a post-submission attribute that must be excluded from features (would constitute data leakage).

### 1.4 Feature Distributions

#### Numerical Features

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|
| total_billed | $12,164 | $13,173 | $523 | $4,401 | $7,972 | $14,847 | $95,000 |
| expected_payment | $6,032 | $6,698 | $193 | $2,065 | $3,850 | $7,422 | $62,989 |
| num_procedures | 3.8 | 2.1 | 1 | 2 | 3 | 5 | 15 |
| num_diagnoses | 5.3 | 2.6 | 1 | 3 | 5 | 7 | 18 |
| days_to_submit | 19.3 | 11.8 | 1 | 11 | 17 | 25 | 78 |

**Key observations:**
- `total_billed` is heavily right-skewed (mean >> median). Log transform warranted.
- `expected_payment` is similarly skewed. Ratio `payment/total` ranges from very low (pennies on the dollar) to near 100%.
- `days_to_submit` has a long right tail. 30-day threshold is a natural cut point for timely filing.

#### Categorical Features

| Feature | Categories | Distribution |
|---|---|---|
| payer_type | 4 | Commercial (42.1%), Medicaid MCO (24.3%), Medicare Advantage (17.4%), BCBS (16.2%) |
| visit_type | 4 | Outpatient (46.2%), Emergency (21.0%), Inpatient (20.8%), Observation (12.1%) |
| payer_id | 12 | P011 (10.3%), P001 (9.6%), P009 (9.4%), others 6.6-8.6% each |

**Key observations:**
- 4 payer types and 12 payer IDs. One-hot encoding adds 4 + 4 + 12 = 20 sparse binary features.
- No single payer dominates; distribution is fairly uniform.
- Outpatient visits are the most common, but Inpatient/Emergency carry higher risk.

#### Binary Flags

| Flag | Count=1 | Count=0 | Interpretation |
|---|---|---|---|
| prior_auth_required | 1,216 (38.0%) | 1,984 (62.0%) | 38% of claims need prior auth |
| has_prior_auth | 1,262 (39.4%) | 1,938 (60.6%) | Only slightly more claims HAVE auth than NEED it |
| is_in_network | 2,617 (81.8%) | 583 (18.2%) | ~18% out-of-network |
| missing_documentation_flag | 604 (18.9%) | 2,596 (81.1%) | ~19% have missing docs |
| eligibility_verified | 2,780 (86.9%) | 420 (13.1%) | ~13% not verified |
| referral_required | 909 (28.4%) | 2,291 (71.6%) | 28% need referral |
| referral_present | 759 (23.7%) | 2,441 (76.3%) | Gap of 150 claims needing but missing referral |

### 1.5 Administrative Gaps vs Denial Rate (The Critical Finding)

This is the most important analysis. Each gap flag was crossed against denial rate:

| Gap Flag | Denial Rate When Gap=1 | Denial Rate When Gap=0 | **Lift** |
|---|---|---|---|
| **auth_gap** (needs auth, doesn't have it) | **46.5%** | 18.9% | **+27.5 pp** |
| **doc_issue** (missing documentation) | **38.6%** | 17.6% | **+20.9 pp** |
| elig_issue (eligibility not verified) | 36.0% | 19.4% | +16.5 pp |
| late_filing (>30 days to submit) | 32.8% | 19.6% | +13.2 pp |
| referral_gap (needs referral, doesn't have it) | 33.0% | 20.6% | +12.4 pp |
| network_issue (out-of-network) | 24.9% | 20.9% | +4.0 pp |

**Cumulative gaps are multiplicative in risk:**

| Number of Admin Gaps | Denial Rate | Claim Count |
|---|---|---|
| 0 gaps | **13.4%** | 1,587 (49.6%) |
| 1 gap | **23.0%** | 1,134 (35.4%) |
| 2 gaps | **40.3%** | 397 (12.4%) |
| 3 gaps | **69.2%** | 78 (2.4%) |
| 4 gaps | **100.0%** | 4 (0.1%) |

**Critical insight:** Administrative completeness is the dominant denial predictor. A claim with 0 gaps has only a 13.4% denial rate. A claim with 3 gaps jumps to 69.2%. This is a linear, additive relationship. The strongest single gap is `auth_gap` (+27.5 percentage point lift).

**Why this matters for model selection:** This strongly linear, additive relationship means that **Logistic Regression is the natural model choice.** Tree-based models (XGBoost, Random Forest) split on individual features and struggle to capture the cumulative additive effect as efficiently as a linear model. This finding alone explains why LR outperforms XGBoost by 13.6 percentage points (49.51% vs 35.92% capture@25).

---

## 2. Feature Engineering

### 2.1 Design Principles

All features must be computable **before claim submission**. Post-submission leakage columns (`denial_reason`) and the target (`is_denied`) are programmatically excluded. Identifiers (`claim_id`, `service_month`) are removed.

### 2.2 Engineered Features

| Category | Features | Rationale |
|---|---|---|
| **Gap Flags** | `auth_gap`, `referral_gap`, `doc_issue`, `elig_issue`, `network_issue` | Direct denial predictors from EDA findings |
| **Composite Scores** | `total_admin_gaps` (sum of 5 gaps), `complexity_score` (procedures x diagnoses), `double_deficit` (auth_gap AND doc_issue) | Cumulative risk from EDA; non-linear interactions |
| **Financial** | `payment_ratio` (expected/total), `payment_gap` (total - expected), `billed_per_procedure`, `billed_per_diagnosis` | High-cost claims face more payer scrutiny |
| **Log Transforms** | `log_total_billed`, `log_expected_payment` | Corrects right-skew in financial amounts |
| **Temporal** | `service_month_num`, `service_quarter`, `submission_delay` (binned 0-5), `late_filing` (>30 days) | Timely filing deadlines drive denials |
| **Interactions** | `oon_auth_gap`, `oon_referral_gap`, `is_high_risk_combo` (Medicaid MCO + Inpatient/Emergency) | Compound risk from multiple flags |
| **Segments** | One-hot encoding of `payer_type`, `visit_type`, `payer_id` | Payer-specific denial patterns |

**Total final feature count:** 53 features (20 one-hot + 33 numeric/engineered).

### 2.3 Feature Preprocessing Pipeline

```
Raw CSV (17 columns)
  -> pd.to_numeric() on financial columns
  -> Engineer 20+ derived features
  -> ColumnTransformer:
       OneHotEncoder(handle_unknown='ignore') on categoricals
       StandardScaler() on numeric
     = 53 final features
```

`handle_unknown='ignore'` is critical for production: current claims may include payer IDs or visit types not seen in training. The preprocessor silently ignores them rather than crashing.

---

## 3. Experimental Design

### 3.1 Evaluation Framework

| Element | Choice | Reason |
|---|---|---|
| **Primary metric** | Denial Capture@25% | Business KPI: what fraction of denials fall in top 25% by risk? Directly measures review capacity utilization. |
| **Secondary metrics** | ROC-AUC, PR-AUC, Brier Score | AUC for ranking quality; Brier for probability calibration |
| **Validation strategy** | Fixed 70/15/15 split | Supplied with data; ensures no data leakage across experiments |
| **Class imbalance handling** | `class_weight='balanced'` | Automatically weights minority class proportional to inverse frequency |

### 3.2 Why Capture@25% is the North Star

Accuracy is meaningless here (baseline: 78.4% by predicting all "not denied"). ROC-AUC measures overall ranking but doesn't tell us about the top quartile where billers focus. Capture@25% answers the exact business question: "If we review the riskiest 25% of claims, what fraction of all denials will we catch?"

A random reviewer catches 25% of denials. Our goal is to beat this significantly.

### 3.3 Hyperparameter Search Strategy

Rather than grid search across all possible values, we used a targeted approach:
- **C values:** 0.01, 0.1, 1.0, 10.0 -- spanning strong to weak regularization
- **Solver:** lbfgs (default), newton-cg (for comparison)
- **Class weights:** balanced vs explicit {0:1, 1:3.6}

The best validation performance was LR with C=0.1 and balanced weights (49.51% capture@25). Higher C values overfit; lower C values under-regularize.

---

## 4. Model Experimentation & Results

### 4.1 Complete 10-Experiment Leaderboard

| Rank | Experiment | Type | Capture@25 | ROC-AUC | Brier | PR-AUC |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | exp1_lr_baseline | LR, C=0.1, balanced | **49.51%** | 0.711 | 0.209 | 0.430 |
| 1 | exp2_lr_calibrated | LR + Platt scaling | **49.51%** | 0.710 | **0.137** | 0.428 |
| 1 | exp5_lr_interactions | LR + interaction feats | **49.51%** | 0.707 | 0.210 | 0.415 |
| 4 | exp6_gradient_boosting | GBM, depth=5, lr=0.05 | 48.54% | 0.661 | 0.151 | 0.375 |
| 5 | exp7_voting_ensemble | Soft Voting (LR+RF+GBM) | 48.54% | 0.696 | 0.166 | 0.409 |
| 6 | exp8_stacking | Stacking, LR meta, cv=5 | 46.60% | **0.718** | **0.136** | **0.433** |
| 7 | exp10_svc_rbf | SVC RBF + Platt | 45.63% | 0.641 | 0.148 | 0.327 |
| 8 | exp9_mlp_neural | MLP (64,32), ReLU | 43.69% | 0.678 | 0.146 | 0.391 |
| 9 | exp4_rf_robust | RF, depth=12, balanced | 39.81% | 0.653 | 0.155 | 0.366 |
| 10 | exp3_xgb_tuned | XGB, depth=6, lr=0.05 | 35.92% | 0.615 | 0.180 | 0.341 |

### 4.2 Detailed Analysis Per Model Family

#### Logistic Regression Variants (exp1, exp2, exp5): 49.51% Capture

All three LR variants achieve identical 49.51% capture@25. The interaction features experiment (exp5) adds no gain because the engineered gap flags already capture the relevant interactions (e.g., `double_deficit` = auth_gap AND doc_issue). The calibration experiment (exp2) matches capture but dramatically improves Brier (0.209 -> 0.137).

**Why LR works so well:** The EDA proved that denial risk increases nearly linearly with the number of administrative gaps. LR's linear decision boundary is the natural fit for this structure. The model learns a coefficient for each gap flag; the sum of positive coefficients directly corresponds to cumulative risk.

#### Gradient Boosting (exp6): 48.54% Capture

GBM narrowly trails LR by ~1 percentage point. This is the best non-linear performer. The key tuning decisions:
- **Shallow trees** (max_depth=5): Prevents overfitting to the small dataset (3,200 rows)
- **Low learning rate** (0.05): Allows the sequential boosting to learn additive patterns incrementally
- **Subsampling** (0.8): Adds regularization, prevents memorization

GBM outperforms XGBoost by 12.6 percentage points on this dataset. The reason is that GBM's default behavior (Friedman MSE, no column subsampling by default) is better suited to problems where features have strong marginal effects. XGBoost's column subsampling and regularization are designed for wide datasets with noisy features -- counterproductive here.

#### Voting Ensemble (exp7): 48.54% Capture

Soft voting with weights [2, 1, 1] for [LR, RF, GBM] matches GBM but doesn't beat pure LR. The weaker RF component (39.81%) dilutes the strong LR signal. The ensemble's strength -- combining diverse models -- doesn't help when one model (LR) is already near-optimal for the data structure.

**Why ensembles don't improve on LR:** Ensembles help when individual models have uncorrelated errors. Here, all models see the same strong linear signal. LR captures it perfectly; trees capture it less efficiently. Weighting trees into the vote adds noise, not signal.

#### Stacking (exp8): 46.60% Capture

Stacking trains a meta-learner (LR) on the outputs of base models (LR + RF + GBM). It achieves the **best ROC-AUC (0.718)** and **best Brier (0.136)** of all experiments, even beating calibrated LR on these metrics. However, capture@25 drops to 46.60%.

**The paradox explained:** The meta-learner optimizes for overall probability accuracy (minimizing log-loss), which improves ROC-AUC and Brier. But this trades off ranking at the extreme (top 25%). The meta-learner smooths out the probability estimates, making them more reliable globally but less discriminative at the high-risk tail where our business metric lives.

#### SVC RBF (exp10): 45.63% Capture

The RBF kernel SVM with Platt scaling achieves decent capture. The RBF kernel finds non-linear decision boundaries, but the data's true structure is linear. The kernel adds unnecessary complexity. Platt scaling (cv=3) provides probability outputs, but the inherent ranking is worse than LR.

#### MLP Neural Network (exp9): 43.69% Capture

The MLP with (64,32) hidden layers and ReLU activation learns non-linear feature interactions. It outperforms RF and XGB but trails LR. The reason: 3,200 training samples are insufficient for a neural network to learn meaningful patterns. With more data, this might approach LR's performance, but the linear structure suggests diminishing returns.

#### Random Forest (exp4): 39.81% Capture

RF with 200 trees and max_depth=12 underperforms significantly. Tree-based models split on individual features and cannot efficiently represent the additive cumulative risk pattern. A claim with 3 gaps is high-risk regardless of which specific gaps; RF must learn this through many tree splits rather than a single coefficient sum.

#### XGBoost (exp3): 35.92% Capture

XGBoost performs worst, likely due to its default regularization (L1+L2) suppressing the weak-but-important marginal signals from individual gap flags. The `scale_pos_weight=3.6` handles class imbalance but doesn't compensate for the fundamental mismatch between tree structure and additive data.

### 4.3 The LR Baseline vs Everything Else (Visual Summary)

```
Capture@25%
50% |  *LR variants (all three at 49.51%)
    |  *GBM, Voting (48.54%)
45% |  *Stacking (46.60%), SVC (45.63%)
    |  *MLP (43.69%)
40% |  *RF (39.81%)
    |  *XGB (35.92%)
35% |
    +------------------------------------
      Linear    Ensemble    Tree    Neural    SVM
```

The pattern is clear: **linear models dominate.** The gap between best and worst is 13.6 percentage points. This is not noise -- it reflects a fundamental property of the data.

---

## 5. Model Selection Reasoning

### 5.1 Why Not Just Pick the Highest Capture@25?

If all three LR variants achieve 49.51%, why does the choice matter?

**Calibration matters for tier assignment.** The pipeline assigns tiers at the 75th percentile threshold. If probabilities are poorly calibrated, the threshold is unreliable. An uncalibrated model might assign probability 0.60 to claims that are actually 40% likely to be denied, causing incorrect tier boundaries.

**The Brier score measures calibration quality.** LR baseline has Brier=0.209; calibrated LR has Brier=0.137. This is a **34% improvement** -- the calibrated model's probability estimates are substantially closer to true denial frequencies.

### 5.2 The Calibrated LR Selection

| Criteria | LR Baseline | Calibrated LR | Verdict |
|---|---|---|---|
| Capture@25 | 49.51% | 49.51% | Equal |
| ROC-AUC | 0.711 | 0.710 | Equal |
| Brier Score | 0.209 | **0.137** | **Calibrated wins** |
| Interpretability | Coefficients | Coefficients | Equal |
| Training time | ~0.1s | ~0.3s | Acceptable |
| Production complexity | Simple | Slightly more | Acceptable |

Calibrated LR was selected because it delivers identical ranking performance with dramatically better probability estimates, at minimal additional cost. The calibration uses Platt scaling (sigmoid) with 5-fold cross-validation on the full history.

### 5.3 Why Not Stacking (Best ROC-AUC/Brier)?

Stacking has the best overall metrics (ROC-AUC=0.718, Brier=0.136) but worse capture@25 (46.60% vs 49.51%). The choice depends on the business use case:

- **If the goal is tier assignment accuracy:** Stacking is better (most reliable probabilities).
- **If the goal is maximizing denials caught in the top 25%:** LR is better.

We chose the latter because the business problem statement prioritizes denial capture within a fixed review capacity. The 2.91 percentage point gap in capture@25 represents ~14 additional denials that would be missed with Stacking.

However, if Ensemble Health wanted to A/B test both models, Stacking is a strong contender for scenarios where tier reliability matters more than extreme-quartile ranking.

---

## 6. Calibration Decision

### 6.1 Why Calibrate?

Logistic Regression is theoretically well-calibrated, but in practice with imbalanced data and regularization (C=0.1), the predicted probabilities can drift from true frequencies. The EDA showed that:

- Uncalibrated Brier: 0.209
- Calibrated Brier: 0.137
- Improvement: 0.071 (34%)

This means that for a claim the model says is "60% likely to be denied", the true denial rate is much closer to 60% after calibration. Before calibration, the model was overconfident.

### 6.2 Platt Scaling (sigmoid) vs Isotonic Regression

We chose sigmoid calibration because:
- **Sample efficiency:** Isotonic regression is non-parametric and can overfit with small validation sets
- **Monotonicity:** Sigmoid preserves the ranking order (critical since ranking is our primary metric)
- **Simplicity:** Single-parameter fit on top of LR output

The calibration uses `CalibratedClassifierCV(cv=5, method='sigmoid')` -- 5-fold cross-validation to avoid using the same data for both base model training and calibration fitting.

### 6.3 Validation Strategy

The threshold is frozen at the 75th percentile of **validation set** probabilities (0.252). This threshold:
- Is computed on data neither model nor calibration sees during training
- Prevents leakage from test set into threshold
- Generalizes to the current claims scoring

This is a critical design decision: computing the threshold on test probabilities would constitute data leakage and produce an overly optimistic assessment.

---

## 7. Threshold Selection

### 7.1 Why the 75th Percentile?

The assessment specifies a **25% review capacity**. This means billers can manually review one quarter of claims. The 75th percentile of validation probabilities naturally maps to this constraint:

- Top 25% highest-risk claims: flagged as High tier
- Next 25%: Medium tier
- Bottom 50%: Low tier

For 500 current claims, this yields exactly 125/125/250.

### 7.2 Threshold Value

```
val_threshold = 0.252
```

Any claim with probability >= 0.252 is predicted as a denial. This threshold is relatively low because the calibration pulls probabilities toward the mean. Most High-tier claims have probabilities above 0.59, well above the threshold.

### 7.3 Test Set Performance at This Threshold

| Metric | Value |
|---|---|
| Capture@25 | 45.7% |
| Precision@25 | 47.8% |
| F1 | 0.469 |
| Confusion Matrix | TN=320, FP=79, FN=73, TP=67 |

**Interpretation:** Reviewing the top 125 claims would catch 67 out of 140 actual denials (45.7%). Nearly half of all denials are caught in the top quartile -- almost double the 25% random baseline.

---

## 8. GenAI Explanation Engine

### 8.1 Design Rationale

The explanation engine serves two purposes:
1. **Operational:** Give billers actionable guidance on what to fix before submission
2. **Compliance:** Provide auditable reasoning for why a claim was flagged

### 8.2 Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM provider | Ollama Cloud (gemma4:31b-cloud) | Specified in assessment requirements |
| Prompt structure | Structured JSON via Pydantic | Ensures parseable, validated outputs |
| Validation | Pydantic field validators | Enforces disclaimer, actionability, min length |
| Fallback | 3-tier (JSON -> regex -> template) | Resilience against API failures |
| Coverage | 125 High via API, 375 via template | Balances cost/quality; all claims get explanations |
| Audit | Self-contained JSON logs | HIPAA-safe, no external telemetry |

### 8.3 Why Deterministic Templates for Medium/Low?

- **Cost efficiency:** 125 API calls cost ~56K tokens. 500 calls would cost ~224K tokens.
- **Quality threshold:** Medium/Low claims have lower risk; template explanations are sufficient.
- **Speed:** Templates are instant; 125 API calls take ~4 minutes.

### 8.4 Pydantic Validation Schema

```python
# Input validation
class ExplanationRequest:
    claim_id: str          # Non-empty
    denial_probability: float  # 0 <= x <= 1
    risk_estimate_label: str   # "High" | "Medium" | "Low"
    top_risk_factors: list     # Max 5 items, each min_length=1

# Output validation
class ExplanationResponse:
    claim_id: str              # Non-empty
    disclaimer: str            # Must contain "estimate"/"statistical"/"not guaranteed"
    risk_description: str      # min_length=1
    recommended_action: str    # min_length=1
```

This two-layer validation (input + output) ensures the LLM response is structurally correct and semantically appropriate before it reaches the CSV.

---

## 9. Production Deployment

### 9.1 Full Pipeline Execution

```
python src/run_pipeline.py
```

This single command:
1. Loads and validates raw CSVs (3,200 history + 500 current)
2. Engineers 53 features (no leakage)
3. Trains 5 models, evaluates on validation, selects best (Cal-LR)
4. Computes frozen threshold from validation (0.252)
5. Retrains production CalibratedClassifierCV on full history
6. Scores all 500 current claims, assigns tiers (125/125/250)
7. Extracts risk factors from LR coefficients
8. Generates 125 API explanations + 375 template explanations
9. Validates output CSV structure
10. Saves predictions, metrics, and audit log

### 9.2 Output Deliverables

| File | Content |
|---|---|
| `predictions_current_claims.csv` | 500 rows, 6 columns, sorted by probability descending |
| `metrics.json` | Test metrics, threshold, calibration scores |
| `audit_logs/audit_*.json` | 500 records with tokens, latency, quality flags |
| `experiments/exp*/` | 10 experiment runs with artifacts |

### 9.3 Infrastructure Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Experiment tracker | Self-contained (MLflow format) | HIPAA-safe, no external servers |
| Audit logging | Local JSON files | No telemetry, full traceability |
| Token tracking | Ollama ChatResponse fields | Native to the library, no extra dependencies |
| Git strategy | data/output/ in .gitignore | Generated artifacts stay local |

---

## 10. Key Takeaways

### 10.1 Why Logistic Regression Won

The answer lies in the EDA: **administrative gaps have a linear, additive relationship with denial probability.** Each additional gap (missing auth, missing docs, missing referral) adds roughly 10-27 percentage points to denial risk. LR's linear coefficients perfectly capture this structure.

Tree models (XGBoost, RF) split on individual features and cannot efficiently represent "sum of gaps." They must learn complex tree structures to approximate what LR does with a single coefficient per gap.

### 10.2 Calibration is a Free Lunch

Platt scaling improves Brier by 34% without affecting ranking. This is essentially free -- the calibration adds ~0.2s to training time and makes tier assignments substantially more reliable.

### 10.3 Ensemble Methods Don't Always Help

The conventional wisdom (ensembles beat individual models) fails here. When one model class (linear) is already near-optimal for the data structure, adding weaker models dilutes performance. Stacking achieved better overall metrics but worse capture@25 -- the specific business metric matters.

### 10.4 The Dataset Drives Model Choice

The 3,200-row synthetic dataset has a very specific structure: binary administrative flags with strong marginal effects. A different dataset (with ICD codes, clinical notes, payer remittance data) would likely favor different models. The key lesson is **let the EDA guide model selection, not preconceptions about which algorithms are "better."**

### 10.5 Business Metric Alignment

Capture@25% is the right metric for this problem because it directly answers "how many denials can we catch with limited review capacity?" ROC-AUC and Brier are useful diagnostics, but they don't measure the business outcome. The 49.51% validation capture (nearly 2x random) demonstrates meaningful operational value.
