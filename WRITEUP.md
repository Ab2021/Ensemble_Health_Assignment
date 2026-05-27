# Ensemble Health Partners -- AI Team Hiring Assessment: Complete Write-Up

## 1. Problem Understanding

Ensemble Health Partners processes millions of healthcare claims annually. Pre-bill denial prediction -- identifying claims likely to be denied before submission -- is a high-value operational capability. Catching denials before they occur reduces rework costs, accelerates revenue cycles, and improves provider-payer relationships.

This assessment requires building an end-to-end system that:
1. Predicts denial probability for 500 current claims using historical data (3,200 claims)
2. Ranks claims by risk for a 25% review capacity
3. Generates plain-English explanations for the highest-risk claims using GenAI
4. Operates in a production-safe manner (no data leakage, auditable, reproducible)

---

## 2. Technical Approach

### 2.1 Architecture Overview

The system is structured as a modular Python pipeline:

```
Raw Data (CSV)
  -> Data Validation (contract assertions)
  -> Feature Engineering (53 pre-submission features, no leakage)
  -> Model Training (10 experiments, calibrated LR selected)
  -> Risk Tier Assignment (High/Medium/Low at 75th percentile threshold)
  -> Explanation Engine (LR coefficient attribution + Ollama gemma4:31b-cloud)
  -> Output (predictions CSV, metrics JSON, audit logs)
```

### 2.2 Why Logistic Regression?

After running 10 experiments across 5 model families, Logistic Regression variants consistently achieved the highest denial capture at 49.51% within the top 25% review window. The reasons:

- **Strong linear signal:** Administrative gap flags (missing prior auth, missing documentation, missing referral) are binary features with monotonic relationships to denials
- **Sample efficiency:** 3,200 claims is adequate for linear models but small for deep neural networks
- **Interpretability:** LR coefficients map directly to business-relevant risk factors, enabling the GenAI explanation engine
- **Calibration:** Platt scaling (sigmoid calibration) improves probability reliability by 34% without sacrificing ranking performance

### 2.3 Feature Engineering Strategy

All features are computable before claim submission (no leakage). Key engineered features:

| Category | Features | Rationale |
|---|---|---|
| Administrative gaps | auth_gap, referral_gap, doc_issue, elig_issue, network_issue | Direct denial predictors |
| Composite scores | total_admin_gaps, complexity_score, double_deficit | Combine multiple risk signals |
| Financial ratios | payment_ratio, payment_gap, billed_per_procedure | High-cost claims face more scrutiny |
| Temporal | late_filing, submission_delay (binned), service_quarter | Timely filing is a common denial reason |
| Risk segments | oon_auth_gap, is_high_risk_combo | Out-of-network + missing auth = compound risk |

---

## 3. Experiment Results & Model Selection

### 3.1 10-Experiment Leaderboard

| Rank | Model | Capture@25 | ROC-AUC | Notes |
|:---:|:---|:---:|:---:|:---|
| 1 | LR Baseline | 49.51% | 0.711 | Highest capture, simplest model |
| 1 | Calibrated LR | 49.51% | 0.710 | Best overall (Brier 0.137) |
| 1 | LR + Interactions | 49.51% | 0.707 | No gain from extra features |
| 4 | Gradient Boosting | 48.54% | 0.708 | Best non-linear alternative |
| 5 | Voting Ensemble | 48.54% | 0.705 | LR+RF+GBM soft voting |
| 6 | Stacking | 46.60% | 0.698 | Meta-learner doesn't help |
| 7 | SVC RBF | 45.63% | 0.688 | RBF kernel + Platt scaling |
| 8 | MLP Neural Net | 43.69% | 0.672 | (64,32) hidden, ReLU |
| 9 | Random Forest | 39.81% | 0.653 | Misses additive signal |
| 10 | XGBoost | 35.92% | 0.615 | Overfits sparse data |

### 3.2 Why Calibrated LR for Production?

The uncalibrated LR baseline achieves 49.51% capture@25 but has a Brier score of 0.209, indicating poorly calibrated probabilities. This matters operationally: if probability estimates are unreliable, the High/Medium/Low tier assignments will misclassify claims.

Calibrated LR (Platt scaling, cv=5) achieves:
- **Same capture@25: 49.51%** (no ranking degradation)
- **34% better Brier: 0.137** (much more reliable probabilities)
- **More stable tier assignments** (claims near the threshold are better distinguished)

The trade-off is ~20% additional training time, which is negligible for a 3,200-claim dataset.

### 3.3 Test Set Performance

Using a frozen validation threshold (75th percentile = 0.252):

| Metric | Value |
|---|---|
| Denial Capture @ 25% | 45.7% |
| Precision @ 25% | 47.8% |
| ROC-AUC | 0.691 |
| Brier Score | 0.171 |
| High-tier claims | 125 |
| Medium-tier claims | 125 |
| Low-tier claims | 250 |

---

## 4. GenAI Explanation Engine

### 4.1 Design Philosophy

Explanations must be:
1. **Actionable** -- tell the biller what to fix, not just what's wrong
2. **Honest** -- qualify with uncertainty language ("statistical estimate", "not guaranteed")
3. **Concise** -- 2-4 sentences, plain English
4. **Auditable** -- every API call logged with tokens, latency, and quality checks

### 4.2 Technical Implementation

```
Risk Factors (LR coefficients)
  -> ExplanationRequest (Pydantic, validated)
  -> Structured prompt (JSON schema in natural language)
  -> ollama.chat('gemma4:31b-cloud')
  -> JSON response
  -> ExplanationResponse (Pydantic, validated)
  -> to_plain_text()
  -> CSV output
```

### 4.3 Pydantic Validation

**Input validation (`ExplanationRequest`):**
- `denial_probability`: float, 0 <= value <= 1
- `risk_estimate_label`: regex `^(High|Medium|Low)$`
- `top_risk_factors`: list of RiskFactorItem, max 5, each with min_length=1

**Output validation (`ExplanationResponse`):**
- `claim_id`: non-empty string
- `disclaimer`: must contain uncertainty keywords ("estimate", "statistical", "not guaranteed")
- `risk_description`: min_length=1
- `recommended_action`: min_length=1

### 4.4 Fallback Strategy

Three-tier fallback ensures explanations are always generated:
1. **JSON parse** -- strip markdown fences, parse JSON, validate with Pydantic
2. **Regex extraction** -- search raw text for JSON-like structure
3. **Deterministic template** -- use `build_fallback_response()` with pre-mapped factor-to-action translations

Low-risk claims (prob < 0.25) skip the API entirely and use the deterministic template.

### 4.5 Production Quality (Verified)

Latest run: **125 API calls (all High-tier), 375 deterministic templates (Medium/Low), 100% Pydantic pass rate, 100% disclaimer rate**

Sample explanation:
> "This is a statistical estimate and not a guaranteed denial outcome. The claim is missing required prior authorization and supporting documentation. Resolve the missing authorization and attach all required documentation before submission."

---

## 5. Active Learning Experiment Results

### 5.1 Experiment Design

Ten experiments were conducted across five model families, varying architecture, hyperparameters, and feature sets. All experiments used identical train/val/test splits and preprocessing pipelines to ensure fair comparison.

### 5.2 Detailed Results

| Experiment | Type | Key Params | Capture@25 | ROC-AUC | Brier |
|---|---|---|---|---|---|
| exp1_lr_baseline | LR | C=0.1, balanced | 49.51% | 0.711 | 0.209 |
| exp2_lr_calibrated | LR-Calib | sigmoid, cv=5 | 49.51% | 0.710 | 0.137 |
| exp5_lr_interactions | LR | C=0.1, interaction feats | 49.51% | 0.707 | 0.210 |
| exp6_gradient_boosting | GBM | depth=5, lr=0.05, subsample=0.8 | 48.54% | 0.708 | 0.165 |
| exp7_voting_ensemble | Voting | soft, LR+RF+GBM, [2,1,1] | 48.54% | 0.705 | 0.172 |
| exp8_stacking | Stacking | LR meta, cv=5, passthrough | 46.60% | 0.698 | 0.181 |
| exp10_svc_rbf | SVC-Calib | RBF, C=1.0, Platt cv=3 | 45.63% | 0.688 | 0.192 |
| exp9_mlp_neural | MLP | (64,32), ReLU, alpha=0.001 | 43.69% | 0.672 | 0.198 |
| exp4_rf_robust | RF | depth=12, balanced | 39.81% | 0.653 | 0.155 |
| exp3_xgb_tuned | XGB | depth=6, lr=0.05, sw=3.6 | 35.92% | 0.615 | 0.180 |

### 5.3 Experiment Artifacts

Each experiment run produces a versioned artifact bundle in `data/output/experiments/<exp_name>/<run_id>/`:

```
params.json          -- hyperparameter snapshot
metrics.json         -- evaluation metrics
model.pkl            -- serialized model
preprocessor.pkl     -- ColumnTransformer
predictions.csv      -- scored current claims
feature_names.json   -- ordered feature list
run_metadata.json    -- combined metadata
```

This layout follows MLflow conventions, enabling easy migration to a full MLflow server if needed.

---

## 6. Production LLM Audit Infrastructure

### 6.1 Architecture

`src/llm_audit.py` provides HIPAA-compliant observability with zero external telemetry:

| Component | Capability |
|---|---|
| `LLMCallRecord` dataclass | Complete per-call metadata (tokens, latency, validation, quality) |
| `LLMAuditLogger` | Accumulates records, flushes to JSON with aggregation summaries |
| `extract_token_counts()` | Reads `prompt_eval_count`, `eval_count` from Ollama `ChatResponse` |
| `extract_latency()` | Converts nanosecond durations to milliseconds |
| `validate_response_quality()` | Checks disclaimer, action, length, hallucination markers |
| `build_audit_record()` | Factory function for complete call records |

### 6.2 Quality Checks

Every LLM response is validated for:
- **Disclaimer presence:** Must contain "estimate", "statistical", or "not guaranteed"
- **Actionable content:** Must contain action verbs (verify, obtain, review, attach, confirm, resolve, submit, ensure, check)
- **Minimum length:** >= 50 characters
- **Hallucination/PII markers:** Flags ICD-, CPT, diagnosis code, procedure code, patient name, SSN, DOB

### 6.3 Audit Output

A typical audit log (`audit_YYYYMMDD_HHMMSS.json`) contains:

```json
{
  "run_id": "20260527_133359",
  "total_calls": 500,
  "api_calls": 125,
  "fallbacks": 0,
  "token_summary": {
    "total_prompt_tokens": 44724,
    "total_completion_tokens": 11277,
    "total_tokens": 56001
  },
  "quality_summary": {
    "api_success_rate": 1.0,
    "json_parse_rate": 1.0,
    "pydantic_pass_rate": 1.0,
    "disclaimer_rate": 1.0,
    "avg_latency_ms": 1942.0
  },
  "records": [...]
}
```

### 6.4 Security & Compliance
- **No external telemetry:** All logging is self-contained to local JSON files
- **No PII in prompts:** Only claim IDs, engineered features, and risk factors are sent
- **Response truncation:** Audit records store first 500 chars of responses to limit log size
- **HIPAA alignment:** No patient identifiers, no clinical data in transit

---

## 7. Deliverables Checklist

| Deliverable | Status | Location |
|---|---|---|
| predictions_current_claims.csv | Complete | data/output/ |
| metrics.json | Complete | data/output/ |
| Source code (all modules) | Complete | src/ |
| Unit tests (82 tests, 6 modules) | Complete | tests/ |
| Prompt template file | Complete | src/prompts/explanation_prompt_v1.txt |
| Experiment runner (10 experiments) | Complete | src/experiment_runner.py |
| LLM audit infrastructure | Complete | src/llm_audit.py |
| Experiment tracker | Complete | src/experiment_tracker.py |
| Audit logs | Complete | data/output/audit_logs/ |
| Experiment artifacts | Complete | data/output/experiments/ |
| README.md | Complete | root |
| ANALYSIS.md | Complete | root |
| WRITEUP.md | Complete | root |
| Project_documentation.docx | Complete | root |
| .env.example | Complete | root |
| requirements.txt | Complete | root |

---

## 8. Future Improvements

### 8.1 Short-Term (Next Sprint)
- A/B test the 25% review workflow against a control group to measure actual denial reduction
- Add cost-weighted metrics (denied claim dollar value, not just denial count)
- Monitor Brier score drift monthly as an early warning system

### 8.2 Medium-Term
- Incorporate ICD-10 and CPT code features when real claims data is available
- Build payer-specific sub-models for the top 5 payers by volume
- Experiment with LightGBM and CatBoost as additional gradient boosting alternatives
- Implement Bayesian hyperparameter optimization (Optuna) for LR and GBM

### 8.3 Long-Term
- Transition from batch predictions to real-time scoring API
- Integrate with Ensemble's existing RCM workflow for automated pre-bill flagging
- Add explanation quality feedback loop (biller ratings) to improve LLM prompts
- Explore multi-class denial prediction (administrative vs medical necessity vs coding)
