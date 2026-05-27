# Ensemble Health -- Claim Denial Risk Prediction System

## Overview
Complete implementation of the **Ensemble Health Partners AI Team Hiring Assessment** -- pre-bill claim denial risk prediction combining classical ML (Logistic Regression + XGB/RF baselines) with a GenAI explanation engine powered by **Ollama Cloud (gemma4:31b-cloud)** using structured JSON prompts validated through **Pydantic** models.

**This Project Contains :**
- **Production LLM audit infrastructure** (`src/llm_audit.py`) -- token tracking, latency logging, response quality validation, HIPAA-safe JSON audit logs
- **MLflow-compatible experiment tracker** (`src/experiment_tracker.py`) -- artifact versioning, parameter/metric logging, run comparison
- **10-iteration active learning experiment loop** (`src/experiment_runner.py`) -- calibrated LR, XGB, RF, interaction features, GBM, Voting, Stacking, MLP, SVC
- **Calibrated Logistic Regression** selected as production model -- same denial capture (45.7%) with 34% better probability calibration (Brier 0.137 vs 0.209)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
#    venv\Scripts\activate       # Windows (cmd)
#    source venv/bin/activate    # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Ollama Cloud API key (already in .env for this submission)
#    Model: gemma4:31b-cloud via ollama.chat()

# 4. Run full pipeline -- single command
python src/run_pipeline.py

# 5. Run active learning experiments (10 model variants)
python src/experiment_runner.py

# 6. Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term --cov-report=html
```

Output:
- `data/output/predictions_current_claims.csv` -- 500 scored claims
- `data/output/metrics.json` -- test metrics and business KPIs
- `data/output/audit_logs/` -- per-call LLM observability (tokens, latency, quality)
- `data/output/experiments/` -- versioned experiment runs (MLflow-compatible layout)
- `coverage_html/` -- HTML coverage report

## Results at a Glance

### Production Model: Calibrated Logistic Regression

| Metric | Value |
|---|---|
| **Selected Model** | **Calibrated Logistic Regression** (sigmoid, cv=5) |
| **Test ROC-AUC** | **0.692** |
| **Test Denial Capture @ 25%** | **45.7%** |
| Test Precision @ 25% | 47.8% |
| Validation Brier Score | 0.209 |
| **Calibrated Brier Score** | **0.137** (34% improvement) |
| Binary Threshold (frozen) | 0.584 |

LR outperforms XGBoost (42.7% capture) and Random Forest (45.6% capture) on validation. Calibration does not hurt capture but makes probability-based tier assignment far more reliable.

### Experiment Leaderboard (10 Experiments)

| Rank | Experiment | Capture@25 | ROC-AUC | Brier | Verdict |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **LR Baseline** | **49.51%** | **0.711** | **0.209** | **Highest capture, simplest model** |
| 1 | LR + Calibration | 49.51% | 0.710 | 0.137 | Best overall -- same capture, best calibration |
| 1 | LR + Interactions | 49.51% | 0.707 | 0.210 | No gain from extra features |
| 4 | Gradient Boosting | 48.54% | 0.708 | 0.165 | Best non-linear model; close to LR |
| 5 | Voting Ensemble | 48.54% | 0.705 | 0.172 | LR+RF+GBM soft voting |
| 6 | Stacking (LR meta) | 46.60% | 0.698 | 0.181 | Meta-learner doesn't improve base models |
| 7 | SVC RBF | 45.63% | 0.688 | 0.192 | RBF kernel + Platt scaling |
| 8 | MLP Neural Net | 43.69% | 0.672 | 0.198 | (64,32) hidden layers, ReLU |
| 9 | Random Forest | 39.81% | 0.653 | 0.155 | Misses additive signal |
| 10 | XGBoost | 35.92% | 0.615 | 0.180 | Overfits sparse data |

**Key Insight:** This dataset has a very strong linear signal -- administrative gaps (missing prior auth, missing documentation, missing referral) are binary flags that directly predict denials. Non-linear models cannot improve on this because there is limited non-linear interaction to capture. Gradient Boosting (48.54%) comes closest to LR, confirming that careful boosting can approach linear performance on additive-feature problems.

## Project Structure

```
ensemble_solution/
|-- data/
|   |-- input/                   # Raw CSV files (gitignored)
|   |   |-- claims_history.csv
|   |   |-- current_claims.csv
|   |-- output/                  # Generated outputs
|       |-- predictions_current_claims.csv
|       |-- metrics.json
|       |-- audit_logs/          # Per-call LLM observability (tokens, latency, quality)
|       |-- experiments/         # Versioned experiment runs (MLflow-compatible layout)
|           |-- exp1_lr_baseline/
|               |-- comparison.json
|               |-- <run_id>/
|                   |-- model.pkl, preprocessor.pkl
|                   |-- predictions.csv, metrics.json
|                   |-- params.json, tags.json, feature_names.json
|-- src/
|   |-- __init__.py
|   |-- run_pipeline.py           # Main entry point
|   |-- experiment_runner.py      # 10-experiment active learning loop
|   |-- experiment_tracker.py     # Lightweight MLflow-compatible experiment versioning
|   |-- explanations.py           # GenAI engine: ollama.chat() + Pydantic validation
|   |-- llm_audit.py              # LLM observability: tokens, latency, quality, audit logs
|   |-- config/                   # Configuration constants
|   |   |-- __init__.py
|   |   |-- settings.py           # Paths, columns, labels, hyperparameters
|   |-- prompts/                  # Pydantic models + prompt templates
|   |   |-- __init__.py
|   |   |-- templates.py          # ExplanationRequest/Response, prompt builder, fallback
|   |-- utils/                    # ML utilities
|       |-- __init__.py
|       |-- validate_data.py      # Data contract assertions + output validation
|       |-- feature_engineering.py # Pre-submission-safe feature generation
|       |-- models.py             # Training, evaluation, metrics, calibration
|       |-- explainability.py     # LR coefficient attribution -> risk factors
|-- tests/                        # Unit tests (separate from src/)
|   |-- conftest.py               # Shared fixtures
|   |-- test_validate_data.py     # 10 tests
|   |-- test_feature_engineering.py # 16 tests
|   |-- test_models.py            # 14 tests
|   |-- test_explainability.py    # 10 tests
|   |-- test_explanations.py      # 22 tests -- Pydantic models, prompts, offline fallback
|-- setup.cfg                     # pytest + coverage configuration
|-- requirements.txt              # Pinned dependencies (ollama, pydantic, pytest, etc.)
|-- .env                          # API credentials (gitignored)
|-- .env.example                  # Template for .env
|-- .gitignore
|-- README.md
```

## GenAI Explanation Engine

### Architecture
```
Claim Data + Risk Factors
        |
  ExplanationRequest (Pydantic)
        |
  build_explanation_prompt()     -> Structured prompt asking for JSON
        |
  ollama.chat('gemma4:31b-cloud') -> Cloud API call
        |
  JSON response
        |
  Strip markdown fences          -> Handle ```json ... ``` wrappers
        |
  ExplanationResponse (Pydantic) -> Validate disclaimer, structure
        |
  to_plain_text()                -> Clean output for CSV
        |
  LLMAuditLogger                 -> Record tokens, latency, quality, raw response
```

### Key Design Decisions

- **Pydantic validation** at both input (`ExplanationRequest`) and output (`ExplanationResponse`)
- **Mandatory uncertainty qualifier** -- `disclaimer` field validator checks for keywords like "estimate", "statistical", "not guaranteed"
- **Markdown fence stripping** -- gemma models often wrap JSON in ` ```json ` blocks; our parser handles this transparently
- **Three-tier fallback**: JSON parse -> regex extraction -> deterministic template
- **Low-risk bypass** -- claims with prob < 0.25 or "No actionable" factors skip the API entirely and use the deterministic fallback
- **Production audit logging** -- every call tracked with token counts, latency, validation results, and quality warnings
- **HIPAA-safe** -- no telemetry to external services; self-contained JSON audit logs stored locally

### Example Outputs

**Top-risk (API-generated):**
> This is a statistical estimate and not a guaranteed denial outcome. The claim is missing the required prior authorization and supporting documentation. Resolve the missing authorization and attach all required documentation before submission.

**Low-risk (template fallback):**
> This claim has an estimated denial risk of 12% - this is a statistical estimate, not a guaranteed outcome. No actionable pre-submission risk flags are evident based on available data. Routine submission with standard verification is recommended.

## LLM Audit Infrastructure

The `src/llm_audit.py` module provides production-grade observability:

| Feature | Details |
|---|---|
| **Token tracking** | Extracts `prompt_eval_count`, `eval_count` from Ollama `ChatResponse` |
| **Latency logging** | `total_duration_ms`, `eval_duration_ms` per call |
| **Response validation** | JSON parse success, Pydantic validation success, error reasons |
| **Quality checks** | Disclaimer presence, actionable suggestion, min length, hallucination markers (ICD/CPT/PII) |
| **Audit output** | `data/output/audit_logs/audit_YYYYMMDD_HHMMSS.json` with full per-call metadata |
| **Aggregates** | Token summary, quality summary (pass rates, avg latency) |

**Latest audit run:**
- 10 API calls, 0 fallbacks, 100% Pydantic pass
- 4,479 total tokens (3,546 prompt + 933 completion)
- Average latency: 3,477 ms
- 0 hallucination warnings

## Experiment Tracking

The `src/experiment_tracker.py` provides a lightweight, self-contained experiment manager following MLflow conventions:

```
experiments/
  <experiment_name>/
    <run_id>/
      params.json        -- hyperparameters
      metrics.json       -- evaluation metrics
      tags.json          -- run metadata
      model.pkl          -- serialized sklearn model
      preprocessor.pkl   -- ColumnTransformer/StandardScaler
      predictions.csv    -- scored current claims
      feature_names.json -- ordered feature list
      run_metadata.json  -- combined run snapshot
```

Each experiment run is fully reproducible with versioned artifacts. The `compare_runs()` method generates a leaderboard for metric comparison.

## Output Schema

`predictions_current_claims.csv` (500 rows, sorted descending by probability):

| Column | Description |
|---|---|
| `claim_id` | Original ID |
| `denial_probability` | LR-calibrated probability [0-1] |
| `predicted_denial` | 0/1 at frozen test-set threshold |
| `risk_tier` | High / Medium / Low (125/125/250) |
| `top_risk_factors` | Top 3 model drivers (LR coefficient x feature) |
| `explanation` | LLM plain-English (top 10) or blank |

## Tests & Coverage

```bash
pytest tests/ -v --cov=src --cov-report=term --cov-report=html
```

**82 tests** across 6 test modules covering:
- Data validation contract (10 tests)
- Feature engineering correctness (16 tests)
- Model training, metrics, evaluation (14 tests)
- Explainability and risk factor extraction (10 tests)
- Pydantic model validation + prompt building + offline fallback (22 tests)
- API integration tests (marked as `api`, skipped by default)

Key coverage: `src/prompts/templates.py` 100%, `src/utils/feature_engineering.py` 100%, `src/config/settings.py` 100%, `src/utils/explainability.py` 97%.

## Limitations

- Synthetic data -- real claims include ICD/CPT codes, 835 remittance data, clinical notes
- Temporal extrapolation: current claims (2025-01) follow training data (2024-01-12)
- Ollama Cloud API requires network; offline fallback ensures resilience
- CalibratedClassifierCV adds ~20% training time but dramatically improves probability reliability
- `predictions_current_claims.csv` must not be distributed outside the hiring process

# Ensemble_Health_Assignment
