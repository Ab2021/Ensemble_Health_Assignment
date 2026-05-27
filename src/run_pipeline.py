"""
End-to-end denial risk pipeline main entry point.

Single command to go from raw CSVs -> predictions CSV with LLM explanations:
    python src/run_pipeline.py

Architecture:
  src/config/         -> Paths, constants, column definitions
  src/utils/          -> Validation, feature engineering, models, explainability
  src/prompts/        -> Pydantic models + prompt templates
  src/explanations.py -> Ollama Cloud (gemma4:31b-cloud) GenAI engine
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Ensure src/ is on the path for modular imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RANDOM_STATE, HISTORY_CSV, CURRENT_CSV, OUTPUT_DIR, AUDIT_LOG_DIR
from src.utils.validate_data import validate_output_csv
from src.utils.models import (
    denial_capture_at_top_k, precision_at_top_k,
    evaluate_model, compute_pr_auc,
)
from src.explanations import generate_explanation
from src.llm_audit import LLMAuditLogger


def engineer(df):
    """Deterministic pre-submission feature engineering."""
    d = df.copy()
    for c in ['total_billed', 'expected_payment', 'num_procedures',
              'num_diagnoses', 'days_to_submit']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for c in ['prior_auth_required', 'has_prior_auth', 'is_in_network',
              'missing_documentation_flag', 'eligibility_verified',
              'referral_required', 'referral_present']:
        if c in d.columns:
            d[c] = d[c].astype(int)
    d['service_month_dt'] = pd.to_datetime(d['service_month'], format='%Y-%m')
    d['service_month_num'] = d['service_month_dt'].dt.month
    d['service_quarter'] = d['service_month_dt'].dt.quarter
    d['auth_gap'] = ((d['prior_auth_required'] == 1) & (d['has_prior_auth'] == 0)).astype(int)
    d['referral_gap'] = ((d['referral_required'] == 1) & (d['referral_present'] == 0)).astype(int)
    d['doc_issue'] = d['missing_documentation_flag'].astype(int)
    d['elig_issue'] = (d['eligibility_verified'] == 0).astype(int)
    d['network_issue'] = (d['is_in_network'] == 0).astype(int)
    d['late_filing'] = (d['days_to_submit'] > 30).astype(int)
    d['payment_ratio'] = d['expected_payment'] / (d['total_billed'] + 1e-9)
    d['payment_gap'] = d['total_billed'] - d['expected_payment']
    d['log_total_billed'] = np.log1p(d['total_billed'])
    d['log_expected_payment'] = np.log1p(d['expected_payment'])
    d['billed_per_procedure'] = d['total_billed'] / (d['num_procedures'] + 1e-9)
    d['billed_per_diagnosis'] = d['total_billed'] / (d['num_diagnoses'] + 1e-9)
    d['complexity_score'] = d['num_procedures'] * d['num_diagnoses']
    d['total_admin_gaps'] = (d['auth_gap'] + d['referral_gap'] +
                             d['doc_issue'] + d['elig_issue'] + d['network_issue'])
    d['oon_auth_gap'] = ((d['is_in_network'] == 0) & (d['auth_gap'] == 1)).astype(int)
    d['oon_referral_gap'] = ((d['is_in_network'] == 0) & (d['referral_gap'] == 1)).astype(int)
    d['double_deficit'] = ((d['auth_gap'] == 1) & (d['missing_documentation_flag'] == 1)).astype(int)
    d['is_high_risk_combo'] = ((d['payer_type'] == 'Medicaid MCO') &
                               (d['visit_type'].isin(['Inpatient', 'Emergency']))).astype(int)
    d['submission_delay'] = pd.cut(d['days_to_submit'],
        bins=[0, 7, 14, 30, 60, 100],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True).astype(int)
    return d


def make_X(df):
    """Prepare feature matrix, dropping leakage columns."""
    drop_cols = ['claim_id', 'split', 'denial_reason', 'service_month', 'service_month_dt']
    leakage = ['is_denied']
    x = df.drop(columns=[c for c in drop_cols + leakage if c in df.columns]).copy()
    return x


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

    print("Loading data...")
    df_hist = pd.read_csv(HISTORY_CSV)
    df_curr = pd.read_csv(CURRENT_CSV)
    print(f"  Historical: {len(df_hist)} rows, Current: {len(df_curr)} rows")

    print("Engineering features...")
    df_hist = engineer(df_hist)
    df_curr = engineer(df_curr)

    drop_cols = ['claim_id', 'split', 'denial_reason', 'service_month', 'service_month_dt']
    train = df_hist[df_hist['split'] == 'train'].copy()
    val = df_hist[df_hist['split'] == 'validation'].copy()
    test = df_hist[df_hist['split'] == 'test'].copy()

    y_train = train['is_denied'].astype(int).values
    y_val = val['is_denied'].astype(int).values
    y_test = test['is_denied'].astype(int).values

    train_X = make_X(train)
    val_X = make_X(val)
    test_X = make_X(test)
    curr_X = make_X(df_curr)

    cat_features = ['payer_type', 'visit_type', 'payer_id']
    num_features = [c for c in train_X.columns if c not in cat_features]

    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64), cat_features),
        ('num', StandardScaler(), num_features),
    ], remainder='drop')
    preprocess.set_output(transform='pandas')

    X_train = preprocess.fit_transform(train_X)
    X_val = preprocess.transform(val_X)
    X_test = preprocess.transform(test_X)
    X_curr = preprocess.transform(curr_X)

    safe_cols = [str(c).replace('cat__', '').replace('num__', '').replace('bin__', '').replace('ord__', '')
                 .replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                 for c in X_train.columns]
    for dset in [X_train, X_val, X_test, X_curr]:
        dset.columns = safe_cols
    feature_names = list(X_train.columns)
    print(f"  Final features: {len(feature_names)}")

    print("\nTraining models...")

    dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_val = dummy.predict_proba(X_val)[:, 1]
    print(f"  Dummy  val cap@25: {denial_capture_at_top_k(y_val, dummy_val):.2%}")

    best_c, best_ap = None, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000,
                                 solver='lbfgs', random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        ap = compute_pr_auc(y_val, lr.predict_proba(X_val)[:, 1])
        if ap > best_ap:
            best_ap = ap
            best_c = C

    lr = LogisticRegression(C=best_c, class_weight='balanced', max_iter=1000,
                             solver='lbfgs', random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_val = lr.predict_proba(X_val)[:, 1]
    print(f"  LR     val cap@25: {denial_capture_at_top_k(y_val, lr_val):.2%}  (best C={best_c})")

    # ---- Calibrated LR (production model) ----
    print("  Calibrating LR...")
    cal_lr = CalibratedClassifierCV(
        LogisticRegression(C=best_c, class_weight='balanced', max_iter=1000,
                           solver='lbfgs', random_state=RANDOM_STATE),
        method='sigmoid', cv=3
    )
    cal_lr.fit(X_train, y_train)
    cal_val_prob = cal_lr.predict_proba(X_val)[:, 1]
    print(f"  Cal-LR val cap@25: {denial_capture_at_top_k(y_val, cal_val_prob):.2%}")

    xgb_prob = None
    try:
        import xgboost as xgb
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3.6,
            eval_metric='logloss', early_stopping_rounds=30,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val = xgb_m.predict_proba(X_val)[:, 1]
        xgb_prob = xgb_val
        print(f"  XGB    val cap@25: {denial_capture_at_top_k(y_val, xgb_val):.2%}")
    except Exception as e:
        print(f"  XGB    skipped: {e}")

    from sklearn.ensemble import RandomForestClassifier
    rf_m = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_m.fit(X_train, y_train)
    rf_val = rf_m.predict_proba(X_val)[:, 1]
    print(f"  RF     val cap@25: {denial_capture_at_top_k(y_val, rf_val):.2%}")

    # Compare on validation; always deploy Calibrated LR per experiment results
    models_val = {'LR': lr_val, 'Cal-LR': cal_val_prob, 'RF': rf_val}
    if xgb_prob is not None:
        models_val['XGB'] = xgb_prob
    best_score = max(denial_capture_at_top_k(y_val, v) for v in models_val.values())
    tied = [k for k, v in models_val.items()
            if abs(denial_capture_at_top_k(y_val, v) - best_score) < 1e-9]
    best_name = 'Cal-LR' if 'Cal-LR' in tied else tied[0]
    print(f"\n* Best model on validation: {best_name}")

    # Derive threshold from VALIDATION probabilities (no leakage)
    val_thresh = float(np.percentile(cal_val_prob, 75))
    print(f"  Validation-derived threshold (75th pct): {val_thresh:.4f}")

    print("\nFinal test evaluation (using calibrated LR + val threshold):")
    test_prob = cal_lr.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model('Calibrated LR', y_test, test_prob, binary_threshold=val_thresh)
    for k, v in test_metrics.items():
        if k != 'confusion':
            print(f"  {k:20s}: {v}")
    print(f"  {'confusion_matrix':20s}: {test_metrics['confusion']}")

    print("\nChecking calibration (Brier score on validation)...")
    from sklearn.metrics import brier_score_loss
    brier_pre = brier_score_loss(y_val, lr.predict_proba(X_val)[:, 1])
    brier_post = brier_score_loss(y_val, cal_lr.predict_proba(X_val)[:, 1])
    print(f"  Brier pre-calibration:  {brier_pre:.4f}")
    print(f"  Brier post-calibration: {brier_post:.4f}  (improvement: {brier_pre - brier_post:.4f})")

    print("\nRetraining on full history for scoring...")
    full_df = pd.concat([train, val, test]).reset_index(drop=True)
    full_X = make_X(engineer(full_df))
    y_full = engineer(full_df)['is_denied'].astype(int).values

    preprocess_full = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64), cat_features),
        ('num', StandardScaler(), num_features),
    ], remainder='drop')
    preprocess_full.set_output(transform='pandas')
    X_full = preprocess_full.fit_transform(full_X)
    X_curr_final = preprocess_full.transform(curr_X)
    safe_cols2 = [str(c).replace('cat__', '').replace('num__', '').replace('bin__', '').replace('ord__', '')
                  .replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                  for c in X_full.columns]
    for dset in [X_full, X_curr_final]:
        dset.columns = safe_cols2
    feature_names_final = list(X_full.columns)

    # Production model: calibrated LR trained on full history
    print("Fitting production CalibratedClassifierCV on full history...")
    prod_lr = LogisticRegression(C=best_c, class_weight='balanced', max_iter=1000,
                                   solver='lbfgs', random_state=RANDOM_STATE)
    prod_lr.fit(X_full, y_full)
    final_model = CalibratedClassifierCV(
        LogisticRegression(C=best_c, class_weight='balanced', max_iter=1000,
                           solver='lbfgs', random_state=RANDOM_STATE),
        method='sigmoid', cv=5
    )
    final_model.fit(X_full, y_full)
    final_probs = final_model.predict_proba(X_curr_final)[:, 1]

    print("\nAssigning risk tiers...")
    scored = pd.DataFrame({
        'claim_id': df_curr['claim_id'].values,
        'denial_probability': final_probs,
    })
    scored = scored.sort_values(by=['denial_probability', 'claim_id'],
                                ascending=[False, True]).reset_index(drop=True)
    scored['predicted_denial'] = (scored['denial_probability'] >= val_thresh).astype(int)
    scored['risk_tier'] = 'Low'
    scored.loc[:124, 'risk_tier'] = 'High'
    scored.loc[125:249, 'risk_tier'] = 'Medium'
    print(f"  High: {(scored['risk_tier'] == 'High').sum()}, "
          f"Medium: {(scored['risk_tier'] == 'Medium').sum()}, "
          f"Low: {(scored['risk_tier'] == 'Low').sum()}")
    print(f"  Predicted denials: {scored['predicted_denial'].sum()}")

    print("\nExtracting risk factors...")
    coef = prod_lr.coef_[0] if hasattr(prod_lr, 'coef_') else None

    top_factors = []
    for i in range(len(scored)):
        cid = scored.loc[i, 'claim_id']
        cidx = df_curr[df_curr['claim_id'] == cid].index[0]
        row = X_curr_final.iloc[[cidx]]
        prob = final_model.predict_proba(row)[0, 1]
        if prob < 0.25:
            factors_text = 'No actionable pre-submission risk flags detected'
        elif coef is not None:
            contributions = row.values[0] * coef
            positive = pd.Series(contributions, index=feature_names_final).sort_values(ascending=False)
            positive = positive[positive > 0]
            labels = []
            for fn in positive.index:
                fn_str = str(fn)
                if 'payer_type_' in fn_str or 'visit_type_' in fn_str or 'payer_id_' in fn_str:
                    name = (fn_str.replace('payer_type_', 'Payer: ')
                           .replace('visit_type_', 'Visit: ')
                           .replace('payer_id_', 'Payer ID: '))
                    labels.append(f"High-risk segment: {name.replace('_', ' ')}")
                elif fn_str.strip().lower() == 'auth_gap':
                    labels.append('Missing Required Prior Authorization')
                elif 'missing_documentation_flag' in fn_str.lower() or 'doc_issue' in fn_str.lower():
                    labels.append('Missing Supporting Documentation')
                elif 'referral_gap' in fn_str.lower():
                    labels.append('Missing Required Referral')
                elif 'eligibility_verified' in fn_str.lower() or 'elig_issue' in fn_str.lower():
                    labels.append('Patient Eligibility Not Verified')
                elif 'is_in_network' in fn_str.lower() or ('network' in fn_str.lower() and 'issue' in fn_str.lower()):
                    labels.append('Provider Not in Payer Network')
                elif 'late_filing' in fn_str.lower() or 'days_to_submit' == fn_str.strip().lower():
                    labels.append('Late Claim Submission (>30 days)')
                elif 'service_month' in fn_str.lower():
                    continue
                elif 'double_deficit' in fn_str.lower():
                    labels.append('Compound deficit: missing auth + documentation')
                elif 'total_admin_gaps' in fn_str.lower():
                    labels.append('Multiple administrative gaps')
                elif 'oon_auth_gap' in fn_str.lower():
                    labels.append('Out-of-network with missing authorization')
                elif 'oon_referral_gap' in fn_str.lower():
                    labels.append('Out-of-network with missing referral')
                elif 'is_high_risk_combo' in fn_str.lower():
                    labels.append('High-risk payer & visit combination')
                else:
                    labels.append(fn_str.replace('_', ' ').title())
            seen = []
            for lbl in labels:
                if lbl not in seen:
                    seen.append(lbl)
            factors_text = ', '.join(seen[:3]) if seen else 'No actionable pre-submission risk flags detected'
        else:
            d_row = df_curr.iloc[cidx]
            parts = []
            if d_row['auth_gap'] == 1: parts.append('Missing Required Prior Authorization')
            if d_row['missing_documentation_flag'] == 1: parts.append('Missing Supporting Documentation')
            if d_row['eligibility_verified'] == 0: parts.append('Patient Eligibility Not Verified')
            if d_row['referral_gap'] == 1: parts.append('Missing Required Referral')
            if d_row['late_filing'] == 1: parts.append('Late Claim Submission (>30 days)')
            if d_row['is_in_network'] == 0: parts.append('Provider Not in Payer Network')
            factors_text = ', '.join(parts[:3]) if parts else 'No actionable pre-submission risk flags detected'
        top_factors.append(factors_text)
    scored['top_risk_factors'] = top_factors

    print('\nGenerating explanations via Ollama Cloud (gemma4:31b-cloud)...')

    audit_logger = LLMAuditLogger(AUDIT_LOG_DIR)
    explanations = []
    for i in range(len(scored)):
        cid = scored.loc[i, 'claim_id']
        prob = scored.loc[i, 'denial_probability']
        tier = scored.loc[i, 'risk_tier']
        factors = scored.loc[i, 'top_risk_factors'].split(', ')
        cidx = df_curr[df_curr['claim_id'] == cid].index[0]
        claim_dict = df_curr.iloc[cidx].to_dict()
        if tier == 'High':
            # All High-risk claims: API-generated explanations
            expl, _ = generate_explanation(claim_dict, prob, factors, use_api=True, audit_logger=audit_logger)
        else:
            # Medium/Low: deterministic template (no API cost, still actionable)
            expl, _ = generate_explanation(claim_dict, prob, factors, use_api=False, audit_logger=audit_logger)
        explanations.append(expl)
    scored['explanation'] = explanations

    audit_path = audit_logger.flush()
    print(f"  Audit: {audit_logger.token_summary}")
    print(f"  Saved audit log: {audit_path}")

    # Preview top 3
    for i in range(min(3, len(scored))):
        print(f'\n  Top {i+1}: {scored.loc[i, "claim_id"]} | '
              f'prob={scored.loc[i, "denial_probability"]:.3f} | tier={scored.loc[i, "risk_tier"]}')
        print(f'    Factors: {scored.loc[i, "top_risk_factors"]}')
        print(f'    Explanation: {scored.loc[i, "explanation"][:200]}...')

    out_path = os.path.join(OUTPUT_DIR, 'predictions_current_claims.csv')
    scored.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    assert len(scored) == 500
    assert scored['denial_probability'].between(0, 1).all()
    assert scored['denial_probability'].is_monotonic_decreasing
    assert (scored['risk_tier'] == 'High').sum() == 125
    assert (scored['risk_tier'] == 'Medium').sum() == 125
    assert (scored['risk_tier'] == 'Low').sum() == 250
    assert scored[scored['risk_tier'] == 'High']['explanation'].str.len().min() > 0, "All High-tier must have explanations"
    assert scored[scored['risk_tier'] != 'High']['explanation'].str.len().min() > 0, "All Medium/Low must have template explanations"
    print("CSV validation passed.")

    metrics = {
        'model': 'Calibrated LR',
        'best_C': best_c,
        'test_roc_auc': test_metrics['roc_auc'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_brier': test_metrics['brier'],
        'test_capture_at_25': test_metrics['capture_at_25'],
        'test_precision_at_25': test_metrics['precision_at_25'],
        'test_f1': test_metrics['f1'],
        'val_threshold': val_thresh,
        'brier_pre_calibration': round(brier_pre, 4),
        'brier_post_calibration': round(brier_post, 4),
        'high_tier_count': int((scored['risk_tier'] == 'High').sum()),
        'llm_model': 'gemma4:31b-cloud',
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    print("\n=== Pipeline Complete ===")
