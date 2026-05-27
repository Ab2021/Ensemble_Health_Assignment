"""
Experiment Runner -- Active Learning loop for model optimization.

This script executes 5 high-value experiments to determine the best predictive model
for denial capture, logging all results via ExperimentTracker.

Goal: Maximize 'denial_capture_at_top_25'.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# Ensure src/ is on the path for modular imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RANDOM_STATE, HISTORY_CSV, CURRENT_CSV, EXPERIMENTS_DIR
from src.utils.models import evaluate_model, denial_capture_at_top_k, compute_pr_auc
from src.experiment_tracker import ExperimentTracker

# -- Replicate pipeline functions --------------------

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


def make_X(df, drop_cols=None):
    """Prepare feature matrix by dropping leakage and identifier columns.

    Removes claim_id, split, denial_reason, service_month identifiers,
    and the target column is_denied to prevent data leakage.
    """
    if drop_cols is None:
        drop_cols = ['claim_id', 'split', 'denial_reason', 'service_month', 'service_month_dt']
    leakage = ['is_denied']
    x = df.drop(columns=[c for c in drop_cols + leakage if c in df.columns]).copy()
    return x


def run_experiment_suite():
    """Execute 5 model experiments to find the optimal denial predictor.

    Trains LR baseline, calibrated LR, XGBoost, Random Forest, and LR
    with interaction features. Logs all results via ExperimentTracker.
    """
    print("Starting Model Experimentation Suite...")
    tracker = ExperimentTracker(EXPERIMENTS_DIR)
    
    # -- Load Data --------------------------------------
    df_hist = pd.read_csv(HISTORY_CSV)
    df_curr = pd.read_csv(CURRENT_CSV)
    
    # Initial engineering
    df_hist = engineer(df_hist)
    df_curr = engineer(df_curr)
    
    # Splits
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

    # -- Pipeline Utility -------------------------------
    def get_processed_data(train_df, val_df, test_df, curr_df):
        preprocess = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64), cat_features),
            ('num', StandardScaler(), num_features),
        ], remainder='drop')
        preprocess.set_output(transform='pandas')
        
        X_train = preprocess.fit_transform(train_df)
        X_val = preprocess.transform(val_df)
        X_test = preprocess.transform(test_df)
        X_curr = preprocess.transform(curr_df)
        
        # Sanitize columns
        safe_cols = [str(c).replace('cat__', '').replace('num__', '').replace('[', '_').replace(']', '_') for c in X_train.columns]
        for d in [X_train, X_val, X_test, X_curr]:
            d.columns = safe_cols
            
        return X_train, X_val, X_test, X_curr, preprocess

    X_train, X_val, X_test, X_curr, default_preprocessor = get_processed_data(train_X, val_X, test_X, curr_X)

    # -- Experiment Definitions -----------------------
    experiments = [
        {
            'name': 'exp1_lr_baseline',
            'model': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            'params': {'C': 0.1, 'class_weight': 'balanced', 'type': 'LR'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp2_lr_calibrated',
            'model': CalibratedClassifierCV(
                LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
                cv=5, method='sigmoid'
            ),
            'params': {'C': 0.1, 'calibration': 'sigmoid', 'type': 'LR-Calib'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp3_xgb_tuned',
            'model': xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, 
                scale_pos_weight=3.6, eval_metric='logloss', random_state=RANDOM_STATE
            ),
            'params': {'max_depth': 6, 'lr': 0.05, 'scale_pos_weight': 3.6, 'type': 'XGB'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp4_rf_robust',
            'model': RandomForestClassifier(
                n_estimators=200, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE
            ),
            'params': {'n_estimators': 200, 'max_depth': 12, 'type': 'RF'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp5_lr_interactions',
            'model': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            'params': {'C': 0.1, 'feature_set': 'interactions', 'type': 'LR'},
            'preprocessor': None # Will be created inside loop
        },
        # --- NEW EXPERIMENTS (v2) ---
        {
            'name': 'exp6_gradient_boosting',
            'model': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20,
                random_state=RANDOM_STATE
            ),
            'params': {'n_estimators': 200, 'max_depth': 5, 'lr': 0.05, 'subsample': 0.8, 'type': 'GBM'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp7_voting_ensemble',
            'model': VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
                    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE)),
                    ('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
                ],
                voting='soft',
                weights=[2, 1, 1]
            ),
            'params': {'voting': 'soft', 'weights': [2,1,1], 'estimators': 'LR+RF+GBM', 'type': 'Voting'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp8_stacking',
            'model': StackingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
                    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE)),
                    ('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
                ],
                final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
                cv=5,
                passthrough=True
            ),
            'params': {'cv': 5, 'passthrough': True, 'meta': 'LR', 'estimators': 'LR+RF+GBM', 'type': 'Stacking'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp9_mlp_neural',
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                alpha=0.001,
                batch_size=64,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                random_state=RANDOM_STATE
            ),
            'params': {'hidden': '(64,32)', 'activation': 'relu', 'alpha': 0.001, 'type': 'MLP'},
            'preprocessor': default_preprocessor
        },
        {
            'name': 'exp10_svc_rbf',
            'model': CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=False, random_state=RANDOM_STATE),
                cv=3, method='sigmoid'
            ),
            'params': {'kernel': 'rbf', 'C': 1.0, 'calibration': 'sigmoid', 'type': 'SVC-Calib'},
            'preprocessor': default_preprocessor
        },
    ]

    # -- Execution --------------------------------------
    for exp in experiments:
        print(f"Running {exp['name']}...")
        run = tracker.create_run(exp['name'])
        
        try:
            # Handling Exp 5 feature modification
            if exp['name'] == 'exp5_lr_interactions':
                # Create interaction features
                def add_interactions(df):
                    d = df.copy()
                    d['billed_x_admin_gaps'] = d['total_billed'] * d['total_admin_gaps']
                    d['billed_x_doc_issue'] = d['total_billed'] * d['doc_issue']
                    return d
                
                t_X = add_interactions(train_X)
                v_X = add_interactions(val_X)
                te_X = add_interactions(test_X)
                c_X = add_interactions(curr_X)
                
                # Adjusted preprocessor for interactions
                all_num = [c for c in t_X.columns if c not in cat_features]
                preprocess = ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64), cat_features),
                    ('num', StandardScaler(), all_num),
                ], remainder='drop')
                preprocess.set_output(transform='pandas')
                
                X_tr, X_v, X_te, X_c = preprocess.fit_transform(t_X), preprocess.transform(v_X), preprocess.transform(te_X), preprocess.transform(c_X)
                # Sanitize cols
                safe_cols = [str(c).replace('cat__', '').replace('num__', '').replace('[', '_').replace(']', '_') for c in X_tr.columns]
                for d in [X_tr, X_v, X_te, X_c]: d.columns = safe_cols
                
                current_X_train, current_X_val, current_X_test, current_X_curr, current_preprocessor = X_tr, X_v, X_te, X_c, preprocess
            else:
                current_X_train, current_X_val, current_X_test, current_X_curr, current_preprocessor = X_train, X_val, X_test, X_curr, exp['preprocessor']

            # Fit
            model = exp['model']
            model.fit(current_X_train, y_train)
            
            # Predict & Evaluate
            probs = model.predict_proba(current_X_val)[:, 1]
            metrics = evaluate_model('ExpModel', y_val, probs)
            
            # Capture @ 25% is our North Star
            cap_25 = denial_capture_at_top_k(y_val, probs)
            
            # Log to tracker
            tracker.log_params(run.run_id, exp['params'])
            tracker.log_metrics(run.run_id, {
                'capture_at_25': cap_25,
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'brier': metrics['brier']
            })
            
            # Save Artifacts
            tracker.save_model(run.run_id, model)
            tracker.save_preprocessor(run.run_id, current_preprocessor)
            tracker.save_feature_names(run.run_id, list(current_X_train.columns))
            
            # Generate Predictions for Current Claims
            curr_probs = model.predict_proba(current_X_curr)[:, 1]
            df_res = pd.DataFrame({'claim_id': df_curr['claim_id'], 'prob': curr_probs})
            tracker.save_predictions(run.run_id, df_res)
            
            tracker.finish_run(run.run_id)
            print(f"  Finished {exp['name']}: Capture@25={cap_25:.3%}")

        except Exception as e:
            print(f"  Failed {exp['name']}: {e}")
            tracker.fail_run(run.run_id, str(e))

    # -- Summary ---------------------------------------
    print("\n" + "=" * 60)
    print("=== FINAL COMPARISON SUMMARY (10 experiments) ===")
    print("=" * 60)
    best_cap = 0.0
    best_name = ""
    import glob
    for exp_dir in sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, 'exp*'))):
        if 'lr_hyperparameter' in exp_dir:
            continue
        runs = sorted(glob.glob(os.path.join(exp_dir, '*')))
        if not runs:
            continue
        latest = runs[-1]
        metrics_path = os.path.join(latest, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                em = json.load(f)
            cap = em.get('denial_capture_at_25', em.get('capture_at_25', 0))
            name = os.path.basename(exp_dir)
            print(f"  {name}: Capture@25={cap:.4f} ({cap*100:.2f}%)")
            if cap > best_cap:
                best_cap = cap
                best_name = name
    print(f"\n  >> Best: {best_name} with Capture@25={best_cap:.4f} ({best_cap*100:.2f}%)")

if __name__ == "__main__":
    run_experiment_suite()
