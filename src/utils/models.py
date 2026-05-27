"""
Model training, evaluation, and selection.

Primary: Logistic Regression (best validated on this synthetic data).
Challengers: XGBoost, Random Forest (to show completeness and robustness).

Business metric: Denial Capture @ Top 25% -- how many actual denials
are flagged within the review capacity constraint.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV


# =======================================================
# Business Metrics
# =======================================================

def denial_capture_at_top_k(y_true, y_prob, k_frac=0.25):
    """Primary business metric: what fraction of actual denials are caught
    in the top k% of predictions?

    Args:
        y_true: Binary ground truth
        y_prob: Model probability scores
        k_frac: Review capacity fraction (default 0.25 = 25%)

    Returns:
        Float in [0, 1]: fraction of denials captured
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    k = max(1, int(np.floor(len(y_true) * k_frac)))
    top_k_idx = np.argsort(-y_prob)[:k]
    actual_denials = y_true.sum()
    captured = y_true[top_k_idx].sum()
    return captured / actual_denials if actual_denials > 0 else 0.0


def precision_at_top_k(y_true, y_prob, k_frac=0.25):
    """Precision within the top k% of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    k = max(1, int(np.floor(len(y_true) * k_frac)))
    top_k_idx = np.argsort(-y_prob)[:k]
    if len(top_k_idx) == 0:
        return 0.0
    return float(y_true[top_k_idx].mean())


def compute_pr_auc(y_true, y_prob):
    """Area under the Precision-Recall curve."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_prob)


# =======================================================
# Evaluation
# =======================================================

def evaluate_model(model_name, y_true, y_prob, k_frac=0.25, binary_threshold=None):
    """Full evaluation suite for a candidate model.

    Args:
        model_name: Human-readable model identifier
        y_true: Binary ground truth array
        y_prob: Model probability scores
        k_frac: Review capacity fraction
        binary_threshold: Optional explicit threshold; if None, uses 75th percentile

    Returns:
        Dict with all metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    if binary_threshold is None:
        binary_threshold = float(np.percentile(y_prob, 75))

    y_pred = (y_prob >= binary_threshold).astype(int)

    results = {
        'model': model_name,
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc': round(compute_pr_auc(y_true, y_prob), 4),
        'brier': round(brier_score_loss(y_true, y_prob), 4),
        'capture_at_25': round(denial_capture_at_top_k(y_true, y_prob, k_frac), 4),
        'precision_at_25': round(precision_at_top_k(y_true, y_prob, k_frac), 4),
        'binary_threshold': round(binary_threshold, 6),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'confusion': confusion_matrix(y_true, y_pred).tolist(),
    }
    return results


# =======================================================
# Model Fitting
# =======================================================

def fit_lr(X_train, y_train, X_val, y_val, class_weight='balanced'):
    """Fit logistic regression with hyperparameter sweep over C.

    Selects best C by validation PR-AUC.

    Returns:
        (fitted_model, best_C)
    """
    best_c, best_ap = None, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lr = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=1000,
            solver='lbfgs', random_state=42
        )
        lr.fit(X_train, y_train)
        val_prob = lr.predict_proba(X_val)[:, 1]
        ap = compute_pr_auc(y_val, val_prob)
        if ap > best_ap:
            best_ap = ap
            best_c = C

    final_lr = LogisticRegression(
        C=best_c, class_weight=class_weight, max_iter=1000,
        solver='lbfgs', random_state=42
    )
    final_lr.fit(X_train, y_train)
    return final_lr, best_c


def fit_xgb(X_train, y_train, X_val, y_val, scale_pos_weight=3.6):
    """Fit XGBoost with early stopping on validation set."""
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_rf(X_train, y_train, class_weight='balanced'):
    """Fit Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight=class_weight, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def calibrate_platt(model, X_val, y_val):
    """Calibrate probabilities with Platt Scaling on validation data only."""
    cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    cal.fit(X_val, y_val)
    return cal
