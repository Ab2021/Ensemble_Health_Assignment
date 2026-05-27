"""
Explainability: produce top_risk_factors per claim using model attribution.

For Logistic Regression: coefficient * feature value = linear contribution.
Maps feature names to human-friendly labels for business users.
"""
import numpy as np
import pandas as pd
from src.config import FRIENDLY_LABELS


def get_top_risk_factors_lr(lr_model, feature_names, claim_row, n=3):
    """Get top positive risk factors from LR coefficients.

    Args:
        lr_model: Fitted logistic regression model
        feature_names: List of feature names (post-preprocessing)
        claim_row: Single-row DataFrame in model-ready form (post-preprocessing)
        n: Number of top factors to return

    Returns:
        List of (feature_name, contribution_value) tuples, sorted descending
    """
    coef = lr_model.coef_[0]
    contributions = claim_row.values[0] * coef

    names = list(feature_names)
    contrib_series = pd.Series(contributions, index=names)
    positive = contrib_series[contrib_series > 0].nlargest(n)

    return list(zip(positive.index, positive.values))


def map_factor_to_friendly(feat_name):
    """Map a single feature name to a human-readable label.

    Handles:
    - Direct label lookups from FRIENDLY_LABELS
    - OneHot encoded categorical names (payer_type_X, visit_type_X)
    - Submission delay bucket names
    - Fallback: title-case the feature name
    """
    if feat_name in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[feat_name]

    if 'payer_type_' in feat_name:
        return f'High-risk payer type: {feat_name.replace("payer_type_", "").replace("_", " ")}'
    if 'visit_type_' in feat_name:
        return f'High-risk visit type: {feat_name.replace("visit_type_", "").replace("_", " ")}'
    if 'payer_id_' in feat_name:
        return f'High-risk segment: Payer ID: {feat_name.replace("payer_id_", "").replace("_", " ")}'
    if 'submission_delay_' in feat_name:
        return f'Submission Delay: {feat_name.replace("submission_delay_", "").replace("_", "-")}'
    if 'payer_id' == feat_name:
        return 'Payer-Specific Risk Profile'

    return feat_name.replace('_', ' ').title()


def build_risk_factors_string(lr_model, feature_names, X_single, n=3):
    """Build comma-separated risk factor string for a single claim.

    For low-risk claims (prob < 0.25), returns a clear 'no flags' message.
    For tree-based models, this will return an appropriate fallback.

    Args:
        lr_model: Fitted LR model (for coefficient attribution)
        feature_names: Post-preprocessing feature names
        X_single: Single-row DataFrame for one claim
        n: Max number of factors

    Returns:
        Comma-separated string of risk factors
    """
    probs = lr_model.predict_proba(X_single)[:, 1]

    if probs[0] < 0.25:
        return 'No actionable pre-submission risk flags detected'

    factors = get_top_risk_factors_lr(lr_model, feature_names, X_single, n=n)
    if len(factors) == 0:
        return 'No actionable pre-submission risk flags detected'

    labels = []
    seen = set()
    for feat_name, _ in factors:
        friendly = map_factor_to_friendly(feat_name)
        if friendly not in seen:
            labels.append(friendly)
            seen.add(friendly)

    return ', '.join(labels[:n])
