"""
Feature engineering - deterministic, pre-submission-safe derived features.

All features are computable BEFORE claim submission:
- No uses is_denied, denial_reason, or any post-adjudication data
- Gap flags (auth, referral, doc, eligibility, network)
- Financial ratios and log transforms
- Temporal features from service_month and days_to_submit
- Interaction terms and risk combo flags
"""
import numpy as np
import pandas as pd


def build_features(df):
    """Return a new DataFrame with all engineered features.

    Safe to call on history OR current - never uses target/denial data.

    Args:
        df: Raw claims DataFrame

    Returns:
        DataFrame with original columns + engineered features
    """
    d = df.copy()

    # -- Ensure numeric types ----------------------
    numeric_cols = ['total_billed', 'expected_payment', 'num_procedures',
                    'num_diagnoses', 'days_to_submit']
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    # -- Ensure binary columns are int -------------
    binary_cols = [
        'prior_auth_required', 'has_prior_auth', 'is_in_network',
        'missing_documentation_flag', 'eligibility_verified',
        'referral_required', 'referral_present'
    ]
    for c in binary_cols:
        if c in d.columns:
            d[c] = d[c].astype(int)

    # -- Temporal features -------------------------
    d['service_month_dt'] = pd.to_datetime(d['service_month'], format='%Y-%m')
    d['service_month_num'] = d['service_month_dt'].dt.month
    d['service_quarter'] = d['service_month_dt'].dt.quarter

    # -- Gap features (highest business value) -----
    d['auth_gap'] = (
        (d['prior_auth_required'] == 1) & (d['has_prior_auth'] == 0)
    ).astype(int)
    d['referral_gap'] = (
        (d['referral_required'] == 1) & (d['referral_present'] == 0)
    ).astype(int)

    # -- Documentation & eligibility gaps ----------
    d['doc_issue'] = d['missing_documentation_flag'].astype(int)
    d['elig_issue'] = (d['eligibility_verified'] == 0).astype(int)
    d['network_issue'] = (d['is_in_network'] == 0).astype(int)

    # -- Timely filing -----------------------------
    d['late_filing'] = (d['days_to_submit'] > 30).astype(int)

    # -- Financial features ------------------------
    d['payment_ratio'] = d['expected_payment'] / (d['total_billed'] + 1e-5)
    d['payment_gap'] = d['total_billed'] - d['expected_payment']
    d['log_total_billed'] = np.log1p(d['total_billed'])
    d['log_expected_payment'] = np.log1p(d['expected_payment'])
    d['billed_per_procedure'] = d['total_billed'] / (d['num_procedures'] + 1e-5)
    d['billed_per_diagnosis'] = d['total_billed'] / (d['num_diagnoses'] + 1e-5)

    # -- Complexity --------------------------------
    d['complexity_score'] = d['num_procedures'] * d['num_diagnoses']

    # -- Administrative gap count -----------------
    d['total_admin_gaps'] = (
        d['auth_gap'] + d['referral_gap'] +
        d['missing_documentation_flag'] +
        d['elig_issue'] + d['network_issue']
    )

    # -- Interaction features ---------------------
    d['oon_auth_gap'] = (
        (d['is_in_network'] == 0) & (d['auth_gap'] == 1)
    ).astype(int)
    d['oon_referral_gap'] = (
        (d['is_in_network'] == 0) & (d['referral_gap'] == 1)
    ).astype(int)
    d['double_deficit'] = (
        (d['auth_gap'] == 1) & (d['missing_documentation_flag'] == 1)
    ).astype(int)

    # -- High-risk combo flag ---------------------
    d['is_high_risk_combo'] = (
        (d['payer_type'] == 'Medicaid MCO') &
        (d['visit_type'].isin(['Inpatient', 'Emergency']))
    ).astype(int)

    # -- Submission delay buckets (ordinal) -------
    d['submission_delay'] = pd.cut(
        d['days_to_submit'],
        bins=[0, 7, 14, 30, 60, 100],
        labels=['<1wk', '1-2wk', '2-4wk', '1-2mo', '>2mo'],
        include_lowest=True
    )

    return d


def prepare_model_matrix(df, drop_target=True):
    """Select feature columns for model input, dropping leakage.

    Args:
        df: DataFrame (raw or engineered)
        drop_target: If True, also drop is_denied (for training data)

    Returns:
        Feature-only DataFrame ready for preprocessing
    """
    d = build_features(df) if 'service_month_num' not in df.columns else df.copy()

    drop_cols = ['claim_id', 'split', 'denial_reason', 'service_month_dt', 'service_month']
    if drop_target:
        drop_cols.append('is_denied')
    drop_cols = [c for c in drop_cols if c in d.columns]

    return d.drop(columns=drop_cols)
