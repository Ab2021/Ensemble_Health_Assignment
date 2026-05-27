"""
Configuration constants for the Ensemble Health denial prediction system.
All paths, column definitions, and hyperparameters in one place.
"""
import os

# -- Paths ------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(_BASE, 'data', 'input')
OUTPUT_DIR = os.path.join(_BASE, 'data', 'output')
HISTORY_CSV = os.path.join(INPUT_DIR, 'claims_history.csv')
CURRENT_CSV = os.path.join(INPUT_DIR, 'current_claims.csv')
FIGURES_DIR = os.path.join(_BASE, 'data', 'figures')
EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, 'experiments')
AUDIT_LOG_DIR = os.path.join(OUTPUT_DIR, 'audit_logs')

# -- Column definitions -----------------------------
LEAKAGE_COLUMNS = ['claim_id', 'split', 'is_denied', 'denial_reason']
IDENTIFIER_COLUMN = 'claim_id'
TARGET_COLUMN = 'is_denied'

CATEGORICAL_COLS = ['payer_type', 'visit_type']
HIGH_CARD_COLS = ['payer_id']
ORDINAL_COLS = ['service_month']

BINARY_COLS = [
    'prior_auth_required', 'has_prior_auth', 'is_in_network',
    'missing_documentation_flag', 'eligibility_verified',
    'referral_required', 'referral_present',
]

NUMERIC_COLS = [
    'total_billed', 'expected_payment',
    'num_procedures', 'num_diagnoses', 'days_to_submit',
]

MONTH_COL = 'service_month'

# -- Model settings ---------------------------------
RANDOM_STATE = 42
REVIEW_CAPACITY_PCT = 0.25

# -- Friendly labels for risk factor display --------
FRIENDLY_LABELS = {
    'auth_gap': 'Missing Required Prior Authorization',
    'referral_gap': 'Missing Required Referral',
    'missing_documentation_flag': 'Missing Supporting Documentation',
    'eligibility_verified': 'Patient Eligibility Not Verified',
    'is_in_network': 'Provider Not in Payer Network',
    'late_filing': 'Late Claim Submission (>30 days)',
    'days_to_submit': 'Extended Days to Submit',
    'log_total_billed': 'High Claim Billed Amount',
    'payment_ratio': 'Low Expected Payment Ratio',
    'complexity_score': 'High Claim Complexity',
    'service_month_num': 'Service Month Factor',
    'total_admin_gaps': 'Multiple Administrative Gaps',
    'oon_auth_gap': 'Out-of-Network + Missing Authorization',
    'oon_referral_gap': 'Out-of-Network + Missing Referral',
    'double_deficit': 'Compound deficit: missing auth + documentation',
    'is_high_risk_combo': 'High-risk payer & visit combination',
}
