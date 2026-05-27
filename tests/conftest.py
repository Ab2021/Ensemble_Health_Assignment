"""
Shared test fixtures for the Ensemble Health test suite.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# -- Sample data fixtures ------------------------------


@pytest.fixture
def sample_claim_row():
    """A single representative claim row as dict."""
    return {
        'claim_id': 'CCLM-TEST01',
        'total_billed': '5000.00',
        'expected_payment': '3500.00',
        'num_procedures': '2',
        'num_diagnoses': '3',
        'days_to_submit': '15',
        'prior_auth_required': '1',
        'has_prior_auth': '0',
        'is_in_network': '1',
        'missing_documentation_flag': '0',
        'eligibility_verified': '1',
        'referral_required': '0',
        'referral_present': '0',
        'payer_type': 'Commercial',
        'visit_type': 'Outpatient',
        'payer_id': 'P001',
        'service_month': '2024-06',
    }


@pytest.fixture
def sample_claims_df():
    """Small DataFrame for feature engineering tests."""
    return pd.DataFrame({
        'claim_id': ['C001', 'C002', 'C003'],
        'total_billed': ['10000', '5000', '2000'],
        'expected_payment': ['8000', '4000', '1800'],
        'num_procedures': ['3', '1', '2'],
        'num_diagnoses': ['2', '1', '3'],
        'days_to_submit': ['5', '45', '15'],
        'prior_auth_required': ['1', '0', '1'],
        'has_prior_auth': ['0', '0', '1'],
        'is_in_network': ['1', '1', '0'],
        'missing_documentation_flag': ['0', '1', '0'],
        'eligibility_verified': ['1', '1', '0'],
        'referral_required': ['0', '1', '1'],
        'referral_present': ['0', '0', '1'],
        'payer_type': ['Commercial', 'Medicare', 'Medicaid MCO'],
        'visit_type': ['Outpatient', 'Inpatient', 'Emergency'],
        'payer_id': ['P001', 'P002', 'P003'],
        'service_month': ['2024-06', '2024-07', '2024-08'],
    })


@pytest.fixture
def mock_model_coefficients():
    """Mock LR coefficients for explainability testing."""
    return np.array([0.5, -0.3, 1.2, -0.1, 0.8])


@pytest.fixture
def mock_feature_names():
    """Mock feature names for explainability testing."""
    return ['auth_gap', 'referral_gap', 'late_filing', 'elig_issue', 'double_deficit']
