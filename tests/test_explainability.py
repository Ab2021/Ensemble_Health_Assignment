"""
Unit tests for src/utils/explainability.py
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.utils.explainability import (
    map_factor_to_friendly,
    get_top_risk_factors_lr,
    build_risk_factors_string,
)


class TestMapFactorToFriendly:
    """Tests for feature name -> human-readable label mapping."""

    def test_known_factor(self):
        assert map_factor_to_friendly('auth_gap') == 'Missing Required Prior Authorization'

    def test_known_factor_referral(self):
        assert map_factor_to_friendly('referral_gap') == 'Missing Required Referral'

    def test_payer_type_feature(self):
        result = map_factor_to_friendly('payer_type_Medicaid_MCO')
        assert 'Medicaid' in result

    def test_visit_type_feature(self):
        result = map_factor_to_friendly('visit_type_Inpatient')
        assert 'Inpatient' in result

    def test_payer_id_feature(self):
        result = map_factor_to_friendly('payer_id_P008')
        assert 'P008' in result
        assert 'Payer ID' in result

    def test_submission_delay_feature(self):
        result = map_factor_to_friendly('submission_delay_1-2wk')
        assert '1-2wk' in result

    def test_unknown_feature_title_cased(self):
        result = map_factor_to_friendly('some_random_feature')
        assert result == 'Some Random Feature'


class TestGetTopRiskFactorsLr:
    """Tests for LR coefficient-based risk factor extraction."""

    def test_returns_positive_only(self):
        """Should only return factors with positive contribution."""
        mock_lr = MagicMock()
        mock_lr.coef_ = np.array([[0.5, -0.3, 1.2, -0.1, 0.0]])
        feature_names = ['auth_gap', 'referral_gap', 'late_filing', 'elig_issue', 'double_deficit']

        # Single row where all features = 1.0
        row = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0]], columns=feature_names)

        factors = get_top_risk_factors_lr(mock_lr, feature_names, row, n=3)
        # Should only include auth_gap (0.5) and late_filing (1.2), not negative/zero ones
        factor_names = [f[0] for f in factors]
        assert 'auth_gap' in factor_names
        assert 'late_filing' in factor_names
        assert 'referral_gap' not in factor_names  # negative coefficient

    def test_respects_n_limit(self):
        mock_lr = MagicMock()
        mock_lr.coef_ = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_names = ['a', 'b', 'c', 'd', 'e']
        row = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0]], columns=feature_names)

        factors = get_top_risk_factors_lr(mock_lr, feature_names, row, n=2)
        assert len(factors) == 2
        # Should be the two with highest coefficient
        assert factors[0][0] == 'e'  # coeff 5.0
        assert factors[1][0] == 'd'  # coeff 4.0


class TestBuildRiskFactorsString:
    """Tests for the full risk factor string builder."""

    def test_low_risk_returns_no_flags(self):
        mock_lr = MagicMock()
        # predict_proba returns (n,2); [:,1] gives denial probability
        mock_lr.predict_proba.return_value = np.array([[0.9, 0.1]])  # denial prob = 0.1
        mock_lr.coef_ = np.array([[0.5, -0.3]])
        feature_names = ['auth_gap', 'referral_gap']
        X = pd.DataFrame([[1.0, 1.0]], columns=feature_names)

        result = build_risk_factors_string(mock_lr, feature_names, X)
        assert 'No actionable' in result

    def test_high_risk_returns_factors(self):
        mock_lr = MagicMock()
        mock_lr.predict_proba.return_value = np.array([[0.2, 0.8]])  # prob = 0.8
        mock_lr.coef_ = np.array([[2.0, -0.5]])
        feature_names = ['auth_gap', 'referral_gap']
        X = pd.DataFrame([[1.0, 0.0]], columns=feature_names)

        result = build_risk_factors_string(mock_lr, feature_names, X)
        assert 'Missing Required Prior Authorization' in result

    def test_no_positive_factors_returns_no_flags(self):
        mock_lr = MagicMock()
        mock_lr.predict_proba.return_value = np.array([[0.4, 0.6]])  # prob = 0.6
        mock_lr.coef_ = np.array([[-0.5, -1.0]])  # all negative
        feature_names = ['auth_gap', 'referral_gap']
        X = pd.DataFrame([[1.0, 1.0]], columns=feature_names)

        result = build_risk_factors_string(mock_lr, feature_names, X)
        assert 'No actionable' in result
