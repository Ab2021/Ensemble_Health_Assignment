"""
Unit tests for src/utils/feature_engineering.py
"""
import pytest
import pandas as pd
import numpy as np
from src.utils.feature_engineering import build_features, prepare_model_matrix


class TestBuildFeatures:
    """Tests for the build_features function."""

    def test_returns_dataframe(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert isinstance(result, pd.DataFrame)

    def test_creates_auth_gap(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'auth_gap' in result.columns
        # C001: auth required (1) but no auth (0) -> gap = 1
        assert result.loc[0, 'auth_gap'] == 1
        # C002: auth not required (0) -> gap = 0
        assert result.loc[1, 'auth_gap'] == 0
        # C003: auth required (1) and has auth (1) -> gap = 0
        assert result.loc[2, 'auth_gap'] == 0

    def test_creates_referral_gap(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'referral_gap' in result.columns
        # C002: referral required (1) but no referral (0) -> gap = 1
        assert result.loc[1, 'referral_gap'] == 1

    def test_creates_late_filing(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'late_filing' in result.columns
        # C002: days_to_submit = 45 > 30 -> late_filing = 1
        assert result.loc[1, 'late_filing'] == 1
        # C001: days_to_submit = 5 -> late_filing = 0
        assert result.loc[0, 'late_filing'] == 0

    def test_creates_financial_features(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'payment_ratio' in result.columns
        assert 'log_total_billed' in result.columns
        assert 'billed_per_procedure' in result.columns
        # payment_ratio = expected / total
        assert abs(result.loc[0, 'payment_ratio'] - 0.8) < 0.01

    def test_creates_interaction_features(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'double_deficit' in result.columns
        assert 'oon_auth_gap' in result.columns
        assert 'is_high_risk_combo' in result.columns

    def test_creates_total_admin_gaps(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'total_admin_gaps' in result.columns
        assert result['total_admin_gaps'].dtype in [np.int64, np.int32, int]

    def test_creates_submission_delay_buckets(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert 'submission_delay' in result.columns

    def test_numeric_conversions(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert pd.api.types.is_numeric_dtype(result['total_billed'])
        assert pd.api.types.is_numeric_dtype(result['num_procedures'])

    def test_all_claims_have_features(self, sample_claims_df):
        result = build_features(sample_claims_df)
        assert len(result) == len(sample_claims_df)

    def test_no_nan_in_gap_features(self, sample_claims_df):
        result = build_features(sample_claims_df)
        gap_cols = ['auth_gap', 'referral_gap', 'doc_issue', 'elig_issue', 'network_issue']
        for col in gap_cols:
            assert not result[col].isna().any(), f"{col} has NaN"


class TestPrepareModelMatrix:
    """Tests for prepare_model_matrix."""

    def test_drops_claim_id(self, sample_claims_df):
        X = prepare_model_matrix(sample_claims_df)
        assert 'claim_id' not in X.columns

    def test_drops_service_month(self, sample_claims_df):
        X = prepare_model_matrix(sample_claims_df)
        assert 'service_month' not in X.columns
        assert 'service_month_dt' not in X.columns

    def test_includes_engineered_features(self, sample_claims_df):
        X = prepare_model_matrix(sample_claims_df)
        assert 'auth_gap' in X.columns
        assert 'late_filing' in X.columns

    def test_drop_target_false_keeps_is_denied(self, sample_claims_df):
        df_with_target = sample_claims_df.copy()
        df_with_target['is_denied'] = [0, 1, 0]
        X = prepare_model_matrix(df_with_target, drop_target=False)
        assert 'is_denied' in X.columns

    def test_drop_target_true_removes_is_denied(self, sample_claims_df):
        df_with_target = sample_claims_df.copy()
        df_with_target['is_denied'] = [0, 1, 0]
        X = prepare_model_matrix(df_with_target, drop_target=True)
        assert 'is_denied' not in X.columns
