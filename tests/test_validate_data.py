"""
Unit tests for src/utils/validate_data.py
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.utils.validate_data import validate_output_csv


class TestValidateOutputCsv:
    """Tests for the output CSV validation function."""

    def _make_valid_df(self):
        """Build a minimal valid output DataFrame."""
        df = pd.DataFrame({
            'claim_id': [f'CCLM-{i:05d}' for i in range(500)],
            'denial_probability': np.linspace(0.95, 0.01, 500),
            'predicted_denial': [1] * 125 + [0] * 375,
            'risk_tier': ['High'] * 125 + ['Medium'] * 125 + ['Low'] * 250,
            'top_risk_factors': ['Missing Auth'] * 500,
            'explanation': ['Test explanation'] * 10 + ['N/A'] * 490,
        })
        return df

    def test_valid_output_passes(self):
        """Valid output should pass all assertions."""
        df = self._make_valid_df()
        validate_output_csv(df)  # should not raise

    def test_wrong_row_count_fails(self):
        df = self._make_valid_df()
        df = df.iloc[:400]
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_wrong_columns_fails(self):
        df = self._make_valid_df()
        df = df.drop(columns=['explanation'])
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_wrong_tier_counts_fails(self):
        df = self._make_valid_df()
        df.loc[0:10, 'risk_tier'] = 'Low'
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_non_binary_predicted_denial_fails(self):
        df = self._make_valid_df()
        df.loc[0, 'predicted_denial'] = 2
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_probability_out_of_bounds_fails(self):
        df = self._make_valid_df()
        df.loc[0, 'denial_probability'] = 1.5
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_not_monotonic_fails(self):
        df = self._make_valid_df()
        # swap first two rows to break monotonicity
        p0 = df.loc[0, 'denial_probability']
        p1 = df.loc[1, 'denial_probability']
        df.loc[0, 'denial_probability'] = p1
        df.loc[1, 'denial_probability'] = p0
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_empty_top10_explanations_fails(self):
        df = self._make_valid_df()
        df.loc[0, 'explanation'] = ''
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_unknown_tier_fails(self):
        df = self._make_valid_df()
        df.loc[0, 'risk_tier'] = 'Unknown'
        with pytest.raises(AssertionError):
            validate_output_csv(df)

    def test_duplicate_claim_ids_passes_if_unique(self):
        df = self._make_valid_df()
        # all IDs are unique by construction
        validate_output_csv(df)  # should not raise
