"""
Unit tests for src/utils/models.py
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.models import (
    denial_capture_at_top_k,
    precision_at_top_k,
    compute_pr_auc,
    evaluate_model,
    fit_lr,
    fit_rf,
)


class TestDenialCaptureAtTopK:
    """Tests for the primary business metric."""

    def test_perfect_capture(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
        # Top 25% = top 2, both are actual denials -> 100% capture
        assert denial_capture_at_top_k(y_true, y_prob, k_frac=0.25) == 1.0

    def test_zero_capture(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.9, 0.8, 0.3, 0.2, 0.1, 0.1])
        # Top 2 have no denials -> 0% capture
        assert denial_capture_at_top_k(y_true, y_prob, k_frac=0.25) == 0.0

    def test_no_denials_returns_zero(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        assert denial_capture_at_top_k(y_true, y_prob) == 0.0

    def test_all_denials_returns_ratio(self):
        """When all are denials, top k% captures exactly k% of all denials."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        # Top 25% of 4 = 1, that 1 is a denial, captures 1/4 = 25% of denials
        assert denial_capture_at_top_k(y_true, y_prob, k_frac=0.25) == 0.25
        # Top 100% captures all denials
        assert denial_capture_at_top_k(y_true, y_prob, k_frac=1.0) == 1.0

    def test_beats_random_baseline(self):
        """With any signal, capture should exceed k_frac (random expectation)."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, size=1000)
        y_prob = y_true * 0.7 + np.random.uniform(0, 0.3, size=1000)
        assert denial_capture_at_top_k(y_true, y_prob, 0.25) > 0.20


class TestPrecisionAtTopK:
    """Tests for precision at top k%."""

    def test_perfect_precision(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.2])
        assert precision_at_top_k(y_true, y_prob, k_frac=0.50) == 1.0

    def test_empty_k_returns_zero(self):
        y_true = np.array([])
        y_prob = np.array([])
        result = precision_at_top_k(y_true, y_prob)
        assert result == 0.0


class TestComputePrAuc:
    """Tests for PR-AUC computation."""

    def test_perfect_returns_one(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])
        assert compute_pr_auc(y_true, y_prob) == 1.0

    def test_output_is_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        result = compute_pr_auc(y_true, y_prob)
        assert isinstance(result, float)
        assert 0 <= result <= 1


class TestEvaluateModel:
    """Tests for the full evaluation function."""

    def test_returns_all_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate_model('TestModel', y_true, y_prob)
        expected_keys = {'model', 'roc_auc', 'pr_auc', 'brier', 'capture_at_25',
                         'precision_at_25', 'binary_threshold', 'f1', 'precision',
                         'recall', 'confusion'}
        assert set(result.keys()) == expected_keys

    def test_binary_threshold_default(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate_model('TestModel', y_true, y_prob)
        # Default should be 75th percentile
        assert 0 < result['binary_threshold'] < 1


class TestFitLr:
    """Tests for logistic regression fitting."""

    def test_returns_model_and_c(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_val = np.random.randn(30, 5)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
        model, best_c = fit_lr(X_train, y_train, X_val, y_val)
        assert isinstance(model, LogisticRegression)
        assert best_c is not None
        assert best_c > 0

    def test_model_can_predict(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = np.random.randn(30, 3)
        y_val = (X_val[:, 0] > 0).astype(int)
        model, _ = fit_lr(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_val)[:, 1]
        assert len(probs) == 30
        assert (probs >= 0).all() and (probs <= 1).all()


class TestFitRf:
    """Tests for random forest fitting."""

    def test_returns_random_forest(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = fit_rf(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)

    def test_model_can_predict(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = fit_rf(X_train, y_train)
        probs = model.predict_proba(X_train)[:, 1]
        assert len(probs) == 50
