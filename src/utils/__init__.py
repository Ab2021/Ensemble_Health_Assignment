"""
Utility modules for the Ensemble Health denial prediction system.
- validate_data: Data contract validation and leakage prevention
- feature_engineering: Pre-submission-safe feature generation
- models: Model training, evaluation, and selection
- explainability: Risk factor extraction from model attributions
"""
from src.utils.validate_data import load_and_validate, validate_output_csv
from src.utils.feature_engineering import build_features, prepare_model_matrix
from src.utils.models import (
    evaluate_model, fit_lr, fit_xgb, fit_rf,
    denial_capture_at_top_k, precision_at_top_k,
    compute_pr_auc, calibrate_platt,
)
from src.utils.explainability import (
    get_top_risk_factors_lr, map_factor_to_friendly,
    build_risk_factors_string,
)

__all__ = [
    'load_and_validate', 'validate_output_csv',
    'build_features', 'prepare_model_matrix',
    'evaluate_model', 'fit_lr', 'fit_xgb', 'fit_rf',
    'denial_capture_at_top_k', 'precision_at_top_k',
    'compute_pr_auc', 'calibrate_platt',
    'get_top_risk_factors_lr', 'map_factor_to_friendly',
    'build_risk_factors_string',
]
