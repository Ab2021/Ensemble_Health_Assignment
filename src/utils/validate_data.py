"""
Data validation with programmatic assertions.
Ensures data contract integrity and leakage prevention.
"""
import pandas as pd


def load_and_validate(history_path, current_path):
    """Load CSVs and run full validation suite.

    Args:
        history_path: Path to claims_history.csv
        current_path: Path to current_claims.csv

    Returns:
        (df_hist, df_curr) tuple of validated DataFrames

    Raises:
        AssertionError on any validation failure
    """
    df_hist = pd.read_csv(history_path, dtype=str)
    df_curr = pd.read_csv(current_path, dtype=str)

    # 1. Row counts
    assert len(df_hist) == 3200, f"Expected 3200 historical rows; got {len(df_hist)}"
    assert len(df_curr) == 500, f"Expected 500 current rows; got {len(df_curr)}"

    # 2. Unique IDs
    assert df_hist['claim_id'].nunique() == len(df_hist), "Duplicate claim_id in history"
    assert df_curr['claim_id'].nunique() == len(df_curr), "Duplicate claim_id in current"

    # 3. No overlap
    overlap = set(df_hist['claim_id']) & set(df_curr['claim_id'])
    assert len(overlap) == 0, f"claim_id overlap found: {overlap}"

    # 4. Required columns
    required_hist = {'claim_id', 'split', 'is_denied', 'denial_reason'}
    required_curr = {'claim_id'}
    missing_hist = required_hist - set(df_hist.columns)
    missing_curr = required_curr - set(df_curr.columns)
    assert len(missing_hist) == 0, f"Missing history columns: {missing_hist}"
    assert len(missing_curr) == 0, f"Missing current columns: {missing_curr}"

    # 5. Split values
    assert set(df_hist['split'].unique()) == {'train', 'validation', 'test'}, "Unexpected split values"

    # 6. Target binary
    assert set(df_hist['is_denied'].unique()).issubset({'0', '1'}), "is_denied contains non-binary values"

    # 7. Expected payment <= total billed
    df_hist_nums = df_hist.astype({'total_billed': float, 'expected_payment': float})
    msg = df_hist_nums[df_hist_nums['expected_payment'] > df_hist_nums['total_billed']]
    assert len(msg) == 0, f"Found {len(msg)} records where expected_payment > total_billed"

    print("OK All data validation checks passed")
    return df_hist, df_curr


def validate_output_csv(df_out):
    """Validate the final predictions CSV against the output contract.

    Checks:
      - 500 rows, correct 6 columns
      - Unique claim_ids
      - Probability bounds [0,1]
      - Monotonic descending by probability
      - Binary predicted_denial (0/1)
      - Risk tiers: 125 High, 125 Medium, 250 Low
      - Top 10 explanations non-empty
    """
    assert len(df_out) == 500, f"Expected 500 rows, got {len(df_out)}"
    assert list(df_out.columns) == [
        'claim_id', 'denial_probability', 'predicted_denial',
        'risk_tier', 'top_risk_factors', 'explanation'
    ], f"Unexpected columns: {list(df_out.columns)}"
    assert df_out['claim_id'].nunique() == 500
    assert df_out['denial_probability'].between(0, 1).all()
    assert df_out['denial_probability'].is_monotonic_decreasing
    assert df_out['predicted_denial'].isin([0, 1]).all()
    assert df_out['risk_tier'].isin(['High', 'Medium', 'Low']).all()
    assert (df_out['risk_tier'] == 'High').sum() == 125
    assert (df_out['risk_tier'] == 'Medium').sum() == 125
    assert (df_out['risk_tier'] == 'Low').sum() == 250
    assert df_out.head(10)['explanation'].notna().all()
    assert (df_out.head(10)['explanation'] != '').all()
    print("OK Output CSV validation passed")
