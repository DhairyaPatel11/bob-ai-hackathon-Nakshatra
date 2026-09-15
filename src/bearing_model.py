"""
bearing_model.py
================
Remaining Useful Life (RUL) prediction model for bearing assets, trained on
the NASA IMS vibration feature table produced by vibration_features.py.

Design intent
-------------
The IMS dataset provides run-to-failure trajectories for 4 bearings per test
run.  We treat the *last snapshot* of each bearing as the failure point
(RUL = 0) and compute RUL for earlier snapshots as:

    RUL(row i) = (total_snapshots − i) * minutes_per_snapshot

This mirrors exactly how C-MAPSS RUL was computed in etl_cmapss.py:
    RUL(row) = max_cycle(unit) − cycle(row)

The model output dict has the same shape as train_model.predict():
    {"predicted_rul": np.ndarray, "is_ready": np.ndarray, "top_features": pd.DataFrame}
so that pipeline.py can call either model without knowing which modality
produced it.

Usage:
    python bearing_model.py            # trains + saves model, prints metrics
    python bearing_model.py --help     # CLI options
"""

import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------

# Feature columns produced by vibration_features.py that are used as inputs
# to the regressor.  Frequency reference columns (bpfo_hz, …) are constant
# for a given bearing and carry no temporal information — excluded.
VIBRATION_FEATURE_COLS = [
    "rms",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
    "bpfo_energy",
    "bpfi_energy",
    "bsf_energy",
    "ftf_energy",
]

# Rolling-window sizes (in snapshot steps) for trend features.
# IMS snapshots are approximately 10 minutes apart → window 5 ≈ 50 min,
# window 10 ≈ 100 min.  Same sizes as C-MAPSS to keep the code symmetric.
ROLLING_WINDOWS = [5, 10]

# Validation fraction (by bearing_id, not by row — same logic as C-MAPSS).
VAL_FRACTION = 0.25

# Mission readiness threshold: if predicted RUL > this value (in snapshot
# steps) the bearing is cleared as "ready".  20 snapshot steps ≈ 3.3 hours
# at the IMS 10-min interval, consistent with the C-MAPSS mission window
# of 20 cycles.
DEFAULT_MISSION_WINDOW = 20

# XGBoost hyperparameters — same as train_model.py for consistency.
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,   # smaller than C-MAPSS because bearing dataset is smaller
    "random_state":     42,
    "n_jobs":           -1,
}

# Minutes between IMS snapshots (approximate; used only in printed summaries)
IMS_MINUTES_PER_SNAPSHOT = 10


# ---------------------------------------------------------------------------
# 1.  RUL labeling
# ---------------------------------------------------------------------------

def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a RUL column to the vibration feature table.

    RUL for bearing b at snapshot i =
        (total snapshots for bearing b) − (index of snapshot i)

    This is the bearing analogue of C-MAPSS's
        RUL = max_cycle(unit) − cycle(row)

    The last snapshot per bearing has RUL = 0 (failure point).

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with columns: bearing_id, timestamp, rms, …
        Rows within each bearing_id must be in chronological order.

    Returns
    -------
    pd.DataFrame with an additional "RUL" column.
    """
    df = df.sort_values(["bearing_id", "timestamp"]).copy()

    rul_list = []
    for _, grp in df.groupby("bearing_id", sort=False):
        n_snaps = len(grp)
        rul_list.append(
            pd.Series(
                np.arange(n_snaps - 1, -1, -1, dtype=float),
                index=grp.index,
            )
        )

    df["RUL"] = pd.concat(rul_list)
    return df


# ---------------------------------------------------------------------------
# 2.  Rolling-window feature engineering
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add per-bearing rolling mean and std for each vibration feature.

    This mirrors train_model.add_rolling_features() to capture degradation
    *trends* in addition to the point-in-time feature values.  A bearing
    whose kurtosis is rising sharply over 10 snapshots is more informative
    than its absolute kurtosis value alone.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with bearing_id, timestamp, and vibration feature columns.
    feature_cols : list[str]
        Columns to compute rolling stats on.
    windows : list[int]
        Rolling window sizes in snapshot steps.

    Returns
    -------
    pd.DataFrame with additional columns:
        {feature}_rmean{w}  for each feature and window w
        {feature}_rstd{w}   for each feature and window w
    """
    df = df.sort_values(["bearing_id", "timestamp"]).copy()
    new_cols: dict = {}

    for feat in feature_cols:
        grp = df.groupby("bearing_id")[feat]
        for w in windows:
            new_cols[f"{feat}_rmean{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{feat}_rstd{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0)
            )

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] = VIBRATION_FEATURE_COLS,
    windows: list[int] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the final feature matrix from raw vibration features + rolling stats.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table (must already have rolling columns added).
    feature_cols : list[str]
        Base vibration feature column names.
    windows : list[int]
        Rolling window sizes used when building rolling columns.

    Returns
    -------
    (X, all_cols) where X is a DataFrame of shape (n_rows, n_features)
    and all_cols is the ordered list of column names.
    """
    rolling_cols = [
        f"{feat}_{stat}{w}"
        for feat in feature_cols
        for w in windows
        for stat in ("rmean", "rstd")
    ]
    all_cols = [c for c in feature_cols + rolling_cols if c in df.columns]
    return df[all_cols].copy(), all_cols


# ---------------------------------------------------------------------------
# 3.  Asset-level train / validation split
# ---------------------------------------------------------------------------

def asset_level_split(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by bearing_id to avoid leakage.

    With only 4 bearings in IMS Test Set 1, one bearing is held for
    validation and three are used for training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with bearing_id column.
    val_fraction : float
        Fraction of bearing_ids to use for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df) — each is a full-row subset of df.
    """
    bearing_ids = df["bearing_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(bearing_ids)

    n_val     = max(1, int(len(bearing_ids) * val_fraction))
    val_ids   = set(bearing_ids[:n_val])
    train_ids = set(bearing_ids[n_val:])

    train_df = df[df["bearing_id"].isin(train_ids)]
    val_df   = df[df["bearing_id"].isin(val_ids)]

    print(
        f"[split] train bearings: {sorted(train_ids)}  ({len(train_df):,} rows)  |  "
        f"val bearings: {sorted(val_ids)}  ({len(val_df):,} rows)"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# 4.  Train XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor on bearing vibration features.

    Uses early stopping on the validation set to prevent overfitting.
    Identical approach to train_model.train_xgboost() for consistency.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for training and validation.
    y_train, y_val : pd.Series
        RUL labels.

    Returns
    -------
    xgb.XGBRegressor — fitted model at the best iteration.
    """
    model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(
        f"\n[train] Best iteration: {model.best_iteration}  "
        f"| Best val RMSE: {model.best_score:.4f}"
    )
    return model


# ---------------------------------------------------------------------------
# 5.  Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """
    Compute RMSE and MAE and print a formatted summary line.

    Returns dict with keys "rmse" and "mae" (float).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    print(f"[eval]  {label:15s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
          f"(snapshot steps; 1 step ~{IMS_MINUTES_PER_SNAPSHOT} min)")
    return {"rmse": rmse, "mae": mae}


def feature_importances(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extract XGBoost feature importances by 'gain' and print the top-N.

    Parameters
    ----------
    model : xgb.XGBRegressor
    feature_cols : list[str]
        Ordered feature names (same order as the training matrix).
    top_n : int

    Returns
    -------
    pd.DataFrame with columns ["feature", "gain"], sorted descending.
    """
    importances = model.get_booster().get_score(importance_type="gain")
    if hasattr(model, "feature_names_in_"):
        fname_map = {f"f{i}": n for i, n in enumerate(model.feature_names_in_)}
    else:
        fname_map = {f"f{i}": n for i, n in enumerate(feature_cols)}

    imp_df = (
        pd.DataFrame(
            [{"feature": fname_map.get(k, k), "gain": v}
             for k, v in importances.items()]
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[importance] Top {top_n} features (by gain):")
    print(imp_df.head(top_n).to_string(index=False))
    return imp_df


# ---------------------------------------------------------------------------
# 6.  Readiness flag
# ---------------------------------------------------------------------------

def compute_readiness(
    predicted_rul: np.ndarray,
    mission_window: float = DEFAULT_MISSION_WINDOW,
) -> np.ndarray:
    """
    Return a boolean array: True if predicted RUL exceeds the mission window.

    Parameters
    ----------
    predicted_rul : array-like, float
        Predicted remaining snapshot steps.
    mission_window : float
        Minimum RUL (snapshot steps) for the asset to be mission-ready.

    Returns
    -------
    np.ndarray of bool.
    """
    rul_clipped = np.maximum(np.asarray(predicted_rul, dtype=float), 0.0)
    return rul_clipped > mission_window


# ---------------------------------------------------------------------------
# 7.  Save / load helpers
# ---------------------------------------------------------------------------

def save_bearing_model(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    output_dir: str,
) -> None:
    """
    Persist the trained bearing RUL model and its feature schema.

    Saved files:
        {output_dir}/bearing_rul_model.json
        {output_dir}/bearing_feature_names.json

    Parameters
    ----------
    model : xgb.XGBRegressor
    feature_cols : list[str]
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path    = os.path.join(output_dir, "bearing_rul_model.json")
    features_path = os.path.join(output_dir, "bearing_feature_names.json")

    model.save_model(model_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\n[save] Bearing model    -> {model_path}")
    print(f"[save] Bearing features -> {features_path}")


def predict(
    asset_features: pd.DataFrame,
    model_dir: str = "./models",
    mission_window: float = DEFAULT_MISSION_WINDOW,
) -> dict:
    """
    Public inference API — same return shape as train_model.predict().

    This function is intentionally signature-compatible with
    train_model.predict() so that pipeline.py can route either modality
    through the same predict() call.

    Parameters
    ----------
    asset_features : pd.DataFrame
        Feature rows for one or more bearing assets.  Columns not in
        bearing_feature_names.json are silently ignored.
    model_dir : str
        Directory containing bearing_rul_model.json and
        bearing_feature_names.json.
    mission_window : float
        Readiness threshold in snapshot steps.

    Returns
    -------
    dict with keys:
        predicted_rul  : np.ndarray  — predicted snapshot steps to failure
        is_ready       : np.ndarray  — bool, True if RUL > mission_window
        top_features   : pd.DataFrame — top-10 feature importances (gain)
    """
    model_path    = os.path.join(model_dir, "bearing_rul_model.json")
    features_path = os.path.join(model_dir, "bearing_feature_names.json")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(features_path) as f:
        feature_cols = json.load(f)

    X = asset_features.reindex(columns=feature_cols, fill_value=0.0)

    predicted_rul = model.predict(X)
    is_ready      = compute_readiness(predicted_rul, mission_window)
    top_feats     = feature_importances(model, feature_cols, top_n=10)

    return {
        "predicted_rul": predicted_rul,
        "is_ready":      is_ready,
        "top_features":  top_feats,
    }


# ---------------------------------------------------------------------------
# 8.  Main orchestrator (standalone training run)
# ---------------------------------------------------------------------------

def main(
    feature_parquet: str,
    output_dir: str,
    mission_window: float,
) -> dict:
    """
    End-to-end bearing model training run.

    Parameters
    ----------
    feature_parquet : str
        Path to the bearing feature parquet produced by etl_ims.py.
    output_dir : str
        Where to save the model artefacts and prediction CSVs.
    mission_window : float
        Readiness threshold (snapshot steps).

    Returns
    -------
    dict with "val_metrics" and "test_metrics" (each has "rmse" and "mae").
    """
    print(f"\n[config] feature_parquet={feature_parquet!r}  output_dir={output_dir!r}")

    # Step 1: Load
    if not os.path.exists(feature_parquet):
        raise FileNotFoundError(
            f"Feature parquet not found: {feature_parquet}\n"
            "Run etl_ims.py first to build the bearing feature table."
        )

    df = pd.read_parquet(feature_parquet)
    print(f"[load]  {df.shape[0]:,} rows, {df.shape[1]} cols, "
          f"{df['bearing_id'].nunique()} unique bearings")

    # Step 2: RUL labels
    df = add_rul_labels(df)
    print(f"[RUL]   min={df['RUL'].min():.0f}  mean={df['RUL'].mean():.1f}  "
          f"max={df['RUL'].max():.0f}")

    # Step 3: Rolling features
    df = add_rolling_features(df, VIBRATION_FEATURE_COLS, windows=ROLLING_WINDOWS)

    # Step 4: Split
    train_df, val_df = asset_level_split(df, VAL_FRACTION)

    # Step 5: Feature matrices
    X_train, feature_cols = build_feature_matrix(train_df)
    X_val,   _            = build_feature_matrix(val_df)

    y_train = train_df["RUL"]
    y_val   = val_df["RUL"]

    print(f"\n[features] Total features: {len(feature_cols)}")

    # Step 6: Train
    print("\n[train] Starting XGBoost training on bearing features ...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Step 7: Evaluate
    val_preds  = model.predict(X_val)
    val_metrics = evaluate(y_val.values, val_preds, "Val (bearing)")

    # Step 8: Feature importances
    feature_importances(model, feature_cols)

    # Step 9: Readiness flags
    val_is_ready = compute_readiness(val_preds, mission_window)

    # Step 10: Save prediction CSV
    os.makedirs(output_dir, exist_ok=True)
    val_out = val_df[["bearing_id", "timestamp", "RUL"]].copy()
    val_out = val_out.rename(columns={"RUL": "true_RUL"})
    val_out["predicted_RUL"] = val_preds
    val_out["is_ready"]      = val_is_ready
    val_out.to_csv(os.path.join(output_dir, "bearing_predictions_val.csv"), index=False)
    print(f"[save]  Predictions -> {output_dir}/bearing_predictions_val.csv")

    # Step 11: Save model
    save_bearing_model(model, feature_cols, output_dir)

    # Step 12: Summary
    print("\n" + "=" * 60)
    print("  BEARING MODEL — RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Validation RMSE : {val_metrics['rmse']:.2f} snapshot steps")
    print(f"  Validation MAE  : {val_metrics['mae']:.2f} snapshot steps")
    print(f"  (1 snapshot ~{IMS_MINUTES_PER_SNAPSHOT} min  -> "
          f"RMSE ~{val_metrics['rmse'] * IMS_MINUTES_PER_SNAPSHOT:.0f} min)")
    print(f"  Mission window  : {mission_window} snapshot steps")
    ready_frac = val_is_ready.mean()
    print(f"  Val ready pct   : {ready_frac:.0%}")
    print("=" * 60)

    return {"val_metrics": val_metrics}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bearing RUL model trainer (IMS dataset)")
    parser.add_argument(
        "--features",
        default="./output_real/ims_bearing_features.parquet",
        help="Path to the bearing feature parquet produced by etl_ims.py",
    )
    parser.add_argument(
        "--output",
        default="./models",
        help="Directory for model artefacts and prediction CSVs",
    )
    parser.add_argument(
        "--mission_window",
        type=float,
        default=DEFAULT_MISSION_WINDOW,
        help="Minimum predicted RUL (snapshot steps) for an asset to be ready",
    )
    args = parser.parse_args()

    main(
        feature_parquet = args.features,
        output_dir      = args.output,
        mission_window  = args.mission_window,
    )
