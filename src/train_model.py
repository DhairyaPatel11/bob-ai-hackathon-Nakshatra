"""
train_model.py
==============
Trains an XGBoost regressor to predict Remaining Useful Life (RUL)
from C-MAPSS sensor data + rolling-window engineered features.

Usage:
    python train_model.py [--train  ./output_real/train_FD001_merged.parquet]
                          [--test   ./output_real/test_FD001_merged.parquet]
                          [--output ./models]
                          [--mission_window 20]

Outputs (written to --output dir):
    rul_model.json          XGBoost model (portable, text-based)
    feature_names.json      Ordered list of feature columns the model expects
    predictions_val.csv     Val-set rows: asset_id, true_RUL, predicted_RUL, is_ready
    predictions_test.csv    Test-set rows: same schema
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TTM forecast features (imported lazily; graceful fallback if unavailable)
try:
    from forecast import forecast_sensors, TTM_CONTEXT_LENGTH, TTM_FORECAST_LENGTH
    _TTM_AVAILABLE = True
except ImportError:
    _TTM_AVAILABLE = False

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------

# Variance below this threshold -> sensor is considered "near-zero variance"
# and will be dropped. Tune if your dataset has different sensor scales.
VARIANCE_THRESHOLD = 0.01

# Rolling window sizes (in cycles) for feature engineering
ROLLING_WINDOWS = [5, 10]

# Fraction of asset_ids to hold out for validation (split is by asset, not row)
VAL_FRACTION = 0.20

# Default mission window: if predicted RUL > this -> asset is ready to fly
DEFAULT_MISSION_WINDOW = 20   # cycles

# XGBoost hyperparameters (good starting point for C-MAPSS FD001)
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state":     42,
    "n_jobs":           -1,
}

# Columns that come from the synthetic service_records table.
# These are intentionally EXCLUDED from the regression model inputs
# (used only in the downstream explanation / prioritisation layer).
SERVICE_RECORD_COLS = [
    "tier",
    "last_inspection_date",
    "cumulative_flight_hours",
    "last_part_replaced",
    "part_replacement_date",
    "open_discrepancies",
    "next_scheduled_maintenance",
    "mission_critical_flag",
]


# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------

def load_data(train_path, test_path):
    """Read the merged parquet files produced by etl_cmapss.py."""
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    print(f"[load] train: {train_df.shape}  |  test: {test_df.shape}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# 2.  Feature engineering
# ---------------------------------------------------------------------------

def drop_low_variance_sensors(train_df, test_df, sensor_cols, threshold=VARIANCE_THRESHOLD):
    """
    Drop sensor columns whose variance on the TRAINING set is below threshold.

    We measure variance on training data only to avoid any information leakage
    from the test set into our feature selection step.

    Sensors with near-zero variance carry no discriminative information
    (they read the same value regardless of engine health) and can confuse
    tree-based models by creating many uninformative splits.
    """
    variances = train_df[sensor_cols].var()
    low_var   = variances[variances < threshold].index.tolist()

    if low_var:
        print(f"\n[feature-eng] Dropping {len(low_var)} near-zero-variance sensors:")
        for col in low_var:
            print(f"   {col:12s}  var={variances[col]:.6f}  (threshold={threshold})")
    else:
        print("\n[feature-eng] No low-variance sensors found -- keeping all.")

    kept = [c for c in sensor_cols if c not in low_var]
    train_df = train_df.drop(columns=low_var)
    test_df  = test_df.drop(columns=low_var)
    return train_df, test_df, kept


def add_rolling_features(df, sensor_cols, windows=ROLLING_WINDOWS):
    """
    For each sensor and each window size, compute per-asset rolling mean and std.

    Rolling stats capture the *trend* of sensor readings -- e.g. a rising rolling
    mean on sensor_11 is more informative than the raw point reading alone.

    We sort by (asset_id, flight_hours) first so the rolling window is always
    computed in chronological order within each asset's lifecycle.

    min_periods=1 ensures early cycles (fewer than window points) still get a
    value rather than NaN.
    """
    # Sort ensures time-ordered rolling within each asset
    df = df.sort_values(["asset_id", "flight_hours"]).copy()

    new_cols = {}
    for sensor in sensor_cols:
        grouped = df.groupby("asset_id")[sensor]
        for w in windows:
            new_cols[f"{sensor}_rmean{w}"] = grouped.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{sensor}_rstd{w}"] = grouped.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0)
            )

    rolled = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, rolled], axis=1)
    print(f"[feature-eng] Added {len(new_cols)} rolling features "
          f"({len(sensor_cols)} sensors x {len(windows)} windows x 2 stats)")
    return df


def add_ttm_forecast_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Augment *df* with TTM zero-shot forecast summary features per row.

    For each asset and each row (cycle), we take that row's position in the
    asset's history and feed the preceding context_length cycles to the TTM
    model.  The resulting 16-step forecast is summarised into three scalars
    per sensor channel (fmean, fstd, fslope) and broadcast to every row of
    that asset so XGBoost can use them alongside rolling-window features.

    Because TTM inference is relatively fast (pure CPU, ~0.5 s per asset) and
    we have 100 assets, the full pass takes roughly 30-60 s.

    Fallback: if the TTM model is unavailable, returns *df* unchanged and
    prints a warning.  train_model.py's build_feature_matrix() will simply
    skip the forecast columns since they won't exist.
    """
    if not _TTM_AVAILABLE:
        print("[TTM] forecast module not available -- skipping forecast features.")
        return df

    df = df.sort_values(["asset_id", "flight_hours"]).copy()
    all_forecast_rows = []

    asset_ids = df["asset_id"].unique()
    print(f"[TTM] Computing forecast features for {len(asset_ids)} assets "
          f"(context={TTM_CONTEXT_LENGTH}, horizon={TTM_FORECAST_LENGTH}) ...")

    for i, aid in enumerate(asset_ids, 1):
        asset_df = df[df["asset_id"] == aid].sort_values("flight_hours")
        # Use the full history of the asset (TTM will take the last 52 cycles)
        fcast_df = forecast_sensors(
            asset_sensor_history=asset_df,
            sensor_cols=sensor_cols,
            context_length=TTM_CONTEXT_LENGTH,
            forecast_length=TTM_FORECAST_LENGTH,
        )
        if fcast_df.empty:
            # TTM unavailable for this asset -- skip
            continue

        # fcast_df is a single-row summary; broadcast to ALL rows of this asset
        for col in fcast_df.columns:
            df.loc[asset_df.index, col] = float(fcast_df[col].iloc[0])

        if i % 20 == 0 or i == len(asset_ids):
            print(f"  ... processed {i}/{len(asset_ids)} assets")

    # Identify which forecast columns were actually added
    fcols = [c for c in df.columns if c.endswith(("_fmean", "_fstd", "_fslope"))]
    if fcols:
        print(f"[TTM] Added {len(fcols)} forecast features "
              f"({len(fcols) // 3} sensors x 3 stats)")
    else:
        print("[TTM] No forecast features added (inference returned empty).")

    return df


def build_feature_matrix(df, sensor_cols, op_cols, windows=ROLLING_WINDOWS):
    """
    Combine raw sensors, op-settings, rolling features, and TTM forecast
    features into the final feature matrix X.

    TTM forecast columns (sensor_N_fmean/fstd/fslope) are included if
    present in df; they are silently skipped if TTM was unavailable.
    """
    rolling_cols = []
    for sensor in sensor_cols:
        for w in windows:
            rolling_cols.append(f"{sensor}_rmean{w}")
            rolling_cols.append(f"{sensor}_rstd{w}")

    # TTM forward-looking features (present only if TTM loaded successfully)
    forecast_cols = [c for c in df.columns if c.endswith(("_fmean", "_fstd", "_fslope"))]

    feature_cols = sensor_cols + op_cols + rolling_cols + forecast_cols
    # Guard: keep only columns that actually exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df[feature_cols], feature_cols


# ---------------------------------------------------------------------------
# 3.  Asset-level train / validation split (no leakage)
# ---------------------------------------------------------------------------

def asset_level_split(df, val_fraction=VAL_FRACTION, seed=42):
    """
    Split df into train and validation sets by asset_id.

    WHY asset-level and not row-level?
    If we split by row, consecutive cycles of the same engine will appear in
    both train and val. The model would then see earlier cycles of an engine
    during training and 'memorise' its degradation trajectory -- leading to
    over-optimistic validation scores that don't reflect real deployment.

    By holding out ENTIRE assets we get a true out-of-sample evaluation.
    """
    asset_ids = df["asset_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(asset_ids)

    n_val = max(1, int(len(asset_ids) * val_fraction))
    val_assets   = set(asset_ids[:n_val])
    train_assets = set(asset_ids[n_val:])

    train_split = df[df["asset_id"].isin(train_assets)]
    val_split   = df[df["asset_id"].isin(val_assets)]

    print(f"[split] train assets: {len(train_assets)}  "
          f"({len(train_split):,} rows)  |  "
          f"val assets: {len(val_assets)}  ({len(val_split):,} rows)")
    return train_split, val_split


# ---------------------------------------------------------------------------
# 4.  Train XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost regressor with early stopping on the validation set.

    early_stopping_rounds=30 means training stops if the validation RMSE
    hasn't improved for 30 consecutive rounds -- prevents overfitting without
    needing to hand-tune n_estimators precisely.
    """
    model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,   # print eval every 50 rounds
    )
    print(f"\n[train] Best iteration: {model.best_iteration}  "
          f"| Best val RMSE: {model.best_score:.4f}")
    return model


# ---------------------------------------------------------------------------
# 5.  Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label):
    """Compute RMSE and MAE and print a formatted summary line."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"[eval]  {label:12s}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return {"rmse": rmse, "mae": mae}


def feature_importances(model, feature_cols, top_n=10):
    """
    Extract and display XGBoost feature importances (by 'gain').

    'gain' = average improvement in loss brought by a feature across all splits
    where it is used -- more interpretable than 'weight' (split count) for
    sensor data where some sensors split far more often than others.
    """
    importances = model.get_booster().get_score(importance_type="gain")
    # XGBoost names features f0, f1, ... -- map back to real column names
    # Map XGBoost internal keys (f0, f1, ...) back to real column names.
    # We build a lookup dict from the model's own feature_names_in_ when
    # available, otherwise fall back to the positional feature_cols list.
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
# 6.  Mission-readiness flag
# ---------------------------------------------------------------------------

def compute_readiness(predicted_rul, mission_window_hours=DEFAULT_MISSION_WINDOW):
    """
    Return a boolean array: True if the asset has enough predicted RUL to
    safely complete the next mission.

    is_ready = (predicted_RUL > mission_window_hours)

    *mission_window_hours* is the minimum RUL required before committing the
    asset to a mission (default: 20 cycles). Adjust this per mission type --
    a 4-hour sortie vs a 12-hour patrol will need different thresholds.

    Predicted RUL is clipped to 0 before comparison so negative predictions
    (model noise) don't accidentally mark an asset as ready.
    """
    rul_clipped = np.maximum(np.asarray(predicted_rul, dtype=float), 0.0)
    return rul_clipped > mission_window_hours


# ---------------------------------------------------------------------------
# 7.  Save model + predict() API
# ---------------------------------------------------------------------------

def save_model(model, feature_cols, output_dir):
    """Persist the trained model and its feature schema to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    model_path    = os.path.join(output_dir, "rul_model.json")
    features_path = os.path.join(output_dir, "feature_names.json")

    model.save_model(model_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\n[save] Model    -> {model_path}")
    print(f"[save] Features -> {features_path}")


def predict(asset_features, model_dir="./models", mission_window_hours=DEFAULT_MISSION_WINDOW):
    """
    Public inference API -- import and call from other scripts.

    Parameters
    ----------
    asset_features : pd.DataFrame
        One or more rows of raw + rolling features (same schema used at
        training time). Columns not in feature_names.json are ignored.
    model_dir : str
        Directory where rul_model.json and feature_names.json live.
    mission_window_hours : float
        Readiness threshold (cycles); override per mission as needed.

    Returns
    -------
    dict with keys:
        predicted_rul  : np.ndarray  -- predicted RUL per row
        is_ready       : np.ndarray  -- bool, True if RUL > mission_window
        top_features   : pd.DataFrame -- top-10 feature importances (gain)

    Example
    -------
    >>> from train_model import predict
    >>> result = predict(my_feature_df, mission_window_hours=30)
    >>> print(result["predicted_rul"], result["is_ready"])
    """
    model_path    = os.path.join(model_dir, "rul_model.json")
    features_path = os.path.join(model_dir, "feature_names.json")

    # Load model and expected feature schema
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(features_path) as f:
        feature_cols = json.load(f)

    # Align columns: keep only what the model expects, in training order
    X = asset_features.reindex(columns=feature_cols, fill_value=0.0)

    predicted_rul = model.predict(X)
    is_ready      = compute_readiness(predicted_rul, mission_window_hours)
    top_feats     = feature_importances(model, feature_cols, top_n=10)

    return {
        "predicted_rul": predicted_rul,
        "is_ready":      is_ready,
        "top_features":  top_feats,
    }


# ---------------------------------------------------------------------------
# 8.  Final summary
# ---------------------------------------------------------------------------

def print_final_summary(val_metrics, test_metrics, val_preds_df, mission_window_hours):
    """Print a compact results table and 5 sample asset predictions."""
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Validation   RMSE : {val_metrics['rmse']:.2f} cycles")
    print(f"  Validation   MAE  : {val_metrics['mae']:.2f} cycles")
    print(f"  Test (FD001) RMSE : {test_metrics['rmse']:.2f} cycles")
    print(f"  Test (FD001) MAE  : {test_metrics['mae']:.2f} cycles")
    print(f"  Mission window    : {mission_window_hours} cycles")
    print("=" * 60)

    # Sample 5 assets (one row per asset -- use their last/most-degraded reading)
    sample = (
        val_preds_df
        .sort_values("flight_hours")
        .groupby("asset_id")
        .last()
        .reset_index()
        [["asset_id", "true_RUL", "predicted_RUL", "is_ready"]]
        .sample(min(5, val_preds_df["asset_id"].nunique()), random_state=42)
    )
    sample["predicted_RUL"] = sample["predicted_RUL"].round(1)
    print("\n  Sample asset predictions (val set, last cycle per asset):")
    print(sample.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# 9.  Main orchestrator
# ---------------------------------------------------------------------------

def main(train_path, test_path, output_dir, mission_window_hours):

    # Step 1: Load
    train_df, test_df = load_data(train_path, test_path)

    # Identify sensor and op-setting columns
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    op_cols         = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_cols     = [c for c in all_sensor_cols if c in train_df.columns]

    # Step 2: Drop low-variance sensors
    train_df, test_df, sensor_cols = drop_low_variance_sensors(
        train_df, test_df, sensor_cols
    )

    # Step 3: Compute rolling features on the full train + test DFs before splitting.
    # Rolling features for an asset only look back within that asset's own history
    # (via groupby), so there is no cross-asset leakage.
    train_df = add_rolling_features(train_df, sensor_cols)
    test_df  = add_rolling_features(test_df,  sensor_cols)

    # Step 3b: TTM zero-shot forecast features (forward-looking analogues of
    # the rolling stats).  Uses the last 52 cycles per asset as context.
    # Falls back gracefully if granite-tsfm is not installed.
    train_df = add_ttm_forecast_features(train_df, sensor_cols)
    test_df  = add_ttm_forecast_features(test_df,  sensor_cols)

    # Step 4: Asset-level train / val split
    train_split, val_split = asset_level_split(train_df)

    # Step 5: Build feature matrices
    X_train, feature_cols = build_feature_matrix(train_split, sensor_cols, op_cols)
    X_val,   _            = build_feature_matrix(val_split,   sensor_cols, op_cols)
    X_test,  _            = build_feature_matrix(test_df,     sensor_cols, op_cols)

    y_train = train_split["RUL"]
    y_val   = val_split["RUL"]
    y_test  = test_df["RUL"]

    print(f"\n[features] Total features used: {len(feature_cols)}")

    # Step 6: Train XGBoost
    print("\n[train] Starting XGBoost training...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Step 7: Evaluate
    print("\n[eval] Computing metrics...")
    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_metrics  = evaluate(y_val.values,  val_preds,  "Val")
    test_metrics = evaluate(y_test.values, test_preds, "Test(FD001)")

    # Step 8: Feature importances
    feature_importances(model, feature_cols, top_n=10)

    # Step 9: Readiness flags
    val_is_ready  = compute_readiness(val_preds,  mission_window_hours)
    test_is_ready = compute_readiness(test_preds, mission_window_hours)

    # Step 10: Save prediction CSVs
    os.makedirs(output_dir, exist_ok=True)

    val_preds_df = val_split[["asset_id", "flight_hours", "RUL"]].copy()
    val_preds_df = val_preds_df.rename(columns={"RUL": "true_RUL"})
    val_preds_df["predicted_RUL"] = val_preds
    val_preds_df["is_ready"]      = val_is_ready
    val_preds_df.to_csv(os.path.join(output_dir, "predictions_val.csv"), index=False)

    test_preds_df = test_df[["asset_id", "flight_hours", "RUL"]].copy()
    test_preds_df = test_preds_df.rename(columns={"RUL": "true_RUL"})
    test_preds_df["predicted_RUL"] = test_preds
    test_preds_df["is_ready"]      = test_is_ready
    test_preds_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)

    print(f"[save] Predictions -> {output_dir}/predictions_val.csv & predictions_test.csv")

    # Step 11: Save model
    save_model(model, feature_cols, output_dir)

    # Step 12: Final summary
    print_final_summary(val_metrics, test_metrics, val_preds_df, mission_window_hours)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C-MAPSS RUL model trainer")
    parser.add_argument("--train",          default="./output_real/train_FD001_merged.parquet")
    parser.add_argument("--test",           default="./output_real/test_FD001_merged.parquet")
    parser.add_argument("--output",         default="./models")
    parser.add_argument("--mission_window", type=float, default=DEFAULT_MISSION_WINDOW,
                        help="Minimum predicted RUL (cycles) for an asset to be ready")
    args = parser.parse_args()

    main(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        mission_window_hours=args.mission_window,
    )

