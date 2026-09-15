"""
pipeline.py
===========
Single orchestration entrypoint for the Fleet Readiness Copilot pipeline.
Supports two asset modalities routed through the same public API:

    asset_type == "engine"  — C-MAPSS turbofan engines (train_model.py)
    asset_type == "gearbox" — IMS bearing / gearbox assets (bearing_model.py)

Public API:
    from pipeline import run_readiness_check
    assets, queue, all_map = run_readiness_check(assets_df)

Parameters
----------
assets_df : pd.DataFrame
    The merged asset dataframe.  May contain engine assets, gearbox assets,
    or both (combined fleet).

    Engine assets must contain:
      - asset_id, flight_hours
      - sensor_N columns, op_setting_N columns
      - RUL
      - Service-record columns: last_inspection_date,
          cumulative_flight_hours, last_part_replaced, part_replacement_date,
          open_discrepancies, next_scheduled_maintenance, mission_critical_flag,
          part_in_stock, part_lead_time_days
      - asset_type == "engine"  (if absent, rows are assumed "engine")

    Gearbox assets must contain:
      - asset_id, flight_hours (snapshot index), timestamp
      - Vibration feature columns: rms, peak_to_peak, kurtosis, crest_factor,
          bpfo_energy, bpfi_energy, bsf_energy, ftf_energy
      - RUL
      - Same service-record columns as above
      - asset_type == "gearbox"

Returns
-------
assets : list[dict]
    One dict per assessed asset.  Every dict contains:
      asset_id, asset_type, predicted_rul, is_ready, flight_hours,
      mission_critical_flag, open_discrepancies,
      next_scheduled_maintenance, last_inspection_date,
      cumulative_flight_hours, last_part_replaced,
      part_in_stock, part_lead_time_days,
      explanation (str), top_features (list of (feature, value) tuples)

queue : list[dict]
    Non-ready assets only, sorted by priority score descending.
    Each dict carries every field from `assets` plus:
      priority_rank, priority_score, score_components

all_map : dict[int, dict]
    asset_id → asset dict for O(1) lookup (covers both ready and not-ready).

Pipeline steps (in order)
--------------------------
1. Split assets_df into engine rows and gearbox rows by asset_type.
2. Engine path: drop low-variance sensors, add rolling-window stats, add TTM
   zero-shot forecast features; call train_model.predict().
3. Gearbox path: add rolling vibration features; call bearing_model.predict().
4. Merge results into a single list of asset dicts.
5. Explanation — call explain.generate_explanation() per asset, passing
   asset_type so the prompt uses modality-appropriate language.
6. Prioritisation — call prioritize.build_maintenance_queue() on all assets.

None of the internals of train_model.py, explain.py, or prioritize.py are
modified by modality routing — this module only orchestrates calls to their
existing functions.
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

# Suppress verbose downstream library output during pipeline runs
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ── Pipeline module imports (internals untouched) ─────────────────────────────
from train_model import (
    predict as _engine_predict,
    add_rolling_features,
    add_ttm_forecast_features,
    drop_low_variance_sensors,
    DEFAULT_MISSION_WINDOW,
    VARIANCE_THRESHOLD,
    ROLLING_WINDOWS,
)
from bearing_model import (
    predict as _bearing_predict,
    add_rolling_features as _bearing_add_rolling,
    build_feature_matrix as _bearing_build_feature_matrix,
    add_rul_labels as _bearing_add_rul,
    VIBRATION_FEATURE_COLS,
    DEFAULT_MISSION_WINDOW as BEARING_MISSION_WINDOW,
)
from explain import generate_explanation
from prioritize import build_maintenance_queue

# Service-record columns that must NOT be passed into the XGBoost feature
# matrix (they belong only to the explanation / prioritisation layers).
_SERVICE_RECORD_COLS = [
    "tier",
    "last_inspection_date",
    "cumulative_flight_hours",
    "last_part_replaced",
    "part_replacement_date",
    "open_discrepancies",
    "next_scheduled_maintenance",
    "mission_critical_flag",
    "part_in_stock",
    "part_lead_time_days",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _process_engine_assets(
    engine_df: pd.DataFrame,
    model_dir: str,
    mission_window: float,
    top_n_features: int,
) -> list[dict]:
    """
    Run the full engine (C-MAPSS) readiness pipeline on *engine_df* and
    return a list of per-asset dicts with asset_type="engine".

    Parameters
    ----------
    engine_df : pd.DataFrame
        Rows where asset_type == "engine" (or inferred).
    model_dir : str
        Directory containing rul_model.json and feature_names.json.
    mission_window : float
        Minimum predicted RUL (cycles) for readiness.
    top_n_features : int
        Number of global top features to resolve per asset.

    Returns
    -------
    list[dict]  — one dict per unique engine asset_id.
    """
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    op_cols         = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_cols     = [c for c in all_sensor_cols if c in engine_df.columns]
    op_cols         = [c for c in op_cols if c in engine_df.columns]

    # Feature engineering
    eng, _, sensor_cols = drop_low_variance_sensors(
        engine_df, engine_df, sensor_cols, threshold=VARIANCE_THRESHOLD,
    )
    eng = add_rolling_features(eng, sensor_cols, windows=ROLLING_WINDOWS)
    eng = add_ttm_forecast_features(eng, sensor_cols)
    eng = eng.sort_values(["asset_id", "flight_hours"]).reset_index(drop=True)

    result = _engine_predict(
        asset_features       = eng,
        model_dir            = model_dir,
        mission_window_hours = mission_window,
    )
    eng["_predicted_rul"] = result["predicted_rul"]
    eng["_is_ready"]      = result["is_ready"]

    global_top_feat_names = list(result["top_features"]["feature"].head(top_n_features))

    last_per_asset = (
        eng.sort_values("flight_hours").groupby("asset_id").last().reset_index()
    )

    return _build_asset_dicts(last_per_asset, global_top_feat_names, "engine")


def _process_gearbox_assets(
    gearbox_df: pd.DataFrame,
    model_dir: str,
    mission_window: float,
    top_n_features: int,
) -> list[dict]:
    """
    Run the bearing vibration readiness pipeline on *gearbox_df* and
    return a list of per-asset dicts with asset_type="gearbox".

    Parameters
    ----------
    gearbox_df : pd.DataFrame
        Rows where asset_type == "gearbox".
    model_dir : str
        Directory containing bearing_rul_model.json and
        bearing_feature_names.json.
    mission_window : float
        Minimum predicted RUL (snapshot steps) for readiness.
    top_n_features : int
        Number of global top features to resolve per asset.

    Returns
    -------
    list[dict]  — one dict per unique gearbox/bearing asset_id.
    """
    # Rolling features on vibration columns
    gb = _bearing_add_rolling(gearbox_df, VIBRATION_FEATURE_COLS, windows=ROLLING_WINDOWS)
    gb = gb.sort_values(["asset_id", "flight_hours"]).reset_index(drop=True)

    result = _bearing_predict(
        asset_features = gb,
        model_dir      = model_dir,
        mission_window = mission_window,
    )
    gb["_predicted_rul"] = result["predicted_rul"]
    gb["_is_ready"]      = result["is_ready"]

    global_top_feat_names = list(result["top_features"]["feature"].head(top_n_features))

    last_per_asset = (
        gb.sort_values("flight_hours").groupby("asset_id").last().reset_index()
    )

    return _build_asset_dicts(last_per_asset, global_top_feat_names, "gearbox")


def _build_asset_dicts(
    last_per_asset: pd.DataFrame,
    global_top_feat_names: list[str],
    asset_type: str,
) -> list[dict]:
    """
    Build the list of per-asset dicts from the last-cycle-per-asset frame.

    Parameters
    ----------
    last_per_asset : pd.DataFrame
        One row per asset (the final / most-degraded snapshot).
    global_top_feat_names : list[str]
        Feature names for which to resolve actual values.
    asset_type : str
        "engine" or "gearbox" — passed through to explanation layer.

    Returns
    -------
    list[dict]
    """
    def _str(v, default=""):
        return str(v)[:10] if hasattr(v, "strftime") else str(v) if v is not None else default

    assets_out = []
    for _, row in last_per_asset.iterrows():
        aid      = int(row["asset_id"])
        pred_rul = float(row["_predicted_rul"])
        is_rdy   = bool(row["_is_ready"])
        fh       = float(row["flight_hours"])

        top_feats = [
            (fname, float(row[fname]))
            for fname in global_top_feat_names
            if fname in row.index and not pd.isna(row[fname])
        ]

        svc_record = {
            "last_inspection_date":       _str(row.get("last_inspection_date",       "")),
            "cumulative_flight_hours":    float(row.get("cumulative_flight_hours",   0)),
            "last_part_replaced":         str(row.get("last_part_replaced",          "")),
            "part_replacement_date":      _str(row.get("part_replacement_date",      "")),
            "open_discrepancies":         int(row.get("open_discrepancies",          0)),
            "next_scheduled_maintenance": _str(row.get("next_scheduled_maintenance", "")),
            "mission_critical_flag":      bool(row.get("mission_critical_flag",      False)),
        }

        explanation = generate_explanation(
            asset_id       = aid,
            predicted_rul  = pred_rul,
            is_ready       = is_rdy,
            top_features   = top_feats,
            service_record = svc_record,
            asset_type     = asset_type,
        )

        assets_out.append({
            "asset_id":                   aid,
            "asset_type":                 asset_type,
            "flight_hours":               fh,
            "predicted_rul":              pred_rul,
            "is_ready":                   is_rdy,
            "mission_critical_flag":      svc_record["mission_critical_flag"],
            "open_discrepancies":         svc_record["open_discrepancies"],
            "next_scheduled_maintenance": svc_record["next_scheduled_maintenance"],
            "last_inspection_date":       svc_record["last_inspection_date"],
            "cumulative_flight_hours":    svc_record["cumulative_flight_hours"],
            "last_part_replaced":         svc_record["last_part_replaced"],
            "part_in_stock":              bool(row.get("part_in_stock",        True)),
            "part_lead_time_days":        int(row.get("part_lead_time_days",   0)),
            "explanation":                explanation,
            "top_features":               top_feats,
        })
    return assets_out


# ── Public API ────────────────────────────────────────────────────────────────

def run_readiness_check(
    assets_df: pd.DataFrame,
    model_dir: str = "./models",
    mission_window: float = DEFAULT_MISSION_WINDOW,
    top_n_features: int = 10,
) -> tuple[list, list, dict]:
    """
    Full end-to-end readiness assessment for all assets in *assets_df*.

    Supports mixed fleets: engine assets are processed by the C-MAPSS
    XGBoost model; gearbox assets are processed by the bearing vibration
    model.  Both modalities produce identical output schemas so that
    prioritize.py and the Streamlit dashboard need zero changes.

    Parameters
    ----------
    assets_df : pd.DataFrame
        Merged asset dataframe (see module docstring for required columns).
        Must contain an "asset_type" column ("engine" or "gearbox").
        Rows without asset_type are assumed "engine" for backward compatibility.
    model_dir : str
        Directory containing model artefacts for both modalities.
    mission_window : float
        Minimum predicted RUL for an engine asset to be classified as ready.
        Gearbox assets use BEARING_MISSION_WINDOW from bearing_model.py.
    top_n_features : int
        How many global top features to resolve per-asset values for.

    Returns
    -------
    (assets, queue, all_map)  — see module docstring for field details.
    """
    assets_df = assets_df.copy()

    # Ensure asset_type column exists; default to "engine" for backward compat
    if "asset_type" not in assets_df.columns:
        assets_df["asset_type"] = "engine"

    engine_df  = assets_df[assets_df["asset_type"] == "engine"].copy()
    gearbox_df = assets_df[assets_df["asset_type"] == "gearbox"].copy()

    assets_out: list[dict] = []

    # ── Engine path ───────────────────────────────────────────────────────────
    if not engine_df.empty:
        assets_out.extend(
            _process_engine_assets(engine_df, model_dir, mission_window, top_n_features)
        )

    # ── Gearbox path ──────────────────────────────────────────────────────────
    if not gearbox_df.empty:
        assets_out.extend(
            _process_gearbox_assets(
                gearbox_df, model_dir, BEARING_MISSION_WINDOW, top_n_features
            )
        )

    # ── Build the ranked maintenance queue ────────────────────────────────────
    queue = build_maintenance_queue(assets_out)

    # O(1) lookup map covering ALL assets (ready + not-ready)
    all_map = {a["asset_id"]: a for a in assets_out}

    return assets_out, queue, all_map
