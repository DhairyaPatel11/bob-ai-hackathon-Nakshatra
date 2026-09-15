"""
forecast.py
===========
Zero-shot sensor forecasting using IBM's Granite Time Series model
(TinyTimeMixer / TTM) running fully locally via the granite-tsfm package.

Model used:  ibm-granite/granite-timeseries-ttm-r2  (branch 52-16-ft-l1-r2.1)
  context_length  : 52 cycles   -- compatible with every C-MAPSS FD001 asset
  prediction_length: 16 cycles  -- 16-step ahead forecast per sensor channel

Public API:
    from forecast import forecast_sensors
    df_fcast = forecast_sensors(
        asset_sensor_history = df_last_52_cycles,   # pd.DataFrame, sensor cols
        sensor_cols          = [...],               # column names to forecast
        context_length       = 52,                  # must match model branch
        forecast_length      = 16,
    )
    # Returns a pd.DataFrame with one row and columns like
    # "sensor_4_fmean", "sensor_4_fstd", "sensor_4_fslope" for each sensor.

Design notes
------------
TTM is a channel-independent multivariate model: each sensor is handled as an
independent channel.  We pass all live sensor columns together so the model
can learn cross-channel correlations via its mixing layers.

The raw 16-step forecast array per channel is summarised into three scalar
features per sensor to keep the XGBoost input space manageable:
  - fmean   : mean of the 16 forecast steps  (expected future level)
  - fstd    : std of the 16 forecast steps   (predicted volatility)
  - fslope  : linear slope over the 16 steps (direction of trend)

These are "forward-looking" analogues of the backward-looking rolling stats
already in the feature set, giving XGBoost a view of where each sensor is
HEADING rather than only where it HAS BEEN.

Fallback behaviour
------------------
If the TTM model cannot be loaded (network/GPU unavailable, dependency
mismatch, etc.), forecast_sensors() returns an empty DataFrame (0 feature
columns).  train_model.py detects the empty return and skips TTM features
cleanly, falling back to the rolling-window-only feature set.
"""

import warnings
import logging
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
# 52-cycle context, 16-step prediction, fine-tuned with L1 loss, revision 2.1.
# Every C-MAPSS FD001 asset has >= 128 cycles, so the 52-cycle context window
# is always satisfiable.  The larger 512-cycle branches are NOT compatible with
# this dataset (0 of 100 assets have >= 512 cycles).
TTM_REVISION = "52-16-ft-l1-r2.1"

TTM_CONTEXT_LENGTH  = 52
TTM_FORECAST_LENGTH = 16

# Suppress overly verbose tsfm / transformers logging at import time
logging.getLogger("tsfm_public").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tsfm_public")


# ---------------------------------------------------------------------------
# Lazy model loader (cached so we pay the load cost only once per process)
# ---------------------------------------------------------------------------

_TTM_PIPELINE = None   # module-level cache


def _load_pipeline(sensor_cols: List[str], context_length: int, forecast_length: int):
    """
    Load the TTM model + tsfm pipeline once and cache it.

    Returns the pipeline object, or None if loading fails.
    """
    global _TTM_PIPELINE
    if _TTM_PIPELINE is not None:
        return _TTM_PIPELINE

    try:
        # Imports are deferred so that a missing granite-tsfm installation
        # does not break the rest of the pipeline at import time.
        from tsfm_public import TinyTimeMixerForPrediction                              # noqa: PLC0415
        from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor # noqa: PLC0415
        from tsfm_public.toolkit.time_series_forecasting_pipeline import (              # noqa: PLC0415
            TimeSeriesForecastingPipeline,
        )

        print(f"[TTM] Loading {TTM_MODEL_ID} @ {TTM_REVISION} "
              f"(ctx={context_length}, pred={forecast_length}, "
              f"channels={len(sensor_cols)}) ...")

        model = TinyTimeMixerForPrediction.from_pretrained(
            TTM_MODEL_ID,
            revision=TTM_REVISION,
            num_input_channels=len(sensor_cols),
        )
        model.eval()

        # Build a dummy dataframe to train the preprocessor (scaling + freq token).
        dummy = pd.DataFrame(
            np.zeros((context_length, len(sensor_cols))), columns=sensor_cols
        )
        dummy["timestamp"] = pd.date_range("2020-01-01", periods=context_length, freq="h")

        tsp = TimeSeriesPreprocessor(
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=sensor_cols,
            context_length=context_length,
            prediction_length=forecast_length,
            freq="h",
            scaling=False,   # XGBoost handles its own normalisation
        )
        tsp.train(dummy)

        pipe = TimeSeriesForecastingPipeline(
            model=model,
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=sensor_cols,
            feature_extractor=tsp,
        )
        _TTM_PIPELINE = pipe
        print("[TTM] Model loaded successfully.")
        return pipe

    except Exception as exc:
        print(f"[TTM] WARNING: Could not load TTM model — {exc}")
        print("[TTM] Falling back to rolling-window-only features.")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_sensors(
    asset_sensor_history: pd.DataFrame,
    sensor_cols: List[str],
    context_length: int = TTM_CONTEXT_LENGTH,
    forecast_length: int = TTM_FORECAST_LENGTH,
) -> pd.DataFrame:
    """
    Run TTM zero-shot inference on one asset's recent sensor history.

    Parameters
    ----------
    asset_sensor_history : pd.DataFrame
        Recent cycles for ONE asset, already sorted ascending by flight_hours.
        Must have at least `context_length` rows and contain all `sensor_cols`.
    sensor_cols : list[str]
        Sensor column names to forecast (must be present in asset_sensor_history).
    context_length : int
        Number of past cycles to feed as context (default: 52).
    forecast_length : int
        Number of future cycles to forecast (default: 16).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with summary statistics of the forecast:
          sensor_N_fmean   — mean of the forecast horizon for sensor N
          sensor_N_fstd    — std dev of the forecast horizon
          sensor_N_fslope  — linear slope across the forecast horizon
        Returns an empty DataFrame (0 rows, 0 cols) if TTM is unavailable.
    """
    pipe = _load_pipeline(sensor_cols, context_length, forecast_length)
    if pipe is None:
        return pd.DataFrame()

    # Take the most recent `context_length` rows of sensor data
    ctx = asset_sensor_history[sensor_cols].tail(context_length).copy()
    if len(ctx) < context_length:
        # Pad with first row if the asset history is shorter than context
        pad_rows = context_length - len(ctx)
        pad_df   = pd.concat([ctx.iloc[[0]]] * pad_rows, ignore_index=True)
        ctx = pd.concat([pad_df, ctx], ignore_index=True)

    ctx = ctx.reset_index(drop=True)
    ctx["timestamp"] = pd.date_range("2020-01-01", periods=context_length, freq="h")

    try:
        result = pipe(ctx)
    except Exception as exc:
        print(f"[TTM] Inference error: {exc}")
        return pd.DataFrame()

    # result is a 1-row DataFrame; each sensor column contains a list of
    # forecast_length values.  Summarise each into 3 scalars.
    feat = {}
    steps = np.arange(forecast_length, dtype=float)
    for col in sensor_cols:
        pred_col = f"{col}_prediction"
        if pred_col not in result.columns:
            continue
        vals = result[pred_col].iloc[0]
        if vals is None or (hasattr(vals, "__len__") and len(vals) == 0):
            continue
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        feat[f"{col}_fmean"]  = float(np.mean(arr))
        feat[f"{col}_fstd"]   = float(np.std(arr))
        # Slope via least-squares over the valid steps
        n = len(arr)
        x = steps[:n]
        slope = float(np.polyfit(x, arr, 1)[0]) if n >= 2 else 0.0
        feat[f"{col}_fslope"] = slope

    return pd.DataFrame([feat]) if feat else pd.DataFrame()
