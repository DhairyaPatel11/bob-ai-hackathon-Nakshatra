"""
etl_cmapss.py
=============
ETL pipeline for NASA C-MAPSS FD001 dataset.
Hackathon context: predictive maintenance for military assets.

Dataset reference:
  Saxena, A. et al. (2008). "Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation." PHM 2008 Conference.

Usage:
  python etl_cmapss.py --data_dir ./data --output_dir ./output

Expected input files (in --data_dir):
  train_FD001.txt, test_FD001.txt, RUL_FD001.txt
"""

import argparse
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Constants / column schema
# ---------------------------------------------------------------------------

# The raw C-MAPSS files have 26 space-separated columns with NO header row.
COLUMN_NAMES = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# We rename two columns for a military-asset narrative.
RENAME_MAP = {
    "unit_number": "asset_id",
    "time_in_cycles": "flight_hours",  # cumulative flight hours at this measurement
}

# Candidate parts for the synthetic service-record table.
PART_LIST = [
    "turbine_bearing",
    "fuel_pump",
    "compressor_blade",
    "sensor_module",
    "none",  # "none" = no part replaced at last service — a red-flag for degraded assets
]

# Reference date for all synthetic date arithmetic.
REFERENCE_DATE = date(2026, 9, 12)   # today's date at time of writing

# Random seed for reproducibility.
RNG_SEED = 42
rng = random.Random(RNG_SEED)
np_rng = np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# 1.  Load raw C-MAPSS files
# ---------------------------------------------------------------------------

def load_raw(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read train, test, and ground-truth RUL files from *data_dir*.

    All three files are plain-text, space-delimited, no header.
    RUL_FD001.txt has one value per line: the RUL of each test unit
    at the last recorded cycle (i.e., the 'cutoff' point).
    """
    def _read_sensor_file(fname):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(
            path,
            sep=r"\s+",      # one or more whitespace chars — handles multiple spaces
            header=None,
            names=COLUMN_NAMES,
            index_col=False,
        )
        # Drop any trailing NaN columns that can appear from a trailing space
        df.dropna(axis=1, how="all", inplace=True)
        return df

    train_df = _read_sensor_file("train_FD001.txt")
    test_df  = _read_sensor_file("test_FD001.txt")

    rul_path = os.path.join(data_dir, "RUL_FD001.txt")
    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul_at_cutoff"],
    )
    # Assign a 1-based unit index so we can join it to test_df's unit_number
    rul_df["unit_number"] = rul_df.index + 1

    print(f"[load]  train rows: {len(train_df):,}  |  test rows: {len(test_df):,}  |  RUL entries: {len(rul_df)}")
    return train_df, test_df, rul_df


# ---------------------------------------------------------------------------
# 2.  Compute RUL labels
# ---------------------------------------------------------------------------

def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engine unit in the training set, the last observed cycle is
    treated as the failure point (RUL = 0 at that row).

    RUL for row i of unit u  =  max_cycle(u) − cycle(u, i)

    This is the standard approach for the C-MAPSS dataset because the training
    trajectories run all the way to failure.
    """
    # Per-unit maximum cycle (failure cycle)
    max_cycle = (
        train_df.groupby("unit_number")["time_in_cycles"]
        .transform("max")
    )
    # RUL counts DOWN to 0 at the failure row
    train_df["RUL"] = max_cycle - train_df["time_in_cycles"]
    return train_df


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test trajectories are TRUNCATED — they stop before failure.
    RUL_FD001.txt gives the true RUL of each unit at its last observed cycle.

    For any earlier row we back-compute:
        RUL(row) = rul_at_cutoff  +  (max_cycle_in_test − current_cycle)

    Because within the test file the units go from cycle 1 up to some cutoff,
    (max_cycle_in_test − current_cycle) is exactly how many cycles remain
    before the cutoff, and rul_at_cutoff is then appended on top.
    """
    # Merge the ground-truth RUL per unit
    test_df = test_df.merge(rul_df, on="unit_number", how="left")

    # Maximum observed cycle for each test unit (= cutoff cycle)
    cutoff_cycle = (
        test_df.groupby("unit_number")["time_in_cycles"]
        .transform("max")
    )

    # Back-compute RUL for every row
    test_df["RUL"] = test_df["rul_at_cutoff"] + (cutoff_cycle - test_df["time_in_cycles"])
    test_df.drop(columns=["rul_at_cutoff"], inplace=True)

    return test_df


# ---------------------------------------------------------------------------
# 3.  Rename columns for military-asset narrative
# ---------------------------------------------------------------------------

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename unit_number → asset_id and time_in_cycles → flight_hours.
    All downstream code uses these names.
    """
    return df.rename(columns=RENAME_MAP)


# ---------------------------------------------------------------------------
# 4.  Synthesise service_records table
# ---------------------------------------------------------------------------

def build_service_records(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one synthetic service-record row per asset_id.

    Correlation design (tune via the constants below):
    ───────────────────────────────────────────────────
    We divide assets into three degradation tiers based on their *final* RUL
    (the RUL at the last recorded measurement, i.e., the most-degraded snapshot):

        HEALTHY   : final_RUL > HIGH_RUL_THRESHOLD   (engine still has lots of life)
        MODERATE  : MID_RUL_THRESHOLD < final_RUL ≤ HIGH_RUL_THRESHOLD
        CRITICAL  : final_RUL ≤ MID_RUL_THRESHOLD    (engine near failure)

    Synthetic correlations applied per tier:
      - last_inspection_date  : CRITICAL assets tend to have been inspected LONGER ago
                                 (inspection_gap_days drawn from a higher range).
      - open_discrepancies    : CRITICAL assets have more open write-ups (higher mean).
      - last_part_replaced    : CRITICAL assets more likely have "none" as last replaced
                                 part (maintenance backlog / deferred action).
      - mission_critical_flag : CRITICAL assets have higher probability of being flagged.

    These are intentional, tunable correlations — not physically causal — designed
    so that downstream ML models can learn the interaction between sensor degradation
    and maintenance record features.
    """

    # ── Tunable thresholds ───────────────────────────────────────────────────
    HIGH_RUL_THRESHOLD = 100   # engine is 'healthy'  if final RUL > 100 cycles
    MID_RUL_THRESHOLD  = 40    # engine is 'moderate' if 40 < final RUL ≤ 100
    # engine is 'critical' if final RUL ≤ 40

    # Inspection gap (days since last inspection) ranges by tier
    INSP_GAP = {
        "healthy":  (5,  45),
        "moderate": (20, 75),
        "critical": (50, 120),
    }
    # open_discrepancies (integer 0–5) ranges by tier
    DISC_RANGE = {
        "healthy":  (0, 2),
        "moderate": (0, 3),
        "critical": (1, 5),
    }
    # P(last_part_replaced == "none") by tier —
    # critical assets are more likely to have deferred maintenance
    NONE_PROB = {
        "healthy":  0.10,
        "moderate": 0.25,
        "critical": 0.55,
    }
    # P(mission_critical_flag == True) by tier
    CRIT_FLAG_PROB = {
        "healthy":  0.20,
        "moderate": 0.40,
        "critical": 0.70,
    }
    # P(part_in_stock == False) by tier and by whether last_part_replaced == "none".
    # "none" implies the depot hasn't been proactively ordering parts for this asset.
    OUT_OF_STOCK_PROB = {
        ("healthy",  False): 0.15,   # part known, mostly stocked
        ("healthy",  True):  0.35,   # "none" replaced → less certainty what's needed
        ("moderate", False): 0.25,
        ("moderate", True):  0.50,
        ("critical", False): 0.35,
        ("critical", True):  0.65,
    }
    # Lead-time range (days) when out of stock
    LEAD_TIME_RANGE = (1, 30)
    # ─────────────────────────────────────────────────────────────────────────

    # Compute per-asset summary statistics we'll need
    asset_summary = (
        sensor_df.groupby("asset_id")
        .agg(
            max_flight_hours=("flight_hours", "max"),
            final_RUL=("RUL", "min"),          # min RUL = most-degraded measurement
        )
        .reset_index()
    )

    records = []
    for _, row in asset_summary.iterrows():
        aid          = int(row["asset_id"])
        final_rul    = float(row["final_RUL"])
        max_fh       = float(row["max_flight_hours"])

        # ── Tier assignment ──────────────────────────────────────────────────
        if final_rul > HIGH_RUL_THRESHOLD:
            tier = "healthy"
        elif final_rul > MID_RUL_THRESHOLD:
            tier = "moderate"
        else:
            tier = "critical"

        # ── last_inspection_date ─────────────────────────────────────────────
        gap_lo, gap_hi = INSP_GAP[tier]
        inspection_gap = rng.randint(gap_lo, gap_hi)          # days ago
        last_inspection_date = REFERENCE_DATE - timedelta(days=inspection_gap)

        # ── cumulative_flight_hours ──────────────────────────────────────────
        # Add ±5 % noise around the sensor-derived max flight hours.
        noise_factor = rng_float(0.95, 1.05)
        cumulative_flight_hours = round(max_fh * noise_factor, 1)

        # ── open_discrepancies ───────────────────────────────────────────────
        d_lo, d_hi = DISC_RANGE[tier]
        open_discrepancies = rng.randint(d_lo, d_hi)

        # ── last_part_replaced ───────────────────────────────────────────────
        if rng.random() < NONE_PROB[tier]:
            last_part_replaced = "none"
        else:
            # Pick any non-"none" part with equal probability
            last_part_replaced = rng.choice([p for p in PART_LIST if p != "none"])

        # ── part_replacement_date (before last_inspection_date) ─────────────
        days_before_inspection = rng.randint(1, 180)   # replaced up to 6 months prior
        part_replacement_date = last_inspection_date - timedelta(days=days_before_inspection)

        # ── next_scheduled_maintenance (0–90 days in future) ─────────────────
        days_until_maint = rng.randint(0, 90)
        next_scheduled_maintenance = REFERENCE_DATE + timedelta(days=days_until_maint)

        # ── mission_critical_flag ────────────────────────────────────────────
        mission_critical_flag = rng.random() < CRIT_FLAG_PROB[tier]

        # ── part_in_stock / part_lead_time_days ──────────────────────────────
        # Bias toward out-of-stock when last_part_replaced == "none" (depot
        # hasn't been tracking which consumable this asset needs).
        is_none_part = (last_part_replaced == "none")
        oos_prob = OUT_OF_STOCK_PROB[(tier, is_none_part)]
        part_in_stock = rng.random() >= oos_prob          # True = in stock
        if part_in_stock:
            part_lead_time_days = 0
        else:
            part_lead_time_days = rng.randint(*LEAD_TIME_RANGE)

        records.append(
            {
                "asset_id":                  aid,
                "tier":                      tier,           # keep for diagnostics
                "last_inspection_date":      last_inspection_date,
                "cumulative_flight_hours":   cumulative_flight_hours,
                "last_part_replaced":        last_part_replaced,
                "part_replacement_date":     part_replacement_date,
                "open_discrepancies":        open_discrepancies,
                "next_scheduled_maintenance": next_scheduled_maintenance,
                "mission_critical_flag":     mission_critical_flag,
                "part_in_stock":             part_in_stock,
                "part_lead_time_days":       part_lead_time_days,
            }
        )

    return pd.DataFrame(records)


def rng_float(lo: float, hi: float) -> float:
    """Uniform float in [lo, hi) using the module-level RNG."""
    return rng.random() * (hi - lo) + lo


# ---------------------------------------------------------------------------
# 5.  Merge sensor data and service records
# ---------------------------------------------------------------------------

def merge_tables(sensor_df: pd.DataFrame, svc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join every sensor+RUL row in *sensor_df* with the corresponding
    service-record row in *svc_df* on asset_id.

    The result is a 'fat' row-level table: each measurement row carries both
    the live sensor snapshot AND the latest maintenance record for that asset.
    This is the primary input for downstream ML feature engineering.
    """
    return sensor_df.merge(svc_df, on="asset_id", how="left")


# ---------------------------------------------------------------------------
# 6.  Write parquet outputs
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.parquet")
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"[write] {path}  ({len(df):,} rows, {df.shape[1]} cols)")
    return path


# ---------------------------------------------------------------------------
# 7.  Summary report
# ---------------------------------------------------------------------------

def print_summary(train_merged: pd.DataFrame, test_merged: pd.DataFrame) -> None:
    """Print a concise diagnostic table to stdout."""
    for split_name, df in [("TRAIN", train_merged), ("TEST", test_merged)]:
        n_assets = df["asset_id"].nunique()
        n_rows   = len(df)
        rul      = df["RUL"]
        print(f"\n{'='*55}")
        print(f"  Split : {split_name}")
        print(f"  Assets: {n_assets:,}   |   Rows: {n_rows:,}")
        print(f"  RUL   : min={rul.min():.0f}  mean={rul.mean():.1f}"
              f"  median={rul.median():.0f}  max={rul.max():.0f}  std={rul.std():.1f}")
        print(f"{'='*55}")

    print("\n-- Sample of 5 merged TRAIN rows --")
    sample_cols = ["asset_id", "flight_hours", "RUL",
                   "sensor_1", "sensor_7", "sensor_11",
                   "tier", "open_discrepancies", "mission_critical_flag"]
    print(train_merged[sample_cols].sample(5, random_state=RNG_SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# 8.  Main orchestrator
# ---------------------------------------------------------------------------

def main(data_dir: str, output_dir: str) -> None:

    print(f"\n[config] data_dir={data_dir!r}  output_dir={output_dir!r}")

    # ── Step 1: Load raw files ───────────────────────────────────────────────
    train_raw, test_raw, rul_raw = load_raw(data_dir)

    # ── Step 2: Compute RUL labels ───────────────────────────────────────────
    train_raw = add_train_rul(train_raw)
    test_raw  = add_test_rul(test_raw, rul_raw)

    # ── Step 3: Rename columns for military-asset context ───────────────────
    train_df = rename_columns(train_raw)
    test_df  = rename_columns(test_raw)

    # ── Step 4: Synthesise service records (based on TRAIN assets) ──────────
    # We generate one service table that covers ALL unique asset_ids across
    # both splits.  In practice the test asset_ids happen to overlap with the
    # train asset_ids for FD001, so a single table is sufficient.
    all_sensor_df = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True,
    )
    svc_df = build_service_records(all_sensor_df)
    print(f"[synth] service_records rows: {len(svc_df)}")

    # ── Step 5: Merge sensor + service records ───────────────────────────────
    train_merged = merge_tables(train_df, svc_df)
    test_merged  = merge_tables(test_df,  svc_df)

    # ── Step 6: Write output files ───────────────────────────────────────────
    write_parquet(train_df,     output_dir, "train_FD001_clean")
    write_parquet(test_df,      output_dir, "test_FD001_clean")
    write_parquet(svc_df,       output_dir, "service_records")
    write_parquet(train_merged, output_dir, "train_FD001_merged")
    write_parquet(test_merged,  output_dir, "test_FD001_merged")

    # ── Step 7: Summary ──────────────────────────────────────────────────────
    print_summary(train_merged, test_merged)
    print("\n[done] ETL complete.\n")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C-MAPSS FD001 ETL pipeline")
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Folder containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Destination folder for parquet files",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
