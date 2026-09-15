"""
etl_ims.py
==========
ETL pipeline for the NASA IMS Bearing Dataset.

Produces:
    output_real/ims_bearing_features.parquet   — per-snapshot vibration
                                                  feature table with RUL labels
    output_real/ims_service_records.parquet    — one synthetic service record
                                                  per bearing asset
    output_real/ims_bearing_merged.parquet     — features + service records joined

Dataset reference:
    J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
    "IMS, University of Cincinnati. 'Bearing Data Set', NASA Ames Prognostics
    Data Repository." NASA Ames Research Center, Moffett Field, CA.

Usage:
    python etl_ims.py [--data_dir ./data_real/ims_bearing/1st_test]
                      [--output_dir ./output_real]

If the raw snapshot directory does not exist the script prints instructions
and exits cleanly — it does not crash.
"""

import argparse
import logging
import os
import random
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from vibration_features import (
    IMS_BEARING_GEOMETRY,
    IMS_SAMPLING_RATE,
    IMS_SHAFT_RPM,
    process_bearing_timeseries,
)
from bearing_model import add_rul_labels, VIBRATION_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------

# IMS Test Set 1 channels to process.
# Bearings 3 and 4 (channels 2 and 3) failed in this test run.
# Bearings 1 and 2 survived — they have "early" failure patterns.
# Processing all 4 gives a mixed-health dataset for training.
IMS_CHANNELS = [0, 1, 2, 3]   # → bearing_id 1, 2, 3, 4

# Asset-ID offset to avoid collisions with C-MAPSS asset IDs (1–100).
# IMS bearings get IDs starting from 1001.
IMS_ASSET_ID_OFFSET = 1000

# Synthetic service-record generation
REFERENCE_DATE = date(2026, 9, 12)
RNG_SEED = 43      # different from C-MAPSS (42) for independent reproducibility
rng  = random.Random(RNG_SEED)
np_rng = np.random.default_rng(RNG_SEED)

# Gearbox-appropriate parts list
GEARBOX_PART_LIST = [
    "outer_race_insert",
    "inner_race_sleeve",
    "roller_cage_assembly",
    "bearing_housing_seal",
    "none",
]


# ---------------------------------------------------------------------------
# 1.  Load and process all bearing channels
# ---------------------------------------------------------------------------

def load_all_bearings(snapshot_dir: str) -> pd.DataFrame:
    """
    Process all 4 bearing channels from the IMS snapshot directory and
    concatenate them into one feature table.

    Each bearing gets a unique asset_id = IMS_ASSET_ID_OFFSET + bearing_id.

    Parameters
    ----------
    snapshot_dir : str
        Path to the directory of IMS snapshot files (e.g. 1st_test/).

    Returns
    -------
    pd.DataFrame with columns:
        asset_id, bearing_id, timestamp, rms, peak_to_peak, kurtosis,
        crest_factor, bpfo_energy, bpfi_energy, bsf_energy, ftf_energy,
        bpfo_hz, bpfi_hz, bsf_hz, ftf_hz
    """
    dfs = []
    for ch in IMS_CHANNELS:
        df = process_bearing_timeseries(
            snapshot_dir    = snapshot_dir,
            channel         = ch,
            bearing_geometry= IMS_BEARING_GEOMETRY,
            sampling_rate   = IMS_SAMPLING_RATE,
            shaft_rpm       = IMS_SHAFT_RPM,
        )
        if df.empty:
            logger.warning("No data for channel %d — skipping.", ch)
            continue
        df["asset_id"] = IMS_ASSET_ID_OFFSET + df["bearing_id"]
        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No bearing data could be loaded from: " + snapshot_dir
        )

    combined = pd.concat(dfs, ignore_index=True).sort_values(
        ["asset_id", "timestamp"]
    )
    print(
        f"[load]  {len(combined):,} total snapshot rows across "
        f"{combined['asset_id'].nunique()} bearings"
    )
    return combined


# ---------------------------------------------------------------------------
# 2.  Synthesise service records for gearbox/bearing assets
# ---------------------------------------------------------------------------

def _rng_float(lo: float, hi: float) -> float:
    """Uniform float in [lo, hi) using the module-level RNG."""
    return rng.random() * (hi - lo) + lo


def build_service_records(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one synthetic service record per bearing asset (asset_id).

    The same tier-based logic as etl_cmapss.build_service_records() is used,
    with final_RUL (snapshot steps) determining the tier.  This ensures
    prioritize.py can handle both engine and gearbox assets with identical
    schema — the only difference is asset_type = "gearbox".

    Tier thresholds are calibrated to IMS snapshot counts:
        IMS Test Set 1 has ~984 snapshots per bearing.
        HEALTHY   : final_RUL > 400   (>66 hours remaining)
        MODERATE  : 100 < final_RUL ≤ 400
        CRITICAL  : final_RUL ≤ 100   (<17 hours remaining)

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature table with asset_id and RUL columns.

    Returns
    -------
    pd.DataFrame — one row per asset_id with all service-record fields plus
        asset_type = "gearbox".
    """
    HIGH_RUL = 400
    MID_RUL  = 100

    INSP_GAP = {"healthy": (5, 45), "moderate": (20, 75), "critical": (50, 120)}
    DISC_RANGE = {"healthy": (0, 2), "moderate": (0, 3), "critical": (1, 5)}
    NONE_PROB  = {"healthy": 0.10, "moderate": 0.25, "critical": 0.55}
    CRIT_PROB  = {"healthy": 0.20, "moderate": 0.40, "critical": 0.70}
    OOS_PROB   = {
        ("healthy", False): 0.15, ("healthy", True): 0.35,
        ("moderate", False): 0.25, ("moderate", True): 0.50,
        ("critical", False): 0.35, ("critical", True): 0.65,
    }
    LEAD_TIME = (1, 30)

    asset_summary = (
        feature_df.groupby("asset_id")
        .agg(
            max_snapshots=("RUL", "max"),
            final_RUL=("RUL", "min"),
        )
        .reset_index()
    )

    records = []
    for _, row in asset_summary.iterrows():
        aid       = int(row["asset_id"])
        final_rul = float(row["final_RUL"])
        max_snaps = float(row["max_snapshots"])

        tier = "healthy" if final_rul > HIGH_RUL else ("moderate" if final_rul > MID_RUL else "critical")

        lo, hi = INSP_GAP[tier]
        last_insp = REFERENCE_DATE - timedelta(days=rng.randint(lo, hi))

        cum_fh = round(max_snaps * 10 / 60 * _rng_float(0.95, 1.05), 1)  # snapshot→hours

        d_lo, d_hi = DISC_RANGE[tier]
        open_disc = rng.randint(d_lo, d_hi)

        if rng.random() < NONE_PROB[tier]:
            last_part = "none"
        else:
            last_part = rng.choice([p for p in GEARBOX_PART_LIST if p != "none"])

        part_date = last_insp - timedelta(days=rng.randint(1, 180))
        next_maint = REFERENCE_DATE + timedelta(days=rng.randint(0, 90))
        mc_flag = rng.random() < CRIT_PROB[tier]

        is_none = (last_part == "none")
        oos_prob = OOS_PROB[(tier, is_none)]
        in_stock = rng.random() >= oos_prob
        lead_time = 0 if in_stock else rng.randint(*LEAD_TIME)

        records.append({
            "asset_id":                   aid,
            "asset_type":                 "gearbox",
            "tier":                       tier,
            "last_inspection_date":       last_insp,
            "cumulative_flight_hours":    cum_fh,
            "last_part_replaced":         last_part,
            "part_replacement_date":      part_date,
            "open_discrepancies":         open_disc,
            "next_scheduled_maintenance": next_maint,
            "mission_critical_flag":      mc_flag,
            "part_in_stock":              in_stock,
            "part_lead_time_days":        lead_time,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3.  Add a snapshot-step index (flight_hours analogue)
# ---------------------------------------------------------------------------

def add_snapshot_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "flight_hours" column that counts the snapshot index within each
    bearing's run, so that pipeline.py can treat it like C-MAPSS flight_hours.

    Values run from 1 (first snapshot) to N (last snapshot) per bearing.
    This is the bearing analogue of time_in_cycles / flight_hours in C-MAPSS.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with asset_id and timestamp columns, sorted by
        (asset_id, timestamp).

    Returns
    -------
    pd.DataFrame with an added "flight_hours" column (int).
    """
    df = df.sort_values(["asset_id", "timestamp"]).copy()
    df["flight_hours"] = df.groupby("asset_id").cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# 4.  Write parquet outputs
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, output_dir: str, name: str) -> str:
    """Write df to {output_dir}/{name}.parquet and print a confirmation line."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.parquet")
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"[write] {path}  ({len(df):,} rows, {df.shape[1]} cols)")
    return path


# ---------------------------------------------------------------------------
# 5.  Main orchestrator
# ---------------------------------------------------------------------------

def main(snapshot_dir: str, output_dir: str) -> None:
    """
    Run the full IMS bearing ETL pipeline.

    Steps
    -----
    1. Load and feature-extract all 4 bearing channels.
    2. Add RUL labels (snapshot steps to failure).
    3. Add flight_hours snapshot index.
    4. Synthesise per-bearing service records.
    5. Merge and write parquet outputs.

    Parameters
    ----------
    snapshot_dir : str
        Path to the IMS 1st_test snapshot directory.
    output_dir : str
        Destination for parquet files.
    """
    print(f"\n[config] snapshot_dir={snapshot_dir!r}  output_dir={output_dir!r}")

    if not os.path.isdir(snapshot_dir):
        print(
            f"\n[ERROR] Snapshot directory not found: {snapshot_dir}\n"
            "  Run download_ims.py first to download the IMS dataset:\n"
            "      python download_ims.py\n"
            "  Then re-run:\n"
            f"      python etl_ims.py --data_dir {snapshot_dir}\n"
        )
        sys.exit(1)

    # Step 1: Load + feature-extract
    feature_df = load_all_bearings(snapshot_dir)

    # Step 2: RUL labels
    feature_df = add_rul_labels(feature_df)
    print(
        f"[RUL]   min={feature_df['RUL'].min():.0f}  "
        f"mean={feature_df['RUL'].mean():.1f}  "
        f"max={feature_df['RUL'].max():.0f}  (snapshot steps)"
    )

    # Step 3: Snapshot index (flight_hours analogue)
    feature_df = add_snapshot_index(feature_df)

    # Step 4: Synthesise service records
    svc_df = build_service_records(feature_df)
    print(f"[synth] service_records rows: {len(svc_df)}")

    # Step 5: Merge
    merged_df = feature_df.merge(svc_df, on="asset_id", how="left")

    # Step 6: Write outputs
    write_parquet(feature_df, output_dir, "ims_bearing_features")
    write_parquet(svc_df,     output_dir, "ims_service_records")
    write_parquet(merged_df,  output_dir, "ims_bearing_merged")

    # Step 7: Summary
    print(f"\n{'='*55}")
    print(f"  IMS Bearing ETL Complete")
    print(f"  Bearings processed  : {feature_df['asset_id'].nunique()}")
    print(f"  Total snapshot rows : {len(feature_df):,}")
    print(f"  RUL range           : {feature_df['RUL'].min():.0f} – "
          f"{feature_df['RUL'].max():.0f} steps")
    print(f"  Service records     : {len(svc_df)}")
    print(f"{'='*55}\n")

    _print_sample(merged_df)
    print("\n[done] IMS ETL complete.\n")


def _print_sample(df: pd.DataFrame) -> None:
    """Print 5 sample rows from the merged table."""
    sample_cols = [
        "asset_id", "flight_hours", "RUL",
        "rms", "kurtosis", "bpfo_energy",
        "tier", "open_discrepancies", "mission_critical_flag",
    ]
    present = [c for c in sample_cols if c in df.columns]
    print("\n-- Sample of 5 merged IMS rows --")
    print(df[present].sample(min(5, len(df)), random_state=RNG_SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMS Bearing Dataset ETL pipeline")
    parser.add_argument(
        "--data_dir",
        default="./data_real/ims_bearing/1st_test",
        help="Directory containing IMS snapshot files",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_real",
        help="Destination directory for parquet outputs",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
