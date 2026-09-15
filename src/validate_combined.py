"""
validate_combined.py
====================
Phase 7 end-to-end validation script.
Runs run_readiness_check() on the combined engine + gearbox fleet and
reports asset counts, readiness breakdown, and queue composition.
"""
import warnings
import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import pandas as pd

# ── Load both datasets ────────────────────────────────────────────────────────
engine_df  = pd.read_parquet("./output_real/train_FD001_merged.parquet")
bearing_df = pd.read_parquet("./output_real/ims_bearing_merged.parquet")

engine_df["asset_type"]  = "engine"
bearing_df["asset_type"] = "gearbox"

combined = pd.concat([engine_df, bearing_df], ignore_index=True)

print(f"Combined fleet: {len(combined):,} rows")
print(f"  Engines  : {engine_df['asset_id'].nunique()} assets ({len(engine_df):,} rows)")
print(f"  Gearboxes: {bearing_df['asset_id'].nunique()} assets ({len(bearing_df):,} rows)")
print()

# ── Run pipeline ──────────────────────────────────────────────────────────────
from pipeline import run_readiness_check
assets, queue, all_map = run_readiness_check(combined)

n_engine  = sum(1 for a in assets if a["asset_type"] == "engine")
n_gearbox = sum(1 for a in assets if a["asset_type"] == "gearbox")
n_ready   = sum(1 for a in assets if a["is_ready"])
n_notready= len(assets) - n_ready
n_eng_nr  = sum(1 for a in assets if not a["is_ready"] and a["asset_type"] == "engine")
n_gb_nr   = sum(1 for a in assets if not a["is_ready"] and a["asset_type"] == "gearbox")

print()
print("=" * 65)
print("  COMBINED FLEET VALIDATION RESULTS")
print("=" * 65)
print(f"  Total assets     : {len(assets)}  ({n_engine} engines + {n_gearbox} gearboxes)")
print(f"  Mission ready    : {n_ready}")
print(f"  Not ready        : {n_notready}  (engines: {n_eng_nr}, gearboxes: {n_gb_nr})")
print(f"  Maintenance queue: {len(queue)} assets ranked")
print()

print("  Top 5 in maintenance queue:")
for q in queue[:5]:
    atype = q.get("asset_type", "?")
    icon  = "eng " if atype == "engine" else "gear"
    crit  = "CRIT" if q.get("mission_critical_flag") else "    "
    print(
        f"    Rank #{q['priority_rank']}  [{icon}]  {crit}  "
        f"ID={q['asset_id']:4d}  "
        f"RUL={q['predicted_rul']:6.1f}  "
        f"score={q['priority_score']:.4f}"
    )

print()
print("  Sample explanations (one engine + one gearbox):")
engine_sample  = next((a for a in assets if a["asset_type"] == "engine"),  None)
gearbox_sample = next((a for a in assets if a["asset_type"] == "gearbox"), None)

for label, sample in [("ENGINE", engine_sample), ("GEARBOX", gearbox_sample)]:
    if sample:
        brief = sample["explanation"][:140]
        print(f"\n  [{label}] Asset {sample['asset_id']}:")
        print(f"    {brief}...")

print()
print("=" * 65)
print("  Bearing model RMSE/MAE (from bearing_predictions_val.csv):")
try:
    bp = pd.read_csv("./models/bearing_predictions_val.csv")
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rmse = float(np.sqrt(mean_squared_error(bp["true_RUL"], bp["predicted_RUL"])))
    mae  = float(mean_absolute_error(bp["true_RUL"], bp["predicted_RUL"]))
    print(f"    Bearing RMSE : {rmse:.2f} snapshot steps")
    print(f"    Bearing MAE  : {mae:.2f} snapshot steps")
    print(f"    (1 step ~10 min -> RMSE ~{rmse*10:.0f} min)")
except Exception as exc:
    print(f"    Could not load bearing predictions: {exc}")
print("=" * 65)
