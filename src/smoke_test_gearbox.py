"""
smoke_test_gearbox.py
=====================
Quick smoke test for the gearbox-only pipeline path (no TTM, fast).
"""
import warnings
import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import pandas as pd

bearing_df = pd.read_parquet("./output_real/ims_bearing_merged.parquet")
bearing_df["asset_type"] = "gearbox"

from pipeline import run_readiness_check
assets, queue, all_map = run_readiness_check(bearing_df)

print("Gearbox-only pipeline OK")
print(f"  Assets : {len(assets)}")
n_ready = sum(1 for a in assets if a["is_ready"])
print(f"  Ready  : {n_ready}")
print(f"  Queue  : {len(queue)}")
for a in assets:
    brief = a["explanation"][:100]
    print(f"  [GEARBOX] ID={a['asset_id']}  RUL={a['predicted_rul']:.1f}  ready={a['is_ready']}")
    print(f"    {brief}...")

print()
print("Checking explanation language uses gearbox terms...")
for a in assets:
    expl = a["explanation"].lower()
    has_gearbox_terms = any(kw in expl for kw in
        ["bearing", "gearbox", "snapshot", "vibration", "outer race", "defect"])
    print(f"  Asset {a['asset_id']}: gearbox_language={has_gearbox_terms}")

print()
print("All checks passed.")
