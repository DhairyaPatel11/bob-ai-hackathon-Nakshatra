"""
download_ims.py
===============
Download the NASA IMS Bearing Dataset (Test Set 1 only) from its public
Kaggle mirror and unpack it to data_real/ims_bearing/.

Dataset reference:
    J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
    "IMS, University of Cincinnati. 'Bearing Data Set', NASA Ames Prognostics
    Data Repository." NASA Ames Research Center, Moffett Field, CA.
    https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

Dataset details:
    - 4 bearings mounted on a shaft, monitored simultaneously at 20 kHz
    - Test Set 1: bearing 3 and 4 failed at end of run (~34 million samples total)
    - Each raw file is one snapshot (~1 second @ 20 480 samples/snapshot)
    - Filenames encode the timestamp: YYYY.MM.DD.HH.mm.ss

Usage:
    pip install kaggle
    kaggle datasets download -d vinayak123tyagi/bearing-dataset --path ./data_raw_ims
    python download_ims.py

    OR (if kaggle API is configured):
    python download_ims.py          # will call kaggle CLI internally

Raw files land in:
    data_real/ims_bearing/1st_test/    (Test Set 1)

The script is idempotent: if the target folder already contains files it
prints a skip message and exits.
"""

import os
import shutil
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Only pull Test Set 1 — it has the most complete run-to-failure story
# for bearings 3 & 4, and a second full test run is not needed here.
KAGGLE_DATASET  = "vinayak123tyagi/bearing-dataset"
DOWNLOAD_DIR    = "./data_raw_ims_zip"          # temporary staging area
TARGET_DIR      = "./data_real/ims_bearing"     # final location
TEST_SET_SUBDIR = "1st_test"                    # subfolder inside the zip
EXPECTED_SUBDIR = os.path.join(TARGET_DIR, TEST_SET_SUBDIR)

MIN_FILES_EXPECTED = 100    # sanity check: Test Set 1 has ~984 snapshot files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _already_downloaded() -> bool:
    """Return True if the target folder already has enough snapshot files."""
    if not os.path.isdir(EXPECTED_SUBDIR):
        return False
    files = [f for f in os.listdir(EXPECTED_SUBDIR) if not f.startswith(".")]
    if len(files) >= MIN_FILES_EXPECTED:
        print(f"[skip] {EXPECTED_SUBDIR} already contains {len(files)} files — nothing to do.")
        return True
    return False


def _call_kaggle_cli() -> str:
    """
    Invoke the kaggle CLI to download the dataset zip.

    Requires:
        pip install kaggle
        ~/.kaggle/kaggle.json  (API token from https://www.kaggle.com/settings)

    Returns the path to the downloaded zip file.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"[download] Downloading '{KAGGLE_DATASET}' via kaggle CLI …")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "--path", DOWNLOAD_DIR,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[ERROR] kaggle CLI failed:")
        print(result.stderr)
        print("\nManual download instructions:")
        print("  1. Create a Kaggle account and go to Account → API → Create New Token")
        print("     to get ~/.kaggle/kaggle.json")
        print("  2. pip install kaggle")
        print(f"  3. kaggle datasets download -d {KAGGLE_DATASET} --path {DOWNLOAD_DIR}")
        print("  4. Re-run this script.")
        sys.exit(1)

    # Find the downloaded zip
    zips = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    if not zips:
        print(f"[ERROR] No zip file found in {DOWNLOAD_DIR} after download.")
        sys.exit(1)
    return os.path.join(DOWNLOAD_DIR, zips[0])


def _unpack(zip_path: str) -> None:
    """
    Extract only the 1st_test subfolder from the zip to TARGET_DIR.

    The Kaggle archive layout is:
        bearing-dataset.zip
        ├── 1st_test/
        │   ├── 2003.10.22.12.06.24
        │   ├── ...
        ├── 2nd_test/
        └── 3rd_test/

    We extract only 1st_test/ to keep disk usage reasonable.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"[unpack] Extracting '{TEST_SET_SUBDIR}' from {zip_path} …")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(TEST_SET_SUBDIR + "/")]
        if not members:
            # Some kaggle zips nest differently — fall back to extracting everything
            print(f"[warn] Could not find '{TEST_SET_SUBDIR}/' prefix — extracting all.")
            members = zf.namelist()
        zf.extractall(TARGET_DIR, members=members)

    n_extracted = len([
        f for f in os.listdir(EXPECTED_SUBDIR)
        if os.path.isfile(os.path.join(EXPECTED_SUBDIR, f))
    ]) if os.path.isdir(EXPECTED_SUBDIR) else 0

    print(f"[unpack] Extracted {n_extracted} files to {EXPECTED_SUBDIR}")
    if n_extracted < MIN_FILES_EXPECTED:
        print(f"[warn] Expected at least {MIN_FILES_EXPECTED} files; got {n_extracted}.")
        print("       The archive may have a different internal layout.")
        print("       Please verify the contents of:", EXPECTED_SUBDIR)


def _cleanup() -> None:
    """Remove the temporary staging directory."""
    if os.path.isdir(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
        print(f"[clean] Removed staging directory {DOWNLOAD_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== NASA IMS Bearing Dataset — Download Script ===\n")

    if _already_downloaded():
        return

    zip_path = _call_kaggle_cli()
    _unpack(zip_path)
    _cleanup()

    print(f"\n[done] Raw IMS snapshot files are in: {EXPECTED_SUBDIR}")
    print("       Run etl_ims.py next to build the feature table.\n")


if __name__ == "__main__":
    main()
