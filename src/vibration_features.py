"""
vibration_features.py
=====================
Signal-processing module for the NASA IMS Bearing Dataset.

Physical background
-------------------
A rolling-element bearing consists of an inner race (rotates with the shaft),
an outer race (fixed in the housing), rolling elements (balls/rollers), and a
cage that spaces the rolling elements.  When a surface defect develops on any
of these components, it produces an *impulsive* force each time a rolling
element passes over (or impacts) the defect.  The rate of these impacts is
determined by the bearing geometry and shaft speed:

    BPFO (Ball-Pass Frequency, Outer race)  — impacts per revolution of shaft
         = (N/2) * (1 - Bd/Pd * cos(α))
    BPFI (Ball-Pass Frequency, Inner race)
         = (N/2) * (1 + Bd/Pd * cos(α))
    BSF  (Ball Spin Frequency)
         = (Pd / (2*Bd)) * (1 - (Bd/Pd * cos(α))²)
    FTF  (Fundamental Train / Cage Frequency)
         = (1/2) * (1 - Bd/Pd * cos(α))

Where:
    N   = number of rolling elements
    Bd  = ball/roller diameter
    Pd  = pitch circle diameter
    α   = contact angle (radians)

All frequencies above are given in units of *shaft revolutions* (order domain).
Multiply by shaft_rpm / 60 to get Hz.

Diagnostic logic:
    - Elevated energy at BPFO / harmonics → outer race surface defect (spalling,
      pitting).  Most common failure mode in the IMS dataset.
    - Elevated energy at BPFI            → inner race defect.
    - Elevated energy at BSF             → rolling element (ball) defect.
    - Elevated energy at FTF             → cage instability.

Energy is computed via the *envelope spectrum* (Hilbert-transform demodulation)
to reveal the modulation sidebands characteristic of bearing faults, which can
be buried by broadband noise in the raw FFT.

IMS Test Set 1 — bearing geometry (from IMS / University of Cincinnati docs)
------------------------------
    Bearing model: Rexnord ZA-2115 double-row bearing
    Number of rolling elements per row : N  = 16
    Ball diameter                       : Bd = 0.331 in  (8.407 mm)
    Pitch circle diameter               : Pd = 2.815 in  (71.501 mm)
    Contact angle                       : α  = 15.17 degrees

These are the *documented* values from the original IMS dataset paper:
    J. Lee et al. (2007), "Intelligent Prognostics Tools and E-Maintenance",
    Computers in Industry, 57(6), 476-489.
    Blocker: the paper quotes slightly different mm values depending on edition.
    We use the inch values (primary in the Rexnord datasheet) and convert.

Channels: each snapshot file has 4 columns (bearing 1, 2, 3, 4).
Sampling rate: 20 000 Hz (20 kHz).  Every file is one second of data.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Bearing geometry and defect-frequency computation
# ---------------------------------------------------------------------------

@dataclass
class BearingGeometry:
    """
    Physical geometry of a rolling-element bearing.

    All dimensions in *inches* to match the Rexnord ZA-2115 datasheet.
    The contact angle is stored as degrees; internally converted to radians.

    Attributes
    ----------
    n_rollers : int
        Number of rolling elements per row.
    ball_diameter_in : float
        Roller / ball diameter in inches.
    pitch_diameter_in : float
        Pitch circle diameter in inches.
    contact_angle_deg : float
        Contact angle in degrees.
    """
    n_rollers: int
    ball_diameter_in: float
    pitch_diameter_in: float
    contact_angle_deg: float


# Documented geometry for the Rexnord ZA-2115 used in IMS Test Set 1.
# Source: Lee et al. (2007) and Rexnord ZA-2115 datasheet (inch values).
IMS_BEARING_GEOMETRY = BearingGeometry(
    n_rollers          = 16,
    ball_diameter_in   = 0.331,
    pitch_diameter_in  = 2.815,
    contact_angle_deg  = 15.17,
)

# IMS sampling rate (Hz)
IMS_SAMPLING_RATE = 20_000

# IMS shaft speed (RPM) — nominally 2000 RPM throughout all test runs
IMS_SHAFT_RPM = 2000


def compute_defect_frequencies(
    geometry: BearingGeometry,
    shaft_rpm: float,
) -> dict[str, float]:
    """
    Compute the four fundamental bearing defect frequencies in Hz.

    Parameters
    ----------
    geometry : BearingGeometry
        Bearing dimensional parameters (see class docstring for physics).
    shaft_rpm : float
        Shaft rotational speed in revolutions per minute.

    Returns
    -------
    dict with keys: "bpfo_hz", "bpfi_hz", "bsf_hz", "ftf_hz"
        Defect frequencies in Hertz.

    Physical meaning
    ----------------
    BPFO: How often a rolling element strikes an outer-race defect.
          Energy at this frequency (and harmonics 2×, 3×) indicates
          developing or established outer-race spalling.
    BPFI: Same for inner-race defect.  Inner-race faults modulate at
          the shaft frequency, so sidebands at BPFI ± shaft_hz are also
          characteristic.
    BSF:  Ball/roller spin defect frequency.  Typically lower amplitude
          because the ball must make contact with both races to ring.
    FTF:  Cage frequency.  Cage distortion or instability shows energy here.
    """
    alpha_rad = np.deg2rad(geometry.contact_angle_deg)
    N         = geometry.n_rollers
    Bd        = geometry.ball_diameter_in
    Pd        = geometry.pitch_diameter_in

    ratio     = (Bd / Pd) * np.cos(alpha_rad)   # dimensionless bearing ratio
    shaft_hz  = shaft_rpm / 60.0                 # shaft frequency in Hz

    bpfo_order = (N / 2.0) * (1.0 - ratio)
    bpfi_order = (N / 2.0) * (1.0 + ratio)
    bsf_order  = (Pd / (2.0 * Bd)) * (1.0 - ratio ** 2)
    ftf_order  = 0.5 * (1.0 - ratio)

    return {
        "bpfo_hz": bpfo_order * shaft_hz,
        "bpfi_hz": bpfi_order * shaft_hz,
        "bsf_hz":  bsf_order  * shaft_hz,
        "ftf_hz":  ftf_order  * shaft_hz,
    }


# ---------------------------------------------------------------------------
# 2.  Signal I/O
# ---------------------------------------------------------------------------

def load_bearing_signal(
    file_path: str,
    channel: int = 0,
) -> Optional[np.ndarray]:
    """
    Read one IMS snapshot file and return the vibration signal for one bearing.

    IMS snapshot files are plain-text, space-delimited, no header row.
    Each row is one sample; columns correspond to bearings 1–4 (0-indexed).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a single IMS snapshot file.
    channel : int
        Column index (0–3) corresponding to bearings 1–4.

    Returns
    -------
    np.ndarray of shape (n_samples,), dtype float64, or None on error.
        Raw vibration acceleration signal in sensor counts / g-units.

    Notes
    -----
    Files that are missing, empty, or contain non-numeric data are logged
    at WARNING level and return None — the caller should skip these files
    rather than crashing.
    """
    try:
        data = np.loadtxt(file_path, delimiter="\t")
        if data.ndim == 1:
            # Single-column file — treat entire array as one channel
            return data.astype(np.float64)
        if channel >= data.shape[1]:
            logger.warning(
                "Channel %d requested but file has only %d columns: %s",
                channel, data.shape[1], file_path,
            )
            return None
        return data[:, channel].astype(np.float64)
    except Exception as exc:
        logger.warning("Could not load %s: %s", file_path, exc)
        return None


# ---------------------------------------------------------------------------
# 3.  Frequency-domain helpers
# ---------------------------------------------------------------------------

def _band_energy(
    power_spectrum: np.ndarray,
    freqs: np.ndarray,
    center_hz: float,
    bandwidth_hz: float = 10.0,
    n_harmonics: int = 3,
) -> float:
    """
    Sum power spectral density in narrow bands around a defect frequency
    and its first *n_harmonics* harmonics.

    Parameters
    ----------
    power_spectrum : np.ndarray
        One-sided power spectral density (magnitude squared of FFT).
    freqs : np.ndarray
        Frequency axis in Hz, aligned with power_spectrum.
    center_hz : float
        Fundamental defect frequency in Hz.
    bandwidth_hz : float
        Half-width of the integration band around each harmonic.
        Default 10 Hz is appropriate for 20 kHz sampling and ~2000 RPM.
    n_harmonics : int
        Number of harmonics to include (1 = fundamental only, 3 = f, 2f, 3f).

    Returns
    -------
    float  — total energy in the selected bands (normalised by band width).

    Physical motivation
    -------------------
    A bearing fault creates a periodic impulse train at the defect frequency.
    In the frequency domain this appears as a series of discrete tones at
    f, 2f, 3f, …  Summing across harmonics captures the full fault energy
    even when higher harmonics are stronger than the fundamental (common for
    BPFO faults with significant structural resonance amplification).
    """
    total_energy = 0.0
    for h in range(1, n_harmonics + 1):
        f_center = center_hz * h
        mask     = (freqs >= f_center - bandwidth_hz) & (freqs <= f_center + bandwidth_hz)
        if mask.any():
            total_energy += float(np.sum(power_spectrum[mask]))
    # Normalise by total number of band bins to make it comparable across
    # different array lengths (different snapshot file sizes)
    n_bins = sum(
        np.sum(
            (freqs >= center_hz * h - bandwidth_hz) &
            (freqs <= center_hz * h + bandwidth_hz)
        )
        for h in range(1, n_harmonics + 1)
    )
    return total_energy / max(n_bins, 1)


# ---------------------------------------------------------------------------
# 4.  Main feature-extraction function
# ---------------------------------------------------------------------------

def extract_vibration_features(
    signal: np.ndarray,
    sampling_rate: float,
    bearing_geometry: BearingGeometry,
    shaft_rpm: float,
    band_bw_hz: float = 10.0,
    n_harmonics: int = 3,
) -> dict:
    """
    Extract time-domain and frequency-domain condition indicators from a raw
    vibration signal snapshot.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Raw vibration acceleration (single channel, single snapshot).
    sampling_rate : float
        Acquisition sampling rate in Hz (20 000 for IMS data).
    bearing_geometry : BearingGeometry
        Bearing dimensional parameters used to compute defect frequencies.
    shaft_rpm : float
        Shaft rotational speed in RPM at the time of this snapshot.
    band_bw_hz : float
        Half-bandwidth of defect-frequency energy bands (default 10 Hz).
    n_harmonics : int
        Number of harmonics to include in each band-energy computation.

    Returns
    -------
    dict with the following keys:

    Time-domain features
    ---------------------
    rms           : Root-mean-square of the signal. Tracks overall vibration
                    energy; rises as bearing surface degrades.
    peak_to_peak  : Max - min amplitude. Sensitive to impulsive events early
                    in the degradation process.
    kurtosis      : Fourth standardised moment of the signal. Kurtosis > 3
                    (excess kurtosis > 0) indicates impulsive behaviour
                    consistent with localized bearing surface defects.
    crest_factor  : Peak / RMS.  High crest factor = spiky signal relative to
                    average energy — characteristic of early-stage pitting.

    Frequency-domain features (envelope spectrum)
    ----------------------------------------------
    The signal is bandpass-filtered via Hilbert transform to extract the
    *envelope* (modulation signal), which highlights the repetitive impact
    pattern.  Energy is then summed in narrow bands around each defect freq.

    bpfo_energy   : Energy at BPFO and harmonics — outer race defect indicator.
    bpfi_energy   : Energy at BPFI and harmonics — inner race defect indicator.
    bsf_energy    : Energy at BSF  and harmonics — rolling element defect.
    ftf_energy    : Energy at FTF  and harmonics — cage defect indicator.

    Also returned for reference:
    bpfo_hz, bpfi_hz, bsf_hz, ftf_hz : Computed defect frequencies in Hz.
    """
    n = len(signal)

    # ── Time domain ──────────────────────────────────────────────────────────
    rms          = float(np.sqrt(np.mean(signal ** 2)))
    peak_to_peak = float(np.max(signal) - np.min(signal))

    # Kurtosis: scipy-style (normalised by n, no bias correction for large n)
    mean_sig = np.mean(signal)
    std_sig  = np.std(signal)
    if std_sig > 0:
        kurtosis = float(np.mean(((signal - mean_sig) / std_sig) ** 4))
    else:
        kurtosis = 0.0

    crest_factor = float(np.max(np.abs(signal)) / rms) if rms > 0 else 0.0

    # ── Envelope spectrum ────────────────────────────────────────────────────
    # 1. Compute the analytic signal via Hilbert transform
    analytic   = hilbert(signal)
    envelope   = np.abs(analytic)                   # amplitude envelope
    envelope  -= np.mean(envelope)                  # remove DC

    # 2. FFT of the envelope
    fft_env    = np.fft.rfft(envelope, n=n)
    power_env  = (np.abs(fft_env) ** 2) / n        # one-sided power
    freqs      = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # ── Defect frequencies ────────────────────────────────────────────────────
    defect_freqs = compute_defect_frequencies(bearing_geometry, shaft_rpm)

    bpfo_energy = _band_energy(power_env, freqs, defect_freqs["bpfo_hz"], band_bw_hz, n_harmonics)
    bpfi_energy = _band_energy(power_env, freqs, defect_freqs["bpfi_hz"], band_bw_hz, n_harmonics)
    bsf_energy  = _band_energy(power_env, freqs, defect_freqs["bsf_hz"],  band_bw_hz, n_harmonics)
    ftf_energy  = _band_energy(power_env, freqs, defect_freqs["ftf_hz"],  band_bw_hz, n_harmonics)

    return {
        # Time domain
        "rms":          rms,
        "peak_to_peak": peak_to_peak,
        "kurtosis":     kurtosis,
        "crest_factor": crest_factor,
        # Frequency domain (envelope)
        "bpfo_energy":  bpfo_energy,
        "bpfi_energy":  bpfi_energy,
        "bsf_energy":   bsf_energy,
        "ftf_energy":   ftf_energy,
        # Defect frequencies used (for traceability in explanations)
        "bpfo_hz":      defect_freqs["bpfo_hz"],
        "bpfi_hz":      defect_freqs["bpfi_hz"],
        "bsf_hz":       defect_freqs["bsf_hz"],
        "ftf_hz":       defect_freqs["ftf_hz"],
    }


# ---------------------------------------------------------------------------
# 5.  Full bearing time-series processor
# ---------------------------------------------------------------------------

def _parse_ims_timestamp(filename: str) -> Optional[pd.Timestamp]:
    """
    Parse a timestamp from an IMS snapshot filename.

    IMS filenames have the format: YYYY.MM.DD.HH.mm.ss
    Some variants omit the seconds field (YYYY.MM.DD.HH.mm).

    Returns pd.Timestamp or None if parsing fails.
    """
    base = os.path.basename(filename)
    parts = base.split(".")
    try:
        if len(parts) == 6:
            return pd.Timestamp(
                year=int(parts[0]), month=int(parts[1]), day=int(parts[2]),
                hour=int(parts[3]), minute=int(parts[4]), second=int(parts[5]),
            )
        if len(parts) == 5:
            return pd.Timestamp(
                year=int(parts[0]), month=int(parts[1]), day=int(parts[2]),
                hour=int(parts[3]), minute=int(parts[4]),
            )
    except (ValueError, IndexError):
        pass
    return None


def process_bearing_timeseries(
    snapshot_dir: str,
    channel: int,
    bearing_geometry: BearingGeometry = IMS_BEARING_GEOMETRY,
    sampling_rate: float = IMS_SAMPLING_RATE,
    shaft_rpm: float = IMS_SHAFT_RPM,
    band_bw_hz: float = 10.0,
    n_harmonics: int = 3,
) -> pd.DataFrame:
    """
    Process all snapshot files in *snapshot_dir* for one bearing channel and
    return a per-timestamp feature table.

    Parameters
    ----------
    snapshot_dir : str
        Directory containing IMS snapshot files (e.g. data_real/ims_bearing/1st_test/).
    channel : int
        0-based column index: 0=bearing1, 1=bearing2, 2=bearing3, 3=bearing4.
    bearing_geometry : BearingGeometry
        Bearing dimensional parameters (defaults to IMS_BEARING_GEOMETRY).
    sampling_rate : float
        Acquisition sampling rate in Hz.
    shaft_rpm : float
        Shaft speed in RPM.
    band_bw_hz : float
        Half-bandwidth for defect-frequency energy bands in Hz.
    n_harmonics : int
        Number of harmonics to include in band-energy sums.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp    : pd.Timestamp (parsed from filename)
        bearing_id   : int (channel + 1, i.e. 1-based)
        rms, peak_to_peak, kurtosis, crest_factor
        bpfo_energy, bpfi_energy, bsf_energy, ftf_energy
        bpfo_hz, bpfi_hz, bsf_hz, ftf_hz

    Snapshot files that cannot be read are skipped with a WARNING log and
    do NOT raise an exception — the pipeline handles partial data gracefully.

    The returned DataFrame is sorted by timestamp ascending (earliest first).
    """
    snapshot_files = sorted(
        [
            os.path.join(snapshot_dir, f)
            for f in os.listdir(snapshot_dir)
            if os.path.isfile(os.path.join(snapshot_dir, f))
            and not f.startswith(".")
        ]
    )

    if not snapshot_files:
        logger.warning("No snapshot files found in %s", snapshot_dir)
        return pd.DataFrame()

    rows = []
    n_skipped = 0

    for fpath in snapshot_files:
        # Parse timestamp from filename
        ts = _parse_ims_timestamp(fpath)
        if ts is None:
            logger.warning("Could not parse timestamp from filename: %s — skipping", fpath)
            n_skipped += 1
            continue

        # Load one channel
        sig = load_bearing_signal(fpath, channel=channel)
        if sig is None or len(sig) < 128:
            logger.warning("Signal too short or unreadable: %s — skipping", fpath)
            n_skipped += 1
            continue

        # Extract features
        try:
            feats = extract_vibration_features(
                signal          = sig,
                sampling_rate   = sampling_rate,
                bearing_geometry= bearing_geometry,
                shaft_rpm       = shaft_rpm,
                band_bw_hz      = band_bw_hz,
                n_harmonics     = n_harmonics,
            )
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s — skipping", fpath, exc)
            n_skipped += 1
            continue

        rows.append({"timestamp": ts, "bearing_id": channel + 1, **feats})

    if n_skipped:
        logger.info("Skipped %d / %d snapshot files.", n_skipped, len(snapshot_files))

    if not rows:
        logger.warning("No valid snapshots processed for channel %d in %s", channel, snapshot_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(
        f"[vibration] Bearing {channel + 1}: {len(df)} snapshots processed "
        f"({n_skipped} skipped)"
    )
    return df


# ---------------------------------------------------------------------------
# 6.  __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ── Quick geometry sanity check ──────────────────────────────────────────
    freqs = compute_defect_frequencies(IMS_BEARING_GEOMETRY, IMS_SHAFT_RPM)
    print("IMS ZA-2115 defect frequencies @ 2000 RPM:")
    for name, hz in freqs.items():
        print(f"  {name:9s}: {hz:.2f} Hz")

    # ── Test on a single synthetic signal ─────────────────────────────────────
    print("\nRunning feature extraction on a synthetic impulse signal …")
    rng  = np.random.default_rng(42)
    sr   = IMS_SAMPLING_RATE
    t    = np.arange(sr) / sr          # 1 second at 20 kHz
    bpfo = freqs["bpfo_hz"]
    # Synthetic signal: noise + sinusoidal BPFO fault + white noise
    sig = (
        0.5 * np.sin(2 * np.pi * bpfo * t)         # outer-race fault tone
        + 0.3 * rng.standard_normal(sr)             # broadband noise
    )
    feats = extract_vibration_features(
        sig, sr, IMS_BEARING_GEOMETRY, IMS_SHAFT_RPM
    )
    print("Features from synthetic BPFO signal:")
    for k, v in feats.items():
        print(f"  {k:14s}: {v:.6f}")

    # ── Test directory processing if data is present ──────────────────────────
    test_dir = "./data_real/ims_bearing/1st_test"
    if os.path.isdir(test_dir):
        print(f"\nProcessing bearing 3 (channel 2) from {test_dir} …")
        df = process_bearing_timeseries(test_dir, channel=2)
        if not df.empty:
            print(df.head(3).to_string())
    else:
        print(f"\n[skip] {test_dir} not found — run download_ims.py first.")
        sys.exit(0)
