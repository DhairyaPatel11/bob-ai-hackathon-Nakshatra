"""
explain.py
==========
Generates human-readable maintenance-readiness briefings via a local Granite
model served by Ollama (OpenAI-compatible endpoint).  The LLM call is isolated
in call_watsonx() so swapping to IBM watsonx.ai cloud is a one-function change.

Usage (standalone demo):
    python explain.py

Import API:
    from explain import generate_explanation
    text = generate_explanation(
        asset_id=42,
        predicted_rul=18.5,
        is_ready=False,
        top_features=[("sensor_4_rmean10", 547.3), ...],
        service_record={...},
    )
"""

import os
import textwrap


# ---------------------------------------------------------------------------
# 1.  LLM backend config
# ---------------------------------------------------------------------------

# --- Ollama local backend (active) -----------------------------------------
# Ollama serves an OpenAI-compatible endpoint at localhost:11434.
# Pull the model once with:  ollama pull granite3.3:2b
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "granite3.3:2b")

# --- IBM watsonx.ai cloud backend (fill in and swap call_watsonx to use) ----
# Set WATSONX_API_KEY and WATSONX_PROJECT_ID via environment variables or .env.
# NOTE: WATSONX_PROJECT_ID must be a UUID like "a1b2c3d4-..." from the
#       watsonx.ai project URL -- NOT an API key string.
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID   = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")


# ---------------------------------------------------------------------------
# 2.  LLM call  (Ollama local -- identical signature to the watsonx version)
# ---------------------------------------------------------------------------


# Module-level Ollama reachability cache.
# None = untested, True = reachable, False = offline (fail fast, skip network).
# This avoids paying a ~4s TCP-timeout per asset when Ollama is not running.
_OLLAMA_REACHABLE: bool | None = None


def _probe_ollama() -> bool:
    """Test Ollama reachability once; cache the result for the process lifetime."""
    import urllib.request
    global _OLLAMA_REACHABLE
    if _OLLAMA_REACHABLE is not None:
        return _OLLAMA_REACHABLE
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        _OLLAMA_REACHABLE = True
    except Exception:
        _OLLAMA_REACHABLE = False
    return _OLLAMA_REACHABLE


def call_watsonx(prompt: str) -> str:
    """Send *prompt* to local Granite via Ollama and return generated text.

    Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint so the model
    can be swapped without touching any other code.  Falls back to the
    deterministic mock if Ollama is unreachable.

    Fast-fail behaviour: Ollama reachability is probed once per process on the
    first call and cached in _OLLAMA_REACHABLE.  Subsequent calls skip the
    network entirely when Ollama is known to be offline, so a fleet of 100+
    assets does not each pay a ~4s TCP timeout.

    To switch to IBM watsonx.ai cloud instead, replace this function body with:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        model = ModelInference(
            model_id    = WATSONX_MODEL_ID,
            credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL),
            project_id  = WATSONX_PROJECT_ID,   # must be a UUID, not an API key
            params={GenParams.MAX_NEW_TOKENS: 200, GenParams.TEMPERATURE: 0.3,
                    GenParams.STOP_SEQUENCES: ["\n\n"]},
        )
        return model.generate_text(prompt=prompt).strip()
    """
    import json
    import urllib.request

    # Fast-fail if Ollama was already found to be offline
    if not _probe_ollama():
        return _MOCK_SENTINEL

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
        "stop": ["\n\n"],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # Ollama dropped mid-session -- fall back to mock for remaining assets
        global _OLLAMA_REACHABLE
        _OLLAMA_REACHABLE = False
        return _MOCK_SENTINEL


_MOCK_SENTINEL = "__MOCK__"


# ---------------------------------------------------------------------------
# 3.  Mock response generator
# ---------------------------------------------------------------------------

def _mock_granite_response(
    asset_id: int,
    predicted_rul: float,
    is_ready: bool,
    top_features: list,
    service_record: dict,
    asset_type: str = "engine",
) -> str:
    """
    Return a realistic Granite-style briefing built from the structured args
    (not by parsing the prompt text). This is a deterministic template, not
    an LLM -- purely for offline pipeline testing.

    For gearbox assets the sensor evidence is recast in vibration/bearing
    language (defect-frequency energy, kurtosis, etc.) rather than generic
    engine sensor language.  The maintenance-history reasoning is identical
    for both modalities -- only the sensor evidence block changes.

    Replace by simply wiring real credentials into call_watsonx(); the mock
    path is never reached once call_watsonx() returns real text.
    """
    discr     = service_record.get("open_discrepancies", 0)
    last_insp = service_record.get("last_inspection_date", "unknown date")
    part      = service_record.get("last_part_replaced", "unknown")
    next_maint= service_record.get("next_scheduled_maintenance", "TBD")
    crit      = service_record.get("mission_critical_flag", False)
    cum_fh    = service_record.get("cumulative_flight_hours", "N/A")

    crit_note = " This is a mission-critical asset." if crit else ""

    if asset_type == "gearbox":
        # Gearbox path: reference vibration physics, not engine sensor numbers.
        # top_features for a gearbox asset are vibration CIs (rms, kurtosis,
        # bpfo_energy, etc.) -- translate them into plain-English bearing language.
        evidence = _humanize_gearbox_evidence(top_features)
        if predicted_rul <= 20:
            return (
                f"Gearbox asset {asset_id} is NOT mission-ready: only {predicted_rul:.0f} "
                f"snapshot steps of predicted bearing life remain, indicating imminent "
                f"failure. {evidence} "
                f"With {discr} open maintenance discrepancies unresolved since {last_insp}, "
                f"this bearing assembly must be replaced before next operation. "
                f"Confirm part availability for '{part}' and expedite inspection "
                f"ahead of the scheduled maintenance on {next_maint}.{crit_note}"
            )
        elif predicted_rul <= 40:
            return (
                f"Gearbox asset {asset_id} is borderline: {predicted_rul:.0f} snapshot "
                f"steps of predicted bearing life remain — sufficient for limited operation "
                f"but with no safety margin. {evidence} "
                f"Resolve {discr} open discrepancies and restrict to low-load duty cycles "
                f"until the scheduled inspection on {next_maint}.{crit_note}"
            )
        else:
            return (
                f"Gearbox asset {asset_id} is mission-ready with {predicted_rul:.0f} "
                f"snapshot steps of predicted remaining bearing life. {evidence} "
                f"Vibration levels and defect-frequency energy are within healthy operating "
                f"bounds. With {discr} open discrepancies and {cum_fh:.0f} cumulative "
                f"operating hours since last inspection on {last_insp}, this asset is "
                f"cleared for unrestricted tasking.{crit_note}"
            )

    # ── Engine path (unchanged) ───────────────────────────────────────────────
    # Pull the single most-important sensor feature name (human-readable)
    top_sensor = _humanize_feature_name(top_features[0][0]) if top_features else "key sensors"
    top_val    = top_features[0][1] if top_features else 0.0

    if predicted_rul <= 20:
        return (
            f"Asset {asset_id} is NOT mission-ready: with only {predicted_rul:.0f} cycles "
            f"of predicted remaining service life, it is at or near end-of-life and must be "
            f"grounded immediately. {top_sensor} is reading {top_val:.2f}, consistent with "
            f"advanced component wear, and {discr} open maintenance discrepancies remain "
            f"unresolved since the last inspection on {last_insp}. Schedule a depot-level "
            f"inspection before {next_maint} and confirm status of the last replaced part "
            f"('{part}').{crit_note}"
        )
    elif predicted_rul <= 40:
        return (
            f"Asset {asset_id} is borderline: {predicted_rul:.0f} cycles of predicted "
            f"remaining life are available, which exceeds the minimum mission window but "
            f"leaves no buffer for unexpected degradation. {top_sensor} shows a value of "
            f"{top_val:.2f}, indicating measurable wear acceleration over recent cycles, "
            f"and {discr} open discrepancies should be resolved before the next sortie. "
            f"Restrict to low-intensity taskings and prioritise the scheduled maintenance "
            f"on {next_maint}.{crit_note}"
        )
    else:
        return (
            f"Asset {asset_id} is mission-ready with {predicted_rul:.0f} cycles of "
            f"predicted remaining service life remaining well above the mission threshold. "
            f"{top_sensor} is at {top_val:.2f}, within normal operating range, and rolling "
            f"sensor averages show no signs of accelerating degradation over recent flight "
            f"cycles. With {discr} open discrepancies and {cum_fh:.0f} cumulative hours "
            f"logged since last inspection on {last_insp}, this asset is cleared for "
            f"unrestricted tasking.{crit_note}"
        )


# ---------------------------------------------------------------------------
# 4.  Prompt builder
# ---------------------------------------------------------------------------

def _humanize_gearbox_evidence(top_features: list) -> str:
    """
    Convert gearbox vibration feature readings into a human-readable sentence
    citing the actual physical evidence available for this asset.

    The features list contains (feature_name, value) pairs.  For gearbox
    assets the relevant names include rms, kurtosis, crest_factor,
    bpfo_energy, bpfi_energy, bsf_energy, ftf_energy (and their rolling
    variants).  We translate these into plain-English bearing diagnostics.

    Examples:
        bpfo_energy > threshold → "elevated energy at the outer-race defect
            frequency, consistent with early-stage outer race spalling"
        kurtosis > 3            → "impulsive signal character (kurtosis >3)
            indicating localised surface defects"

    Parameters
    ----------
    top_features : list of (str, float)

    Returns
    -------
    str : one-sentence evidence summary.
    """
    if not top_features:
        return "Vibration data available but no dominant feature identified."

    # Build a lookup of raw values for key features
    fdict: dict[str, float] = {}
    for fname, fval in top_features:
        base = fname.split("_rmean")[0].split("_rstd")[0]
        if base not in fdict:
            fdict[base] = fval

    parts = []

    bpfo = fdict.get("bpfo_energy", 0)
    bpfi = fdict.get("bpfi_energy", 0)
    bsf  = fdict.get("bsf_energy",  0)
    ftf  = fdict.get("ftf_energy",  0)
    kurt = fdict.get("kurtosis",    0)
    rms  = fdict.get("rms",         0)
    cf   = fdict.get("crest_factor",0)

    # Identify dominant defect-frequency channel (if any are elevated)
    defect_energies = {"outer race (BPFO)": bpfo, "inner race (BPFI)": bpfi,
                       "rolling element (BSF)": bsf, "cage (FTF)": ftf}
    max_defect, max_val = max(defect_energies.items(), key=lambda x: x[1])

    if max_val > 1e-6:
        parts.append(
            f"Elevated vibration energy at the bearing's {max_defect} defect frequency "
            f"({max_val:.2e} normalised units), consistent with developing surface damage "
            f"on that component"
        )

    if kurt > 4.0:
        parts.append(
            f"impulsive signal character (kurtosis {kurt:.1f}) indicating "
            f"localised bearing surface defects"
        )
    elif kurt > 3.0:
        parts.append(
            f"mildly elevated kurtosis ({kurt:.1f}) suggesting early-stage "
            f"surface irregularities"
        )

    if cf > 4.0:
        parts.append(f"high crest factor ({cf:.1f}) confirming periodic impact events")

    if not parts:
        parts.append(
            f"RMS vibration at {rms:.4f} — no dominant defect-frequency energy detected"
        )

    return "; ".join(parts) + "."


def _build_prompt(
    asset_id: int,
    predicted_rul: float,
    is_ready: bool,
    top_features: list,
    service_record: dict,
    asset_type: str = "engine",
) -> str:
    """
    Construct the structured prompt sent to Granite.

    For gearbox assets the prompt references bearing physics and vibration
    condition indicators; for engine assets it uses the existing sensor
    terminology.  Maintenance-history fields are identical in both cases.

    Design choices:
    - Brief system-role header so Granite stays in the right persona.
    - Sensor/vibration values are humanised (no internal feature-name jargon).
    - Instruction is explicit about length (2-3 sentences), audience
      (non-technical operator), and tone (direct, no hedging).
    """
    readiness_str = "YES" if is_ready else "NO"

    if asset_type == "gearbox":
        readable_fn = _humanize_vibration_feature_name
        asset_label = "bearing/gearbox"
        rul_unit    = "snapshot steps (~10 min each)"
        system_role = "military ground vehicle / rotorcraft gearbox and bearing assembly"
    else:
        readable_fn = _humanize_feature_name
        asset_label = "engine"
        rul_unit    = "flight cycles"
        system_role = "military aviation turbofan engine"

    feature_lines = []
    for rank, (fname, fval) in enumerate(top_features, start=1):
        readable = readable_fn(fname)
        feature_lines.append(f"  {rank}. {readable}: {fval:.3f}")
    features_block = "\n".join(feature_lines) or "  (none available)"

    sr = service_record
    prompt = textwrap.dedent(f"""
    [SYSTEM]
    You are a maintenance analysis assistant for {system_role} assets.
    Write only the briefing text. No bullet points, headers, or caveats.

    [ASSET DATA]
    Asset ID              : {asset_id}  ({asset_label})
    Predicted RUL         : {predicted_rul:.1f} {rul_unit}
    Mission-ready         : {readiness_str}
    Cumulative hours      : {sr.get('cumulative_flight_hours', 'N/A')}
    Mission-critical flag : {"YES" if sr.get('mission_critical_flag') else "NO"}

    Top contributing condition indicators (this asset's actual readings):
{features_block}

    Maintenance history:
      Last inspection          : {sr.get('last_inspection_date', 'N/A')}
      Last part replaced       : {sr.get('last_part_replaced', 'N/A')} (on {sr.get('part_replacement_date', 'N/A')})
      Open discrepancies       : {sr.get('open_discrepancies', 0)}
      Next scheduled maint.    : {sr.get('next_scheduled_maintenance', 'N/A')}

    [INSTRUCTION]
    Write a 2-3 sentence maintenance officer briefing explaining WHY this asset
    is or is not mission-ready. Reference specific sensor/vibration trends and
    maintenance history. Be direct and concrete -- no hedging language. Write
    for a non-technical operator, not an engineer. Translate numbers into
    operational meaning; do not repeat raw values verbatim.

    [BRIEFING]
    """).strip()
    return prompt


def _humanize_feature_name(fname: str) -> str:
    """
    Convert internal engine feature names into plain English labels.

    sensor_4_rmean10  -> "Sensor 4 (10-cycle rolling avg)"
    sensor_11_rstd5   -> "Sensor 11 (5-cycle rolling std dev)"
    sensor_4          -> "Sensor 4 (raw)"
    op_setting_1      -> "Op Setting 1"
    """
    if fname.startswith("op_setting_"):
        return f"Op Setting {fname.split('_')[-1]}"
    for stat, label in [("_rmean", "rolling avg"), ("_rstd", "rolling std dev")]:
        if stat in fname:
            parts  = fname.split(stat)
            sensor = parts[0].replace("sensor_", "Sensor ")
            return f"{sensor} ({parts[1]}-cycle {label})"
    if fname.startswith("sensor_"):
        return f"Sensor {fname.replace('sensor_', '')} (raw)"
    return fname


def _humanize_vibration_feature_name(fname: str) -> str:
    """
    Convert internal vibration feature names into plain English labels.

    bpfo_energy_rmean10 -> "Outer-race energy (10-step rolling avg)"
    kurtosis_rstd5      -> "Kurtosis (5-step rolling std dev)"
    rms                 -> "RMS vibration"
    bpfo_energy         -> "Outer-race defect energy (BPFO)"
    """
    VIBRATION_LABELS = {
        "rms":          "RMS vibration",
        "peak_to_peak": "Peak-to-peak amplitude",
        "kurtosis":     "Kurtosis (impulsiveness)",
        "crest_factor": "Crest factor",
        "bpfo_energy":  "Outer-race defect energy (BPFO)",
        "bpfi_energy":  "Inner-race defect energy (BPFI)",
        "bsf_energy":   "Rolling element defect energy (BSF)",
        "ftf_energy":   "Cage defect energy (FTF)",
    }
    for stat, label in [("_rmean", "rolling avg"), ("_rstd", "rolling std dev")]:
        if stat in fname:
            parts = fname.split(stat)
            base  = parts[0]
            window = parts[1]
            base_label = VIBRATION_LABELS.get(base, base.replace("_", " ").title())
            return f"{base_label} ({window}-step {label})"
    return VIBRATION_LABELS.get(fname, fname.replace("_", " ").title())


# ---------------------------------------------------------------------------
# 5.  Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    asset_id: int,
    predicted_rul: float,
    is_ready: bool,
    top_features: list,
    service_record: dict,
    asset_type: str = "engine",
) -> str:
    """
    Generate a human-readable maintenance-readiness briefing for one asset.

    Parameters
    ----------
    asset_id       : int
    predicted_rul  : float   -- XGBoost-predicted remaining useful life
                               (cycles for engines; snapshot steps for gearboxes)
    is_ready       : bool    -- True if predicted_rul > mission_window
    top_features   : list    -- [(feature_name: str, actual_value: float), ...]
                               Use this ASSET's actual feature values (not global
                               importance scores) so the briefing is asset-specific.
    service_record : dict    -- keys: last_inspection_date, cumulative_flight_hours,
                               last_part_replaced, part_replacement_date,
                               open_discrepancies, next_scheduled_maintenance,
                               mission_critical_flag
    asset_type     : str     -- "engine" (default) or "gearbox".
                               Controls which sensor language the prompt uses.
                               "gearbox" explanations reference bearing defect
                               frequencies and vibration condition indicators
                               rather than generic engine sensor numbers.
                               Maintenance-history reasoning is identical for
                               both modalities.

    Returns
    -------
    str : 2-3 sentence operator briefing
    """
    prompt   = _build_prompt(asset_id, predicted_rul, is_ready, top_features,
                              service_record, asset_type=asset_type)
    response = call_watsonx(prompt)

    # If call_watsonx is still in mock mode, generate structured response
    if response == _MOCK_SENTINEL:
        return _mock_granite_response(
            asset_id, predicted_rul, is_ready, top_features, service_record,
            asset_type=asset_type,
        )

    # Live mode: return Granite's text, stripping any trailing whitespace
    return response.strip()


# ---------------------------------------------------------------------------
# 6.  __main__ demo -- 3 samples from predictions_val.csv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pandas as pd

    # Load artefacts
    val_preds    = pd.read_csv("./models/predictions_val.csv")
    svc_df       = pd.read_parquet("./output_real/service_records.parquet")
    train_merged = pd.read_parquet("./output_real/train_FD001_merged.parquet")

    with open("./models/feature_names.json") as f:
        feature_cols = json.load(f)

    TOP_N = 10
    global_top_features = feature_cols[:TOP_N]   # proxy for per-asset SHAP top-10

    # ── Select 3 representative rows ─────────────────────────────────────────
    # NOT READY  : last cycle of a val asset (all near RUL=0 at end)
    last_per_asset = (
        val_preds.sort_values("flight_hours")
        .groupby("asset_id").last()
        .reset_index()
    )
    # BORDERLINE : a mid-lifecycle row where predicted_RUL is just above threshold
    mid_rows = val_preds[val_preds["predicted_RUL"].between(21, 40)]
    # HEALTHY    : first cycle of the val asset with highest early predicted_RUL
    early_per_asset = (
        val_preds.sort_values("flight_hours")
        .groupby("asset_id").first()
        .reset_index()
    )

    non_ready_row  = last_per_asset.sort_values("predicted_RUL").iloc[0]
    borderline_row = mid_rows.sort_values("predicted_RUL").iloc[len(mid_rows) // 2]
    healthy_row    = early_per_asset.sort_values("predicted_RUL", ascending=False).iloc[0]

    samples = [
        ("NOT READY  (end of life)",  non_ready_row),
        ("BORDERLINE (moderate wear)", borderline_row),
        ("HEALTHY    (mission-ready)", healthy_row),
    ]

    print("=" * 70)
    print("  MAINTENANCE READINESS BRIEFINGS  [Ollama / Granite local]")
    print("=" * 70)

    for label, row in samples:
        aid      = int(row["asset_id"])
        pred_rul = float(row["predicted_RUL"])
        is_rdy   = bool(row["is_ready"])
        fh       = float(row["flight_hours"])

        # ── Fetch this asset's actual sensor values for the top features ─────
        # Use the training-set row closest in flight_hours to the sample row.
        asset_rows = train_merged[train_merged["asset_id"] == aid]
        if not asset_rows.empty:
            nearest_idx = (asset_rows["flight_hours"] - fh).abs().idxmin()
            source_row  = asset_rows.loc[nearest_idx]
        else:
            source_row = row   # fallback: val row itself

        top_feats = [
            (fname, float(source_row[fname]))
            for fname in global_top_features
            if fname in source_row and not pd.isna(source_row[fname])
        ]

        # ── Pull service record ───────────────────────────────────────────────
        svc_row = svc_df[svc_df["asset_id"] == aid]
        service_record = svc_row.iloc[0].to_dict() if not svc_row.empty else {}
        # Ensure date objects are strings
        for k, v in service_record.items():
            if hasattr(v, "strftime"):
                service_record[k] = v.strftime("%Y-%m-%d")

        # ── Generate briefing ─────────────────────────────────────────────────
        briefing = generate_explanation(
            asset_id       = aid,
            predicted_rul  = pred_rul,
            is_ready       = is_rdy,
            top_features   = top_feats,
            service_record = service_record,
        )

        print(f"\n[{label}]")
        print(f"  Asset {aid}  |  Predicted RUL: {pred_rul:.1f} cycles  "
              f"|  Ready: {is_rdy}  |  Flight hrs: {fh:.0f}")
        print("-" * 70)
        for line in textwrap.wrap(briefing, width=70):
            print(line)

    print("\n" + "=" * 70)
    print("  LLM backend: Ollama granite3.3:2b  (localhost:11434)")
    print("  To switch to watsonx.ai cloud: see call_watsonx() docstring.")
    print("=" * 70)
